import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class DomainClassifier(nn.Module):
    def __init__(self,
                 args,
                 input_size=768,
                 n_class=2):
        super(DomainClassifier, self).__init__()
        if args.dc_layers == 1:
            layers = [
                nn.Linear(input_size, n_class, bias=False)
            ]
        elif args.dc_layers == 2:
            layers = [
                nn.Linear(input_size, 200),
                nn.ReLU(),
                nn.Linear(200, n_class, bias=False)
            ]
        elif args.dc_layers == 3:
            layers = [
                nn.Linear(input_size, 200),
                nn.ReLU(),
                nn.Linear(200, 200),
                nn.ReLU(),
                nn.Linear(200, n_class, bias=False)
            ]
        else:
            raise NotImplementedError()
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs, labels=None):
        """
        it doens't work for run_warmup_da.py for now
        since it no longer supports lamb and gradient reversal
        """
        x = inputs        
        for layer in self.layers:
            x = layer(x)
        logits = torch.clamp(x, min=-5.0, max=5.0)

        if labels is None:
            return logits
        elif type(labels) is str:
            assert labels == 'uniform'
            return (
                logits,
                self.uniform_loss(logits),
                None,
            )
        else:
            return (
                logits,
                F.cross_entropy(logits, labels),
                self.get_acc(logits, labels)
            )
    
    @staticmethod
    def uniform_loss(logits):
        batch_size = logits.shape[0]
        device = logits.device
        return (
            F.cross_entropy(logits, torch.tensor([0] * batch_size, device=device)) + \
            F.cross_entropy(logits, torch.tensor([1] * batch_size, device=device))
        ) / 2
    
    @staticmethod
    def get_acc(logits, labels):
        preds = torch.argmax(logits, dim=1)
        total = int(len(labels))
        correct = int(sum(labels==preds))
        return (total, correct, correct/total)


class DummyModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DummyModule, self).__init__()
        self.register_parameter(name='dummy', param=nn.Parameter(torch.randn(1)))
    
    def forward(self, inputs, *args, **kwargs):
        pass


def dry_test(model, device, test_dataset):
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    n_datasets = 2
    total, correct = 0, 0
    class_total, class_correct = [0 for _ in range(n_datasets)], [0 for _ in range(n_datasets)]
    for batch_idx, batch in enumerate(test_dataloader):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        total += len(labels)
        correct += sum(labels==preds)
        for class_id in range(n_datasets):
            class_total[class_id] += sum(labels==class_id)
            class_correct[class_id] += sum(torch.logical_and(labels==class_id, preds==class_id))
        
    result_dict = {'total_acc': int(correct)/total}
    for class_id in range(n_datasets):
        result_dict[f'class {class_id} acc'] = int(class_correct[class_id]) / int(class_total[class_id])
    return {k: f'{v:.5f}' for k, v in result_dict.items()}


def dry_dc_evaluation(args, dc_model, query_embs, passage_embs,
                      prev_dc_state_dict):
    
    # we take all queries from both domains
    # and discard passages from one of the domains
    # so that each domain has the same number of vectors (query+passage)

    single_domain_query_size = min([x.shape[0] for x in query_embs])
    single_domain_passage_size = min([x.shape[0] for x in passage_embs])

    srd_query = query_embs[0][:single_domain_query_size]
    srd_passage = passage_embs[0][:single_domain_passage_size]
    tgd_query = np.concatenate([x[:single_domain_query_size] for x in query_embs[1:]])
    tgd_passage = np.concatenate([x[:single_domain_passage_size] for x in passage_embs[1:]])

    train_ratio = 0.7

    srd_query_train_size = int(train_ratio * single_domain_query_size)
    srd_passage_train_size = int(train_ratio * single_domain_passage_size)
    tgd_query_train_size = int(train_ratio * single_domain_query_size) * (len(query_embs) - 1)
    tgd_passage_train_size = int(train_ratio * single_domain_passage_size) * (len(query_embs) - 1)
    train_query_dataset = TensorDataset(
        torch.tensor(np.concatenate(
            [srd_query[:srd_query_train_size],
             tgd_query[:tgd_query_train_size]]
        )),
        torch.tensor(np.concatenate(
            [[0] * srd_query_train_size,
             [1] * tgd_query_train_size]
        ))
    )
    train_passage_dataset = TensorDataset(
        torch.tensor(np.concatenate(
            [srd_passage[:srd_passage_train_size],
             tgd_passage[:tgd_passage_train_size]]
        )),
        torch.tensor(np.concatenate(
            [[0] * srd_passage_train_size,
             [1] * tgd_passage_train_size]
        ))
    )

    srd_query_test_size = single_domain_query_size - srd_query_train_size
    srd_passage_test_size = single_domain_passage_size - srd_passage_train_size
    tgd_query_test_size = single_domain_query_size * (len(query_embs) - 1) - tgd_query_train_size
    tgd_passage_test_size = single_domain_passage_size * (len(query_embs) - 1) - tgd_passage_train_size
    test_query_dataset = TensorDataset(
        torch.tensor(np.concatenate(
            [srd_query[srd_query_train_size:],
             tgd_query[tgd_query_train_size:]]
        )),
        torch.tensor(np.concatenate(
            [[0] * srd_query_test_size,
             [1] * tgd_query_test_size]
        ))
    )
    test_passage_dataset = TensorDataset(
        torch.tensor(np.concatenate(
            [srd_passage[srd_passage_train_size:],
             tgd_passage[tgd_passage_train_size:]]
        )),
        torch.tensor(np.concatenate(
            [[0] * srd_passage_test_size,
             [1] * tgd_passage_test_size]
        ))
    )

    if prev_dc_state_dict is not None:
        prev_dc_model = DomainClassifier(args)
        prev_dc_model.to(args.device)
        prev_dc_model.load_state_dict(prev_dc_state_dict)
        prev_test_query_results = dry_test(prev_dc_model, args.device, test_query_dataset)
        prev_test_passage_results = dry_test(prev_dc_model, args.device, test_passage_dataset)
    else:
        prev_test_query_results = {'total_acc': None}
        prev_test_passage_results = {'total_acc': None}

    optimizer = torch.optim.Adam(dc_model.parameters(), lr=5e-4)
    # if args.fp16:
    #     try:
    #         from apex import amp
    #     except ImportError:
    #         raise ImportError(
    #             "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #     dc_model, optimizer = amp.initialize(
    #         dc_model, optimizer, opt_level=args.fp16_opt_level)

    step = 0
    total_step = 500  # actually it's 50 query and 50 passage, so 100 steps in total
    train_query_dataloader = DataLoader(train_query_dataset, batch_size=48, shuffle=True)
    train_passage_dataloader = DataLoader(train_passage_dataset, batch_size=48, shuffle=True)
    query_iterator = iter(train_query_dataloader)
    passage_iterator = iter(train_passage_dataloader)
    while step < total_step:
        try:
            query_batch = next(query_iterator)
        except StopIteration:
            query_iterator = iter(train_query_dataloader)
            query_batch = next(query_iterator)
        try:
            passage_batch = next(passage_iterator)
        except StopIteration:
            passage_iterator = iter(train_passage_dataloader)
            passage_batch = next(passage_iterator)

        step += 1
        for batch in [query_batch, passage_batch]:
            inputs, labels = batch[0].to(args.device), batch[1].to(args.device)
            outputs = dc_model(inputs)
            optimizer.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            # if args.fp16:
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            loss.backward()
            optimizer.step()
            
    test_query_results = dry_test(dc_model, args.device, test_query_dataset)
    test_passage_results = dry_test(dc_model, args.device, test_passage_dataset)
    return (
        [test_query_results['total_acc'], test_passage_results['total_acc']],
        [prev_test_query_results['total_acc'], prev_test_passage_results['total_acc']],
        dc_model.state_dict()
    )
