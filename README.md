# MoDIR: Zero-Shot Dense Retrieval with Momentum Adversarial Domain Invariant Representations
Ji Xin, Chenyan Xiong, Ashwin Srinivasan, Ankita Sharma, Damien Jose, Paul Bennett

This repo provides the code for reproducing the experiments in [Zero-Shot Dense Retrieval with Momentum Adversarial Domain Invariant Representations](https://aclanthology.org/2022.findings-acl.316/). Please read the paper for the background.

## Installing Packages

Create your own conda environment, and then install [pytorch](https://pytorch.org/), e.g.:

`conda install pytorch cudatoolkit=11.3 -c pytorch`

For training with fp16, install apex from https://github.com/NVIDIA/apex. If errors occur with apex, try to build pytorch from source so that Apex can be built with CUDA and C++ extensions (the CUDA version must be exactly the same between pytorch and apex).

Then run:

`pip install pandas transformers==2.3.0 pytrec_eval faiss-cpu wget scikit-learn tensorboardX tokenizers=0.9.2 jupyter`

## Data Preparation

There are 3 parts: source domain data download, target domain data download, and preprocessing both domains' data.

### Source Domain Data Download

Download the source domain (MS MARCO) raw data:

`bash commands/data_download.sh`

### Target Domain Data Download

Check https://www.github.com/ji-xin/get-beir-data for instructions. Remember to copy the generated files back to this repo.

### Preprocessing Both Domains' Data

First:

`cd data`

For preprocessing msmarco:

`bash preprocess_marco.sh`

For preprocessing target domain dataset (treccovid is used as the example here):

`bash preprocess_beir.sh treccovid`

We have also created a dataset called `tinymsmarco`, which can be useful for debugging. [Download](https://jxin.blob.core.windows.net/release/datasets/tinymsmarco.zip) and put it into `data/tinymsmarco`.

The `treccovid` dataset after our processing, including `marco-format`, `preprocessed_data`, and `triples.simple.tsv`, can be downloaded at https://jxin.blob.core.windows.net/release/datasets/treccovid.zip.

## Download Checkpoints

These checkpoints are useful for continuing training and/or inference. Download, unzip, and put them into `checkpoints`.

ANCE warmup (which is essentially DPR): https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/warmup_checpoint.zip

ANCE passage: https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/Passage_ANCE_FirstP_Checkpoint.zip

MoDIR-ANCE (treccovid, 10k steps): https://jxin.blob.core.windows.net/release/checkpoints/modir-treccovid-10k.zip

MoDIR-ANCE (treccovid, 50k steps): https://jxin.blob.core.windows.net/release/checkpoints/modir-treccovid-50k.zip

## Run Inference and Evaluation

`cd commands`, and then

`bash inference.sh`.

Check `inference.sh` for selections of checkpoint and dataset. This script generates the embeddings for each query and passage. We can then use the jupyter notebook `evaluation/metrics.ipynb` (change paths inside the notebook if necessary) to compute the metrics.

`cqa_inference.sh` is for the `cqadupstack` dataset only, which consists of multiple subsets. There's also a special part for it in the notebook.

## Training MoDIR based on DPR

`cd commands`, and then

`bash modir_dpr.sh`

## Training MoDIR based on ANCE

Step 1:

Generate `first_ann` data for the source domain (used for ANCE's negative sampling). This step is irrelevant to the target domain and can be used for any target domain dataset. Then run:

`cd commands && bash first_ann_gen.sh`

Step 2:

Continue training MoDIR from ANCE-passage:

`bash modir_ance.sh`

## Citation

Please consider citing out paper:
```
@inproceedings{xin-etal-2022-zero,
    title = "Zero-Shot Dense Retrieval with Momentum Adversarial Domain Invariant Representations",
    author = "Xin, Ji  and
      Xiong, Chenyan  and
      Srinivasan, Ashwin  and
      Sharma, Ankita  and
      Jose, Damien  and
      Bennett, Paul",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.316",
    pages = "4008--4020",
    abstract = "Dense retrieval (DR) methods conduct text retrieval by first encoding texts in the embedding space and then matching them by nearest neighbor search. This requires strong locality properties from the representation space, e.g., close allocations of each small group of relevant texts, which are hard to generalize to domains without sufficient training data. In this paper, we aim to improve the generalization ability of DR models from source training domains with rich supervision signals to target domains without any relevance label, in the zero-shot setting. To achieve that, we propose Momentum adversarial Domain Invariant Representation learning (MoDIR), which introduces a momentum method to train a domain classifier that distinguishes source versus target domains, and then adversarially updates the DR encoder to learn domain invariant representations. Our experiments show that MoDIR robustly outperforms its baselines on 10+ ranking datasets collected in the BEIR benchmark in the zero-shot setup, with more than 10{\%} relative gains on datasets with enough sensitivity for DR models{'} evaluation. Source code is available at https://github.com/ji-xin/modir.",
}
```
