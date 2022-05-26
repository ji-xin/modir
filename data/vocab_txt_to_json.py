import os
import sys
import json

vocab_dir = sys.argv[1]

vocab_dict = {}
with open(os.path.join(vocab_dir, "vocab.txt")) as fin:
    for i_line, line in enumerate(fin):
        vocab_dict[line.strip()] = i_line

with open(os.path.join(vocab_dir, "vocab.json"), 'w') as fout:
    print(json.dumps(vocab_dict), file=fout)
