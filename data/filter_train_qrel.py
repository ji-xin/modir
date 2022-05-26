import sys

fname = sys.argv[1]

qrel = []
with open(fname) as fin:
    for line in fin:
        a = line.strip().split('\t')
        if int(a[-1]) > 0:
            a[-1] = '1'
            qrel.append('\t'.join(a))

with open(fname, 'w') as fout:
    for line in qrel:
        print(line, file=fout)
