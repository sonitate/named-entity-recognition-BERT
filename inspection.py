import argparse
import glob

from Corpus import load_ann, brat

path='data/PGxCorpus'

"""
sample=glob.glob(path + "/*.ann")[:10]
for ann_id in sample:
    id=ann_id.split('.ann')[0]
    txt_path=id+'.txt'
    ann_path=id+'.ann'
    ann=load_ann(ann_path)
    txt = open(txt_path).read().replace('\n',"")
    print(txt)
    for a in ann:
        print(a,txt[a[0][0]:a[0][1]])
"""

X_app,Y_app,Tokens=brat(path)
X_app,Y_app,Tokens=X_app[:10],Y_app[:10],Tokens[:10]
for token,Y in zip(Y_app,Tokens):
    for t,l in zip(token,Y):
             print(t,l)





