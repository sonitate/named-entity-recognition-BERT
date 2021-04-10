import argparse
import glob

from Corpus import load_ann, brat

path='Sample'
id='/8834564_2'
txt_path=path+id+'.txt'
ann_path=path+id+'.ann'
ann=load_ann(ann_path)
txt = open(txt_path).read().replace('\n',"")
print(txt)
for a in ann:
    print(a,txt[a[0][0]:a[0][1]])

X_app,Y_app,Tokens=brat(path)
print(Y_app)

print(len(Tokens[0]))
print(len(Y_app[0]))
for t,l in zip(Tokens[0],Y_app[0]):
     print(t,l)





