import argparse
import glob

from Corpus import load_ann, brat

path='Sample'
id='/26481697_2'
txt_path=path+id+'.txt'
ann_path=path+id+'.ann'
ann=load_ann(ann_path)
txt = open(txt_path).read().replace('\n',"")
for a in ann:
    print(a,txt[a[0][0]:a[0][1]])

X_app,Y_app,Tokens=brat(path)
print(Y_app)



