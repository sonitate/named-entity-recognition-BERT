import glob
import re
import string
import jsonlines
import csv
import torch

from Embedding import Text2tokens, tokenizer, get_bert_inputs

Entity_types={
'Chemical' : 1,
'Genomic_factor' : 2 ,
'Gene_or_protein' : 3 ,
'Genomic_variation' : 4 ,
'Limited_variation' : 5 ,
'Haplotype' : 6 ,
'Phenotype' : 7 ,
'Disease' : 8 ,
'Pharmacodynamic_phenotype': 9 ,
'Pharmacokinetic_phenotype': 10
}

classes=[
'O',
'Chemical',
'Genomic_factor',
'Gene_or_protein',
'Genomic_variation',
'Limited_variation',
'Haplotype',
'Phenotype',
'Disease',
'Pharmacodynamic_phenotype',
'Pharmacokinetic_phenotype'
]


def remove(s,l):
    for i in l:
        s = s.replace(i, "")
    return s

punctuation = string.punctuation + '\t'
punctuation1=['(',')',':',';',',','?','!','.','%','*','+','=','"','#','~','@','$','0','1','2','3','4','5','6','7','8','9','^','{','}','-','_']

def string_normaliz(s,mode=0):
    l = punctuation if mode==0 else punctuation1
    s = s.translate(str.maketrans('', '',l))
    return s

def load_txt(txt_path):
    txt = open(txt_path).read().replace('\n',"")
    return txt


def load_ann(ann_path):
    ann = open(ann_path).read().split('\n')
    labels = []
    T = [i for i in ann if i.startswith('T')]
    for i in T:

        label = Entity_types[i.split()[1]]

        end_i=3
        start=i.split()[2]
        end=i.split()[end_i]
        while(';' in end):
            range=[int(start),int(end.split(';')[0])]
            labels.append((range, label))
            start=int(end.split(';')[1])
            end_i+=1
            end = i.split()[end_i]

        range = [int(start), int(end.split(';')[0])]
        labels.append((range, label))


    return labels


def brat(path):
    ann_txt_files = [(f.split(path)[1], (f.split(path)[1]).split('ann')[0] + "txt") for f in glob.glob(path + "/*.ann")]

    Dataset_X = []
    Dataset_Y = []
    Dataset_Tokens = []

    for ann, txt in ann_txt_files:

        sentence = load_txt(path + txt)
        labels = load_ann(path + ann)
        tokens = tokenizer.tokenize(sentence)
        # E range shift
        for w in range(len(sentence)):
            if(sentence[w]==' '):
               for l in labels:
                 if w in range(l[0][0], l[0][1]):
                     l[0][1]-=1
                 if w <l[0][0] :
                     l[0][0]-=1
                     l[0][1]-=1


        target = [0]*len(tokens)
        pointer=0
        i=0
        for t in tokens :
            for l in labels:
                if pointer in range(l[0][0], l[0][1]) and l[1]>target[i]:
                    target[i]=l[1]
            pointer+=len(t.replace('#', ''))
            i+=1

        Dataset_X.append(get_bert_inputs(sentence))
        Dataset_Y.append([0]+target+[0])
        Dataset_Tokens.append(tokens)
        Dataset_Tokens.append(tokens)

    return Dataset_X, Dataset_Y, Dataset_Tokens



def words2IOBES(words_labels_dataset):
    iobes_dataset=[]
    for words in words_labels_dataset:
        iobes=[classes[words[0]] if words[0]==0 else 'B-'+classes[words[0]]]
        for i in range(1,len(words)-1):
            if(words[i]==0):
                iobes.append(classes[words[i]])
            elif(words[i-1]!=words[i]):
                iobes.append('B-'+classes[words[i]])
            elif(words[i+1]!=words[i]):
                iobes.append('E-'+classes[words[i]])
            else:
                iobes.append('I-'+classes[words[i]])

        prefix='E-' if words[-1]==words[-2] else 'B-'
        iobes.append(classes[words[-1]] if words[-1]==0 else prefix + classes[words[-1]])
        iobes_dataset.append(iobes)
    return iobes_dataset



class Corpus():

    def __init__(self, path, name):
        self.Entityes_types = Entity_types
        self.Nb_class = len(Entity_types.keys())+1
        self.path = path
        self.name = name
        self.data = None

    def get_data(self):
        if self.name == 'pgx':
            self.data = brat(self.path)

        return self.data

