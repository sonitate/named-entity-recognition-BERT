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

    return Dataset_X, Dataset_Y, Dataset_Tokens


def word_level(labels_dataset, tokens_dataset):
   words_labels_dataset=[]
   for  labels, tokens in zip(labels_dataset, tokens_dataset):
        words = []
        ws, we, i = 0, 0, 0
        for token in tokens:
            if '##' in token:
                we += 1
            else:
                words.append((ws, we))
                ws, we = i, i
            i += 1
        words.append((ws, we))
        words = words[1:]
        words_labels = []
        for ws, we in words:
            words_labels.append(max(labels[ws:we + 1]))
        words_labels_dataset.append(words_labels)
   return words_labels_dataset


def entities_ranges(Y):
    entities,ranges=[],[]
    es, e, i = 0, -1, 0
    for y in Y:
        if(y!=e):
            ranges.append((es,i))
            entities.append(e)
            e,es=y,i
        i+=1
    ranges.append((es, i))
    entities.append(e)
    entities,ranges=entities[1:],ranges[1:]
    return entities,ranges

def exact(pred):
    for p in pred:
        if p!=pred[0]:
          return 0
    return pred[0]

def entity_level(preds,Ys):
    pred_entities, true_entities = [],[]
    for pred,Y in zip(preds,Ys):
        entities, ranges =entities_ranges(Y)
        for e,range in zip(entities,ranges):
             if(e!=0):
                 true_entities.append(e)
                 pred_entities.append(exact(pred[range[0]:range[1]]))
    return pred_entities,true_entities


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

