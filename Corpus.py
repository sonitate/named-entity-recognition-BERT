import glob
import re
import string
import jsonlines
import csv
import torch
import random

from Embedding import Text2tokens, tokenizer, get_bert_inputs

Entity_types = {
    "Chemical": 1,
    "Genomic_factor": 2,
    "Gene_or_protein": 3,
    "Genomic_variation": 4,
    "Limited_variation": 5,
    "Haplotype": 6,
    "Phenotype": 7,
    "Disease": 8,
    "Pharmacodynamic_phenotype": 9,
    "Pharmacokinetic_phenotype": 10,
}

labels_ranges = {
    "chem": [1],
    "gen": range(2, 7),
    "phen": range(7, 11),
    "all": range(1, 11),
}

Entity_types_pub={
'Chemical' : 12,
'Disease' : 13 ,
'Gene' : 14,
'Mutation' :15
}
labels_ranges_pub={
   'chem':[12],
   'phen':range(13),
   'gen':range(14,15),
   'all':range(12,15)
}
classes_pub=[
'O',
'Chemical',
'Disease',
'Gene',
'Mutation'
]

shift = 0


def head_label(label, head):
    if label in labels_ranges[head]:
        return 1 + label - labels_ranges[head][0]
    else:
        return 0


classes = [
    "O",
    "Chemical",
    "Genomic_factor",
    "Gene_or_protein",
    "Genomic_variation",
    "Limited_variation",
    "Haplotype",
    "Phenotype",
    "Disease",
    "Pharmacodynamic_phenotype",
    "Pharmacokinetic_phenotype",
]


def remove(s, l):
    for i in l:
        s = s.replace(i, "")
    return s


punctuation = string.punctuation + "\t"
punctuation1 = [
    "(",
    ")",
    ":",
    ";",
    ",",
    "?",
    "!",
    ".",
    "%",
    "*",
    "+",
    "=",
    '"',
    "#",
    "~",
    "@",
    "$",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "^",
    "{",
    "}",
    "-",
    "_",
]


def string_normaliz(s, mode=0):
    l = punctuation if mode == 0 else punctuation1
    s = s.translate(str.maketrans("", "", l))
    return s


def load_txt(txt_path):
    txt = open(txt_path).read().replace("\n", "")
    return txt


def load_ann(ann_path, head="all"):
    ann = open(ann_path).read().split("\n")
    labels = []
    T = [i for i in ann if i.startswith("T")]
    for i in T:

        label = Entity_types[i.split()[1]]
        label = head_label(label, head=head)

        end_i = 3
        start = i.split()[2]
        end = i.split()[end_i]
        while ";" in end:
            range = [int(start), int(end.split(";")[0])]
            labels.append((range, label))
            start = int(end.split(";")[1])
            end_i += 1
            end = i.split()[end_i]

        range = [int(start), int(end.split(";")[0])]
        labels.append((range, label))

    return labels


def load_ann_pub(ann_path, head="all"):
    ann = open(ann_path).read().split("\n")
    labels = []
    T = [i for i in ann if i.startswith("T")]
    for i in T:

        label = Entity_types_pub[i.split()[1]]
        label = head_label(label, head=head)

        end_i = 3
        start = i.split()[2]
        end = i.split()[end_i]
        while ";" in end:
            range = [int(start), int(end.split(";")[0])]
            labels.append((range, label))
            start = int(end.split(";")[1])
            end_i += 1
            end = i.split()[end_i]

        range = [int(start), int(end.split(";")[0])]
        labels.append((range, label))

    return labels


def pointer_step(pointer, token, sent):
    return len(sent[:pointer]) + sent[pointer:].find(token.replace("#", ""))


def label_level(labels, label):
    level = -1
    lb, hb = label[0][0], label[0][1]
    for l in labels:
        if lb >= l[0][0] and hb <= l[0][1]:
            level += 1
            if level == 2:
                break
    return level


def brat(path, head):
    global shift
    shift = labels_ranges[head][0]

    ann_txt_files = [
        (f.split(path)[1], (f.split(path)[1]).split("ann")[0] + "txt")
        for f in glob.glob(path + "/*.ann")
    ]

    random.shuffle(ann_txt_files)

    Dataset_X = []
    Dataset_Y = []
    Dataset_Tokens = []

    for ann, txt in ann_txt_files:

        sentence = load_txt(path + txt)
        labels = load_ann(path + ann, head=head)
        tokens = tokenizer.tokenize(sentence)

        target = [[0] * len(tokens) for i in range(3)]
        pointer = 0
        i = 0
        for t in tokens:
            pointer = pointer_step(pointer, t, sentence)
            for l in labels:
                level = label_level(labels, l)
                if pointer in range(l[0][0], l[0][1] + 1):
                    target[level][i] = l[1]

            i += 1

        Dataset_X.append(get_bert_inputs(sentence))
        Dataset_Y.append(
            [0] + target[0] + [0] + [0] + target[1] + [0] + [0] + target[2] + [0]
        )
        Dataset_Tokens.append(tokens)

        """
        print('##########################')
        print(txt)
        print(target[0])
        print(target[1])
        print(target[2])
        print('##########################\n\n\n')
        """

    return Dataset_X, Dataset_Y, Dataset_Tokens


def headc(label):
    return 0 if label == 0 else label + shift - 1


def words2IOBES(words_labels_dataset):
    iobes_dataset = []
    for words in words_labels_dataset:
        iobes = [
            classes[headc(words[0])]
            if words[0] == 0
            else "B-" + classes[headc(words[0])]
        ]
        for i in range(1, len(words) - 1):
            if words[i] == 0:
                iobes.append(classes[headc(words[i])])
            elif words[i - 1] != words[i]:
                iobes.append("B-" + classes[headc(words[i])])
            elif words[i + 1] != words[i]:
                iobes.append("E-" + classes[headc(words[i])])
            else:
                iobes.append("I-" + classes[headc(words[i])])

        prefix = "E-" if words[-1] == words[-2] else "B-"
        iobes.append(
            classes[headc(words[-1])]
            if words[-1] == 0
            else prefix + classes[headc(words[-1])]
        )
        iobes_dataset.append(iobes)
    return iobes_dataset


def brat_pub(path, path_pub, head):
    global shift
    shift = labels_ranges_pub[head][0]

    ann_txt_files = [
        (f.split(path)[1], (f.split(path)[1]).split("ann")[0] + "txt")
        for f in glob.glob(path + "/*.ann")
    ]

    random.shuffle(ann_txt_files)
    Dataset_X = []
    Dataset_Y = []
    Dataset_Tokens = []
    # Dataset_pub = []
    for ann, txt in ann_txt_files:
        sentence = load_txt(path + txt)
        # labels = load_ann_pub(path+ann,head=head)
        labels = load_ann(path + ann, head=head)
        labels_pub = load_ann_pub(path_pub + ann + "_pubtator", head=head)
        tokens = tokenizer.tokenize(sentence)
        target = [[0] * len(tokens) for i in range(3)]
        target_pub = [[0] * len(tokens) for i in range(3)]
        pointer = 0
        pointer_pub = 0
        i = 0
        j = 0
        for t in tokens:
            pointer = pointer_step(pointer, t, sentence)
            for l in labels:
                level = label_level(labels, l)
                if pointer in range(l[0][0], l[0][1] + 1):
                    target[level][j] = l[1]
            j += 1
        for t in tokens:
            pointer_pub = pointer_step(pointer_pub, t, sentence)
            for l in labels_pub:
                level = label_level(labels_pub, l)
                if pointer_pub in range(l[0][0], l[0][1] + 1):
                    target_pub[level][i] = l[1]
            i += 1
        Dataset_X.append(
            {
                "bert_inputs": get_bert_inputs(sentence),
                "pub_inputs": [0]
                + target_pub[0]
                + [0]
                + [0]
                + target_pub[1]
                + [0]
                + [0]
                + target_pub[2]
                + [0],
            }
        )
        # Dataset_pub.append()
        Dataset_Y.append(
            [0] + target[0] + [0] + [0] + target[1] + [0] + [0] + target[2] + [0]
        )
        Dataset_Tokens.append(tokens)
        # Dataset_pub.append([0]+target_pub[0]+[0]+[0]+target_pub[1]+[0]+[0]+target_pub[2]+[0])
        # print('##########################')
        # print(txt)
        # print(target[0])
        # print(target[1])
        # print(target[2])
        # print('##########################\n\n\n')
        # print(txt)
        # print(target_pub[0])
        # print(target_pub[1])
        # print(target_pub[2])
        # print('##########################\n\n\n')
    # print(Dataset_X_pub[-1])
    # print(len(Dataset_Y_pub[-1]))
    # print(Dataset_Tokens_pub[-1])

    return Dataset_X, Dataset_Y, Dataset_Tokens


class Corpus:
    def __init__(self, path, name, head="all", path_pub=None):
        if path_pub != None:
            self.Entityes_types_pub = Entity_types_pub
            self.Nb_class = len(Entity_types_pub.keys()) + 1
        else:
            self.Entityes_types = Entity_types
            self.Nb_class = len(Entity_types.keys()) + 1
        self.Entityes_types = Entity_types
        self.Nb_class = len(Entity_types.keys()) + 1
        self.path = path
        self.path_pub = path_pub
        self.name = name
        self.data = None
        self.head = head

    def get_data(self):
        if self.name == "pgx":
            self.data = brat(self.path, self.head)
        if self.name == "pgx_pub":
            self.data = brat_pub(self.path, self.path_pub, self.head)

        return self.data
