from torch.utils import data
from torch.utils.data import Dataset
from Embedding import *

class Data_set(Dataset):

    def __init__(self, X,ann_pub, Y):

        self.X = X
        self.Y = Y
        self.ann_pub=ann_pub
        # print("data", X[0].size())

    def __getitem__(self, index):

        return [self.X[index], self.ann_pub[index], self.Y[index]]

    def __len__(self):
        return len(self.Y)


def collate(batch):

    input = [item[0] for item in batch]
    input = torch.nn.utils.rnn.pad_sequence(input, batch_first=True)
    input_pub =[torch.tensor(item[1]) for item in batch]
    input_pub = torch.nn.utils.rnn.pad_sequence(input_pub, batch_first=True)
    targets = [torch.tensor(item[2]) for item in batch]
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)

    return input,input_pub, targets


def torch_loader(X, ann_pub, Y, shuffle=True, batch_size=32):

    corpus = Data_set(X,ann_pub, Y)
    loader = data.DataLoader(corpus, shuffle=shuffle, batch_size=batch_size, collate_fn=collate, pin_memory=True)


    return loader

