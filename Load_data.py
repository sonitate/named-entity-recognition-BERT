from torch.utils import data
from torch.utils.data import Dataset
from Embedding import *

class Data_set(Dataset):

    def __init__(self, X, Y):

        self.X = X
        self.Y = Y
        # print("data", X[0].size())

    def __getitem__(self, index):

        return [self.X[index], self.Y[index]]

    def __len__(self):
        return len(self.Y)


def collate(batch):
    # print(batch[0])
    input_map=[item[0] for item in batch]
    bert=[i['bert_inputs'] for i in input_map]
    bert=torch.nn.utils.rnn.pad_sequence(bert, batch_first=True)
    pub=[torch.tensor(i['pub_inputs']) for i in input_map]
    pub=torch.nn.utils.rnn.pad_sequence(pub, batch_first=True)
    targets = [torch.tensor(item[1]) for item in batch]
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)

    return bert,pub, targets


def torch_loader(X, Y, shuffle=True, batch_size=32):

    corpus = Data_set(X, Y)
    loader = data.DataLoader(corpus, shuffle=shuffle, batch_size=batch_size, collate_fn=collate, pin_memory=True)


    return loader

