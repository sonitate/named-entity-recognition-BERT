import torch
from torch import nn
import Bert
from torch.nn.utils import fusion as F

class SeqClassifier(nn.Module):

    def __init__(self, H=200, emb_size=768, out=11):
        super(SeqClassifier, self).__init__()
        self.lstm = nn.LSTM(bidirectional=True, num_layers=2, dropout=0.5,
                            input_size=emb_size, hidden_size=H,
                            batch_first=True)

        self.classifier = nn.LSTM(bidirectional=False, num_layers=1, dropout=0.5,
                            input_size=H*2, hidden_size=out,
                            batch_first=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        logits,_ = self.classifier(x)
        return logits

class MLP(nn.Module):
    def __init__(self,inputs=768,H=200,out=11):
        super(MLP,self).__init__()
        self.l1=nn.Linear(inputs,H)
        self.l2=nn.Linear(H,out)
    
    def forward(self, x):
        x=self.l1(x)
        x=self.l2(x)
        return x



class BertRecNER(nn.Module):
    def __init__(self, out=11,pub=5,dim_em=6, bert_type='bert'):
        super(BertRecNER, self).__init__()
        _, self.bert_model = Bert.get_bert(bert_type=bert_type)

        # self.level1=SeqClassifier()
        # self.level2=SeqClassifier(emb_size=768+out)
        # self.level3=SeqClassifier(emb_size=768+out)

        # self.level1=nn.Linear(768,out)
        # self.pub_em = nn.Embedding(pub,5)
        # self.level2=nn.Linear(768+out,out)
        # self.level3=nn.Linear(768+out,out)


        self.level1=MLP(inputs=768)
        self.pub_em = nn.Embedding(pub,dim_em)
        self.level2=MLP(inputs=768+out)
        self.level3=MLP(inputs=768+out)

    def forward(self, x):
        rep_vects, _ = self.bert_model(x['bert_inputs'])
        # rep_pub = self.pub(x['pub_inputs']) //error index out of range

        # rep_vects, _ = self.bert_model(x['pub_inputs'])
        rep_pub = self.pub_em(x['pub_inputs'])
        if not isinstance(rep_vects, torch.Tensor):
            rep_vects = rep_vects[-1]
        # print(type(rep_vects))
        print(rep_vects.shape)
        # print(rep_pub.size())
        print(rep_pub.shape)
        # )
        # concat_vects=torch.stack((rep_vects,rep_pub),dim=0)
        # print(concat_vects.size())
        level1_logits= self.level1(rep_vects)
        print(level1_logits.shape)
        # print(rep_pub.size())

        level2_inputs = torch.cat((rep_vects,level1_logits),dim=2)
        level2_logits = self.level2(level2_inputs)
        # print(level2_logits[0].shape)

        level3_inputs = torch.cat((rep_vects,level2_logits),dim=2)
        level3_logits = self.level3(level3_inputs)
        print(level3_logits.size())
        outputs=torch.cat((level1_logits, level2_logits, level3_logits),dim=1)
        concat_vects=torch.cat((outputs,rep_pub),dim=2)
        # print(concat_vects[0])
        # print(outputs[0])
        print(concat_vects.size())
        # print(outputs.shape)


        return concat_vects
