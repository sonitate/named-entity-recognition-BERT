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
    def __init__(self, out=11,pub=5,dim_em=5, bert_type='bert'):
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
        self.three_levels_pub=MLP(out+pub)

    def forward(self, x,corpus):
        if(corpus=='pgx_pub'):
            print(x['bert_inputs'][0])
            rep_vects, _ = self.bert_model(x['bert_inputs'])
            rep_pub = self.pub_em(x['pub_inputs'])
            if not isinstance(rep_vects, torch.Tensor):
                rep_vects = rep_vects[-1]
            level1_logits= self.level1(rep_vects)
            print(level1_logits.shape)

            # l1_lo =self.level1_pub(rep_pub)
            # print(l1_lo.shape)
            # con_level1_logits=torch.cat((level1_logits,l1_lo),dim=1)
            # print(con_level1_logits.shape)
            level2_inputs = torch.cat((rep_vects,level1_logits),dim=2)
            # l2_in = torch.cat((rep_pub,l1_lo),dim=2)
            level2_logits = self.level2(level2_inputs)
            # l2_lo =self.level2_pub(l2_in)
            # con_level2_logits=torch.cat((level2_logits,l2_lo),dim=1)
            # print(con_level2_logits.shape)
            # l3_in = torch.cat((rep_pub,l2_lo),dim=2)
            # l3_lo=self.level3_pub(l3_in)
            level3_inputs = torch.cat((rep_vects,level2_logits),dim=2)
            level3_logits = self.level3(level3_inputs)
            # con_level3_logits=torch.cat((level3_logits,l3_lo),dim=1)
            # print(con_level3_logits.shape)
            all_level=torch.cat((level1_logits, level2_logits, level3_logits),dim=1)
            # all_level_pub=torch.cat((con_level1_logits, con_level2_logits, con_level3_logits),dim=1)
            # print(all_level_pub.shape)
            outputs=torch.cat((all_level,rep_pub),dim=2)

            print(outputs.shape)
            output=self.three_levels_pub(outputs)
            print(output.shape)
            return output

        elif(corpus=='pgx'):
            rep_vects, _ = self.bert_model(x)
            if not isinstance(rep_vects, torch.Tensor):
                rep_vects = rep_vects[-1]
            level1_logits= self.level1(rep_vects)
            level2_inputs = torch.cat((rep_vects,level1_logits),dim=2)
            level2_logits = self.level2(level2_inputs)
            level3_inputs = torch.cat((rep_vects,level2_logits),dim=2)
            level3_logits = self.level3(level3_inputs)
            outputs=torch.cat((level1_logits, level2_logits, level3_logits),dim=1)

            return outputs
