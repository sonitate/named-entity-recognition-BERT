import argparse

import torch

masks_type={
    "simple_bert":["[unused_1]","[unused_1]"],
    "bio_bert":["@GENE$","@DISEASE$"]
}

encapsulate_items={
    "sci_bert":["<<",">>","[[","]]"]
}

class Hyperparameter :
    def __init__(self):

        parser = argparse.ArgumentParser()

        parser.add_argument('-bert', default='bert', choices=['bert','scibert'],
                            dest='bert',
                            help='Bert model : bert, biobert, scibert')

        parser.add_argument('-head', default='all', choices=['all','gen','chem','phen'],
                            dest='head',
                            help='The head of NER model')

        parser.add_argument('-F_type', default='macro', choices=['micro','macro'],
                            dest='F_type',
                            help='Type of F-mesure avg (micro,macro)')

        parser.add_argument('-lr', default=0.001, type=float,
                            dest='lr',
                            help='learning rate ( 0.001 for frozen, 5e-5 / 3e-5  for fine-tune')

        parser.add_argument('-num_ep',default=5, type=int,
                            dest='num_ep',
                            help='number of epochs ( 5/8 : for fine tuning ) / (30/ 60) for frozen ')

        param = parser.parse_args()


        self.model_param={
        'bert': param.bert,
        'head' : param.head
        }

        self.corpus_param={
        'corpus':'pgx'
        }


        method='fine_tuning'
        method+='_'+self.model_param['bert']
        method+='_' + self.model_param['head']
        method+='_'+self.corpus_param['corpus']
        

        self.traning_param={
        'num_ep':param.num_ep,
        'batch_size':32,
        'F_type':param.F_type,
        'exp_tag':method,
        'lr':param.lr
        }
        
        method+='_ep'+str(self.traning_param['num_ep'])
        method+='_lr'+str(self.traning_param['lr'])
        self.traning_param['exp_tag']=method
        print(method)

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            print("°°°°°°°°°°°°°°°°°°°°  GPU  °°°°°°°°°°°°°°°°°°")
            self.device = torch.device('cuda')
        else:
            print("°°°°°°°°°°°°°°°°°°°°  CPU  °°°°°°°°°°°°°°°°°°")
            self.device = torch.device('cpu')




global_param=Hyperparameter()
