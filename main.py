import os
#from sklearn.metrics import precision_recall_fscore_support, classification_report
from seqeval.metrics import classification_report
from seqeval.metrics.v1 import precision_recall_fscore_support as precision_recall_fscore_support
from seqeval.scheme import IOBES
import Bert
import Cross_validation
from Corpus import Corpus,words2IOBES,words2IOBES_pub
from Model import BertRecNER
from Parameters import global_param
from Train import train_save, prediction
from itertools import chain

head =global_param.model_param['head']
corpus_path =global_param.corpus_param['corpus']
if(corpus_path=='pgx'):
    corpus=Corpus('data/PGxCorpus','pgx',head=head)
elif(corpus_path=='pgx_pub'):
    corpus=Corpus('data/PGxCorpus','pgx_pub',head=head,path_pub='data/PGxCorpus_pubtator')
X_app,Y_app,Tokens= corpus.get_data()

###### param sup #######
do_valid=True
fold_num=10
do_cross_valid=True

nb_epoch =global_param.traning_param['num_ep'] # 5
lr= global_param.traning_param['lr'] # 3e-5
bert_type= global_param.model_param['bert'] # 'bert'
F_type= global_param.traning_param['F_type']  # 'macro'
save=global_param.traning_param['save']
exp_name= global_param.traning_param['exp_tag']



machine_name = os.uname()[1]

X_valid,Y_valid=[],[]
X_test,Y_test=[],[]


global_param.model_param['bert']

def Experence():
    #_,model=Bert.get_bert(bert_type=bert_type)

    model=BertRecNER(bert_type=bert_type)
    
    model.to(global_param.device)
    print(model)

    train_param = {
            'model': model,
            'X_app': X_app,
            'Y_app': Y_app,
            'nb_epoch': nb_epoch,
            'F_type': F_type,
            'lr': lr,
            'do_valid':do_valid,
            'save':save,
            'corpus':corpus_path
        }


    if not do_cross_valid:
        print("///////////////////////////////////////////////////////////////////////////////")
        print("///////////////////////////////////////////////////////////////////////////////")
        print("/////////////////////////         Training       ///////////////////////////////")

        best_model, path = train_save(**train_param)

        print("///////////////////////////////////////////////////////////////////////////////")
        print("///////////////////////////////////////////////////////////////////////////////")
        print("/////////////////////////         TEST          ///////////////////////////////")

        pred = prediction(best_model, X_test)
        true=Y_test

    else:

        print("///////////////////////////////////////////////////////////////////////////////")
        print("///////////////////////////////////////////////////////////////////////////////")
        print("/////////////////////////         CROSS RESULT          ///////////////////////////////")

        pred, true = Cross_validation.cross_validation(train_param, train_save, fold_num)
    pred_e, true_e= words2IOBES(pred), words2IOBES(true)




    #pred_entities,true_entities=entity_level(pred_words,Y_words)


    #chain.from_iterable([[1]])

    #pred_entities, true_entities = list(chain.from_iterable(pred_entities)), list(chain.from_iterable(true_entities))
        

    raport = classification_report(y_pred=pred_e, y_true=true_e,scheme=IOBES,mode='strict')

    print(raport)

    file = open("result_" + machine_name + "_" + F_type + ".pred", "a+")
    y, p = "", ""
    for xp, xy in zip(pred_e,true_e):
        y += ' ' + str(xy)
        p += ' ' + str(xp)
    print("Y" + y + '\nP' + p, file=file)
    file.close()

    return precision_recall_fscore_support(y_pred=pred_e, y_true=true_e, average=F_type,scheme=IOBES,mode='strict'), raport

for k in range(10):
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     Experience {}      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~".format(k))

    raport, raport_det = Experence()
    file = open(exp_name + "result_" + machine_name + "_" + F_type + ".res", "a+")
    print(str(raport[0]) + " " + str(raport[1]) + " " + str(raport[2]), file=file)
    file.close()

    file = open(exp_name + "result_" + machine_name + "_" + F_type + ".det", "a+")
    print(str(raport_det), file=file)
    file.close()
    # os.rmdir(path)
