from itertools import chain

import torch
from tqdm import tqdm

from Load_data import torch_loader

from Checkpoint import ModelCheckpoint, save_path
from Parameters import global_param
import random

import torch.nn.functional as F


def train(model, loader,f_loss, optimizer):
    
    
    model.to(global_param.device)

    model.train()

    nb_batch=len(loader)

    pbar = tqdm(total=nb_batch, desc="Training batch : ")

    N = 0
    tot_loss, correct = 0.0, 0.0
    # print(loader[0])
    for berts,pubs, targets in loader:

        berts = berts.to(global_param.device)
        # print(berts)
        # print(type(berts))
        pubs = pubs.to(global_param.device)
        targets = targets.to(global_param.device)
        #print(inputs.size())

        outputs = model({'bert_inputs':berts,'pub_inputs':pubs})
        result=outputs.argmax(dim=2)
        # print(result[:3])
        # print(outputs.permute(0,2,1).argmax(dim=2))
        # print(targets[:3])
        #print(outputs[0].permute(0,2,1).size())
        #print(targets.size())
        #print(outputs.size())

        #loss = f_loss(outputs[0].permute(0,2,1), targets)
        loss = f_loss(outputs.permute(0,2,1), targets)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        pbar.update(1)

        N+=targets.size(0)*targets.size(1)
        tot_loss += targets.size(0)*targets.size(1)*loss.item()
        #predicted_targets = outputs[0].argmax(dim=2)
        predicted_targets = outputs.argmax(dim=2)
        correct +=(predicted_targets==targets).sum().item()



    pbar.close()

    return tot_loss / N, correct / N



def prediction(model,X):

    Y=[]
    # print(type(X))
    # input_map=[item[0] for item in X]
    bert=[i['bert_inputs'] for i in X]
    # bert=torch.nn.utils.rnn.pad_sequence(bert, batch_first=True)
    # print
    pub=[torch.tensor(i['pub_inputs']) for i in X]
    # pub=torch.nn.utils.rnn.pad_sequence(pub, batch_first=True)
    X_in=[{'bert_inputs':i,'pub_inputs':j} for i,j in zip(bert,pub)]
    # print(X_in[0])
    # print(len(X_in))
    pbar = tqdm(total=len(X_in), desc=" Prediction : ")
    for inputs in X_in:
        berts=torch.stack([inputs['bert_inputs']])
        pubs=torch.stack([inputs['pub_inputs']])
        # input=input.to(global_param.device)
        # print(berts)
        # print(pubs)
        berts = berts.to(global_param.device)
        pubs = pubs.to(global_param.device)
    # print(bert[1])
        # input=input.to(global_param.device)
        with torch.no_grad():
            model.to(global_param.device)
            model.eval()
            output = model({'bert_inputs':berts,'pub_inputs':pubs})
            #predicted_targets = output[0].argmax(dim=2)
            predicted_targets = output.argmax(dim=2)
            #predicted_targets=torch.argmax(F.log_softmax(output[0],dim=2),dim=2)
            Y.append(predicted_targets.tolist()[0])
        pbar.update(1)
        continue
    pbar.close()
    return Y



def train_save(model,X_app,Y_app,nb_epoch=30,batch_size=32,X_valid=[],Y_valid=[],F_type='macro',lr= 0.001,do_valid=True,save=False):


    if(len(Y_valid)==0):
        X_valid,Y_valid=X_app,Y_app
        print('##')

    path = save_path()
    checkpoint = ModelCheckpoint(path, model,F_type=F_type,save=save)


    loader_app = torch_loader(X_app,Y_app,batch_size=batch_size)

    f_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=0.0,amsgrad=False)


    for i in range(nb_epoch):
       loss ,acc = train(model,loader_app, f_loss, optimizer)
       pred=prediction(model,X_valid)
       #pred=Y_valid
       checkpoint.update(pred, Y_valid, i, loss, acc,do_valid=do_valid)
       #checkpoint.update(list(chain.from_iterable(pred)),list(chain.from_iterable(Y_valid)),i,loss,acc,do_valid=do_valid)

    return checkpoint.get_best_model(),checkpoint.filepath
