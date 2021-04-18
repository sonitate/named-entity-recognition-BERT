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

    for inputs , targets in loader:

        inputs, targets = inputs.to(global_param.device),targets.to(global_param.device)

        #print(inputs.size())

        outputs = model(inputs)

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

    pbar = tqdm(total=len(X), desc=" Prediction : ")
    Y=[]
    for x in X:
        input=torch.stack([x])
        input=input.to(global_param.device)
        with torch.no_grad():
            model.to(global_param.device)
            model.eval()
            output = model(input)
            #predicted_targets = output[0].argmax(dim=2)
            predicted_targets = output.argmax(dim=2)
            #predicted_targets=torch.argmax(F.log_softmax(output[0],dim=2),dim=2)
            Y.append(predicted_targets.tolist()[0])
        pbar.update(1)
        continue
    pbar.close()
    return Y



def train_save(model,X_app,Y_app,nb_epoch=30,batch_size=32,X_valid=[],Y_valid=[],F_type='macro',lr= 0.001,do_valid=True):


    if(len(Y_valid)==0):
        X_valid,Y_valid=X_app,Y_app

    path = save_path()
    checkpoint = ModelCheckpoint(path, model,F_type=F_type)


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
