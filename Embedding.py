"""

This module offering word embadding using BERT model

"""

import torch
import Bert
from Parameters import global_param

bert_type=global_param.model_param['bert']
tokenizer,model=Bert.get_bert(bert_type=bert_type)

def Text2tokens(text):
    marked_text = "[CLS] " + text +" [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0] * len(tokenized_text)
    return indexed_tokens,segments_ids

def Token2tonsor(token):

    indexed_tokens,segments_ids = token[0],token[1]
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    return tokens_tensor

def Bert_Forward(inputs):

    tokens_tensor,segments_tensors=inputs[0],inputs[1]
    model.eval()
    with torch.no_grad():
        activity_layers, _ = model(tokens_tensor, segments_tensors)
        if(isinstance(activity_layers,torch.Tensor)):
            activity_layers=[activity_layers]
        #print(len(activity_layers),len(activity_layers[0]),len(activity_layers[0][0]),len(activity_layers[0][0][0]))
    return activity_layers


def get_bert_inputs(text):
    tokenized_text = Text2tokens(text)
    inputs=Token2tonsor(tokenized_text)[0].view(-1)
    return inputs
