from collections import OrderedDict
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertConfig,BertForTokenClassification
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForTokenClassification
from pytorch_transformers import BertModel as bm



def get_bert(bert_type='bert',num_labels=11):
    tokenizer, model = None, None
    if (bert_type == 'bert'):
        ######## bert ###########

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=num_labels)

        ########################

    if (bert_type == 'scibert'):
        #### sci bert #########
        config = AutoConfig.from_pretrained('allenai/scibert_scivocab_uncased', output_hidden_states=False, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        model = AutoModelForTokenClassification.from_pretrained('allenai/scibert_scivocab_uncased', config=config)

        #######################

    return tokenizer, model





