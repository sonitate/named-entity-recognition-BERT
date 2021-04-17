"""
from collections import OrderedDict
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertConfig,BertForTokenClassification
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForTokenClassification
from pytorch_transformers import BertModel as bm



def get_bert(bert_type='bert',num_labels=11):
    tokenizer, model = None, None
    if (bert_type == 'bert'):
        ######## bert ###########

        tokenizer = BertTokenizer.from_pretrained('bert-base-cased',do_lower_case=False)
        model = BertForTokenClassification.from_pretrained('bert-base-cased',num_labels=num_labels)

        ########################

    if (bert_type == 'scibert'):
        #### sci bert #########
        config = AutoConfig.from_pretrained('allenai/scibert_scivocab_cased', num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased',do_lower_case=False)
        model = AutoModelForTokenClassification.from_pretrained('allenai/scibert_scivocab_cased', config=config)

        #######################

    return tokenizer, model

"""
from pytorch_pretrained_bert import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel, AutoConfig
from pytorch_transformers import BertModel as bm


def get_bert(bert_type='bert'):
    tokenizer, model = None, None
    if (bert_type == 'bert'):
        ######## bert ###########

        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        model = BertModel.from_pretrained('bert-base-cased')

        ########################

    if (bert_type == 'biobert'):
        #### Bio BERT #########

        model = bm.from_pretrained('biobert_v1.1_pubmed')
        tokenizer = BertTokenizer(vocab_file="biobert_v1.1_pubmed/vocab.txt", do_lower_case=False)

        #### Bio BERT #########

    if (bert_type == 'scibert'):
        #### sci bert #########


        config = AutoConfig.from_pretrained('allenai/scibert_scivocab_cased', output_hidden_states=False)
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased', do_lower_case=False)
        model = AutoModel.from_pretrained('allenai/scibert_scivocab_cased', config=config)

        #######################

    return tokenizer, model



