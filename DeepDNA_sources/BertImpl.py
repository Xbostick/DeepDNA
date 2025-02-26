import torch
from torch import nn
from torch.autograd import Variable 
import transformers
from transformers import BertTokenizer, BertForTokenClassification
import numpy as np
from Bio import SeqIO
from io import StringIO, BytesIO
from tqdm import tqdm
import pickle
import scipy
from scipy import ndimage
from .__vocabs import dna_vocab_revers


TOKENIZER = None
MODEL = None
model_confidence_threshold = 0.5 #@param {type:"number"}
minimum_sequence_length = 10 #@param {type:"integer"}


def seq2kmer(seq, k):
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    return kmer

def split_seq(seq, length = 512, pad = 16):
    res = []
    for st in range(0, len(seq), length - pad):
        end = min(st+512, len(seq))
        res.append(seq[st:end])
    return res

def stitch_np_seq(np_seqs, pad = 16):
    res = np.array([])
    for seq in np_seqs:
        res = res[:-pad]
        res = np.concatenate([res,seq])
    return res

def init_bert():
    global TOKENIZER
    global MODEL
    TOKENIZER = BertTokenizer.from_pretrained('./bert/')
    MODEL = BertForTokenClassification.from_pretrained('./bert/')
    MODEL.cuda()

def check_seq(seq_list):
    num_hits = 0
    num_hits_no_filter = 0
    
    out = {}
    for num, seq in enumerate(seq_list):
        result_dict = {}
        out[num] = {"is_exist" : False, "list_hits" : []}
        kmer_seq = seq2kmer("".join(seq), 6)
        #print(kmer_seq)
        seq_pieces = split_seq(kmer_seq)

        with torch.no_grad():
            preds = []
            for seq_piece in seq_pieces:
                input_ids = torch.LongTensor(TOKENIZER.encode(' '.join(seq_piece), add_special_tokens=False))
                outputs = torch.softmax(MODEL(input_ids.cuda().unsqueeze(0))[-1],axis = -1)[0,:,1]
                preds.append(outputs.cpu().numpy())
        result_dict[num] = stitch_np_seq(preds)



        labeled, max_label = scipy.ndimage.label(result_dict[num]>model_confidence_threshold)

        for label in range(1, max_label+1):
            candidate = np.where(labeled == label)[0]
            candidate_length = candidate.shape[0]
            if candidate_length>minimum_sequence_length:
                out[num]["is_exist"] = True
                out[num]["list_hits"].append([candidate[0], candidate[-1]])

    return out

def Bert_Test(generator, data_sample, batch_size = 1):
    batch = []
    output = {"rate" : 0,
              "Bert_Output" : None}
    
    for i in range(batch_size):
        filt = generator(Variable(torch.randn(data_sample.shape)).cuda())
        OheSeq = filt.detach().cpu().numpy()
        Seq = [dna_vocab_revers[one] for one in np.argmax(OheSeq[0],0)]
        batch.append(Seq)
        
    output["Bert_Output"] = check_seq(batch)
    output["rate"] = sum([one["is_exist"] for one in output["Bert_Output"].values()]) / batch_size
    output["batch"] = batch
    return output
