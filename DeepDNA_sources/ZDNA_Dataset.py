from torch.utils.data import Dataset
from Bio import SeqIO
from torch import Tensor
import numpy as np


class GenomicData_ZDNA(Dataset):
    def __init__(self, fasta_path, markers,max_len, out_col = None, has_strand = 0):
        self.coded_seq = []
        self.data = []
        self.chr_seq = []
        for chr in SeqIO.parse(fasta_path, 'fasta'):  # (generator)
            for _,one in markers[markers["chr"] == chr.id].iterrows():
                start_pos = int(np.floor((max_len - (one.end - one.start)) * np.random.random()))
                end_pos = start_pos + (one.end - one.start)
                #print(f"{start_pos} {end_pos} {one.start} {chr.seq[one.start - start_pos : one.end + int(max_len) - end_pos]}")
                self.chr_seq.append(chr.seq[one.start - start_pos : one.end + int(max_len) - end_pos].upper())
                if has_strand and one.strand == "-": # if change places args - wil be bad
                    new_chr_seq = []
                    for i in reversed(self.chr_seq[-1]):
                        new_chr_seq.extend(dna_complimentar[i])
                    self.chr_seq[-1] = new_chr_seq
                coded = []
                for symbol in self.chr_seq[-1]:
                      coded.append(dna_vocab[symbol])
                coded = ohe_DNA(coded)
                self.coded_seq.append(Tensor(coded).cuda())
                if out_col:
                    self.data.append(Tensor([one[out_col]])[0].float().cuda())
            

    def __len__(self):
        return len(self.coded_seq)#int(np.round((self.len - self.seq_len)/100))
    
    def __getitem__(self, idx):
        return self.coded_seq[idx]

