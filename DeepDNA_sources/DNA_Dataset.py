from torch.utils.data import Dataset
from Bio import SeqIO
from torch import Tensor
from .__vocabs import  ohe_DNA, ohe_DNA_NO_N, dna_complimentar, dna_vocab


class GenomicData(Dataset):
    def __init__(self, fasta_path, markers,left,right, out_col, keep_none = False):
        self.coded_seq = []
        self.data = []
        if keep_none:
            encode_dna_function = ohe_DNA
        else:
            encode_dna_function = ohe_DNA_NO_N

        for chr in SeqIO.parse(fasta_path, 'fasta'):  # (generator)
            for _,one in markers[markers["chr"] == chr.id].iterrows():
                chr_seq = chr.seq[one.start-left:one.start+right].upper()
                if one.strand == "-":
                    new_chr_seq = []
                    for i in reversed(chr_seq):
                        new_chr_seq.extend(dna_complimentar[i])
                    chr_seq = new_chr_seq
                coded = []
                for symbol in chr_seq:
                      coded.append(dna_vocab[symbol])
                coded = encode_dna_function(coded)
                self.coded_seq.append(Tensor(coded).cuda())
                self.data.append(Tensor([one[out_col]])[0].float().cuda())
            

    def __len__(self):
        return len(self.coded_seq)#int(np.round((self.len - self.seq_len)/100))
    
    def __getitem__(self, idx):
        return [self.coded_seq[idx],self.data[idx]]

