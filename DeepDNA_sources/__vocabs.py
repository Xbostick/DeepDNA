dna_vocab_revers = {0: "A",
             1:"C",
             2:"G",
             3:"T",
             4:"N"}


dna_vocab = {"A":0,
             "C":1,
             "G":2,
             "T":3,
             "N":4} # catch-all auxiliary token
dna_complimentar = {
                    "A":"T",
                    "T":"A",
                    "G":"C",
                    "C":"G",
                    "N":"N"
}


def ohe_DNA(part):
    seq = []
    for i in part:
        match i:
                case 0:
                    seq.append([1,0,0,0,0])
                case 1:
                    seq.append([0,1,0,0,0])
                case 2:
                    seq.append([0,0,1,0,0])
                case 3:
                    seq.append([0,0,0,1,0])
                case 4:
                    seq.append([0,0,0,0,1])
    return seq


dna_vocab_no_n = {"A":0,
             "C":1,
             "G":2,
             "T":3} # catch-all auxiliary token
dna_complimentar_no_n = {
                    "A":"T",
                    "T":"A",
                    "G":"C",
                    "C":"G"
}

def ohe_DNA_NO_N(part):
    seq = []
    for i in part:
        match dna_vocab_no_n[i.upper()]:
                case 0:
                    seq.append([1,0,0,0])
                case 1:
                    seq.append([0,1,0,0])
                case 2:
                    seq.append([0,0,1,0])
                case 3:
                    seq.append([0,0,0,1])
    return seq