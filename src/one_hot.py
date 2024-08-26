import numpy as np
import torch


class OneHotEncodeTransform:
    def __call__(self, sequence):
        dictionary = {
            "A": [1, 0, 0, 0],
            "T": [0, 1, 0, 0],
            "G": [0, 0, 1, 0],
            "C": [0, 0, 0, 1],
            'N': [0.25, 0.25, 0.25, 0.25],  # Handling 'N'
        }
        onehot = np.array([dictionary.get(nucleotide, [0, 0, 0, 0]) for nucleotide in sequence])
        return torch.tensor(onehot, dtype=torch.float32).T
        
        
def readInputs(f1,f2):
    lines_pos = open(f1).readlines()
    lines_neg = open(f2).readlines()

    X = []
    Y = []

    for l in convertLines(lines_pos):
        X.append(l)
        Y.append(1)
    for l in convertLines(lines_neg):
        X.append(l)
        Y.append(0)

    return torch.tensor(X), torch.tensor(Y)

def convertLines(lines):
    newLines = []
    for line in lines:
        newline = []
        for c in line.strip():
            if c == 'A':
                v = [1,0,0,0]
            elif c == 'C':
                v = [0,1,0,0]
            elif c == 'G':
                v = [0,0,1,0]
            elif c == 'T':
                v = [0,0,0,1]
            newline.append(v)
        newLines.append(newline)
    return newLines
