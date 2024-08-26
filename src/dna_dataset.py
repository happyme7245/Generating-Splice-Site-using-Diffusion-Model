from torch.utils.data import Dataset


class DNADataset(Dataset):
    def __init__(self, sequences, transform=None):
        self.sequences = sequences
        self.transform = transform
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        if self.transform:
            sequence = self.transform(sequence)
        return sequence
    