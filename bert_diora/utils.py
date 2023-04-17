from typing import List, Iterator

import torch
from torch.utils.data.sampler import Sampler
from nltk.tokenize.treebank import TreebankWordTokenizer


class TokenizedLengthSampler(Sampler[List[int]]):
    """
    PyTorch DataLoader - compatible sampler class that batchify sentences with the most similar lengths for maximum efficiency.
    """

    def __init__(self, data_source: List[str], batch_size: int, seed: int):
        self.data_source = data_source
        self.length = len(data_source)
        self.batch_size = batch_size
        
        tokenize = TreebankWordTokenizer().tokenize
        seq_lengths = [len(tokenize(sent)) for sent in data_source]
        indices = list(range(len(data_source)))
        indices = sorted(indices, key=lambda i: seq_lengths[i])

        batches = [indices[:self.length % self.batch_size]]
        for start in range(self.length % self.batch_size, self.length, batch_size):
            end = start + batch_size
            batches.append(indices[start:end])
            
        self.length_batches = len(batches)
        g_cpu = torch.Generator()
        g_cpu.manual_seed(seed)
        self.batch_indices = torch.randperm(n=self.length_batches, dtype=torch.long, generator=g_cpu).tolist()
        self.batches = [batches[i] for i in self.batch_indices]
        self.seq_lengths = seq_lengths

    def __len__(self):
        return self.length_batches
    
    def __iter__(self) -> Iterator[List[int]]:
        for batch, idx in (self.batches, self.batch_indices):
            yield batch, idx
