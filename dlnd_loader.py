"""
Pytorch Dataset class for DLND data
"""

import logging
import pickle
import random
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate

LOG = logging.getLogger()
# GLOBAL VARIABLES
DLND_DATA = 'dlnd_data.p'
RANDOM_SEED = 1234

def oversample_novel(data):
    """
    Random oversampling of minority class (novel class: 1)
    """
    random.seed(RANDOM_SEED)
    answers = data[-1]
    n_samples_add = len(answers) - (2 * sum(answers))
    minority_ids = [idx for idx, answer in enumerate(answers) if answer]
    random_ids = random.choices(minority_ids, k=n_samples_add)
    for item_no, _ in enumerate(data):
        data[item_no] += [data[item_no][idx] for idx in random_ids]
    return data

def split_stratify_data(data):
    """
    Split data into train and test as 90% and 10% respectively
    """
    answers = data[-1]
    splitting = train_test_split(*data,
                                 test_size=0.1,
                                 random_state=RANDOM_SEED,
                                 shuffle=True,
                                 stratify=answers)
    train_ids = [idx for idx in range(len(splitting)) if idx%2 == 0]
    test_ids = [idx for idx in range(len(splitting)) if idx%2 == 1]
    train_data = [splitting[idx] for idx in train_ids]
    test_data = [splitting[idx] for idx in test_ids]
    return train_data, test_data

def get_dlnd_data():
    """
    Read the dlnd data from the pickle file and pre-process it
    """
    # Add encoding='latin1' to unpickle a file in python3 which was picklized in python2
    # data = [src_docs, tgt_docs, src_ids, tgt_ids, vocab, golds]
    src_docs, tgt_docs, src_ids, tgt_ids, vocab, golds = pickle.load(open(DLND_DATA, 'rb'), encoding='latin1')
    # random oversampling of minority class (novel class: 1)
    data = [src_docs, tgt_docs, src_ids, tgt_ids, golds]
    return oversample_novel(data), vocab

def split_data(data):
    """
    Split data into training, validation and testing data
    """
    # Train, Test split is 90%, 10% respectively
    train_data, test_data = split_stratify_data(data)
    # Valid data is 10% of Train data
    train_data, valid_data = split_stratify_data(train_data)
    # Print split sizes
    train_size = len(train_data[0])
    LOG.debug(f'Train data size: {train_size}')
    valid_size = len(valid_data[0])
    LOG.debug(f'Valid data size: {valid_size}')
    test_size = len(test_data[0])
    LOG.debug(f'Test data size: {test_size}')
    data_size = len(data[0])
    LOG.debug(f'Total data size: {data_size}')
    return train_data, valid_data, test_data

def pad_collate(batch, vocab):
    """
    1. Pad each of source and target based on the max size
    of source and target in the batch respectively

    2. Convert the sources and targets in the batch into matrices
    based on the sentence embeddings specified in the vocab
    """
    # vocab = (#docs * #sents, #embedding)
    embedding_size = vocab.shape[1]
    pad_sent = np.zeros((embedding_size,), dtype='float32')
    max_src_size = 0
    max_tgt_size = 0
    for i, elem in enumerate(batch):
        _, _, src_ids , tgt_ids, _ = elem
        src_size = len(src_ids)
        tgt_size = len(tgt_ids)
        if src_size > max_src_size:
            max_src_size = src_size
        if tgt_size > max_tgt_size:
            max_tgt_size = tgt_size
    for i, elem in enumerate(batch):
        src_docs, tgt_docs, src_ids, tgt_ids, answers = elem
        # Pad each source and target per their maximum sizes
        src_ids = [None] * (max_src_size - len(src_ids)) + src_ids
        tgt_ids = [None] * (max_tgt_size - len(tgt_ids)) + tgt_ids
        # Convert the sources and targets into matrices based on vocab
        # src_ids = (#batch, max_src_size)
        # tgt_ids = (#batch, max_tgt_size)
        src_vec = list()
        for sent_idx in src_ids:
            if sent_idx:
                src_vec.append(vocab[sent_idx])
            else:
                src_vec.append(pad_sent)
        src = np.vstack(src_vec)
        tgt_vec = list()
        for sent_idx in tgt_ids:
            if sent_idx:
                tgt_vec.append(vocab[sent_idx])
            else:
                tgt_vec.append(pad_sent)
        tgt = np.vstack(tgt_vec)
        batch[i] = (src_docs, tgt_docs, src, tgt, answers)
    return default_collate(batch)

class DLND(Dataset):
    """
    Pytorch Dataset class for DLND data
    """
    def __init__(self, mode='train'):
        self.mode = mode
        # self.data = docs, src_ids, tgt_ids, golds
        self.data, self.vocab = get_dlnd_data()
        # Set self.hidden = embedding size = self.vocab.shape[1]
        self.hidden = self.vocab.shape[1]
        LOG.debug(f'Sentence Embedding dimension: {self.hidden}')
        # Set self.{train|valid|test}
        self.train, self.valid, self.test = split_data(self.data)

    def set_mode(self, mode):
        """
        Sets the type of data to be return by the Dataset object
        mode = {train|valid|split}
        """
        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return len(self.train[0])
        if self.mode == 'valid':
            return len(self.valid[0])
        return len(self.test[0])

    def __getitem__(self, index):
        if self.mode == 'train':
            data = [item[index] for item in self.train]
        elif self.mode == 'valid':
            data = [item[index] for item in self.valid]
        elif self.mode == 'test':
            data = [item[index] for item in self.test]
        return data
