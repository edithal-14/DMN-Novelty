"""
Pytorch Dataset class for input to the DMN model
"""

from itertools import chain
import logging
import pickle
import random
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate

LOG = logging.getLogger()
# GLOBAL VARIABLES
DLND_DATA = 'dlnd/dlnd_data.p'
APWSJ_DATA = 'apwsj/apwsj_data.p'
RANDOM_SEED = 1234

def oversample(data, minority_class):
    """
    Random oversampling of minority class
    class labels:
    0: non-novel class
    1: novel class
    """
    random.seed(RANDOM_SEED)
    answers = data[-1]
    n_minority = len([answer for answer in answers if answer == minority_class])
    n_samples_add = len(answers) - (2 * n_minority)
    minority_ids = [idx for idx, answer in enumerate(answers) if answer == minority_class]
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
    src_docs, tgt_docs, src_ids, tgt_ids, vocab, golds \
            = pickle.load(open(DLND_DATA, 'rb'), encoding='latin1')
    # random oversampling of minority class (novel class: 1)
    data = [src_docs, tgt_docs, src_ids, tgt_ids, golds]
    return oversample(data, 1), vocab


def get_apwsj_data():
    """
    Read the apwsj data from the pickle file and pre-process it
    """
    # Consider only the latest max_context_size number
    # of sentences as the context for a document
    max_context_size = 200
    # Consider only the latest max_question_size number
    # of sentences in a document
    max_question_size = 100
    # Add encoding='latin1' to unpickle a file in python3 which was picklized in python2
    # data = [docs, contexts, questions, vocab, golds]
    docs, contexts, questions, vocab, golds = pickle.load(open(APWSJ_DATA, 'rb'), encoding='latin1')
    # Trim the number of sentences in each question
    for idx, question in enumerate(questions):
        questions[idx] = question[-max_question_size:]
    # Stack all the sentence_ids of each context document for a target document
    # Trim the number of sentences
    remove_ids = list()
    # context documents for each question
    context_docs = list()
    for idx, context in enumerate(contexts):
        # Record the indices which dont have any context
        # to delete them later
        if len(context) == 0:
            remove_ids.append(idx)
        sent_ids = [questions[question_idx] for question_idx in context]
        contexts[idx] = list(chain.from_iterable(sent_ids))[-max_context_size:]
        context_doc = [docs[question_idx] for question_idx in context]
        context_docs.append(context_doc)
    # Remove the indices in remove_ids
    for idx in remove_ids[::-1]:
        del context_docs[idx]
        del docs[idx]
        del contexts[idx]
        del questions[idx]
        del golds[idx]
    # random oversampling of minority class (non-novel class: 0)
    data = [context_docs, docs, contexts, questions, golds]
    return oversample(data, 0), vocab

def split_data(data):
    """
    Split data into training, validation and testing data
    """
    # Train, Test split is 90%, 10% respectively
    train_data, test_data = split_stratify_data(data)
    # Valid data is 10% of Train data
    train_data, valid_data = split_stratify_data(train_data)
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
        _, _, src_ids, tgt_ids, _ = elem
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

class DmnData(Dataset):
    """
    Pytorch Dataset class for input to the DMN model:

    Supports k fold cross validation when folds > 1
    """
    def __init__(self, dataset_name, mode='train', folds=1):
        self.mode = mode
        if dataset_name == 'DLND':
            self.data, self.vocab = get_dlnd_data()
        elif dataset_name == 'APWSJ':
            self.data, self.vocab = get_apwsj_data()
        else:
            raise Exception('Dataset name %s is not supported!' % dataset_name)
        # Set self.hidden = embedding size = self.vocab.shape[1]
        self.hidden = self.vocab.shape[1]
        if folds <= 1:
            # Set splitting type
            self.split_type = 'ninety-ten'
            self.train, self.valid, self.test = split_data(self.data)
            LOG.debug('Will do ninety ten split')
        else:
            # Set splitting type
            self.split_type = 'kfold'
            # golds = data[-1]
            skf = StratifiedKFold(
                n_splits=folds,
                random_state=RANDOM_SEED,
                shuffle=True)
            self.folds = skf.split(np.zeros(len(self.data[0])), self.data[-1])
            self.next_fold()
            LOG.debug('Will do %d fold cross validation', folds)
        # Log data set size details
        LOG.debug('Sentence Embedding dimension: %d', self.hidden)
        LOG.debug('Total data size: %d', len(self.data[0]))
        LOG.debug('Train data size: %d', len(self.train[0]))
        LOG.debug('Valid data size: %d', len(self.valid[0]))
        LOG.debug('Test data size: %d', len(self.test[0]))

    def set_mode(self, mode):
        """
        Sets the type of data to be return by the Dataset object
        mode = {train|valid|split}
        """
        self.mode = mode

    def next_fold(self):
        """
        Sets self.{train|valid|test} from the next fold
        """
        if not self.split_type == 'kfold':
            LOG.error('KFold splitting is not enabled')
            return
        try:
            train_idx, test_idx = next(self.folds)
        except StopIteration:
            LOG.error('No more folds to process')
            return
        random.shuffle(train_idx)
        random.shuffle(test_idx)
        train_size = int(len(train_idx) * 0.9)
        valid_idx = train_idx[train_size:]
        train_idx = train_idx[:train_size]
        self.train = [[item[idx] for idx in train_idx] for item in self.data]
        self.valid = [[item[idx] for idx in valid_idx] for item in self.data]
        self.test = [[item[idx] for idx in test_idx] for item in self.data]

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