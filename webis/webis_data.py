"""
Serailize Webis CPC 11 documents to include infersent sentence
embeddings such that we can represent documents as matrices.
Dumps relevant data into a pickle file, which will be loaded for
creating a model.

Run this script with python3 and pytorch 0.3 with GPU support
"""

from itertools import tee
import nltk
import numpy as np
import logging
import os
import pickle
import sys

# Path Variables
GPU_ID = 1
HOME_DIR = "/home1/tirthankar"
WEBIS_PATH = os.path.join(HOME_DIR, 'Vignesh/dmn/code/webis/webis_corpus')
GLOVE_PATH = os.path.join(HOME_DIR, 'glove.840B.300d.txt')
ENCODER_DIR = os.path.join(HOME_DIR, 'Vignesh/InferSent')
ENCODER_PATH = os.path.join(ENCODER_DIR, 'models/model_2048_attn.pickle')
LOGGING_MODULE_PATH = os.path.join(HOME_DIR, 'Vignesh/dmn/code')
LOG_PATH = os.path.join(LOGGING_MODULE_PATH, 'webis/webis_logs')
SAVE_FILE_PATH = 'webis_data.p'

# Initialize logger
sys.path.append(LOGGING_MODULE_PATH)
from logger import init_logger
LOG = logging.getLogger()
init_logger(LOG, LOG_PATH)

# Load pytorch
import torch
torch.cuda.set_device(GPU_ID)

# Infersent should be in the path to load the encoder
sys.path.append(ENCODER_DIR)

def list_dataset():
    """
    List doc_ids (topics), source sentences, target sentences and gold values.
    """
    topics = list(set([fn.split('-')[0] for fn in os.listdir(WEBIS_PATH)]))
    topics_processed = list()
    targets = list()
    sources = list()
    gold = list()
    print('Reading data')
    for topic in topics:
        src = open('{}/{}-original.txt'.format(WEBIS_PATH, topic), 'r').read().encode("ascii","ignore").decode()
        src = nltk.sent_tokenize(src)
        tgt = open('{}/{}-paraphrase.txt'.format(WEBIS_PATH, topic), 'r').read().encode("ascii","ignore").decode()
        tgt = nltk.sent_tokenize(tgt)
        # Do not consider documents whose original or paraphrase have more than 50 sentences
        if (len(src) <= 50 and len(tgt) <= 50):
            sources.append(src)
            targets.append(tgt)
            topics_processed.append(topic)
        else:
            continue
        for line in open('{}/{}-metadata.txt'.format(WEBIS_PATH, topic), 'r').read().encode("ascii","ignore").decode().splitlines():
            lines = line.encode("ascii", "ignore").decode()
            if 'Paraphrase:' in line:
                gold.append(0 if line.split()[1] == 'Yes' else 1)
    return topics_processed, sources, targets, gold

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def get_sent_embs(tgt_docs, src_docs):
    """
    Produce sentence embeddings for target and source documents.
    """
    LOG.debug('Loading encoder')
    infersent = torch.load(ENCODER_PATH, map_location='cuda:{}'.format(GPU_ID))
    # use the encoder network only
    infersent = infersent.encoder
    LOG.debug('Loading glove vectors')
    infersent.set_glove_path(GLOVE_PATH)
    LOG.debug('Building vocab')
    sents = [sent for doc in (tgt_docs + src_docs) for sent in doc]
    infersent.build_vocab(sents, tokenize=True)
    LOG.debug('Encoding sentences')
    all_vecs = infersent.encode(sents, tokenize=True)
    LOG.debug('Encoding complete!')
    # Build indices for target and source documents
    cumulative_tgt_docs_length = np.cumsum([0] + [len(doc) for doc in tgt_docs])
    tgt_ids = [list(range(start, end))
               for start, end in pairwise(cumulative_tgt_docs_length)]
    cumulative_src_docs_length = np.cumsum([cumulative_tgt_docs_length[-1]] + [len(doc) for doc in src_docs])
    src_ids = [list(range(start, end))
               for start, end in pairwise(cumulative_src_docs_length)]
    return tgt_ids, src_ids, all_vecs

if __name__ == "__main__":
    LOG.debug('Reading Webis dataset...')
    topics, source_sentences, target_sentences, golds = list_dataset()
    LOG.debug('Creating sentence embeddings...')
    tgt_ids, src_ids, vocab = get_sent_embs(target_sentences, source_sentences)
    LOG.debug('Dumping data...')
    pickle.dump([topics, tgt_ids, src_ids, vocab, golds], open(SAVE_FILE_PATH, "wb"))
