"""
Serailize Stack Exchange documents to include infersent sentence
embeddings such that we can represent documents as matrices.
Dumps relevant data into a pickle file, which will be loaded for
creating a model.

Run this script with python3 and pytorch 0.3 with GPU support
"""

from itertools import tee
from nltk import sent_tokenize, word_tokenize
import numpy as np
import logging
import os
import pickle
import sys

# Path Variables
GPU_ID = 1
HOME_DIR = "/home1/tirthankar"
STE_PATH = os.path.join(HOME_DIR, 'Vignesh/dmn/code/ste/stack_exchange_data/corpus')
GLOVE_PATH = os.path.join(HOME_DIR, 'glove.840B.300d.txt')
ENCODER_DIR = os.path.join(HOME_DIR, 'Vignesh/InferSent')
ENCODER_PATH = os.path.join(ENCODER_DIR, 'models/model_2048_attn.pickle')
LOGGING_MODULE_PATH = os.path.join(HOME_DIR, 'Vignesh/dmn/code')
LOG_PATH = os.path.join(LOGGING_MODULE_PATH, 'ste/ste_logs')
SAVE_FILE_PATH = 'ste_data.p'

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
    List subtopics, source sentences, target sentences and gold values.
    """
    # Accumlator lists
    subtopics = []
    source_sentences = []
    target_sentences = []
    golds = []
    for topic in os.listdir(STE_PATH):
        # Dont process non directories
        if not os.path.isdir(os.path.join(STE_PATH, topic)):
            continue
        # Keep track of the current subtopic
        # being processed within the csv file
        current_subtopic = None
        source_sentence = None
        with open(os.path.join(STE_PATH, topic, 'results.csv'), 'r') as fp:
            LOG.debug('Processing topic: %s', topic)
            reader = csv.reader(fp)
            # ignore header
            reader.__next__()
            for row in reader:
                # ignore empty row
                if not row:
                    continue
                sentence, _, _, gold, order, subtopic = row
                subtopic = os.path.join(topic, subtopic)
                if order == '0':
                    current_subtopic = subtopic
                    source_sentence = sentence
                    continue
                if order == '1' and subtopic == current_subtopic:
                    # Filter out sentences which have >100 words
                    source_sents = [sent
                                    for sent in sent_tokenize(source_sentence)
                                    if len(word_tokenize(sent)) <= 100]
                    target_sents = [sent
                                    for sent in sent_tokenize(sentence)
                                    if len(word_tokenize(sent)) <= 100]
                    if source_sents and target_sents:
                        subtopics.append(subtopic)
                        source_sentences.append(source_sents)
                        target_sentences.append(target_sents)
                        golds.append(1 if gold == 'True' else 0)
                    current_subtopic = None
                    source_sentence = None
            LOG.debug('Completed processing topic: %s', topic)
    return subtopics, source_sentences, target_sentences, golds

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
    infersent = torch.load(ENCODER_PATH)
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
    LOG.debug('Reading Stack Exchange dataset...')
    subtopics, source_sentences, target_sentences, golds = list_dataset()
    LOG.debug('Creating sentence embeddings...')
    tgt_ids, src_ids, vocab = get_sent_embs(target_sentences, source_sentences)
    LOG.debug('Dumping data...')
    pickle.dump([subtopics, tgt_ids, src_ids, vocab, golds], open(SAVE_FILE_PATH, "wb"))
