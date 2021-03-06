"""
Serialize DLND2 documents to include infersent sentence embeddings
such that we can represent documents as matrices.
Dumps relevant data into a pickle file, which will be loaded for
creating a model.

Run this script with python3 and pytorch 0.3 with GPU support
"""

import logging
import nltk
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pickle
import sys
sys.path.append(ROOT_DIR)
import torch
torch.cuda.set_device(2)
import xml.etree.ElementTree as ET
from logger import init_logger
from itertools import chain

LOG = logging.getLogger()
init_logger(LOG, os.path.join(ROOT_DIR, 'dlnd2/dlnd2_logs'))

# Config variables
HOME_DIR = "/home1/tirthankar"
DLND2_PATH = os.path.join(HOME_DIR, "Vignesh/DLND2/TAPNew")
GLOVE_PATH = os.path.join(HOME_DIR, "glove.840B.300d.txt")
ENCODER_DIR = os.path.join(HOME_DIR, "Vignesh/InferSent")
ENCODER_PATH = os.path.join(ENCODER_DIR, "models/model_2048_attn.pickle")
SAVE_FILE_PATH = "dlnd2_data.p"

# Infersent should be in the path to load the encoder
sys.path.append(ENCODER_DIR)

def list_docs():
    """
    List source and target documents
    """
    source_docs = list()
    tgt_docs = list()
    for genre in os.listdir(DLND2_PATH):
        genre_dir = os.path.join(DLND2_PATH, genre)
        if os.path.isdir(genre_dir) and not genre_dir.startswith('.'):
            for topic in os.listdir(genre_dir):
                topic_dir = os.path.join(genre_dir, topic)
                if not os.path.isdir(topic_dir):
                    continue
                tgt_dir = os.path.join(topic_dir, 'target')
                source_dir = os.path.join(topic_dir, 'source')
                for doc in os.listdir(tgt_dir):
                    if doc.endswith('.txt'):
                        tgt_docs.append([os.path.join(tgt_dir, doc)])
                rel_docs = list()
                for doc in os.listdir(source_dir):
                    if doc.endswith('.txt'):
                        rel_docs.append(os.path.join(source_dir, doc))
                source_docs.append(rel_docs)
    return tgt_docs, source_docs

def read_contents(new_docs):
    """
    Tokenize content of a document into sentences
    """
    new_contents = list()
    for doc_list in new_docs:
        doc_content = list()
        for doc in doc_list:
            content = open(doc, 'rb').read().decode('utf-8', 'ignore')
            doc_content.append(nltk.sent_tokenize(content))
        new_contents.append(list(chain.from_iterable(doc_content)))
    return new_contents

def get_golds(new_docs):
    """
    Get the list of redundant (including partial) documents
    and use it populate golds answers for each document
    """
    new_golds = list()
    for doc in new_docs:
        for tag in ET.parse(doc[0][:-4] + '.xml').findall('feature'):
            if 'SLNS' in tag.attrib:
                new_golds.append(float(tag.attrib['SLNS']))
    return new_golds

def get_sent_embs(tgt_docs, src_docs):
    """
    Convert the given documents into
    document sentence matrices using sentence
    embeddings from infersent
    """
    # Read the contents
    tgt_contents = read_contents(tgt_docs)
    src_contents = read_contents(src_docs)
    LOG.debug('Loading encoder')
    infersent = torch.load(ENCODER_PATH)
    # use the encoder network only
    infersent = infersent.encoder
    LOG.debug('Loading glove vectors')
    infersent.set_glove_path(GLOVE_PATH)
    LOG.debug('Building vocab')
    sents = [sent for doc in (tgt_contents + src_contents) for sent in doc]
    infersent.build_vocab(sents, tokenize=True)
    LOG.debug('Encoding sentences')
    all_vecs = infersent.encode(sents, tokenize=True)
    LOG.debug('Encoding complete!')
    # Convert documents into sentence indices
    tgt_ids = list()
    i = 0
    for doc in tgt_contents:
        tgt_ids.append(list(range(i, i+len(doc))))
        i += len(doc)
    src_ids = list()
    j = 0
    for doc in tgt_docs:
        tgt_topic = doc[0].split('/')[-3]
        src_topic = src_docs[j][0].split('/')[-3]
        if not tgt_topic == src_topic:
            j += 1
            i += len(src_contents[j])
        src_ids.append(list(range(i, i+len(src_contents[j]))))
    return tgt_ids, src_ids, all_vecs

if __name__ == "__main__":
    # List the documents
    tgt_docs, src_docs = list_docs()
    # Get sentence embeddings
    tgt_ids, src_ids, vocab = get_sent_embs(tgt_docs, src_docs)
    # Get Novelty judegements
    golds = get_golds(tgt_docs)
    LOG.debug('Dumping data')
    # Expand src_docs to the same size as tgt_docs
    j = 0
    new_src_docs = []
    for doc in tgt_docs:
        tgt_topic = doc[0].split('/')[-3]
        src_topic = src_docs[j][0].split('/')[-3]
        if not tgt_topic == src_topic:
            j += 1
        new_src_docs.append(src_docs[j])
    # Source docs, Target docs, Source indices, Target indices, Vocabulary, Gold values
    # src_ids = (#docs, #sents)
    # tgt_ids = (#docs, #sents)
    # vocab = (#docs * #sents, #embedding)
    pickle.dump([new_src_docs, tgt_docs, src_ids, tgt_ids, vocab, golds], open(SAVE_FILE_PATH, "wb"))
