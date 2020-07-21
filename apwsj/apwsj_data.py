"""
Serialize APWSJ documents to include infersent sentence embeddings
such that we can represent documents as matrices.
Dumps relevant data into a pickle file, which will be loaded for
creating a model.

Run this script with python3 and pytorch 0.3 with GPU support
"""

import logging
import nltk
import os
import pickle
import sys
import torch
torch.cuda.set_device(2)

# Config variables
HOME_DIR = "/home1/tirthankar"
APWSJ_PATH = os.path.join(HOME_DIR, "Vignesh/dmn/dlnd/apwsj/apwsj_parsed_documents")
APWSJ_RED_RESULT = 'redundancy.apwsj.result'
APWSJ_ORDER = 'apwsj88-90.rel.docno.sorted'
GLOVE_PATH = os.path.join(HOME_DIR, "glove.840B.300d.txt")
ENCODER_DIR = os.path.join(HOME_DIR, "Vignesh/InferSent")
ENCODER_PATH = os.path.join(ENCODER_DIR, "models/model_2048_attn.pickle")
SAVE_FILE_PATH = "apwsj_data.p"

# Infersent should be in the path to load the encoder
sys.path.append(ENCODER_DIR)
# dlnd directory should be in the path to initialize logger
sys.path.append(os.path.dirname(os.getcwd()))
from dlnd_logger import init_logger
LOG = logging.getLogger(filename='apwsj_logs')
init_logger(LOG)

def list_docs():
    """
    List of documents of the form <topic>/<doc_name>
    """
    new_docs = list()
    for topic in os.listdir(APWSJ_PATH):
        topic_path = os.path.join(APWSJ_PATH, topic)
        for doc in os.listdir(topic_path):
            new_docs.append(os.path.join(topic, doc))
    return new_docs

def read_contents(new_docs):
    """
    Tokenize content of a document into sentences
    """
    new_contents = list()
    for doc in new_docs:
        doc_path = os.path.join(APWSJ_PATH, doc)
        content = open(doc_path, "r").read().decode("utf-8", "ignore")
        new_contents.append(nltk.sent_tokenize(content))
    return new_contents

def get_golds(new_docs):
    """
    Get the list of redundant (including partial) documents
    and use it populate golds answers for each document
    """
    red_docs = list()
    lines = open(APWSJ_RED_RESULT, 'r').read().splitlines()
    red_docs = ['/'.join(line.split()[:2]) for line in lines]
    # Prepare a list of gold values
    new_golds = list()
    for doc in new_docs:
        if doc in red_docs:
            # Redundant documents are non-novel
            new_golds.append(0)
        else:
            new_golds.append(1)
    return new_golds

def get_sent_embs(new_docs):
    """
    Convert the given documents into
    document sentence matrices using sentence
    embeddings from infersent
    """
    # Read the contents
    new_contents = read_contents(new_docs)
    LOG.debug('Loading encoder')
    infersent = torch.load(ENCODER_PATH)
    # use the encoder network only
    infersent = infersent.encoder
    LOG.debug('Loading glove vectors')
    infersent.set_glove_path(GLOVE_PATH)
    LOG.debug('Building vocab')
    sents = [sent for doc in new_contents for sent in doc]
    infersent.build_vocab(sents, tokenize=True)
    LOG.debug('Encoding sentences')
    all_vecs = infersent.encode(sents, tokenize=True)
    LOG.debug('Encoding complete!')
    # Convert documents into sentence indices
    new_questions = list()
    i = 0
    for doc in new_contents:
        new_questions.append(list(range(i, i+len(doc))))
        i += len(doc)
    return new_questions, all_vecs

def get_context_ids(doc, new_docs, order):
    """
    Index all documents with the same topic as the given document.
    Prepare list of contexts indices according to the order file.
    """
    # Look for all docs having the same topic and index them
    idx_map = dict()
    for idx, new_item in enumerate(new_docs):
        if os.path.dirname(new_item) == os.path.dirname(doc):
            idx_map[os.path.basename(new_item)] = idx
    # Prepare a list of ordered indices
    indices = list()
    for new_item in order:
        # Only consider items that occur before doc
        if new_item == os.path.basename(doc):
            break
        idx = idx_map.get(new_item, None)
        if idx:
            indices.append(idx)
    return indices

def get_contexts(new_docs):
    """Get context indices for each question
       Then stack sentence indices of each context

    Params:
    new_docs: List of documents

    Return:
    contexts: Context ids
    """
    order = open(APWSJ_ORDER, 'r').read().splitlines()
    new_contexts = list()
    for doc in new_docs:
        ctx_ids = get_context_ids(doc, new_docs, order)
        new_contexts.append(ctx_ids)
    return new_contexts

if __name__ == "__main__":
    # List the documents
    docs = list_docs()
    # Get contexts
    contexts = get_contexts(docs)
    # Get sentence embeddings
    questions, vocab = get_sent_embs(docs)
    # Get Novelty judegements
    golds = get_golds(docs)
    LOG.debug('Dumping data')
    # Documents, Contexts indices, Question indices, Vocabulary, Gold values
    # contexts = (#docs, #contexts)
    # questions = (#docs, #sents)
    # vocab = (#docs * #sents, #embedding)
    pickle.dump([docs, contexts, questions, vocab, golds], open(SAVE_FILE_PATH, "wb"))
