"""
Serialize DLND documents to include infersent/quickthought sentence embeddings
such that we can represent documents as matrices.
Dumps relevant data into a pickle file, which will be loaded for
creating a model.

Run this script with python3 and pytorch 0.3 with GPU support for infersent
Run this script with python2 and tensorflow-gpu 1.14.0 with GPU support for quickthought
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
from logging import init_logger
from itertools import chain

LOG = logging.getLogger()
init_logger(LOG, os.path.join(ROOT_DIR, 'dlnd/dlnd_logs'))

# Config variables
HOME_DIR = "/home1/tirthankar"
DLND_PATH = os.path.join(HOME_DIR, "Vignesh/TAP-DLND-1.0_LREC2018_modified")
GLOVE_PATH = os.path.join(HOME_DIR, "glove.840B.300d.txt")
INFERSENT_ENCODER_DIR = os.path.join(HOME_DIR, "Vignesh/InferSent")
INFERSENT_ENCODER_PATH = os.path.join(ENCODER_DIR, "models/model_2048_attn.pickle")
QT_ENCODER_DIR = os.path.join(HOME_DIR, 'Vignesh/S2V')
QT_ENCODER_PATH = os.path.join(QT_ENCODER_DIR, 's2v_models')
QT_ENCODER_CONFIG = os.path.join(QT_ENCODER_DIR, 'model_configs/eval.json')
SAVE_FILE_PATH = "dlnd_data.p"
ENCODER = 'infersent'

# Infersent/Quickthought should be in the path to load the encoder
sys.path.append(INFERSENT_ENCODER_DIR, QT_ENCODER_DIR, '%s/src' % QT_ENCODER_DIR)

def list_docs():
    """
    List source and target documents
    """
    source_docs = list()
    tgt_docs = list()
    for genre in os.listdir(DLND_PATH):
        genre_dir = os.path.join(DLND_PATH, genre)
        if os.path.isdir(genre_dir):
            for topic in os.listdir(genre_dir):
                topic_dir = os.path.join(genre_dir, topic)
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

def read_contents(new_docs, encoder='infersnet'):
    """
    Tokenize content of a document into sentences
    """
    new_contents = list()
    for doc_list in new_docs:
        doc_content = list()
        for doc in doc_list:
            content = open(doc, 'rb').read().decode('utf-8', 'ignore')
            if encoder == 'infersent':
                sents = nltk.sent_tokenize(content)
            elif encoder == 'quickthought':
                # Try to split a sentence which was left unsplit by nltk
                sents = [subsent for sent in sents for subsent in sent.splitlines()]
                # Filter out sentences with zero length
                sents = [sent for sent in sents if len(sent.strip()) > 0]
                # Trim sentences to have a maximum of 90 tokens only.
                sents = [' '.join(sent.split(' ')[:90]) for sent in sents]
            doc_content.append(sents)
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
            if 'DLA' in tag.attrib:
                if tag.attrib['DLA'] == 'Novel':
                    new_golds.append(1)
                else:
                    new_golds.append(0)
    return new_golds

def get_sent_embs(tgt_docs, src_docs, encoder='infersent'):
    """
    Convert the given documents into
    document sentence matrices using sentence
    embeddings from infersent
    """
    # Read the contents
    tgt_contents = read_contents(tgt_docs, encoder)
    src_contents = read_contents(src_docs, encoder)
    if encoder == 'infersent':
        LOG.debug('Loading encoder')
        infersent = torch.load(INFERSENT_ENCODER_PATH)
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
    elif encoder == 'quickthought':
        LOG.debug('Loading encoder')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
        import tensorflow as tf
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        tf.flags.DEFINE_float("uniform_init_scale", 0.1, "Random init scale")
        tf.flags.DEFINE_string("results_path", QT_ENCODER_PATH, "Model results path")
        from src import encoder_manager, configuration
        qt_encoder = encoder_manager.EncoderManager()
        model_cfg = json.load(open(QT_ENCODER_CONFIG, 'r'))
        for mdl_cfg in model_cfg:
            qt_encoder.load_model(configuration.model_config(mdl_cfg, mode='encode'))
        LOG.debug('Encoding sentences')
        all_vecs = qt_encoder.encode(sents, use_norm=False, batch_size=400)
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
    tgt_ids, src_ids, vocab = get_sent_embs(tgt_docs, src_docs, encoder=ENCODER)
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
