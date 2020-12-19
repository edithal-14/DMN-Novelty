# Run in base environment base in python 2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
import json
import pickle
import gc
import numpy as np
import logging
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from spacy_decomposable_attention import _Entailment
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import (
    Bidirectional,
    Embedding,
    Flatten,
    Input,
    LSTM,
    GRU
)
from keras import backend as K

def pad_or_truncate1(arr, max_sents):
    pad_sent_idx = vocab.shape[0] - 1
    arr_len = len(arr)
    new_arr = arr + [pad_sent_idx for i in range(max_sents - arr_len)]
    return new_arr

def init_logger(logger, filename):
    """
    Init given logger object
    """
    log_format = logging.Formatter("%(asctime)s %(message)s")

    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)

def process_topic():
#     LOG.debug('Processing topic: %s', topic)
#     ids = [idx for idx in range(len(subtopics)) if subtopics[idx].split('/')[0] == topic]
#     tgt_ids = [all_tgt_ids[idx] for idx in ids]
#     src_ids = [all_src_ids[idx] for idx in ids]
#     golds = [all_golds[idx] for idx in ids]
    tgt_ids = all_tgt_ids
    src_ids = all_src_ids
    golds = all_golds

    max_sents= max([len(doc) for doc in src_ids + tgt_ids])
    LOG.debug("Max sentences: %d", max_sents)

    # target_vecs = [np.vstack([vocab[sent_idx] for sent_idx in doc]) for doc in tgt_ids]
    # source_vecs = [np.vstack([vocab[sent_idx] for sent_idx in doc]) for doc in src_ids]
    # target_vecs = np.array([pad_or_truncate1(mat,max_sents) for mat in target_vecs])
    # source_vecs = np.array([pad_or_truncate1(mat,max_sents) for mat in source_vecs])
    tgt_ids = np.array([pad_or_truncate1(doc, max_sents) for doc in tgt_ids])
    src_ids = np.array([pad_or_truncate1(doc, max_sents) for doc in src_ids])
    gold_list = [i for i in golds]
    golds = to_categorical(golds)

    train, test = train_test_split(
                    list(range(len(gold_list))),
                    train_size=0.8,
                    random_state=9274,
                    shuffle=True,
                    stratify=gold_list)
    LOG.debug("Compiling model")
    emb = Embedding(
            vocab.shape[0],
            vocab.shape[1],
            weights=[vocab],
            input_length=max_sents,
            trainable=False)
    # tgt = Input(shape=(max_sents,SENT_DIM), dtype='float32')
    # srcs = Input(shape=(max_sents,SENT_DIM), dtype='float32')
    tgt = Input(shape=(max_sents,), dtype='int32')
    with tf.device('/device:CPU:0'):
        tgt_emb = emb(tgt)

    srcs = Input(shape=(max_sents,), dtype='int32')
    with tf.device('/device:CPU:0'):
        srcs_emb = emb(srcs)

    encode = Bidirectional(LSTM(SENT_DIM/2, return_sequences=False,
                                     dropout_W=0.0, dropout_U=0.0),
                                     input_shape=(max_sents, SENT_DIM))
    # tgt_vec = encode(tgt)
    # src_vec = encode(srcs)
    tgt_vec = encode(tgt_emb)
    src_vec = encode(srcs_emb)
    pds = _Entailment(SENT_DIM,NUM_CLASSES,dropout=0.2)(tgt_vec,src_vec)
    model = Model(input=[tgt,srcs],output=pds)
    model.summary()
    model.compile(loss="categorical_crossentropy",optimizer=Adam(lr=0.001),metrics=["accuracy"])

    NUM_EPOCHS = 25 
    BATCH_SIZE = 32 

    LOG.debug("Training model")
    model.fit(x=[tgt_ids[train],src_ids[train]],y=golds[train],batch_size=BATCH_SIZE,nb_epoch=NUM_EPOCHS,shuffle=True,verbose=2)

    preds = model.predict([tgt_ids[test],src_ids[test]])
    preds = np.argmax(preds,axis=1)
    gold_test = np.argmax(golds[test],axis=1)
    test_acc = accuracy_score(gold_test, preds)
    LOG.debug("Testing accuracy: %0.3f", test_acc)
    LOG.debug("Confusion matrix:\n%s\n\n", confusion_matrix(gold_test,preds))

    # Write completed topic
    # with open(completed_topics_file, 'a') as fh:
    #     fh.write('\n{}'.format(topic))

    # Cleanup memory
    del model
    gc.collect()
    K.clear_session()
    tf.reset_default_graph()

def oversample(data, minority_class):
    """
    Random oversampling of minority class
    class labels:
    0: non-novel class
    1: novel class
    """
    random.seed(1234)
    answers = data[-1]
    n_minority = len([answer for answer in answers if answer == minority_class])
    n_samples_add = len(answers) - (2 * n_minority)
    minority_ids = [idx for idx, answer in enumerate(answers) if answer == minority_class]
    random_ids = [random.choice(minority_ids) for i in range(n_samples_add)]
    for item_no, _ in enumerate(data):
        data[item_no] += [data[item_no][idx] for idx in random_ids]
    return data

# main
SENT_DIM = 2048
NUM_CLASSES = 2
LOG = logging.getLogger()
init_logger(LOG, 'ste_bilstm_mlp_baseline.log')

with open('ste-subtopics-tgt_ids-src_ids-golds.json', 'r') as fh:
    contents = json.loads(fh.read())
    all_tgt_ids, all_src_ids, all_golds, subtopics = [
            contents[key] for key in ['tgt_ids', 'src_ids', 'golds', 'subtopics']]
vocab = np.load('ste_vocab.npy')
# Add an empty sentence embedding at the end
vocab = np.pad(vocab,((0,1),(0,0)), 'constant') 


# Oversample minority class
all_tgt_ids, all_src_ids, subtopics, all_golds = oversample(
    [all_tgt_ids, all_src_ids, subtopics, all_golds], 0)

# unique_topics = set([subtopic.split('/')[0] for subtopic in subtopics])
# completed_topics_file = 'ste_bilstm_mlp_baseline_completed_topics.txt'
# with open(completed_topics_file, 'r') as fh:
#     completed_topics = fh.read().splitlines()

# for topic in unique_topics:
#     if topic in completed_topics:
#         continue
process_topic()
