# Run in base environment base in python 2
from collections import defaultdict
import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
from nltk import sent_tokenize, word_tokenize
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
from keras.layers import Input,Flatten,Bidirectional, GRU, LSTM
from keras import backend as K
from gensim.models.doc2vec import Doc2Vec
from nltk.corpus import stopwords
import string

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
            next(reader)
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
                    source_sentence = source_sentence.decode('utf-8').encode('ascii', 'ignore')
                    sentence = sentence.decode('utf-8').encode('ascii', 'ignore')
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

def doc_to_mat(docs,max_sents):
    if_word_in_stopwords = defaultdict(int)
    for word in stopwords:
        if_word_in_stopwords[word] = 1
    mat = np.zeros((len(docs),max_sents,SENT_DIM),dtype="float32")
    for i in range(len(docs)):
        docs[i] = docs[i][:max_sents]
        for j in range(len(docs[i])):
            words = [word for word in docs[i][j] if if_word_in_stopwords[word]==0]
            sent_vec = pv_model.infer_vector(doc_words=words, alpha=0.1, min_alpha=0.0001, steps=5)
            mat[i,max_sents-len(docs[i])+j] = sent_vec
    return mat

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
    LOG.debug('Processing topic: %s', topic)
    ids = [idx for idx in range(len(subtopics)) if subtopics[idx].split('/')[0] == topic]
    max_sents = max([len(doc) for doc in source_sentences + target_sentences])
    LOG.debug("Max sentences: %d", max_sents)

    target_vecs = doc_to_mat([target_sentences[idx] for idx in ids], max_sents)
    source_vecs = doc_to_mat([source_sentences[idx] for idx in ids], max_sents)
    golds = [all_golds[idx] for idx in ids]
    gold_list = [i for i in golds]
    golds = to_categorical(golds)

    train, test = train_test_split(
                    list(range(len(gold_list))),
                    train_size=0.8,
                    random_state=9274,
                    shuffle=True,
                    stratify=gold_list)
    LOG.debug("Compiling model")
    tgt = Input(shape=(max_sents,SENT_DIM), dtype='float32')
    srcs = Input(shape=(max_sents,SENT_DIM), dtype='float32')
    encode = Bidirectional(LSTM(SENT_DIM/2, return_sequences=False,
                                     dropout_W=0.0, dropout_U=0.0),
                                     input_shape=(max_sents, SENT_DIM))
    tgt_vec = encode(tgt)
    src_vec = encode(srcs)
    pds = _Entailment(SENT_DIM,NUM_CLASSES,dropout=0.2)(tgt_vec,src_vec)
    model = Model(input=[tgt,srcs],output=pds)
    model.summary()
    model.compile(loss="categorical_crossentropy",optimizer=Adam(lr=0.001),metrics=["accuracy"])

    NUM_EPOCHS = 25 
    BATCH_SIZE = 32 

    LOG.debug("Training model")
    model.fit(x=[target_vecs[train],source_vecs[train]],y=golds[train],batch_size=BATCH_SIZE,nb_epoch=NUM_EPOCHS,shuffle=True,verbose=2)

    preds = model.predict([target_vecs[test],source_vecs[test]])
    preds = np.argmax(preds,axis=1)
    gold_test = np.argmax(golds[test],axis=1)
    test_acc = accuracy_score(gold_test, preds)
    LOG.debug("Testing accuracy: %0.3f", test_acc)
    LOG.debug("Confusion matrix:\n%s\n\n", confusion_matrix(gold_test,preds))

    # Write completed topic
    with open(completed_topics_file, 'a') as fh:
        fh.write('\n{}'.format(topic))

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
HOME_DIR = "/home1/tirthankar"
STE_PATH = os.path.join(HOME_DIR, 'Vignesh/dmn/code/ste/stack_exchange_data/corpus')
SENT_DIM = 300
NUM_CLASSES = 2
CHECKPOINT1 = 'ste_doc2vec_mlp_baseline_checkpoint1.p'

pv_model = Doc2Vec.load(os.path.join(HOME_DIR, 'enwiki_dbow/doc2vec.bin'))
stopwords = list(string.punctuation)+list(set(stopwords.words('english')))

LOG = logging.getLogger()
init_logger(LOG, 'ste_doc2vec_mlp_baseline.log')

# Checkpoint 1
if os.path.isfile(CHECKPOINT1):
    subtopics, source_sentences, target_sentences, all_golds = pickle.load(open(CHECKPOINT1, 'rb'))
else:
    subtopics, source_sentences, target_sentences, all_golds = list_dataset()
    pickle.dump([subtopics, source_sentences, target_sentences, all_golds], open(CHECKPOINT1, 'wb'))

# Oversample minority class
subtopics, source_sentences, target_sentences, all_golds = oversample(
    [subtopics, source_sentences, target_sentences, all_golds], 0)

unique_topics = set([subtopic.split('/')[0] for subtopic in subtopics])
completed_topics_file = 'ste_doc2vec_mlp_baseline_completed_topics.txt'
with open(completed_topics_file, 'r') as fh:
    completed_topics = fh.read().splitlines()

for topic in unique_topics:
    if topic in completed_topics:
        continue
    process_topic()
