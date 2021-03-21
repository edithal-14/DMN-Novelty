"""
Run DMN model
Command: python dmn.py <dataset_name>

Usage notes: Run in python3 in conda environment named vignesh
"""

# Config variables
import os
import sys
import gc

NUM_FOLDS = 5
NUM_HOPS = 4
NUM_EPOCHS = 10
BATCH_SIZE = 16
EARLY_STOP_THRESHOLD = 4
PRE_TRAINED_MODEL = None
EPOCH_OFFSET = 1
GPU_ID = 6
DATASET_NAME = sys.argv[1]
IS_REGRESSION = False
if DATASET_NAME == 'DLND':
    LOGFILE = 'dlnd/dlnd_logs'
    IS_SENTENCE_LEVEL = True
elif DATASET_NAME == 'APWSJ':
    LOGFILE = 'apwsj/apwsj_logs'
    IS_SENTENCE_LEVEL = True
elif DATASET_NAME == 'STE':
    LOGFILE = 'ste/ste_logs'
    IS_SENTENCE_LEVEL = False
elif DATASET_NAME == 'WEBIS':
    LOGFILE = 'webis/webis_logs'
    IS_SENTENCE_LEVEL = True
elif DATASET_NAME == 'DLND2':
    LOGFILE = 'dlnd2/dlnd2_sent_logs'
    IS_SENTENCE_LEVEL = True
    IS_REGRESSION = True
else:
    raise Exception('Dataset name %s is not supported!' % DATASET_NAME)
HOME_DIR = "/home1/tirthankar"
ENCODER_DIR = os.path.join(HOME_DIR, "Vignesh/InferSent")
ENCODER_PATH = os.path.join(ENCODER_DIR, "models/model_2048_attn.pickle")
# Infersent should be in the path
sys.path.append(ENCODER_DIR)

import logging
import numpy as np
import pickle
from scipy.spatial import distance
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
torch.cuda.set_device(GPU_ID)
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import warnings
from functools import partial
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from loader import DmnData, pad_collate
from logger import init_logger

# Suppress all warnings
warnings.simplefilter('ignore')

# Initialize logging
LOG = logging.getLogger()
init_logger(LOG, LOGFILE)

class AttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.Wr = nn.Linear(input_size, hidden_size)
        init.xavier_normal(self.Wr.state_dict()['weight'])
        self.Ur = nn.Linear(hidden_size, hidden_size)
        init.xavier_normal(self.Ur.state_dict()['weight'])
        self.W = nn.Linear(input_size, hidden_size)
        init.xavier_normal(self.W.state_dict()['weight'])
        self.U = nn.Linear(hidden_size, hidden_size)
        init.xavier_normal(self.U.state_dict()['weight'])

    def forward(self, fact, C, g):
        '''
        fact.size() -> (#batch, #hidden = #embedding)
        c.size() -> (#hidden, ) -> (#batch, #hidden = #embedding)
        r.size() -> (#batch, #hidden = #embedding)
        h_tilda.size() -> (#batch, #hidden = #embedding)
        g.size() -> (#batch, )
        '''

        r = F.sigmoid(self.Wr(fact) + self.Ur(C))
        h_tilda = F.tanh(self.W(fact) + r * self.U(C))
        g = g.unsqueeze(1).expand_as(h_tilda)
        h = g * h_tilda + (1 - g) * C
        return h

class AttentionGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.AGRUCell = AttentionGRUCell(input_size, hidden_size)

    def forward(self, facts, G):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        fact.size() -> (#batch, #hidden = #embedding)
        G.size() -> (#batch, #sentence)
        g.size() -> (#batch, )
        C.size() -> (#batch, #hidden)
        '''
        batch_num, sen_num, embedding_size = facts.size()
        C = Variable(torch.zeros(self.hidden_size)).cuda()
        for sid in range(sen_num):
            fact = facts[:, sid, :]
            g = G[:, sid]
            if sid == 0:
                C = C.unsqueeze(0).expand_as(fact)
            C = self.AGRUCell(fact, C, g)
        return C

class EpisodicMemory(nn.Module):
    def __init__(self, hidden_size):
        super(EpisodicMemory, self).__init__()
        # Store attention values for each hop
        self.att_vals = list()
        self.AGRU = AttentionGRU(hidden_size, hidden_size)
        self.z1 = nn.Linear(4 * hidden_size, hidden_size)
        self.z2 = nn.Linear(hidden_size, 1)
        self.next_mem = nn.Linear(3 * hidden_size, hidden_size)
        init.xavier_normal(self.z1.state_dict()['weight'])
        init.xavier_normal(self.z2.state_dict()['weight'])
        init.xavier_normal(self.next_mem.state_dict()['weight'])

    def clear_stored_att_vals(self):
        '''
        Clear the attention values stored in att_vals before the start
        of the episodic memory module
        '''
        self.att_vals = list()

    def make_interaction(self, facts, question, prevM):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        question.size() -> (#batch, 1, #hidden)
        prevM.size() -> (#batch, #sentence = 1, #hidden = #embedding)
        z.size() -> (#batch, #sentence, 4 x #embedding)
        G.size() -> (#batch, #sentence)
        '''
        batch_num, sen_num, embedding_size = facts.size()
        question = question.expand_as(facts)
        prevM = prevM.expand_as(facts)

        z = torch.cat([
            facts * question,
            facts * prevM,
            torch.abs(facts - question),
            torch.abs(facts - prevM)
        ], dim=2)

        z = z.view(-1, 4 * embedding_size)

        G = F.tanh(self.z1(z))
        G = self.z2(G)
        G = G.view(batch_num, -1)
        G = F.softmax(G)

        return G

    def forward(self, facts, question, prevM):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        question.size() -> (#batch, #sentence = 1, #hidden)
        prevM.size() -> (#batch, #sentence = 1, #hidden = #embedding)
        G.size() -> (#batch, #sentence)
        C.size() -> (#batch, #hidden)
        concat.size() -> (#batch, 3 x #embedding)
        '''
        G = self.make_interaction(facts, question, prevM)
        # Store attention values for each hop
        self.att_vals.append(G)
        C = self.AGRU(facts, G)
        concat = torch.cat([prevM.squeeze(1), C, question.squeeze(1)], dim=1)
        next_mem = F.relu(self.next_mem(concat))
        next_mem = next_mem.unsqueeze(1)
        return next_mem

class QuestionModule(nn.Module):
    def __init__(self, hidden_size):
        super(QuestionModule, self).__init__()
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, questions):
        '''
        questions.size() -> (#batch, #sentence, #embedding)
        gru() -> (1, #batch, #hidden)
        '''
        _, questions = self.gru(questions)
        questions = questions.transpose(0, 1)
        return questions

class InputModule(nn.Module):
    def __init__(self, hidden_size, enable_pruning=False):
        super(InputModule, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        for name, param in self.gru.state_dict().items():
            if 'weight' in name: init.xavier_normal(param)
        self.dropout = nn.Dropout(0.1)
        self.enable_pruning = enable_pruning
        if self.enable_pruning:
            infersent = torch.load(ENCODER_PATH, map_location=torch.device('cuda')).cuda()
            self.entailment = infersent.classifier

    def prune_contexts(self, contexts, question):
        """
        1. Prune the top 10 context sentences based on the entailment score w.r.t question
        2. Add +/- 5 sentences as context for each sentence selected above
        3. Pad the final context matrix to the size of the biggest context matrix
        """
        num_batch, num_sent, embedding_dim = contexts.size()
        # select topn context sentences
        topn = min(num_sent, 20)
        # context size for topn context sentences
        topn_context_size = 5
        # question and contexts should have same dimensionality for feature extraction
        question = question.expand(contexts.size())
        features = torch.cat(
            (contexts, question, torch.abs(contexts - question), contexts * question), 2)
        # unbatch features and calculate entailment scores then batch up the entailment scores
        squashed_features = features.view(num_batch * num_sent, 4 * embedding_dim)
        entailment_scores = self.entailment(squashed_features)[:, 0]
        entailment_scores = entailment_scores.view(num_batch, num_sent)
        _, sentence_ids = torch.topk(entailment_scores, topn)
        # Add +/- 5 sentences to each id
        contextualized_sentence_ids = []
        max_num_ids = 0
        for i in range(num_batch):
            ids = torch.cuda.LongTensor([])
            for idx in sentence_ids[i]:
                prev_ids = torch.arange(max(0, idx - topn_context_size), idx).cuda()
                ids = torch.cat((ids, prev_ids))
                next_ids = torch.arange(idx, min(idx + topn_context_size + 1, num_sent)).cuda()
                ids = torch.cat((ids, next_ids))
            # Remove duplicate sentence indices
            ids = torch.unique(ids, sorted=True)
            num_ids = ids.size()[0]
            if num_ids > max_num_ids:
                max_num_ids = num_ids
            contextualized_sentence_ids.append(ids)
        # apply padding based on max_num_ids
        padded_contextualized_pruned_contexts = \
                torch.zeros(num_batch, max_num_ids, embedding_dim).cuda()
        for i in range(num_batch):
            num_ids = contextualized_sentence_ids[i].size()[0]
            padded_contextualized_pruned_contexts[i][max_num_ids - num_ids : max_num_ids] = \
                    contexts[i].index_select(0, contextualized_sentence_ids[i])
        return padded_contextualized_pruned_contexts

    def forward(self, contexts, question=None):
        '''
        contexts.size() -> (#batch, #context, #embedding)
        # question.size() -> (#batch, #embedding)
        facts.size() -> (#batch, #context, #hidden)
        '''
        if self.enable_pruning:
            contexts = self.prune_contexts(contexts, question)
        contexts = self.dropout(contexts)
        facts, hdn = self.gru(contexts)
        facts = facts[:, :, :self.hidden_size] + facts[:, :, self.hidden_size:]
        return facts

class AnswerModule(nn.Module):
    def __init__(self, hidden_size, output_dim=2):
        super(AnswerModule, self).__init__()
        self.z = nn.Linear(2 * hidden_size, output_dim)
        init.xavier_normal(self.z.state_dict()['weight'])
        self.dropout = nn.Dropout(0.1)

    def forward(self, memory, question):
        memory = self.dropout(memory)
        concat = torch.cat([memory, question], dim=2).squeeze(1)
        return self.z(concat)

class DMNEncoder(nn.Module):
    def __init__(self, hidden_size, num_hop, isEncoder=False, non_encoder_output_dim=2):
        super(DMNEncoder, self).__init__()
        self.num_hop = num_hop
        self.memory = EpisodicMemory(hidden_size)
        self.isEncoder = isEncoder
        if self.isEncoder:
            self.input_module = InputModule(hidden_size, enable_pruning=True)
        else:
            self.input_module = InputModule(hidden_size)
            self.question_module = QuestionModule(hidden_size)
            self.answer_module = AnswerModule(hidden_size, output_dim=non_encoder_output_dim)

    def forward(self, src, question):
        if self.isEncoder:
            facts = self.input_module(src, question)
        else:
            facts = self.input_module(src)
            question = self.question_module(question)
        memory = question
        # Before the start of a Episodic Memory step, clear the stored attention values
        self.memory.clear_stored_att_vals()
        for hop in range(self.num_hop):
            memory = self.memory(facts, question, memory)
        if self.isEncoder:
            output = torch.cat([memory, question], dim=2).squeeze(1)
        else:
            output = self.answer_module(memory, question)
        attention_values = np.vstack(
            [att_val.detach().cpu().numpy()
             for att_val in self.memory.att_vals])
        # print('Attention values:')
        # print(attention_values)
        return output

class CNNEncoder(nn.Module):
    """
    Apply convolution + max pool
    """
    def __init__(self, hidden_size, num_filters, filter_sizes, output_dim=2):
        super(CNNEncoder, self).__init__()
        self.convs = nn.ModuleList()
        for filter_size in filter_sizes:
            self.convs.append(nn.Conv2d(1, num_filters, (filter_size, hidden_size)))
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        init.xavier_normal(self.fc.state_dict()['weight'])
        self.dropout = nn.Dropout(0.1)

    def forward(self, rdv):
        """
        rdv = (#batch, #sentence, #embedding)
        feature_vector = (#batch, len(filter_sizes) * #num_filters)
        output = (#batch, 2)
        """
        # Add a channel dimension
        rdv = rdv.unsqueeze(1)
        feature_vectors = list()
        for conv in self.convs:
            feature = F.relu(conv(rdv))
            feature = feature.squeeze(-1)
            feature = F.max_pool1d(feature, feature.size()[2])
            feature = feature.squeeze(-1)
            feature_vectors.append(feature)
        feature_vector = torch.cat(feature_vectors, 1)
        feature_vector = self.dropout(feature_vector)
        output = self.fc(feature_vector)
        return output

class DMNPlus(nn.Module):
    def __init__(self, hidden_size, num_hop=3, isSentenceLevel=False, isRegression=False):
        super(DMNPlus, self).__init__()
        self.num_hop = num_hop
        self.isSentenceLevel = isSentenceLevel
        self.isRegression = isRegression
        if isRegression:
            output_dim = 1
        else:
            output_dim = 2
        if isSentenceLevel:
            self.mem_encoder = DMNEncoder(hidden_size, num_hop, isEncoder=True, non_encoder_output_dim=output_dim)
            self.cnn_encoder = CNNEncoder(2 * hidden_size, num_filters=200, filter_sizes=[3, 4, 5], output_dim=output_dim)
            # For documents with less than 5 target sentences use filter of size 1
            # self.cnn_encoder = CNNEncoder(2 * hidden_size, num_filters=200, filter_sizes=[1])
        else:
            self.mem_encoder = DMNEncoder(hidden_size, num_hop, non_encoder_output_dim=output_dim)
        if isRegression:
            self.criterion = nn.L1Loss(size_average=False)
        else:
            self.criterion = nn.CrossEntropyLoss(size_average=False)

    def forward(self, src, tgt):
        '''
        tgt.size() -> (#batch, #sentence, #embedding) -> (#batch, 1, #hidden)
        src.size() -> (#batch, #sentence, #embedding) -> (#batch, #sentence, #hidden)
        encoding.size() -> (#batch, 2 * #embedding)
        rdv.size() -> (#batch, #sentence, 2 * #embedding)
        output.size() -> (#batch, 2)
        '''
        if self.isSentenceLevel:
            batch_num, sent_num, embedding_dim = tgt.size()
            # rdv = Relative document vector
            rdv = torch.zeros(batch_num, sent_num, 2 * embedding_dim).cuda()
            for i in range(sent_num):
                question = tgt.index_select(1, torch.tensor([i]).cuda())
                encoding = checkpoint(self.mem_encoder, src, question)
                for j in range(batch_num):
                    rdv[j][i] = encoding[j]
            output = self.cnn_encoder(rdv)
        else:
            output = self.mem_encoder(src, tgt)
        return output

    def get_loss(self, src, tgt, answers):
        output = self.forward(src, tgt)
        reg_loss = 0
        for param in self.parameters():
            reg_loss += 0.001 * torch.sum(param * param)
        if self.isRegression:
            predictions = F.leaky_relu(output)
            predictions = predictions.squeeze()
            loss = self.criterion(predictions, answers)
            answers_data = answers.detach().cpu().numpy()
            predictions = predictions.detach().cpu().numpy()
            mae = mean_absolute_error(answers_data, predictions)
            mse = mean_squared_error(answers_data, predictions)
            cos_sim = 1 - distance.cosine(answers_data, predictions)
            pearson_r, _ = pearsonr(answers_data, predictions)
            metrics = {'pearson_r': pearson_r, 'mae': mae, 'mse': mse, 'cos_sim': cos_sim}
        else:
            predictions = F.softmax(output)
            loss = self.criterion(output, answers)
            _, predictions = torch.max(predictions, dim=1)
            answers_data = answers.detach().cpu().numpy()
            predictions = predictions.detach().cpu().numpy()
            corrects = (predictions == answers_data)
            acc = torch.mean(corrects.float())
            metrics = {'acc': acc}
        output = output.detach().cpu().numpy()
        return loss + reg_loss, metrics, predictions, output

def step(dataloader, model, optim, train=True):
    """Runs single step on the dataset

    Parmeters:
    dataloader: DataLoader instance for the data to be processed
    model:      Model instance
    optim:      Optimizer
    train:      Boolean indicating whether to optimize model or just calculate loss

    Returns:
    result: dict containing performance metrics

    Performance metrics:
    Regression model: pearson_r, mae, mse, cos_sim
    Classification model: acc
    """
    if model.isRegression:
        metrics = {'pearson_r': 0.0, 'mae': 0.0, 'mse': 0.0, 'cos_sim': 0.0}
    else:
        metrics = {'acc': 0.0}
    total_loss = 0
    cnt = 0
    all_preds = []
    all_outputs = []
    for batch_idx, data in enumerate(dataloader):
        optim.zero_grad()
        src, tgt, answers = data[-3:]
        batch_size = answers.shape[0]
        src = Variable(src.cuda())
        tgt = Variable(tgt.cuda())
        answers = Variable(answers.cuda())
        loss, step_metrics, preds, output = model.get_loss(src, tgt, answers)
        all_preds.extend(preds.tolist())
        all_outputs.append(output)
        total_loss += loss.data.item() * batch_size
        cnt += batch_size
        for metric_type in step_metrics:
            metrics[metric_type] += step_metrics[metric_type] * batch_size
        if train:
            loss.backward()
            optim.step()
            if batch_idx % 80 == 0:
                perf_str = ', '.join([f'{metric_type}: {metrics[metric_type] / cnt: {5}.{4}}' for metric_type in metrics])
                LOG.debug(f'Training loss : {total_loss / cnt: {5}.{4}}, {perf_str}, batch_idx: {batch_idx}')
    for metric_type in metrics:
        total_loss /= cnt
        metrics[metric_type] /= cnt
    perf_str = ', '.join([f'{metric_type}: {metrics[metric_type]: {5}.{4}}' for metric_type in metrics])
    if train:
        LOG.debug(f'Training loss : {total_loss: {5}.{4}}, {perf_str}, batch_idx: {batch_idx}')
    else:
        LOG.debug(f'Loss : {total_loss: {5}.{4}}, {perf_str}, batch_idx: {batch_idx}')
    return metrics, all_preds, all_outputs


def pretty_size(size):
    """Pretty prints a torch.Size object"""
    assert isinstance(size, torch.Size)
    return " x ".join(map(str, size))


def dump_tensors(gpu_only=True):
    """Prints a list of the Tensors being tracked by the garbage collector."""
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    LOG.debug("%s:%s%s %s" % (type(obj).__name__,
                                          " GPU" if obj.is_cuda else "",
                                          " pinned" if obj.is_pinned else "",
                                          pretty_size(obj.size())))
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    LOG.debug("%s -> %s:%s%s%s%s %s" % (type(obj).__name__,
                                                    type(obj.data).__name__,
                                                    " GPU" if obj.is_cuda else "",
                                                    " pinned" if obj.data.is_pinned else "",
                                                    " grad" if obj.requires_grad else "",
                                                    " volatile" if obj.volatile else "",
                                                    pretty_size(obj.data.size())))
                    total_size += obj.data.numel()
        except:
            pass
    LOG.debug("Total size: %d", total_size)


def print_memory_stats():
    LOG.debug('GPU memory usage: %f' % torch.cuda.memory_allocated())
    LOG.debug('Caching allocator memory usage: %f' % torch.cuda.memory_cached())


def run_fold(fold_num):
    # define the model
    model = DMNPlus(dset.hidden, num_hop=NUM_HOPS, isSentenceLevel=IS_SENTENCE_LEVEL, isRegression=IS_REGRESSION)
    model.cuda()
    if PRE_TRAINED_MODEL:
        with open(PRE_TRAINED_MODEL, 'rb') as fp:
            model.load_state_dict(torch.load(fp))
    early_stopping_cnt = 0
    early_stopping_flag = False
    best_result = -float('inf')
    optim = Adam(model.parameters())
    lr_decay = ReduceLROnPlateau(optim, mode='max', patience=3,
                                 threshold_mode='abs', threshold=0.01, verbose=True)
    if fold_num > 0:
        # Get the current fold of data to use
        dset.next_fold()
    LOG.debug('Fold no: %d', fold_num + 1)
    # Training starts here
    for epoch in range(EPOCH_OFFSET, NUM_EPOCHS+EPOCH_OFFSET):
        # Training step
        dset.set_mode('train')
        train_loader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_func)
        model.train()
        LOG.debug(f'Epoch {epoch}')
        step(train_loader, model, optim, train=True)
        # Validation step
        dset.set_mode('valid')
        valid_loader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_func)
        model.eval()
        valid_metrics, _, _ = step(valid_loader, model, optim, train=False)
        for metric_type in valid_metrics:
            LOG.debug(f'Validation {metric_type}: {valid_metrics[metric_type]: {5}.{4}}')
        # Decay Learning rate
        if IS_REGRESSION:
            # less mae is better
            valid_result = -1 * valid_metrics['mae']
            lr_decay.step(valid_result)
        else:
            valid_result = valid_metrics['acc']
            lr_decay.step(valid_result)
        # Save best model and stop early
        if valid_result > best_result:
            best_result = valid_result
            best_state = model.state_dict()
            early_stopping_cnt = 0
            LOG.debug(f'Improvement in Epoch {epoch}')
        else:
            early_stopping_cnt += 1
            if early_stopping_cnt >= EARLY_STOP_THRESHOLD:
                early_stopping_flag = True
        if early_stopping_flag:
            LOG.debug(f'Early Stopping at Epoch {epoch}, no improvement in {EARLY_STOP_THRESHOLD} epochs')
            break
    # Testing starts here
    dset.set_mode('test')
    test_loader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_func)
    # Load the best performing model
    model.load_state_dict(best_state)
    model.eval()
    test_metrics, test_preds, _  = step(test_loader, model, optim, train=False)
    for metric_type in test_metrics:
        LOG.debug(f'Testing {metric_type}: {test_metrics[metric_type]: {5}.{4}}')
    # Save the best model
    os.makedirs(os.path.join(DATASET_NAME.lower(), 'models'), exist_ok=True)
    if IS_REGRESSION:
        test_result = test_metrics['mae']
    else:
        test_result = test_metrics['acc']
    test_filename = f'fold{fold_num+1}_test_result{test_result:{5}.{4}}'
    with open(f'{DATASET_NAME.lower()}/models/{test_filename}.pth', 'wb') as fp:
        torch.save(best_state, fp)
    # Save the predictions
    os.makedirs(os.path.join(DATASET_NAME.lower(), 'predictions'), exist_ok=True)
    # test_preds is an array containing the predictions for the test data for the current fold
    pickle.dump([test_preds], open(f'{DATASET_NAME.lower()}/predictions/{test_filename}.p', "wb"))
    # delete model to free GPU memory
    del model
    del optim
    return test_metrics


def perform_folds(dset, collate_func):
    # collect acuracy across all the folds
    if IS_REGRESSION:
        all_folds_metrics = {'pearson_r': 0.0, 'mae': 0.0, 'mse': 0.0, 'cos_sim': 0.0}
    else:
        all_folds_metrics = {'acc': 0.0}
    # Training and validation starts here
    for fold_num in range(NUM_FOLDS):
        print_memory_stats()
        dump_tensors()
        fold_metrics = run_fold(fold_num)
        for metric_type in all_folds_metrics:
            all_folds_metrics[metric_type] += fold_metrics[metric_type]
    for metric_type in all_folds_metrics:
        LOG.debug('Overall test %s over %d folds: %5.4f', metric_type, NUM_FOLDS, all_folds_metrics[metric_type] / NUM_FOLDS)

if __name__ == "__main__":
    # Initialise data
    dset = DmnData(DATASET_NAME, folds=NUM_FOLDS)
    collate_func = partial(pad_collate, vocab=dset.vocab)
    # Perform folds
    perform_folds(dset, collate_func)
