"""
Run DMN model
Command: python dmn.py <dataset_name>
"""

# Config variables
import os
import sys

NUM_FOLDS = 10
NUM_HOPS = 4
NUM_EPOCHS = 25
BATCH_SIZE = 32
EARLY_STOP_THRESHOLD = 10
PRE_TRAINED_MODEL = None
EPOCH_OFFSET = 1
GPU_ID = 2
DATASET_NAME = sys.argv[1]
if DATASET_NAME == 'DLND':
    LOGFILE = 'dlnd/dlnd_logs'
elif DATASET_NAME == 'APWSJ':
    LOGFILE = 'apwsj/apwsj_logs'
else:
    raise Exception('Dataset name %s is not supported!' % DATASET_NAME)
HOME_DIR = "/home1/tirthankar"
ENCODER_DIR = os.path.join(HOME_DIR, "Vignesh/InferSent")
ENCODER_PATH = os.path.join(ENCODER_DIR, "models/model_2048_attn.pickle")
# Infersent should be in the path
sys.path.append(ENCODER_DIR)

import logging
import pickle
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
        self.AGRU = AttentionGRU(hidden_size, hidden_size)
        self.z1 = nn.Linear(4 * hidden_size, hidden_size)
        self.z2 = nn.Linear(hidden_size, 1)
        self.next_mem = nn.Linear(3 * hidden_size, hidden_size)
        init.xavier_normal(self.z1.state_dict()['weight'])
        init.xavier_normal(self.z2.state_dict()['weight'])
        init.xavier_normal(self.next_mem.state_dict()['weight'])

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
        C = self.AGRU(facts, G)
        concat = torch.cat([prevM.squeeze(1), C, question.squeeze(1)], dim=1)
        next_mem = F.relu(self.next_mem(concat))
        next_mem = next_mem.unsqueeze(1)
        return next_mem

# class QuestionModule(nn.Module):
#     def __init__(self, hidden_size):
#         super(QuestionModule, self).__init__()
#         self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
#
#     def forward(self, questions):
#         '''
#         questions.size() -> (#batch, #sentence, #embedding)
#         gru() -> (1, #batch, #hidden)
#         '''
#         _, questions = self.gru(questions)
#         questions = questions.transpose(0, 1)
#         return questions

class InputModule(nn.Module):
    def __init__(self, hidden_size):
        super(InputModule, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        for name, param in self.gru.state_dict().items():
            if 'weight' in name: init.xavier_normal(param)
        self.dropout = nn.Dropout(0.1)
        infersent = torch.load(ENCODER_PATH).cuda()
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

    def forward(self, contexts, question):
        '''
        contexts.size() -> (#batch, #context, #embedding)
        # question.size() -> (#batch, #embedding)
        facts.size() -> (#batch, #context, #hidden)
        '''
        contexts = self.prune_contexts(contexts, question)
        contexts = self.dropout(contexts)
        facts, hdn = self.gru(contexts)
        facts = facts[:, :, :self.hidden_size] + facts[:, :, self.hidden_size:]
        return facts

# class AnswerModule(nn.Module):
#     def __init__(self, hidden_size):
#         super(AnswerModule, self).__init__()
#         # self.z = nn.Linear(2 * hidden_size, 2)
#         # init.xavier_normal(self.z.state_dict()['weight'])
#         # self.dropout = nn.Dropout(0.1)
#
#     def forward(self, memory, question):
#         memory = self.dropout(memory)
#         concat = torch.cat([memory, question], dim=2).squeeze(1)
#         # z = self.z(concat)
#         return concat

class DMNEncoder(nn.Module):
    def __init__(self, hidden_size, num_hop):
        super(DMNEncoder, self).__init__()
        self.input_module = InputModule(hidden_size)
        self.memory = EpisodicMemory(hidden_size)
        self.num_hop = num_hop

    def forward(self, src, question):
        facts = self.input_module(src, question)
        memory = question
        for hop in range(self.num_hop):
            memory = self.memory(facts, question, memory)
        output = torch.cat([memory, question], dim=2).squeeze(1)
        return output

class CNNEncoder(nn.Module):
    """
    Apply convolution + max pool
    """
    def __init__(self, hidden_size, num_filters, filter_sizes):
        super(CNNEncoder, self).__init__()
        self.convs = nn.ModuleList()
        for filter_size in filter_sizes:
            self.convs.append(nn.Conv2d(1, num_filters, (filter_size, hidden_size)))
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 2)
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
    def __init__(self, hidden_size, num_hop=3):
        super(DMNPlus, self).__init__()
        self.num_hop = num_hop
        self.criterion = nn.CrossEntropyLoss(size_average=False)
        self.mem_encoder = DMNEncoder(hidden_size, num_hop)
        self.cnn_encoder = CNNEncoder(2 * hidden_size, num_filters=200, filter_sizes=[3, 4, 5])

    def forward(self, src, tgt):
        '''
        tgt.size() -> (#batch, #sentence, #embedding) -> (#batch, 1, #hidden)
        src.size() -> (#batch, #sentence, #embedding) -> (#batch, #sentence, #hidden)
        encoding.size() -> (#batch, 2 * #embedding)
        rdv.size() -> (#batch, #sentence, 2 * #embedding)
        output.size() -> (#batch, 2)
        '''
        batch_num, sent_num, embedding_dim = tgt.size()
        # rdv = Relative document vector
        rdv = torch.zeros(batch_num, sent_num, 2 * embedding_dim).cuda()
        for i in range(sent_num):
            question = tgt.index_select(1, torch.tensor([i]).cuda())
            encoding = checkpoint(self.mem_encoder, src, question)
            for j in range(batch_num):
                rdv[j][i] = encoding[j]
        output = self.cnn_encoder(rdv)
        return output

    def get_loss(self, src, tgt, answers):
        output = self.forward(src, tgt)
        loss = self.criterion(output, answers)
        reg_loss = 0
        for param in self.parameters():
            reg_loss += 0.001 * torch.sum(param * param)
        preds = F.softmax(output)
        _, pred_ids = torch.max(preds, dim=1)
        corrects = (pred_ids.data == answers.data)
        acc = torch.mean(corrects.float())
        return loss + reg_loss, acc, pred_ids.data

def step(dataloader, model, optim, train=True):
    """Runs single step on the dataset

    Parmeters:
    dataloader: DataLoader instance for the data to be processed
    model:      Model instance
    optim:      Optimizer
    train:      Boolean indicating whether to optimize model or just calculate loss

    Returns:
    float: accuracy of the model
    """
    total_acc = 0
    total_loss = 0
    cnt = 0
    all_preds = []
    for batch_idx, data in enumerate(dataloader):
        optim.zero_grad()
        _, _, src, tgt, answers = data
        batch_size = answers.shape[0]
        src = Variable(src.cuda())
        tgt = Variable(tgt.cuda())
        answers = Variable(answers.cuda())
        loss, acc, preds = model.get_loss(src, tgt, answers)
        all_preds.extend(preds.tolist())
        total_acc += acc * batch_size
        total_loss += loss.data.item() * batch_size
        cnt += batch_size
        if train:
            loss.backward()
            if batch_idx % 80 == 0:
                LOG.debug(f'Training loss : {total_loss / cnt: {5}.{4}}, acc: {total_acc / cnt: {5}.{4}}, batch_idx: {batch_idx}')
            optim.step()
    if train:
        LOG.debug(f'Training loss : {total_loss / cnt: {5}.{4}}, acc: {total_acc / cnt: {5}.{4}}, batch_idx: {batch_idx}')
    else:
        LOG.debug(f'Loss : {total_loss / cnt: {5}.{4}}, acc: {total_acc / cnt: {5}.{4}}, batch_idx: {batch_idx}')
    return total_acc / cnt, all_preds

if __name__ == "__main__":
    # Initialise data
    dset = DmnData(DATASET_NAME, folds=NUM_FOLDS)
    collate_func = partial(pad_collate, vocab=dset.vocab)
    # Model parameters
    # hidden_size = sentence embedding dimension
    hidden_size = dset.hidden
    # collect acuracy across all the folds
    all_folds_acc = 0
    # Training and validation starts here
    for fold_num in range(NUM_FOLDS):
        # define the model
        model = DMNPlus(hidden_size, num_hop=NUM_HOPS)
        model.cuda()
        if PRE_TRAINED_MODEL:
            with open(PRE_TRAINED_MODEL, 'rb') as fp:
                model.load_state_dict(torch.load(fp))
        early_stopping_cnt = 0
        early_stopping_flag = False
        best_acc = 0
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
            acc, _ = step(valid_loader, model, optim, train=False)
            LOG.debug(f'Validation Accuracy : {acc: {5}.{4}}')
            # Decay Learning rate
            lr_decay.step(acc)
            # Save best model and stop early
            if acc > best_acc:
                best_acc = acc
                best_state = model.state_dict()
                early_stopping_cnt = 0
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
        test_acc, test_preds = step(test_loader, model, optim, train=False)
        all_folds_acc += test_acc
        LOG.debug(f'Testing Accuracy : {test_acc: {5}.{4}}')
        # Save the best model
        os.makedirs(os.path.join(DATASET_NAME.lower(), 'models'), exist_ok=True)
        test_filename = f'fold{fold_num+1}_test_acc{test_acc:{5}.{4}}'
        with open(f'{DATASET_NAME.lower()}/models/{test_filename}.pth', 'wb') as fp:
            torch.save(best_state, fp)
        # Save the predictions
        os.makedirs(os.path.join(DATASET_NAME.lower(), 'predictions'), exist_ok=True)
        # test_preds is an array containing the predictions for the test data for the current fold
        pickle.dump([test_preds], open(f'{DATASET_NAME.lower()}/predictions/{test_filename}.p', "wb"))
    LOG.debug('Overall test accuracy over %d folds: %5.4f', NUM_FOLDS, all_folds_acc/NUM_FOLDS)
