"""
Train DLND model
"""

import logging
import os
import torch
torch.cuda.set_device(2)
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dlnd_loader import DLND, pad_collate
from dlnd_logger import init_logger

LOG = logging.getLogger()
init_logger(LOG)

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

    def make_interaction(self, facts, questions, prevM):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, 1, #hidden)
        prevM.size() -> (#batch, #sentence = 1, #hidden = #embedding)
        z.size() -> (#batch, #sentence, 4 x #embedding)
        G.size() -> (#batch, #sentence)
        '''
        batch_num, sen_num, embedding_size = facts.size()
        questions = questions.expand_as(facts)
        prevM = prevM.expand_as(facts)

        z = torch.cat([
            facts * questions,
            facts * prevM,
            torch.abs(facts - questions),
            torch.abs(facts - prevM)
        ], dim=2)

        z = z.view(-1, 4 * embedding_size)

        G = F.tanh(self.z1(z))
        G = self.z2(G)
        G = G.view(batch_num, -1)
        G = F.softmax(G)

        return G

    def forward(self, facts, questions, prevM):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, #sentence = 1, #hidden)
        prevM.size() -> (#batch, #sentence = 1, #hidden = #embedding)
        G.size() -> (#batch, #sentence)
        C.size() -> (#batch, #hidden)
        concat.size() -> (#batch, 3 x #embedding)
        '''
        G = self.make_interaction(facts, questions, prevM)
        C = self.AGRU(facts, G)
        concat = torch.cat([prevM.squeeze(1), C, questions.squeeze(1)], dim=1)
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
    def __init__(self, hidden_size):
        super(InputModule, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        for name, param in self.gru.state_dict().items():
            if 'weight' in name: init.xavier_normal(param)
        self.dropout = nn.Dropout(0.1)

    def forward(self, contexts):
        '''
        contexts.size() -> (#batch, #context, #embedding)
        facts.size() -> (#batch, #context, #hidden)
        '''
        batch_num, context_num, embedding_dim = contexts.size()
        contexts = self.dropout(contexts)
        facts, hdn = self.gru(contexts)
        facts = facts[:, :, :self.hidden_size] + facts[:, :, self.hidden_size:]
        return facts

class AnswerModule(nn.Module):
    def __init__(self, hidden_size):
        super(AnswerModule, self).__init__()
        self.z = nn.Linear(2 * hidden_size, 2)
        init.xavier_normal(self.z.state_dict()['weight'])
        self.dropout = nn.Dropout(0.1)

    def forward(self, M, questions):
        M = self.dropout(M)
        concat = torch.cat([M, questions], dim=2).squeeze(1)
        z = self.z(concat)
        return z

class DMNPlus(nn.Module):
    def __init__(self, hidden_size, num_hop=3, qa=None):
        super(DMNPlus, self).__init__()
        self.num_hop = num_hop
        self.qa = qa
        self.criterion = nn.CrossEntropyLoss(size_average=False)

        self.input_module = InputModule(hidden_size)
        self.question_module = QuestionModule(hidden_size)
        self.memory = EpisodicMemory(hidden_size)
        self.answer_module = AnswerModule(hidden_size)

    def forward(self, src, tgt):
        '''
        src.size() -> (#batch, #sentence, #embedding) -> (#batch, #sentence, #hidden)
        tgt.size() -> (#batch, #sentence, #embedding) -> (1, #batch, #hidden)
        '''
        facts = self.input_module(src)
        questions = self.question_module(tgt)
        M = questions
        for hop in range(self.num_hop):
            M = self.memory(facts, questions, M)
        preds = self.answer_module(M, questions)
        return preds

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
        return loss + reg_loss, acc

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
    cnt = 0
    for batch_idx, data in enumerate(dataloader):
        optim.zero_grad()
        _, _, src, tgt, answers = data
        batch_size = answers.shape[0]
        src = Variable(src.cuda())
        tgt = Variable(tgt.cuda())
        answers = Variable(answers.cuda())
        loss, acc = model.get_loss(src, tgt, answers)
        total_acc += acc * batch_size
        cnt += batch_size
        if train:
            loss.backward()
            if batch_idx % 20 == 0:
                LOG.debug(f'Training loss : {loss.data.item(): {10}.{8}}, acc : {total_acc / cnt: {5}.{4}}, batch_idx : {batch_idx}')
            optim.step()
    return total_acc / cnt

if __name__ == "__main__":
    # Config variables
    NUM_EPOCHS = 25
    BATCH_SIZE = 32
    EARLY_STOP_THRESHOLD = 10
    PRE_TRAINED_MODEL = None
    EPOCH_OFFSET = 1

    # Initialise DLND data
    dset = DLND()
    collate_func = partial(pad_collate, vocab=dset.vocab)

    # Model parameters
    # hidden_size = sentence embedding dimension
    hidden_size = dset.hidden

    model = DMNPlus(hidden_size, num_hop=6)
    model.cuda()
    if PRE_TRAINED_MODEL:
        with open(PRE_TRAINED_MODEL, 'rb') as fp:
            model.load_state_dict(torch.load(fp))
    early_stopping_cnt = 0
    early_stopping_flag = False
    best_acc = 0
    optim = torch.optim.Adam(model.parameters())

    # Training starts here
    for epoch in range(EPOCH_OFFSET, NUM_EPOCHS+EPOCH_OFFSET):
        # Training step
        dset.set_mode('train')
        train_loader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_func)
        model.train()
        LOG.debug(f'Epoch {epoch}')
        acc = step(train_loader, model, optim, train=True)

        # Validation step
        dset.set_mode('valid')
        valid_loader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_func)
        model.eval()
        acc = step(valid_loader, model, optim, train=False)
        LOG.debug(f'Epoch {epoch}: [Validate] Accuracy : {acc: {5}.{4}}')
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
    # update epoch to the best performing epoch
    epoch -= early_stopping_cnt
    # load the state from the best performing epoch
    model.load_state_dict(best_state)
    test_loader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_func)
    test_acc = step(test_loader, model, optim, train=False)
    LOG.debug(f'Epoch {epoch}: [Test] Accuracy : {test_acc : {5}.{4}}')
    # Save the best model
    os.makedirs('models', exist_ok=True)
    with open(f'models/6hops_epoch{epoch}_acc{best_acc}.pth', 'wb') as fp:
        torch.save(best_state, fp)
