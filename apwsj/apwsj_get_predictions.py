from functools import partial
from itertools import chain
import pickle
import torch
torch.cuda.set_device(0)
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from apwsj_loader import APWSJ, pad_collate
from apwsj_main import DMNPlus

model_name = 'models/epoch7_acc0.6506_20200408_0652.pth'
dset = APWSJ()
dset.set_mode('valid')
collate_func = partial(pad_collate, vocab=dset.vocab)
BATCH_SIZE = 50
valid_loader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_func)
model = DMNPlus(dset.hidden, num_hop=3)
model.cuda()
with open(model_name, 'rb') as fp:
    model.load_state_dict(torch.load(fp))
model.eval()
predictions = list()
golds = list()
for batch_idx, data in enumerate(valid_loader):
    _, contexts, questions, answers = data
    batch_size = answers.shape[0]
    contexts = Variable(contexts.cuda())
    questions = Variable(questions.cuda())
    answers = Variable(answers.cuda())
    output = model.forward(contexts, questions)
    preds = F.softmax(output)
    _, pred_ids = torch.max(preds, dim=1)
    predictions.append(pred_ids.data.tolist())
    golds.append(answers.data.tolist())
predictions = list(chain.from_iterable(predictions))
golds = list(chain.from_iterable(golds))
# Make confusion matrix
tn, fp, fn, tp = confusion_matrix(golds, predictions).ravel()
print('True positive: %d' % tp)
print('False positive: %d' % fp)
print('True negative: %d' % tn)
print('False negative: %d' % fn)
