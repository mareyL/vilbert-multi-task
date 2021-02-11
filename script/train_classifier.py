import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
from statistics import mean
import matplotlib.pyplot as plt
import _pickle as cPickle
from tqdm import tqdm
from scipy.stats import spearmanr
import pandas as pd

def evaluate(model, inp, target):
    loss_func = torch.nn.MultiLabelMarginLoss()
    torch_dataset_val = Data.TensorDataset(inp, target)

    loader_val = Data.DataLoader(
        dataset=torch_dataset_val,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False
    )

    dataiter_val = iter(loader_val)
    in_, out_ = dataiter_val.next()

    model.eval()
    pred_scores = model(in_)

    loss = loss_func(pred_scores, out_)
    model.train()
    r_1, _ = spearmanr(
        pred_scores.cpu().detach().numpy()[:,0],
        out_.cpu().detach().numpy()[:,0],
        axis=0
    )

    r_2, _ = spearmanr(
        pred_scores.cpu().detach().numpy()[:,1],
        out_.cpu().detach().numpy()[:,1],
        axis=0
    )

    return r_1, r_2, loss.item()

train_save_path = "/data/moviescope/outputs/train_features.pkl"
pooled_dict = cPickle.load(open(train_save_path, 'rb'))
pooled_output_mul = pooled_dict["pooled_output_mul"]
pooled_output_sum = pooled_dict["pooled_output_sum"]
pooled_output_t = pooled_dict["pooled_output_t"]
pooled_output_v = pooled_dict["pooled_output_v"]
concat_pooled_output = pooled_dict["concat_pooled_output"]
targets_ = cPickle.load(open("/data/moviescope/cache/ME_trainval_23_cleaned.pkl", 'rb'))
targets = []
for el in targets_:
    genre = []
    genre_str = str(el["scores"].cpu().detach().numpy()[0])
    for c in genre_str:
        genre.append(int(c))
    while len(genre)<13:
        genre.insert(0,0)
    targets.append(genre)
targets = torch.tensor(targets,device='cuda:0')

test_save_path = "/data/moviescope/outputs/test_features.pkl"
pooled_dict_test = cPickle.load(open(test_save_path, 'rb'))
pooled_output_mul_test = pooled_dict_test["pooled_output_mul"]
pooled_output_sum_test = pooled_dict_test["pooled_output_sum"]
pooled_output_t_test = pooled_dict_test["pooled_output_t"]
pooled_output_v_test = pooled_dict_test["pooled_output_v"]
concat_pooled_output_test = pooled_dict_test["concat_pooled_output"]
targets_test_ = cPickle.load(open("/data/moviescope/cache/ME_test_23_cleaned.pkl", 'rb'))
targets_test = []
for el in targets_test_:
    genre = []
    genre_str = str(el["scores"].cpu().detach().numpy()[0])
    for c in genre_str:
        genre.append(int(c))
    while len(genre)<13:
        genre.insert(0,0)
    targets_test.append(genre)
targets_test = torch.tensor(targets_test,device='cuda:0')

class SigLinNet(nn.Module):
    def __init__(self, input_size,
        hidden_size_1,
        hidden_size_2,
        hidden_size_3,
        num_scores):
            super(SigLinNet, self).__init__()
            self.out = nn.Sequential(
                nn.Linear(input_size, hidden_size_1),
                nn.Sigmoid(),
                nn.Linear(hidden_size_1, hidden_size_2),
                nn.Sigmoid(),
                nn.Linear(hidden_size_2, hidden_size_3),
                nn.Sigmoid(),
                nn.Linear(hidden_size_3, num_scores),
                nn.Sigmoid(),
            )   

    def forward(self, x):
        return self.out(x)

BATCH_SIZE = 128
VAL_BATCH_SIZE = 128
EPOCH = 100
lr = 4e-4

torch.manual_seed(42)


torch_dataset = Data.TensorDataset(concat_pooled_output, targets)


loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

net = SigLinNet(1024*2, 512, 64, 32, 13)
net.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
loss_func = torch.nn.MultiLabelMarginLoss()

summary = {
    "losses" : [],
    "r1s" : [],
    "r2s" : [],
    "eval_losses" : []
}

net.train()
for _ in tqdm(range(EPOCH)):
    errors = []
    # r1s, r2s, ls = list(), list(), list()
    for step, (batch_in, batch_out) in enumerate(loader):
        optimizer.zero_grad()

        b_in = Variable(batch_in)
        b_out = Variable(batch_out)
        
        prediction = net(b_in)

        loss = loss_func(prediction, b_out)
        errors.append(loss.item())

        loss.backward()
        optimizer.step()

    """r1, r2, _ = evaluate(net, pooled_output_mul_test, targets_test)
    summary["r1s"].append(r1)
    summary["r2s"].append(r2)"""
    summary["losses"].append(mean(errors))

plt.figure()
plt.plot(summary["losses"][3:])
plt.savefig('/data/moviescope/outputs/train_loss.png')

losses = list()

torch_dataset_val = Data.TensorDataset(concat_pooled_output_test, targets_test)

loader_val = Data.DataLoader(
    dataset=torch_dataset_val, 
    batch_size=VAL_BATCH_SIZE, 
    shuffle=False
)

dataiter_val = iter(loader_val)
in_, out_ = dataiter_val.next()

net.eval()
pred_scores = net(in_)

loss = loss_func(pred_scores, out_)
losses.append(loss.item())

mAP=0
out_list = out_.cpu().detach().numpy()
scores_list = pred_scores.cpu().detach().numpy()

for i in range(scores_list.shape[0]):
    for j in range(len(scores_list[i])):
        if scores_list[i][j] > 0.5:
            prediction = 1
        else:
            prediction = 0
        if prediction==out_list[i][j]:
            mAP += 1
mAP = mAP/(out_list.shape[0]*out_list.shape[1])

print("mAP=",mAP)
