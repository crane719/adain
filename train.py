import numpy as np
import configparser
import shutil
import joblib
import pickle
import sys
import gc
import random

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import optim
from torch import nn

from config import *
import preprocess
import hoge

# 引数
args = hoge.get_args()
is_preprocess = args.preprocess
is_player_emb = args.player_emb

# 結果のdirの再作成
if hoge.is_dir_existed("train_result"):
    print("delete file...")
    print("- train_result")
    shutil.rmtree("./train_result")
required_dirs = ["param", "train_result"]
hoge.make_dir(required_dirs)

if is_preprocess:
    preprocess.Preprocess(train_c_dir, train_s_dir, train_dataset)

content_dirs = hoge.get_filedirs(train_dataset+"/content/*")
style_dirs = hoge.get_filedirs(train_dataset+"/style/*")

# 少ない方のdataset数に合わせる
if len(content_dirs)<len(style_dirs):
    style_dirs = random.sample(style_dirs, len(content_dirs))
else:
    content_dirs = random.sample(content_dirs, len(style_dirs))
train_c_dirs, valid_c_dirs, train_s_dirs, valid_s_dirs =\
    train_test_split(content_dirs, style_dirs, test_size=0.2)

dd

# model define
lstm = model.Lstm(input_size, int(input_size*dim_ratio), 8, layer_num, dropout)
opt = optim.Adam(lstm.parameters(), lr=lr)
f = nn.LogSoftmax(dim=1)
criterion = nn.NLLLoss(ignore_index=ignore_label, reduction="sum")

train_losses = []
valid_losses = []
train_accs = []
valid_accs = []
min_loss = 1e10
for epoch in range(epoch_num):
    train_loss = 0
    train_acc = []
    valid_loss = 0
    valid_acc = []
    print("Epoch[%d/%d]"%(epoch+1, epoch_num))
    train_data_num = 0
    valid_data_num = 0
    # train
    for train_step, (label, data) in enumerate(train_dl):
        lstm.train()
        opt.zero_grad()
        output, _ = lstm(data)

        tmp_loss = 0
        for dim in range(output.shape[1]):
            loss = criterion(f(output[:,dim,:]), label[:, dim])
            # accの計算
            pred = np.argmax(output[:, dim, :].detach().numpy(), 1) # argmax. labelに変換
            correct = label[:,dim].detach().numpy()
            mask = np.where(correct==ignore_label)[0] # ignore labelの引数ら
            if len(mask)!=len(correct): # 全部ignore labelじゃなきゃ
                # ignore labelの部分の削除
                pred = np.delete(pred, mask)
                correct = np.delete(correct, mask)
                train_acc.extend(list(pred==correct)) # predとcorrectが一致していれば1, してなきゃ0
            tmp_loss += loss
        loss=tmp_loss
        train_data_num+=data.shape[0]
        """
        f = nn.LogSoftmax(dim=2)
        output = f(output)
        loss = criterion(output.transpose(2,1), label)
        """
        loss.backward()
        opt.step()
        train_loss += loss.item()

    # valid
    for valid_step, (label, data) in enumerate(valid_dl):
        lstm.eval()
        opt.zero_grad()
        output, _ = lstm(data)
        #loss = criterion(f(output), label)
        tmp_loss = 0
        for dim in range(output.shape[1]):
            loss = criterion(f(output[:,dim,:]), label[:, dim])
            # accの計算
            pred = np.argmax(output[:, dim, :].detach().numpy(), 1) # argmax. labelに変換
            correct = label[:,dim].detach().numpy()
            mask = np.where(correct==ignore_label)[0] # ignore labelの引数ら
            if len(mask)!=len(correct): # 全部ignore labelじゃなきゃ
                # ignore labelの部分の削除
                pred = np.delete(pred, mask)
                correct = np.delete(correct, mask)
                valid_acc.extend(list(pred==correct)) # predとcorrectが一致していれば1, してなきゃ0
            tmp_loss+=loss
        loss = tmp_loss
        valid_loss += loss.item()
        valid_data_num+=data.shape[0]

    # 重みの吐き出し
    if valid_loss<min_loss:
        min_loss = valid_loss
        torch.save(lstm.state_dict(), "param" + dir_id + "/weight")

    # 諸々average
    train_loss/=train_data_num
    valid_loss/=valid_data_num
    train_acc = np.average(train_acc)
    valid_acc = np.average(valid_acc)
    # 諸々append
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)
    # 整形, 可視化
    losses = {
        "train": train_losses,
        "valid": valid_losses
        }
    accs = {
        "train": train_accs,
        "valid": valid_accs
        }
    x = list(range(len(train_losses)))
    io.time_draw(x, losses, "train_result" + dir_id + "/loss.png")
    io.time_draw(x, accs, "train_result" + dir_id + "/acc.png")

    print("----------------------------")
    print(" loss:")
    print("     train: %lf"%(train_loss))
    print("     valid: %lf"%(valid_loss))
    print(" acc:")
    print("     train: %lf"%(train_acc))
    print("     valid: %lf"%(valid_acc))
    print("----------------------------")

# validationが最小だった重みの読み込み
lstm.load_state_dict(torch.load("param" + dir_id + "/weight"))

del train_dl, train_labels, train_losses, valid_losses, train_accs, valid_accs
gc.collect()
#f = open(test_data_dir, "rb")
#test_datas, args = pickle.load(f)
#test_datas, args = joblib.load(test_data_dir)
test_datas, args = joblib.load(test_data_dir)

# test data
print("TEST:")
f = nn.Softmax(dim=1)
result = None
sort_args = []
sums = len(test_datas)
#while len(test_datas)!=0:
for i, (test_data, arg) in enumerate(zip(test_datas, args)):
    if i % 1000 == 0:
        print("seq_num[%d/%d]: "%(i, sums))
        gc.collect()
    """
    test_data = test_datas[0]
    arg = args[0]
    del args[0]
    test_datas = test_datas[1:, :, :]
    """
    lstm.eval()
    test_data = torch.unsqueeze(test_data, 0)
    output, _ = lstm(test_data)
    output = torch.squeeze(output, 0)
    output = f(output)
    output = output[:len(arg), :] # padding箇所の削除
    output = output.detach().numpy()
    if result is None:
        result = output
        sort_args = arg
    else:
        result = np.concatenate((result, output), 0)
        sort_args.extend(arg)

# test dataのid順に並び替え
result = result[np.argsort(sort_args), :]
result = pd.DataFrame(result)
result.to_csv("train_result" + dir_id + "/result.csv", index=True, header=False)

