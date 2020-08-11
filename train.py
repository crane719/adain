import numpy as np
import configparser
import shutil
import joblib
import pickle
import sys
import gc
import random
import time
import cv2

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import optim
from torch import nn

from config import *
import preprocess
import hoge
import model
import function as func
import IO as io

# 引数
args = hoge.get_args()
is_preprocess = args.preprocess

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

# model define
encoder = hoge.try_gpu(model.Encoder(fixed_point, style_outputs))
decoder = hoge.try_gpu(model.Decoder())

dopt = optim.Adam(decoder.parameters(), lr=lr)

criterion = nn.MSELoss()

train_losses = []
valid_losses = []
closses = []
slosses = []
min_loss = 1e10
iteration = 1
for epoch in range(epoch_num):
    train_loss = 0
    valid_loss = 0
    closs = 0
    sloss = 0
    print("Epoch[%d/%d]"%(epoch+1, epoch_num))
    train_data_num = 0
    valid_data_num = 0

    content_dl = DataLoader(TensorDataset(torch.LongTensor(range(len(train_c_dirs)))), shuffle=True, batch_size=batchsize)
    style_dl = DataLoader(TensorDataset(torch.LongTensor(range(len(train_s_dirs)))), shuffle=True, batch_size=batchsize)
    train_total_step = len(train_c_dirs)//batchsize
    start_time = time.time()
    # train
    #for train_step, (label, data) in enumerate(train_dl):
    for train_step, (cdir_args, sdir_args) in enumerate(zip(content_dl, style_dl), 1):
        if train_step%100==0:
            print("     train step[%d/%d]"%(train_step, train_total_step))
        encoder.eval()
        decoder.train()

        dopt.zero_grad()

        # data reading
        cdirs = cdir_args[0].detach().numpy()
        cdirs = np.array(train_c_dirs)[cdirs]
        sdirs = sdir_args[0].detach().numpy()
        sdirs = np.array(train_s_dirs)[sdirs]
        cdatas, sdatas = hoge.get_pics(cdirs, sdirs)

        # normalization
        #cdatas = cdatas/256
        #sdatas = sdatas/256

        # reshape
        #cdatas = hoge.try_gpu(cdatas.view(-1, 3, 256, 256))
        #sdatas = hoge.try_gpu(sdatas.view(-1, 3, 256, 256))
        cdatas = hoge.try_gpu(cdatas.permute(0, 3, 1, 2).contiguous())
        sdatas = hoge.try_gpu(sdatas.permute(0, 3, 1, 2).contiguous())

        # encode
        crep, couts = encoder(cdatas)
        crep = couts[-1]
        srep, souts = encoder(sdatas)
        srep = souts[-1]

        # decode
        rep = func.adain(crep, srep)
        pic = decoder(rep)
        train_pic = pic

        # loss calc
        _, reconstruct_outs = encoder(pic)
        reconstruct_rep = reconstruct_outs[-1]
        lc = criterion(rep, reconstruct_rep)

        mu_loss = 0
        sigma_loss = 0
        for sout, reconstruct_out in zip(souts, reconstruct_outs):
            smu = func.mu(sout, False)
            rmu = func.mu(reconstruct_out, False)
            ssigma = func.sigma(sout, False)
            rsigma = func.sigma(reconstruct_out, False)
            mu_loss+=criterion(rmu, smu)
            sigma_loss+=criterion(rsigma, ssigma)
        loss = lc + (mu_loss+sigma_loss)*style_weight
        loss.backward()
        dopt.step()
        train_loss += loss.item()
        closs+=lc.item()
        sloss+=(mu_loss+sigma_loss).item()
        train_data_num+=cdatas.shape[0]

        # lrの更新
        renew_lr = func.renew_lr(lr, iteration, lr_decay)
        iteration+=1
        for param_group in dopt.param_groups:
            param_group["lr"] = renew_lr

        # 重みを吐き出す
        if loss<min_loss:
            min_loss = loss
            torch.save(encoder.state_dict(), eweight_dir)
            torch.save(decoder.state_dict(), dweight_dir)

    # valid
    content_dl = DataLoader(TensorDataset(torch.LongTensor(range(len(valid_c_dirs)))), shuffle=True, batch_size=batchsize)
    style_dl = DataLoader(TensorDataset(torch.LongTensor(range(len(valid_s_dirs)))), shuffle=True, batch_size=batchsize)
    valid_total_step = len(valid_c_dirs)//batchsize
    for valid_step, (cdir_args, sdir_args) in enumerate(zip(content_dl, style_dl), 1):
        if valid_step%100==0:
            print("     valid step[%d/%d]"%(valid_step, valid_total_step))
        encoder.eval()
        decoder.eval()

        dopt.zero_grad()

        # data reading
        cdirs = cdir_args[0].detach().numpy()
        cdirs = np.array(valid_c_dirs)[cdirs]
        sdirs = sdir_args[0].detach().numpy()
        sdirs = np.array(valid_s_dirs)[sdirs]
        cdatas, sdatas = hoge.get_pics(cdirs, sdirs)

        # normalization
        #cdatas = cdatas/256
        #sdatas = sdatas/256

        # reshape
        #cdatas = hoge.try_gpu(cdatas.view(-1, 3, 256, 256))
        #sdatas = hoge.try_gpu(sdatas.view(-1, 3, 256, 256))
        cdatas = hoge.try_gpu(cdatas.permute(0, 3, 1, 2).contiguous())
        sdatas = hoge.try_gpu(sdatas.permute(0, 3, 1, 2).contiguous())

        # encode
        _, couts = encoder(cdatas)
        crep = couts[-1]
        _, souts = encoder(sdatas)
        srep = souts[-1]

        # decode
        rep = func.adain(crep, srep)
        pic = decoder(rep)

        # loss calc
        _, reconstruct_outs = encoder(pic)
        reconstruct_rep = reconstruct_outs[-1]
        lc = criterion(reconstruct_rep, rep)

        mu_loss = 0
        sigma_loss = 0
        for sout, reconstruct_out in zip(souts, reconstruct_outs):
            smu = func.mu(sout, False)
            rmu = func.mu(reconstruct_out, False)
            ssigma = func.sigma(sout, False)
            rsigma = func.sigma(reconstruct_out, False)
            mu_loss+=criterion(rmu, smu)
            sigma_loss+=criterion(rsigma, ssigma)
        loss = lc + (mu_loss+sigma_loss)*style_weight
        valid_loss += loss.item()
        valid_data_num+=cdatas.shape[0]


    # 諸々average
    train_loss/=train_data_num
    valid_loss/=valid_data_num
    closs/=train_data_num
    sloss/=train_data_num
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    closses.append(closs)
    slosses.append(sloss)
    # 整形, 可視化
    losses = {
        "train": train_losses,
        "valid": valid_losses
        }
    x = list(range(len(train_losses)))
    io.time_draw(x, losses, "train_result/loss.png")
    io.time_draw(x, {"train": slosses}, "train_result/sloss.png")
    io.time_draw(x, {"train": closses}, "train_result/closs.png")

    print("----------------------------")
    print(" loss:")
    print("     train: %lf"%(train_loss))
    print("     valid: %lf"%(valid_loss))
    print("implemented time:")
    print("     time: %lf"%(time.time()-start_time))
    print("----------------------------")

    """
    sdatas = sdatas[0].permute(1, 2, 0).contiguous().cpu().detach().numpy()*256
    cdatas = cdatas[0].permute(1, 2, 0).contiguous().cpu().detach().numpy()*256
    pic = pic[0].permute(1, 2, 0).contiguous().cpu().detach().numpy()*256
    train_pic = train_pic[0].permute(1, 2, 0).contiguous().cpu().detach().numpy()*256
    """
    sdatas = sdatas[0].permute(1, 2, 0).contiguous().cpu().detach().numpy()
    cdatas = cdatas[0].permute(1, 2, 0).contiguous().cpu().detach().numpy()
    pic = pic[0].permute(1, 2, 0).contiguous().cpu().detach().numpy()
    train_pic = train_pic[0].permute(1, 2, 0).contiguous().cpu().detach().numpy()
    cv2.imwrite("train_result/%d_s.jpg"%(epoch), sdatas)
    cv2.imwrite("train_result/%d_c.jpg"%(epoch), cdatas)
    cv2.imwrite("train_result/%d_t_valid.jpg"%(epoch), pic)
    cv2.imwrite("train_result/%d_t_train.jpg"%(epoch), train_pic)

