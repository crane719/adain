import sys
import argparse
import glob
import os
from collections import OrderedDict
import collections
import matplotlib.pyplot as plt
import numpy as np
import shutil

def get_pics(content_dirs, style_dirs):
    import torch
    import cv2
    cdatas = []
    sdatas = []
    for content_dir in content_dirs:
        cdatas.append(cv2.imread(content_dir))
    for style_dir in style_dirs:
        sdatas.append(cv2.imread(style_dir))
    return torch.Tensor(cdatas), torch.Tensor(sdatas)

def try_gpu(obj):
    import torch
    if torch.cuda.is_available():
        return obj.cuda()
    return obj

def convert2onehot(vec, dim):
    """
    特徴量のnumpy配列をonehotベクトルに変換
    :param vec: 特徴量のnumpy行列, int型 (サンプル数分の1次元行列)．
    :param dim: onehot vectorの次元
    :return: onehot vectorのnumpy行列
    """
    return np.identity(dim)[vec]

def zero_padding(vecs, flow_len, value=0):
    """
    flowの長さを最大flow長に合わせるためにzeropadding
    :param vecs: flow数分のリスト. リストの各要素はflow長*特徴量長の二次元numpy配列
    :param flow_len: flow長. int
    :param value: paddingするvectorの要素値 int
    :return: データ数*最大flow長*特徴量長の3次元配列
    """
    for i in range(len(vecs)):
        flow = vecs[i]
        diff_vec = np.ones((flow_len-flow.shape[0], flow.shape[1]))
        diff_vec *= value
        vecs[i] = np.concatenate((flow, diff_vec), 0)
    return np.array(vecs)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", action="store_true", help="前処理を行う場合")
    parser.add_argument("--player_emb", action="store_true", help="プレイヤー情報の埋め込み")
    parser.add_argument("--dir_id", default='', type=str, help="出力ディレクトリにつけるID")
    args = parser.parse_args()
    return args

def methods(obj):
    for method in dir(obj):
        print(method)

def get_filedirs(required_dir):
    return glob.glob(required_dir)

def make_dir(required_dirs):
    for required_dir in required_dirs:
        if not os.path.exists(required_dir):
            print("generate file in current dir...")
            print("+ "+required_dir)
            os.mkdir(required_dir)

def recreate_dir(directory):
    for dir in directory:
        shutil.rmtree(dir)
    make_dir(directory)

def is_dir_existed(directory):
    dirs = glob.glob("*")
    if directory in dirs:
        return True
    else:
        return False

def flatten(nested_list):
    result = []
    for element in nested_list:
        if isinstance(element, collections.Iterable) and not isinstance(element, str):
            result.extend(flatten(element))
        else:
            result.append(element)
    return result

if __name__ == '__main__':
    # onehotのtest
    print("onehot test")
    print("before")
    vec = np.random.randint(0, 10, 10)
    print(vec)
    onehot = convert2onehot(vec, 10)
    print("after")
    print(onehot)

    # zero_paddingのtest
    print("zeropadding test")
    print("before")
    flows = [np.array([[1], [2], [3]]), np.array([[1], [2]])]
    print(flows)

    print("after")
    flows = zero_padding(flows, 3)
    print(flows.shape)
    print(flows)
