import matplotlib.pyplot as plt
import numpy as np


def time_draw(x, ys, directory):
    """
    複数の時系列を可視化
    :param x: x軸のデータ
    :param ys: y軸のデータら. dictionaryでkeyを時系列のlabel, valueをデータとする
    :param directory: 出力するdirectory
    """
    plt.figure()
    for label, y in ys.items():
        plt.plot(y, label=label)
    plt.legend()
    plt.savefig(directory)
    plt.close()
