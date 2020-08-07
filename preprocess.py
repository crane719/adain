import hoge
import cv2
import random

class Preprocess():
    def __init__(self, content_dir, style_dir):
        content_dirs = hoge.get_filedirs(content_dir+"/*")
        style_dirs = hoge.get_filedirs(style_dir+"/*")

        print("content images: %d"%(len(content_dirs)))
        print("style images: %d"%(len(style_dirs)))

        for content_dir in content_dirs:
            pic = cv2.imread(content_dir)
            pic = cv2.resize(pic, (512,512))
            pic = self.random_trim(pic, 256)

    def random_trim(self, pic, size):
        w = random.choice(range(0, pic.shape[1]))
        h = random.choice(range(0, pic.shape[0]))

        if w>=size:
            w_range = [w-size, w]
        else:
            w_range = [w, w+size]
        if h>=size:
            h_range = [h-size, h]
        else:
            h_range = [h, h+size]

        pic = pic[:, w_range[0]:w_range[1], :]
        pic = pic[h_range[0]:h_range[1], :, :]
        return pic


if __name__ == '__main__':
    from config import *
    Preprocess(train_c_dir, train_s_dir)

