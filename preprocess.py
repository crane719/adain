import hoge
import cv2
import random
import joblib

class Preprocess():
    def __init__(self, content_dir, style_dir, save_dir):
        content_dirs = hoge.get_filedirs(content_dir+"/*")
        print("content images: %d"%(len(content_dirs)))
        for i, content_dir in enumerate(content_dirs):
            if i%1000==0:
                print(" [%d/%d]"%(i, len(content_dirs)))
            pic = cv2.imread(content_dir)
            pic = cv2.resize(pic, (512,512))
            pic = self.random_trim(pic, 256)
            pic = pic/256 # normalization
            #joblib.dump(pic, save_dir+"/content/"+"%d"%(i))
            cv2.imwrite(save_dir+"/content/"+"%d.jpg"%(i), pic)

        style_dirs = hoge.get_filedirs(style_dir+"/*")
        print("style images: %d"%(len(style_dirs)))
        for i, style_dir in enumerate(style_dirs):
            if i%1000==0:
                print(" [%d/%d]"%(i, len(style_dirs)))
            pic = cv2.imread(style_dir)
            pic = cv2.resize(pic, (512,512))
            pic = self.random_trim(pic, 256)
            pic = pic/256 # normalization
            #joblib.dump(pic, save_dir+"/style/"+"%d"%(i))
            cv2.imwrite(save_dir+"/style/"+"%d.jpg"%(i), pic)

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
    Preprocess(train_c_dir, train_s_dir, train_dataset)

