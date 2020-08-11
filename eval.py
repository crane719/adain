import shutil
import cv2
import torch

import model
import hoge
from config import *
import function as func

if hoge.is_dir_existed("test_result"):
    print("delete file...")
    print("- test_result")
    shutil.rmtree("./test_result")
required_dirs = ["test_result"]
hoge.make_dir(required_dirs)

content_dirs = hoge.get_filedirs(test_c_dir+"/*")
style_dirs = hoge.get_filedirs(test_s_dir+"/*")

encoder = hoge.try_gpu(model.Encoder(fixed_point, style_outputs))
decoder = hoge.try_gpu(model.Decoder())
if torch.cuda.is_available():
    encoder.load_state_dict(torch.load(eweight_dir))
    decoder.load_state_dict(torch.load(dweight_dir))
else:
    encoder.load_state_dict(torch.load(eweight_dir,  map_location="cpu"))
    decoder.load_state_dict(torch.load(dweight_dir,  map_location="cpu"))

# 全組み合わせ
for i, content_dir in enumerate(content_dirs):
    for j, style_dir in enumerate(style_dirs):
        encoder.eval()
        decoder.eval()

        content = cv2.imread(content_dir)
        style = cv2.imread(style_dir)

        # shapeをあとで直すために保存
        cshape = content.shape
        sshape = style.shape

        # 256*256
        content = cv2.resize(content, (256,256))
        style = cv2.resize(style, (256,256))

        content = hoge.try_gpu(torch.Tensor(content)).permute(2, 0, 1).contiguous().unsqueeze(0)
        style = hoge.try_gpu(torch.Tensor(style)).permute(2, 0, 1).contiguous().unsqueeze(0)

        # input
        _, couts = encoder(content)
        crep = couts[-1]
        _, souts = encoder(style)
        srep = souts[-1]
        rep = func.adain(crep, srep)
        pic = decoder(rep)

        pic = pic.permute(0, 2, 3, 1).contiguous().squeeze().cpu().detach().numpy()
        pic = cv2.resize(pic, cshape[:2])
        cv2.imwrite("test_result/%d_%d.jpg"%(i, j), pic)
