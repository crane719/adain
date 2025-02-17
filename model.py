import torch
import torch.nn as nn
import hoge
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, fixed_point, style_outputs):
        """
        Args:
            fixed_point: 重みを固定するargの最大値
            style_outputs: loss calcのために渡すargs
        """
        super(Encoder, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True).features
        self.style_outputs = style_outputs

        # 重みを固定
        for i, param in enumerate(self.vgg19.parameters()):
            param.requires_grad=False
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))

        # convのpaddingを削除
        for i, layer in enumerate(self.vgg19):
            if "Conv" in str(layer):
                self.vgg19[i].padding = (0, 0)

    def forward(self, x):
        """
        Args:
            x: 入力画像
        Returns:
            x: 最終的な埋め込み
            outputs: loss calc用のreluの出力ら
        """
        outputs = []
        for i, layer in enumerate(self.vgg19):
            if "Conv" in str(layer):
                x = self.pad(x)
            x = layer(x)
            if i in self.style_outputs:
                outputs.append(x)
        return x, outputs

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        """
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        nn.ReLU(inplace=True),
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        nn.ReLU(inplace=True),
        """
        self.model = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=(3,3), stride=(1,1), padding=(0,0)),
                nn.ReLU(inplace=True),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(0,0)),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(0,0)),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(0,0)),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=(3,3), stride=(1,1), padding=(0,0)),
                nn.ReLU(inplace=True),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(128, 128, kernel_size=(3,3), stride=(1,1), padding=(0,0)),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=(3,3), stride=(1,1), padding=(0,0)),
                nn.ReLU(inplace=True),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(0,0)),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 3, kernel_size=(3,3), stride=(1,1), padding=(0,0)),)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))

    def forward(self, x):
        for layer in self.model:
            if "Conv" in str(layer):
                x = self.pad(x)
            x = layer(x)
        return x

if __name__=="__main__":
    from config import *
    encoder = Encoder(fixed_point, style_outputs)
    x = torch.zeros(1, 3, 256, 256)
    x, outputs = encoder(x)
    decoder = Decoder()
    pic = decoder(x)


