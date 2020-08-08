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

        # 前の方の層の重みを固定
        for i, param in enumerate(self.vgg19.parameters()):
            if i<=fixed_point:
                param.requires_grad=False

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
            x = layer(x)
            for i in self.style_outputs:
                outputs.append(x)
        return x, outputs


if __name__=="__main__":
    from config import *
    encoder = Encoder(fixed_point, style_outputs)
    x = torch.zeros(1, 3, 256, 256)
    x, outputs = encoder(x)
    print(x.shape)


