import importlib


from torch import nn
from torch.nn.functional import relu

from summary import summary
from utils import Conv2D


class BoundaryRefinement(nn.Module):
    def __init__(self, inchannels=21, channels=21, kernel_size=(3, 3)):
        super(BoundaryRefinement, self).__init__()

        self.conv_1 = Conv2D(in_channels=inchannels, out_channels=channels, kernel_size=kernel_size, padding='same')
        self.conv_2 = Conv2D(in_channels=inchannels, out_channels=channels, kernel_size=kernel_size, padding='same')

    def forward(self, x):
        x1 = self.conv_1(x)
        x1 = relu(x1, inplace=True)
        x1 = self.conv_2(x1)

        out = x + x1

        return out


class GCN(nn.Module):
    def __init__(self, inchannels, channels=21, k=3):
        super(GCN, self).__init__()

        self.conv_l1 = Conv2D(in_channels=inchannels, out_channels=channels, kernel_size=(k, 1), padding='same')
        self.conv_l2 = Conv2D(in_channels=channels, out_channels=channels, kernel_size=(1, k), padding='same')

        self.conv_r1 = Conv2D(in_channels=inchannels, out_channels=channels, kernel_size=(1, k), padding='same')
        self.conv_r2 = Conv2D(in_channels=channels, out_channels=channels, kernel_size=(k, 1), padding='same')

    def forward(self, x):
        x1 = self.conv_l1(x)
        x1 = self.conv_l2(x1)

        x2 = self.conv_r1(x)
        x2 = self.conv_r2(x2)

        out = x1 + x2

        return out


class NetworkPath(nn.Module):
    def __init__(self, inchannels, channels=21, gcn_kernel=3, br_kernel=3):
        super(NetworkPath, self).__init__()

        self.gcn = GCN(inchannels=inchannels, channels=channels, k=gcn_kernel)
        self.br1 = BoundaryRefinement(inchannels=channels, channels=channels, kernel_size=br_kernel)
        self.br2 = BoundaryRefinement(inchannels=channels, channels=channels, kernel_size=br_kernel)

        self.deconv = nn.ConvTranspose2d(in_channels=channels, out_channels=channels, kernel_size=2, stride=2)

    def forward(self, x, add_inp=None):
        x = self.gcn(x)
        x = self.br1(x)

        if add_inp is not None:
            x = x + add_inp
            x = self.br2(x)

        out = self.deconv(x)

        return out


class LKM(nn.Module):
    def __init__(self, resnet_extractor='resnet50'):
        """

        :param resnet_extractor: modification of resnet that will be used as feature extractor
        """
        super(LKM, self).__init__()

        resnet = importlib.import_module("torchvision.models.resnet")

        # load resnet in order to use it as feature extractor

        model = eval("resnet.{}(pretrained=True)".format(resnet_extractor))

        self.conv0 = nn.Sequential(*list(model.children())[:4])

        # resnet blocks
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        # turn off learning for resnet blocks
        for param in self.parameters():
            param.requires_grad_(False)

        # network's scales paths

        self.path1 = NetworkPath(inchannels=256, channels=21, gcn_kernel=3, br_kernel=3)
        self.path2 = NetworkPath(inchannels=512, channels=21, gcn_kernel=3, br_kernel=3)
        self.path3 = NetworkPath(inchannels=1024, channels=21, gcn_kernel=3, br_kernel=3)
        self.path4 = NetworkPath(inchannels=2048, channels=21, gcn_kernel=3, br_kernel=3)

        # end of the network

        self.br5 = BoundaryRefinement(inchannels=21, channels=21)
        self.deconv5 = nn.ConvTranspose2d(in_channels=21, out_channels=21, kernel_size=2, stride=2)
        self.br6 = BoundaryRefinement(inchannels=21, channels=21)

    def forward(self, x):
        x = self.conv0(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x4 = self.path4(x4)
        x3 = self.path3(x3, x4)
        x2 = self.path2(x2, x3)
        x1 = self.path1(x1, x2)

        out = self.br5(x1)
        out = self.deconv5(out)
        out = self.br6(out)

        return out




model = LKM(resnet_extractor="resnet152")

# from torchvision.models.resnet import resnet152
#
# model = resnet152(pretrained=True)
#
summary((3, 256, 256), model)





