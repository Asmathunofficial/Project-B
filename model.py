import torch.nn as nn


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                                   bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_channeld, out_channels):
        super(ResidualBlock, self).__init__()

        self.residual_conv = nn.Conv2d(in_channels=in_channeld, out_channels=out_channels, kernel_size=1, stride=2,
                                       bias=False)
        self.residual_bn = nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)

        self.sepConv1 = SeparableConv2d(in_channels=in_channeld, out_channels=out_channels, kernel_size=3, bias=False,
                                        padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        self.relu = nn.ReLU()

        self.sepConv2 = SeparableConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, bias=False,
                                        padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        res = self.residual_conv(x)
        res = self.residual_bn(res)
        x = self.sepConv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.sepConv2(x)
        x = self.bn2(x)
        x = self.maxp(x)
        return res + x


class Model(nn.Module):

    def __init__(self, num_classes):
        super(Model, self).__init__()

        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride = 1, bias=False)
        self.bn1 = nn.BatchNorm2d(8, momentum=0.99, eps=1e-3)
        self.relu1 = nn.ReLU()


        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride = 1, bias=False)
        self.bn2 = nn.BatchNorm2d(8, momentum=0.99, eps=1e-3)
        self.relu2 = nn.ReLU()

        self.ResBlock1 = ResidualBlock(8, 16)
        self.ResBlock2 = ResidualBlock(16, 32)
        self.ResBlock3 = ResidualBlock(32, 64)
        self.ResBlock4 = ResidualBlock(64, 128)
        
        self.conv3 = nn.Conv2d(128, out_channels = num_classes, kernel_size=3, padding=1, stride=1)
        self.AdapAvg = nn.AdaptiveAvgPool2d((1,1))


    def forward(self, input):


        x=self.conv1(input)
        x=self.bn1(x)
        x=self.relu1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu2(x)
        x=self.ResBlock1(x)
        x=self.ResBlock2(x)
        x=self.ResBlock3(x)
        x=self.ResBlock4(x)
        
        x=self.conv3(x)
        x=self.AdapAvg(x)
        x=x.view((x.shape[0],-1))
        
        return x


