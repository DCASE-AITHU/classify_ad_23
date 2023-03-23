from torch import nn
import math


Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 2],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]


class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super().__init__()
        self.connect = stride == 1 and inp == oup
        #
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # dw
            nn.Conv2d(
                inp * expansion,
                inp * expansion, 3,
                stride, 1,
                groups=inp * expansion,
                bias=False
            ),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super().__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


class MobileFacenet(nn.Module):
    def __init__(
        self,
        freq_bins=None,
        tframe=None,
        emb_size=128,
        featmodule=None,
        lossfn=None,
    ):
        super().__init__()
        self.featmodule = featmodule
        self.conv1 = ConvBlock(1, 64, 3, 2, 1)
        fmf = (freq_bins-1)//2 + 1
        fmt = (tframe-1)//2 + 1
        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)
        self.inplanes = 64
        block = Bottleneck
        bottleneck_setting = Mobilefacenet_bottleneck_setting
        self.blocks = self._make_layer(block, bottleneck_setting)
        self.conv2 = ConvBlock(128, 512, 1, 1, 0)
        for it in bottleneck_setting:
            s = it[3]
            fmf = (fmf-1)//s + 1
            fmt = (fmt-1)//s + 1
        self.GDConv = ConvBlock(
            512, 512, k=(fmf, fmt),
            s=1, p=0, dw=True, linear=True
        )
        self.linear1 = ConvBlock(512, emb_size, 1, 1, 0, linear=True)
        self.lossfn = lossfn
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x, labels=None):
        if self.featmodule is not None:
            x = self.featmodule(x)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.GDConv(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        embs = x
        if self.training:
            out = self.lossfn(x, labels)
        else:
            out = {'embedding': embs}
        return out
