import torch
from torch import nn
from torch import Tensor
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.models._utils import _make_divisible
from typing import Callable, Optional, List
import torch.nn.functional as F


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(Conv2dNormActivation(
                inp, hidden_dim, kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.ReLU6
            ))
        layers.extend([
            # dw
            Conv2dNormActivation(
                hidden_dim, hidden_dim, stride=stride,
                groups=hidden_dim, norm_layer=norm_layer,
                activation_layer=nn.ReLU6
            ),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        emb_size: int = 1280,
        featmodule=None,
        norm_layer=None,
        lossfn=None
    ) -> None:
        super().__init__()
        self.featmodule = featmodule
        self.lossfn = lossfn
        # width_mult (float): Width multiplier - adjusts number of
        # channels in each layer by this amount
        width_mult: float = 1.0
        # round_nearest (int): Round the number of channels in each
        # layer to be a multiple of this number
        # Set to 1 to turn off rounding
        round_nearest: int = 8
        # block: Module specifying inverted residual building block
        # for mobilenet
        block = InvertedResidual
        # norm_layer: Module specifying the normalization layer to use
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 64
        last_channel = emb_size
        # inverted_residual_setting: Network structure
        inverted_residual_setting = [
            # t, c, n, s
            [2, 64, 5, 2],
            [4, 128, 1, 2],
            [2, 128, 6, 2],
            [4, 128, 1, 2],
            [2, 128, 2, 1],

            # [1, 256, 1, 2],
            # [1, 128, 1, 2],
            # [1, 128, 11, 1],
            # [1, 128, 1, 2],
            # [1, 64, 1, 2],

            # [1, 2*input_channel, 1, 2],
            # [1, 2*input_channel, 1, 2],
            # [1, 4*input_channel, 6, 1],
            # [1, 8*input_channel, 1, 2],
            # [1, 8*input_channel, 1, 2],

            # [1, 8*input_channel, 1, 2],
            # [1, 8*input_channel, 1, 2],
            # [1, 4*input_channel, 6, 1],
            # [1, 2*input_channel, 1, 2],
            # [1, 2*input_channel, 1, 2],

            # [1, 8*input_channel, 1, 2],
            # [1, 6*input_channel, 1, 2],
            # [1, 3*input_channel, 6, 1],
            # [1, 6*input_channel, 1, 2],
            # [1, 8*input_channel, 1, 2],

            # [1, 3*input_channel, 1, 2],
            # [1, 4*input_channel, 1, 2],
            # [1, 5*input_channel, 6, 1],
            # [1, 4*input_channel, 1, 2],
            # [1, 3*input_channel, 1, 2],

            # [1, 4*input_channel+160, 1, 2],
            # [1, 4*input_channel+160, 1, 2],
            # [1, 4*input_channel+160, 6, 1],
            # [1, 4*input_channel+160, 1, 2],
            # [1, 4*input_channel+160, 1, 2],

            # [1, 8*input_channel, 1, 2],
            # [1, 8*input_channel, 1, 2],
            # [1, 8*input_channel, 1, 2],
            # [1, 8*input_channel, 1, 2],

            # [1, 1*input_channel, 1, 2],
            # [1, 2*input_channel, 1, 2],
            # [1, 3*input_channel, 20, 1],
            # [1, 2*input_channel, 1, 2],
            # [1, 1*input_channel, 1, 2],

        ]
        self.inverted_residual_setting = inverted_residual_setting
        # only check the first element,
        # assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(
            inverted_residual_setting[0]
        ) != 4:
            raise ValueError(
                "inverted_residual_setting should be non-empty "
                "or a 4-element list, got {}".format(inverted_residual_setting)
            )

        # building first layer
        input_channel = _make_divisible(
            input_channel * width_mult,
            round_nearest
        )
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult),
            round_nearest
        )
        features: List[nn.Module] = [
            Conv2dNormActivation(
                1, input_channel, stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.ReLU6
            )
        ]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(
                    input_channel, output_channel,
                    stride, expand_ratio=t,
                    norm_layer=norm_layer
                ))
                input_channel = output_channel
        # building last several layers
        features.append(Conv2dNormActivation(
            input_channel, self.last_channel,
            kernel_size=1, norm_layer=norm_layer,
            activation_layer=nn.ReLU6
        ))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor, labels=None) -> Tensor:
        # This exists since TorchScript doesn't support inheritance,
        # so the superclass method
        # (this one) needs to have a name other than `forward` that
        # can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        # x = self.GDConv(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # x = F.adaptive_max_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        embs = x
        if self.training:
            out = self.lossfn(x, labels)
        else:
            out = {'embedding': embs}
        return out

    def forward(self, x: torch.Tensor, labels=None):
        if self.featmodule is not None:
            x = 10 * torch.log10(self.featmodule(x))
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return self._forward_impl(x, labels)
