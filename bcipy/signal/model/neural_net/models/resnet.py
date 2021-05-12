"""adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py"""
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bcipy.signal.model.neural_net.utils import get_activation

from .base_model import Classifier


def conv(in_planes: int, out_planes: int, kernel_size: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """3x3 convolution with padding"""
    padding = kernel_size // 2
    assert padding * 2 + 1 == kernel_size  # true for odd kernel size

    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size1: int,
        kernel_size2: int,
        act_name: str,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv(in_planes, out_planes, kernel_size1, stride)
        self.bn1 = norm_layer(out_planes)
        self.act = get_activation(act_name)
        self.conv2 = conv(out_planes, out_planes, kernel_size2)
        self.bn2 = norm_layer(out_planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out


class ResNet1D(Classifier):
    def __init__(
        self,
        layers: List[int],
        num_classes: int,
        in_channels: int,
        device: torch.device,
        groups: int = 1,
        act_name: str = "ELU",
        width_per_group: int = 64,
        kernel_size1: int = 5,
        kernel_size2: int = 3,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        criterion=F.nll_loss,
    ):
        super().__init__()
        self.device = device
        self.norm_layer = nn.BatchNorm1d

        self.criterion = criterion

        self.in_planes = 64
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.act_name = act_name
        self.act = get_activation(self.act_name)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False),
            self.norm_layer(self.in_planes),
            self.act,
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            self._make_layer(64, layers[0]),
            self._make_layer(128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]),
            self._make_layer(256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]),
            self._make_layer(512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Linear(512, num_classes),
            nn.LogSoftmax(dim=-1),
        )
        self.to(self.device)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1, dilate: bool = False) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_planes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                conv1(self.in_planes, planes * BasicBlock.expansion, stride),
                self.norm_layer(planes * BasicBlock.expansion),
            )

        layers = []
        layers.append(
            BasicBlock(
                self.in_planes,
                planes,
                kernel_size1=self.kernel_size1,
                kernel_size2=self.kernel_size2,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                act_name=self.act_name,
                base_width=self.base_width,
                dilation=previous_dilation,
                norm_layer=self.norm_layer,
            )
        )
        self.in_planes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    self.in_planes,
                    planes,
                    kernel_size1=self.kernel_size1,
                    kernel_size2=self.kernel_size2,
                    groups=self.groups,
                    act_name=self.act_name,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=self.norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _trace(self, example_x):
        self.net = torch.jit.trace(self.net, example_x.unsqueeze(0).to(self.device))

    # git blame torchvision/models/resnet.py - on the line of "def forward()" and "def _forward_impl()"
    # 227027d5abc8eacb110c93b5b5c2f4ea5dd401d6
    def forward(self, x):
        return self.net(x)

    def get_outputs(self, data, labels):
        data, labels = data.to(self.device), labels.to(self.device)
        log_probs = self.forward(data)
        loss = self.criterion(log_probs, labels)
        acc = self.get_acc(log_probs, labels)
        return {"loss": loss, "log_probs": log_probs, "acc": acc}

    def get_acc(self, log_probs, labels):
        return torch.tensor(100 * log_probs.argmax(1).eq(labels).sum().item() / labels.shape[0])
