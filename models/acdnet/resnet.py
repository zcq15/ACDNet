import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import sys
sys.path.append('.')
from ..modules import _make_pad

from .acdconv.acdconv import ACDConv

__all__ = ['ResNet', 'ResEncoder']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel='resnet',input_height=512):
        super(BasicBlock, self).__init__()
        self.pad_3 =  _make_pad(1) if kernel in ['resnet'] else nn.Sequential()
        conv1_dict = {
            'resnet': nn.Conv2d(inplanes,planes,kernel_size=3,padding=0,stride=stride,bias=False),
            'acdnet': ACDConv(inplanes,planes,stride=stride,bias=False)
        }
        self.conv1 = conv1_dict[kernel]
        self.bn1 = nn.Sequential() if kernel in ['acdnet'] else nn.BatchNorm2d(planes)
        conv2_dict = {
            'resnet': nn.Conv2d(planes,planes,kernel_size=3,padding=0,stride=1,bias=False),
            'acdnet': ACDConv(planes,planes,stride=1,bias=False)
        }
        self.conv2 = conv2_dict[kernel]
        self.bn2 = nn.Sequential() if kernel in ['acdnet'] else nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.kernel = kernel
        self.stride = stride
        del conv1_dict,conv2_dict
    def forward(self, x):
        identity = x

        out = self.conv1(self.pad_3(x))
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(self.pad_3(out))
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel='resnet', padding='sppad', input_height=128):
        super(Bottleneck, self).__init__()
        self.pad_3 =  _make_pad(1) if kernel in ['resnet'] else nn.Sequential()
        self.conv1 = nn.Conv2d(inplanes, planes,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        conv2_dict = {
            'resnet': nn.Conv2d(planes,planes,kernel_size=3,padding=0,stride=stride,bias=False),
            'acdnet': ACDConv(planes,planes,stride=stride,bias=False)
        }
        self.conv2 = conv2_dict[kernel]
        self.bn2 = nn.Sequential() if kernel in ['acdnet'] else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.kernel = kernel
        self.stride = stride
        del conv2_dict

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(self.pad_3(out))
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True, kernel='resnet', **kwargs):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.layer1 = self._make_layer(block, 64, layers[0], kernel=kernel)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, kernel=kernel)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, kernel=kernel)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, kernel=kernel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    if hasattr(m.bn3,'weight'):
                        nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    if hasattr(m.bn2,'weight'):
                        nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, kernel, stride=1):
        # print([stride,size])
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, kernel=kernel))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            # print(['sub',stride,size])
            layers.append(block(self.inplanes, planes, kernel=kernel))
        # if strip:
        #     layers.append(StripPool(planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']),strict=False)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']),strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']),strict=False)
    return model

def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']),strict=False)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

class ResEncoder(nn.Module):
    def __init__(self, layers, encoder='resnet', output_size=(512,1024), in_channels=3, pretrained=True, log=False, **kwargs):
        super().__init__()
        assert encoder in ['resnet','acdnet']
        assert layers in [18, 34, 50, 101, 152]
        assert output_size in [(1024,2048),(512,1024),(256,512)]

        self.output_size = output_size
        if layers <= 34:
            self.out_channels = [64, 128, 256, 512]
        else:
            self.out_channels = [256, 512, 1024, 2048]

        self.feat_heights = [self.output_size[0]//4//(2**i) for i in range(4)]

        pretrained_model = eval('resnet{}'.format(layers))(pretrained=not(encoder=='acdnet'), kernel=encoder)

        self.layer0 = nn.Sequential(
            _make_pad(3),
            pretrained_model._modules['conv1'] if in_channels==3 else nn.Conv2d(
                            in_channels, 64, kernel_size=7, stride=2, padding=0, bias=False),
            pretrained_model._modules['bn1'] if in_channels==3 else nn.BatchNorm2d(64),
            pretrained_model._modules['relu'],
            _make_pad(1),
            pretrained_model._modules['maxpool']
        )

        if layers <= 34:
            ch_lst = [64, 64, 128, 256, 512]
        else:
            ch_lst = [64, 256, 512, 1024, 2048]

        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        del pretrained_model
        torch.cuda.empty_cache()
        init_first = in_channels==3
        if log:
            print('Initialize model with pretrained ResNet{} ...'.format(layers,'' if init_first else 'not '))
        if encoder in ['acdnet']:
            self.load_pretrained_state_dict(layers,init_first=init_first)

    def __key__(self):
        state_dict = self.state_dict()
        return [_ for _ in state_dict]

    def load_pretrained_state_dict(self,layers,init_first=True):
        state_dict = self.state_dict()
        # resnet = eval('resnet{}'.format(layers))(pretrained=True,kernel='resnet', padding=padding)
        resnet = ResEncoder(layers,kernel='resnet', pretrained=True)
        resnet_state_dict = resnet.state_dict()
        # for _ in state_dict.keys():
        #     print(_)
        # for _ in resnet_state_dict.keys():
        #     print(_)
        for key in resnet_state_dict.keys():
            ids = key.split('.')
            if ids[0] == 'layer0' and init_first:
                state_dict[key] = resnet_state_dict[key]
                continue
            if ids[0] in ['layer1','layer2','layer3','layer4'] and layers<=34:
                if ids[2] == 'downsample':
                    state_dict[key] = resnet_state_dict[key]
                    continue
                l,b,n,k = ids
                if n in ['conv1','conv2']:
                    for i in range(4):
                        new_key = '{}.{}.{}.convs.{}.0.conv.{}'.format(l,b,n,i,k)
                        # assert new_key in state_dict
                        state_dict[new_key] = resnet_state_dict[key]
                elif n in ['bn1','bn2']:
                    for i in range(4):
                        new_key = '{}.{}.conv{}.convs.{}.1.{}'.format(l,b,n[-1],i,k)
                        # assert new_key in state_dict
                        state_dict[new_key] = resnet_state_dict[key]
                continue
            if ids[0] in ['layer1','layer2','layer3','layer4'] and layers>=50:
                if ids[2] in ['conv1','bn1','conv3','bn3','downsample']:
                    state_dict[key] = resnet_state_dict[key]
                    continue
                l,b,n,k = ids
                if n == 'conv2':
                    for i in range(4):
                        new_key = '{}.{}.conv2.convs.{}.0.conv.{}'.format(l,b,i,k)
                        # assert new_key in state_dict
                        state_dict[new_key] = resnet_state_dict[key]
                elif n == 'bn2':
                    for i in range(4):
                        new_key = '{}.{}.conv2.convs.{}.1.{}'.format(l,b,i,k)
                        # assert new_key in state_dict
                        state_dict[new_key] = resnet_state_dict[key]
                continue
        self.load_state_dict(state_dict)
        del state_dict,resnet_state_dict,resnet

    def forward(self, inputs):
        # resnet
        x = inputs
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        # x5 = self.layre5(x4)
        return [x1,x2,x3,x4]


