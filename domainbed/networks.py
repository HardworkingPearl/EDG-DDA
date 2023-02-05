# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from domainbed.lib import wide_resnet
import copy
from abc import abstractmethod

def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=True)
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if 'pure_liner' in hparams:
        return MLP_Empty(input_shape[0], 128, hparams)
    elif len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError

class MNIST_CNN_Decoder(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    # n_outputs = 128

    def __init__(self, input_dim, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 1024, bias=False), nn.BatchNorm1d(1024), nn.ReLU())
        self.up1 = nn.Upsample(8)
        self.de1 = nn.Sequential(nn.ConvTranspose2d(64, 128, kernel_size=5, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(128), nn.ReLU())
        self.up2 = nn.Upsample(24)
        self.de2 = nn.Sequential(nn.ConvTranspose2d(128, 256, kernel_size=5, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(256), nn.ReLU())
        self.de3 = nn.Sequential(nn.Conv2d(256, output_shape[0], kernel_size=1, stride=1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = out.view(-1, 64, 4, 4)
        out = self.up1(out)
        out = self.de1(out)
        out = self.up2(out)
        out = self.de2(out)
        out = self.de3(out)
        out = self.sigmoid(out)
        out = out.view(out.shape[0], *self.output_shape)
        return out

class MLP_Empty(nn.Module):
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP_Empty, self).__init__()
        self.n_outputs = n_inputs
        self.tmp = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = x + self.tmp
        # return self.relu(x)
        return x


def Classifier(in_features, out_features, is_nonlinear=False, num_domains=None):
    if is_nonlinear:
        if in_features <= 4:
            return torch.nn.Linear(in_features, out_features)
        else:
            return torch.nn.Sequential(
                torch.nn.Linear(in_features, in_features // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features // 2, in_features // 4),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


def Decoder(output_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    # print(hparams)
    # if 'pure_liner' in hparams:
    #     return MLP_Empty(hparams['zc_dim'] + hparams['zdy_dim'], hparams['zc_dim'] + hparams['zdy_dim'], hparams)
    if len(output_shape) == 1:
        return MLP(hparams['zc_dim'] + hparams['zdy_dim'], output_shape[0], hparams)
    elif output_shape[1:3] == (28, 28):
        return MNIST_CNN_Decoder(hparams['zc_dim'] + hparams['zdy_dim'], output_shape=output_shape)  # TODO
    elif output_shape[1:3] == (32, 32):
        return Wide_ResNet_Decoder(hparams['zc_dim'] + hparams['zdy_dim'], output_shape=output_shape)
    elif output_shape[1:3] == (224, 224):
        return ResNet_Decoder(hparams['zc_dim'] + hparams['zdy_dim'], output_shape=output_shape)
    else:
        raise NotImplementedError


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)

class DomainAdaptor(nn.Module):
    def __init__(self, hdim, hparams, n_heads=16, MLP=None):
        super(DomainAdaptor, self).__init__()
        self.attentive_module = nn.ModuleList([nn.MultiheadAttention(embed_dim=hdim, num_heads=n_heads) for _ in range(hparams['attn_depth'])])
        self.bn = nn.BatchNorm1d(hdim)
        self.layers = hparams['attn_depth']
        self.batch_size = hparams['batch_size']
        self.n_heads = hparams['attn_head'] if 'attn_head' in hparams else n_heads
        self.env_number = hparams['env_number']
        self.MLP = MLP

    def forward(self, x, x_kv, attn_mask=None, attend_to_domain_embs=False):
        x_0 = x
        x = x.unsqueeze(1)
        x_kv = x_kv.unsqueeze(1)
        if attend_to_domain_embs:
            for i in range(self.layers):
                residual = x
                x = self.attentive_module[i](x, x_kv, x_kv, attn_mask=attn_mask)[0]
            if self.MLP:
                return self.bn(x.squeeze(1) + self.MLP(x_0))
            else:
                return x.squeeze(1) 
        else:
            for i in range(self.layers):
                residual = x
                x = self.attentive_module[i](x, x, x, attn_mask=attn_mask)[0]
                x = x + residual 
            return self.bn(x.squeeze(1) + self.MLP(x_0))  


class Wide_ResNet_Decoder(nn.Module):
    """
    Hand-tuned architecture for Wide_ResNet_Decoder.
    """
    # n_outputs = 128

    def __init__(self, input_dim, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 1024, bias=False), nn.BatchNorm1d(1024), nn.ReLU())
        self.up1 = nn.Upsample(8)
        self.de1 = nn.Sequential(nn.ConvTranspose2d(64, 128, kernel_size=5, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(128), nn.ReLU())
        self.up2 = nn.Upsample(24)
        self.de2 = nn.Sequential(nn.ConvTranspose2d(128, 256, kernel_size=5, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(256), nn.ReLU())
        self.de28_32 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=5, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(256), nn.ReLU())
        self.de3 = nn.Sequential(nn.Conv2d(256, output_shape[0], kernel_size=1, stride=1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = out.view(-1, 64, 4, 4)
        out = self.up1(out)     # (-1, 64, 8, 8)
        out = self.de1(out)     # (-1, 128, 12, 12)
        out = self.up2(out)     # (-1, 128, 24, 24)
        out = self.de2(out)     # (-1, 256, 28, 28)
        out = self.de28_32(out) # (-1, 256, 32, 32)
        out = self.de3(out)     # (-1, 1, 32, 32)
        out = self.sigmoid(out) # (-1, 256, 32, 32)
        out = out.view(out.shape[0], *self.output_shape)
        return out

class ResNet_Decoder(nn.Module):
    """
    Hand-tuned architecture for Wide_ResNet_Decoder.
    """
    # n_outputs = 128

    def __init__(self, input_dim, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 1024, bias=False), nn.BatchNorm1d(1024), nn.ReLU())
        self.up1 = nn.Upsample(8)
        self.de1 = nn.Sequential(nn.ConvTranspose2d(64, 128, kernel_size=5, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(128), nn.ReLU())
        self.up2 = nn.Upsample(24)
        self.de2 = nn.Sequential(nn.ConvTranspose2d(128, 256, kernel_size=5, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(256), nn.ReLU())
        self.up3 = nn.Upsample(56)
        self.de3 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=5, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(256), nn.ReLU())
        self.up4 = nn.Upsample(112)
        self.de4 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=5, stride=1, padding='same', bias=False),
                                 nn.BatchNorm2d(256), nn.ReLU())
        self.up5 = nn.Upsample(224)
        self.de5 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=5, stride=1, padding='same', bias=False),
                                 nn.BatchNorm2d(256), nn.ReLU())
        self.de6 = nn.Sequential(nn.Conv2d(256, output_shape[0], kernel_size=1, stride=1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = out.view(-1, 64, 4, 4)
        out = self.up1(out)     # (-1, 64, 8, 8)
        out = self.de1(out)     # (-1, 128, 12, 12)
        out = self.up2(out)     # (-1, 128, 24, 24)
        out = self.de2(out)     # (-1, 256, 28, 28)
        out = self.up3(out)     # (-1, 256, 56, 56)
        out = self.de3(out)     # (-1, 256, 56, 56)
        out = self.up4(out)     # (-1, 256, 112, 112)
        out = self.de4(out)     # (-1, 256, 112, 112)
        out = self.up5(out)     # (-1, 256, 224, 224)
        out = self.de5(out)     # (-1, 256, 224, 224)
        out = self.de6(out)     # (-1, 256, 224, 224)
        out = self.sigmoid(out) # (-1, 3, 224, 224)
        out = out.view(out.shape[0], *self.output_shape)
        return out

        