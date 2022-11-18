import torch, math
import torch.nn as nn

import torch.nn.functional as F
from collections import OrderedDict
from util import util

### ----------------------- Modulos usados -----------------------------

class BaseNetwork(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def print_architecture(self):
        name = type(self).__name__
        result = '-------------------%s---------------------\n' % name
        total_num_params = 0

        for i, (name, child) in enumerate(self.named_children()):
            num_params = sum([p.numel() for p in child.parameters()])
            total_num_params += num_params
            result += "%s: %3.3fM\n" % (name, (num_params / 1e6))

            for i, (name, grandchild) in enumerate(child.named_children()):
                num_params = sum([p.numel() for p in grandchild.parameters()])
                result += "\t%s: %3.3fM\n" % (name, (num_params / 1e6))
                
        result += '[Network %s] Total number of parameters : %.3f M\n' % (name, total_num_params / 1e6)
        result += '-----------------------------------------------\n'
        print(result)


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True, lr_mul=1.0):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2) * lr_mul

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, lr_mul=1, activation=False):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias = None

        self.activation = activation
        self.relu = nn.LeakyReLU(0.2)

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        if self.activation:
            out = self.relu(out)

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ConvLayer(nn.Sequential):
    def __init__(self, in_c, out_c, kernel, stride, padding, bias=True, activate=True, pixelnorm=True, batchnorm=False, reflection_pad=False):
        layers = []

        if reflection_pad:
            layers.append(("pad", nn.ReflectionPad2d(padding)))
            padding = 0

        layers.append(("conv", EqualConv2d(in_c, out_c, kernel, padding=padding, stride=stride, bias=bias and not activate)))

        if batchnorm:
            layers.append(("norm", nn.BatchNorm2d(out_c)))
        elif pixelnorm:
            layers.append(("norm", PixelNorm()))

        if activate:
            layers.append(("act", nn.LeakyReLU(0.2)))

        super().__init__(OrderedDict(layers))

    def forward(self, x):
        out = super().forward(x)
        return out

class ResBlock(nn.Module):
    def __init__(self, inc, outc, batchnorm=False):
        super().__init__()

        self.etapa1 = ConvLayer(inc, outc, 3, 1, 1, pixelnorm = not batchnorm, batchnorm = batchnorm)
        self.etapa2 = ConvLayer(outc, outc, 3, 1, 1, pixelnorm = not batchnorm, batchnorm = batchnorm)

        if inc != outc:
            self.skip = ConvLayer(inc, outc, 1, 1, 0, bias=False, activate=False, pixelnorm=False)
        else:
            self.skip = None

    def forward(self, x):
        if self.skip is not None:
            identity_data = self.skip(x)
        else:
            identity_data = x

        out = self.etapa1(x)
        out = self.etapa2(out)
        out = (out + identity_data) / math.sqrt(2)

        return out


class UpsamplingResBlock(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.resblock = ResBlock(inc, outc)

    def forward(self, x):
        out = self.up(x)
        out = self.resblock(out)

        return out


class DownsamplingResBlock(nn.Module):
    def __init__(self, inc, outc, batchnorm=False):
        super().__init__()

        self.down = nn.AvgPool2d(2)
        self.resblock = ResBlock(inc, outc, batchnorm=batchnorm)

    def forward(self, x):
        out = self.down(x)
        out = self.resblock(out)

        return out


class UpsamplingBlock(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.etapa1 = ConvLayer(inc, outc, 3, 1, 1)
        self.etapa2 = ConvLayer(outc, outc, 3, 1, 1)

    def forward(self, x):
        out = self.up(x)

        out = self.etapa1(out)
        out = self.etapa2(out)

        return out


### ----------------------- Modelos Utilizados -----------------------------

class Prior(BaseNetwork):
    def __init__(self, opt):
        super().__init__(opt)

        # calculo das dimensões

        n_camadas = int(math.log(opt.size, 2))

        vec_ch = [32, 64, 128, 256, 256]
        channels = vec_ch + [256] * max((n_camadas - len(vec_ch)), 0)

        inc = opt.channels_pose
        outc = channels[0]

        # camadas

        self.camadas = nn.Sequential()
        self.camadas.add_module("PrevLayer", ConvLayer(inc, outc, 3, 1, 1, batchnorm=True))

        inc = outc

        for i, c in enumerate(channels[1:]):
            self.camadas.add_module("DownResBlock%d"%i, DownsamplingResBlock(inc, c, batchnorm=True))

            inc = c

        self.fc_mu = EqualLinear(inc*2*2, opt.latent_dims, activation=False)
        self.fc_logvar = EqualLinear(inc*2*2, opt.latent_dims, activation=False)

        self.tanH = nn.Tanh()

        self.print_architecture()

    def forward(self, pose):
        out = self.camadas(pose)
        
        out = torch.flatten(out, 1)

        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)

        return self.tanH(mu), self.tanH(logvar)


class Encoder(BaseNetwork):
    def __init__(self, opt):
        super().__init__(opt)

        # calculo das dimensões

        n_camadas = int(math.log(opt.size, 2))

        vec_ch = [64, 64, 128, 128, 256, 256]
        channels = vec_ch + [380] * max((n_camadas - len(vec_ch)), 0)

        inc = opt.channels_img + opt.channels_pose
        outc = channels[0]

        # camadas

        self.camadas = nn.Sequential()
        self.camadas.add_module("PrevLayer", ConvLayer(inc, outc, 3, 1, 1, batchnorm=True))

        inc = outc

        for i, c in enumerate(channels[1:]):
            self.camadas.add_module("DownResBlock%d"%i, DownsamplingResBlock(inc, c, batchnorm=True))

            inc = c

        self.fc_mu = EqualLinear(inc*2*2, opt.latent_dims, activation=False)
        self.fc_logvar = EqualLinear(inc*2*2, opt.latent_dims, activation=False)

        self.print_architecture()

    def forward(self, img, pose):
        ip = torch.cat([img, pose], dim=1)

        out = self.camadas(ip)
        
        out = torch.flatten(out, 1)

        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)

        return mu, logvar


class Generator(BaseNetwork):
    def __init__(self, opt):
        super().__init__(opt)

        self.opt = opt
        #self.n_camadas = int(math.log(self.opt.size, 2))

        self.channels = {
            4: 512,
            8: 480,
            16: 420,
            32: 256,
            64: 128,
            128: 64,
            256: 64
        }

        inc = self.opt.latent_dims
        outc = self.channels[4] * 4 * 4

        self.linear = EqualLinear(inc, outc, activation=True)

        inc = 4
        self.camadas = []

        #upsampling
        for i, dim in enumerate(self.channels):
            if dim == 4:
                continue
            if dim > self.opt.size:
                break

            self.add_module("UpsamplingBlock_%d"%i,
                UpsamplingBlock(self.channels[inc] + self.opt.channels_pose, self.channels[dim])
            )
            self.camadas.append([i, inc])

            inc = dim

        inc = self.channels[inc]    # 64
        out = inc // 2              # 32

        #out
        for i in range(self.opt.ch_gen_out):
            self.add_module("ResBlock_%d"%i, ResBlock(inc, out))

            inc = out
            out = max(out // 2, 16)

        self.out = ConvLayer(inc, 3, 3, 1, 1, activate=False, pixelnorm=False)
            
        self.print_architecture()

    def forward(self, z, pose):
        # -------------- colocar a pose em todas as dimensões

        pdim = pose.shape[-1]
        pose = {pdim: pose}

        while pdim > 4:
            p = F.avg_pool2d(pose[pdim], 2, 2, 0)
            pdim = p.shape[-1]
            pose[pdim] = p
        
        # --------------- inicio da rede

        out = self.linear(z)
        out = out.view(out.size(0), self.channels[4], 4, 4)

        # --------------- restante da rede
        for i, dims in self.camadas:
            layer = getattr(self, "UpsamplingBlock_%d"%i)
            out = layer(torch.cat([out, pose[dims]], dim=1))

        for i in range(self.opt.ch_gen_out):
            layer = getattr(self, "ResBlock_%d"%i)
            out = layer(out)

        out = self.out(out)

        return out


class Discriminator(BaseNetwork):
    def __init__(self, opt):
        super().__init__(opt)

        # calculo das dimensões

        n_camadas = int(math.log(opt.size, 2))

        vec_ch = [16, 32, 64, 128, 256]
        channels = vec_ch + [380] * max((n_camadas - len(vec_ch)), 0)

        inc = opt.channels_img
        outc = channels[0]

        # camadas

        self.camadas = nn.Sequential()
        self.camadas.add_module("PrevLayer", ConvLayer(inc, outc, 3, 1, 1, batchnorm=True))

        inc = outc

        for i, c in enumerate(channels[1:]):
            self.camadas.add_module("DownResBlock%d"%i, DownsamplingResBlock(inc, c, batchnorm=True))

            inc = c

        self.out = EqualLinear(inc*2*2, 1, activation=False)

        self.print_architecture()

    def forward(self, img):
        pred = self.camadas(img)

        pred = self.out(torch.flatten(pred, 1))

        return pred