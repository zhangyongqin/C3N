import math

import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct
from torchvision import models
from audioop import bias


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun
    
class Gaussian(nn.Module):
    def forward(self, input):
        return torch.exp(-torch.mul(input, input))


#sUnit: Super activation Unit
class sUnit(nn.Module):
    def __init__(self, out_channels=64, kernel_size=3, skernel_size=5, dilation=True):
        super(sUnit, self).__init__()
        if dilation:
            d_rate = 2
        else:
            d_rate = 1
        self.lrelu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=((kernel_size-1)//2))
        self.dconv1 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=d_rate, dilation=d_rate)
        self.conv2 = nn.Conv2d(out_channels*2, out_channels, kernel_size=skernel_size, stride=1, padding=((skernel_size-1)//2), groups=out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.Gauss = Gaussian()

    def forward(self,x):
        x1 = self.lrelu1(x)
        x2t = self.conv1(x1)
        x2d = self.dconv1(x1)
        x2 = torch.cat((x2t,x2d),1)
        x3 = self.conv2(x2)
        x3n = self.bn2(x3)
        x4 = self.Gauss(x3n)
        x = torch.mul(x,x4)
        return x


class sLUnit(nn.Module):
    def __init__(self, out_channels=64, kernel_size=3, skernel_size=5, dilation=True):
        super(sLUnit, self).__init__()
        if dilation:
            d_rate = 2
        else:
            d_rate = 1
        self.lrelu1 = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=((kernel_size-1)//2))
        self.dconv1 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=d_rate, dilation=d_rate)
        self.conv2 = nn.Conv2d(out_channels*2, out_channels, kernel_size=skernel_size, stride=1, padding=((skernel_size-1)//2), groups=out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.Gauss = Gaussian()

    def forward(self,x):
        x1 = self.lrelu1(x)
        x2t = self.conv1(x1)
        x2d = self.dconv1(x1)
        x2 = torch.cat((x2t,x2d),1)
        x3 = self.conv2(x2)
        x3n = self.bn2(x3)
        x4 = self.Gauss(x3n)
        x = torch.mul(x,x4)
        return x



class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

# User modules
class downsample(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride1=1, stride2=2, padding=0, dilation=1, groups=1, bias=True ):
        super(downsample, self).__init__()
        self.sconv1 = nn.Conv2d(in_chs, out_chs, kernel_size, stride1, padding,dilation, groups,bias)
        self.fconv1 = nn.Conv2d(in_chs, out_chs, kernel_size, stride1, padding,dilation, groups,bias)
        self.sconv2 = nn.Conv2d(out_chs * 2, out_chs, kernel_size, stride2, padding,dilation, groups,bias)
        self.sconv1.apply(weights_init('xavier'))
        self.fconv1.apply(weights_init('xavier'))
        self.sconv2.apply(weights_init('kaiming'))  
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_chs))
        else:
            self.register_parameter('bias', None)
        
    def forward(self,x):
        x1 = self.sconv1(x)
        y = dct.dct_2d(x)
        y1 = self.fconv1(y)
        y1s = dct.idct_2d(y1)
        x2 = torch.cat((x1,y1s),1)
        out = self.sconv2(x2)
        return out

class PartialConv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride1=1, stride2=2, padding=0, dilation=1, groups=1, bias=True):
        super(PartialConv1, self).__init__()
        self.input_conv = downsample(in_channels, out_channels, kernel_size,stride1,stride2, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride2, padding, dilation, groups, bias=False)                   
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)   
        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask

class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(PartialConv, self).__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask


class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu',
                 conv_bias=False):
        super(PCBActiv, self).__init__()
        if sample == 'down-5':
            self.conv = PartialConv1(in_ch, out_ch, 5, 1, 2, 2, bias=conv_bias)
        elif sample == 'down-7':
            self.conv = PartialConv1(in_ch, out_ch, 7, 1, 2, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv1(in_ch, out_ch, 3, 1, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = sUnit(out_channels=out_ch, kernel_size=3, skernel_size=5, dilation=True)
        elif activ == 'leaky':
            self.activation = sLUnit(out_channels=out_ch, kernel_size=3, skernel_size=5, dilation=True)
            #self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask


class PConvUNet(nn.Module):
    def __init__(self, layer_size=7, input_channels=3, upsampling_mode='nearest'):
        super(PConvUNet,self).__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample='down-7')
        self.enc_2 = PCBActiv(64, 128, sample='down-5')
        self.enc_3 = PCBActiv(128, 256, sample='down-5')
        self.enc_4 = PCBActiv(256, 512, sample='down-3')
        for i in range(4, self.layer_size):
            name = 'enc_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv(512, 512, sample='down-3'))

        for i in range(4, self.layer_size):
            name = 'dec_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv(512 + 512, 512, activ='leaky'))
        self.dec_4 = PCBActiv(512 + 256, 256, activ='leaky')
        self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_2 = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_1 = PCBActiv(64 + input_channels, input_channels,
                              bn=False, activ=None, conv_bias=True)

    def forward(self, input, input_mask):
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N

        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask

        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
                h_dict[h_key_prev], h_mask_dict[h_key_prev])
            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]

        # concat upsampled output of h_enc_N-1 and dec_N+1, then do dec_N
        # (exception)
        #                            input         dec_2            dec_1
        #                            h_enc_7       h_enc_8          dec_8

        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)

            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)
            h_mask = F.interpolate(
                h_mask, scale_factor=2, mode='nearest')

            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)
            h, h_mask = getattr(self, dec_l_key)(h, h_mask)

        return h, h_mask

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """

        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()


if __name__ == '__main__':
    size = (1, 3, 5, 5)
    input = torch.ones(size)
    input_mask = torch.ones(size)
    input_mask[:, :, 2:, :][:, :, :, 2:] = 0

    conv = PartialConv(3, 3, 3, 1, 1)
    l1 = nn.L1Loss()
    input.requires_grad = True

    output, output_mask = conv(input, input_mask)
    loss = l1(output, torch.randn(1, 3, 5, 5))
    loss.backward()

    assert (torch.sum(input.grad != input.grad).item() == 0)
    assert (torch.sum(torch.isnan(conv.input_conv.weight.grad)).item() == 0)
    assert (torch.sum(torch.isnan(conv.input_conv.bias.grad)).item() == 0)

    # model = PConvUNet()
    # output, output_mask = model(input, input_mask)
