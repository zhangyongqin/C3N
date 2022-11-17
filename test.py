import argparse
import os
import torch


from torchvision import transforms

import opt
from places2 import Places2
from evaluation import evaluate
from net import PConvUNet
from util.io import load_ckpt

# Choose GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"


parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='./Places2')
parser.add_argument('--mask_root', type=str, default='./masks')
parser.add_argument('--snapshot', type=str, default='./snapshots1/default/ckpt/950000.pth')
parser.add_argument('--image_size', type=int, default=256)
args = parser.parse_args()

device = torch.device('cuda')

size = (args.image_size, args.image_size)
img_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])
    
dataset_val = Places2(args.root, args.mask_root, img_transform, mask_transform, 'val')

model = PConvUNet().to(device)
load_ckpt(args.snapshot, [('model', model)])

model.eval()
evaluate(model, dataset_val, device, 'result0.jpg')