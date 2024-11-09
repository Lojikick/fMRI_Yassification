import sys
sys.path.append('versatile_diffusion')
import os
import PIL
from PIL import Image
import numpy as np

import torch
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.experiments.sd_default import color_adjust, auto_merge_imlist
from torch.utils.data import DataLoader, Dataset

from lib.model_zoo.vd import VD
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
import torchvision.transforms as T

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]

cfgm_name = 'vd_noema'

pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.clip = net.clip.to(device)


class batch_generator_external_images(Dataset):
    def __init__(self, data_path, chunk_size=1000):
        self.data_path = data_path
        self.chunk_size = chunk_size
        
        # Get file size and shape without loading the whole array
        with open(data_path, 'rb') as f:
            version = np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        self.shape = shape
        self.dtype = dtype
        self.length = shape[0]

    def __getitem__(self, idx):
        # Memory-efficient loading of single image
        offset = idx * np.prod(self.shape[1:]) * self.dtype.itemsize
        with open(self.data_path, 'rb') as f:
            f.seek(offset)
            data = np.fromfile(f, dtype=self.dtype, count=np.prod(self.shape[1:]))
            img_array = data.reshape(self.shape[1:])
        
        img = Image.fromarray(img_array.astype(np.uint8))
        img = T.functional.resize(img, (64, 64))
        img = torch.tensor(np.array(img)).float()
        return img

    def __len__(self):
        return self.length

    

image_path = 'data/processed_data/subj{:02d}/nsd_train_stim_sub{}.npy'.format(sub,sub)
train_images = batch_generator_external_images(data_path = image_path)

image_path = 'data/processed_data/subj{:02d}/nsd_test_stim_sub{}.npy'.format(sub,sub)
test_images = batch_generator_external_images(data_path = image_path)

trainloader = DataLoader(train_images,batch_size=32,shuffle=False)
testloader = DataLoader(test_images,batch_size=32,shuffle=False)

num_embed, num_features, num_test, num_train = 257, 768, len(test_images), len(train_images)

# Process in batches
train_features = []
test_features = []

print("Processing training images...")
with torch.no_grad():
    for i, batch in enumerate(trainloader):
        print(f"Processing batch {i+1}/{len(trainloader)}")
        features = net.clip.encode_image(batch.to(device))
        train_features.append(features.cpu().numpy())

    train_features = np.concatenate(train_features, axis=0)
    np.save(f'data/extracted_features/subj{sub:02d}/nsd_clipvision_train.npy', train_features)

print("Processing test images...")
with torch.no_grad():
    for i, batch in enumerate(testloader):
        print(f"Processing batch {i+1}/{len(testloader)}")
        features = net.clip.encode_image(batch.to(device))
        test_features.append(features.cpu().numpy())

    test_features = np.concatenate(test_features, axis=0)
    np.save(f'data/extracted_features/subj{sub:02d}/nsd_clipvision_test.npy', test_features)


    
