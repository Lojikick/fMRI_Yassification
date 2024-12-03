import sys
sys.path.append('versatile_diffusion')
import os
import numpy as np

import torch
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from torch.utils.data import DataLoader, Dataset

from lib.model_zoo.vd import VD
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
import matplotlib.pyplot as plt
import torchvision.transforms as T

# import argparse
# parser = argparse.ArgumentParser(description='Argument Parser')
# parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
# args = parser.parse_args()
# sub=int(args.sub)
# assert sub in [1,2,5,7]


def extract_features(model_path, train_path, test_path, save_path):
    cfgm_name = 'vd_noema'
    pth = model_path #Path to pretrained vd-four-flow-v1-0-fp16-deprecated.pth
    cfgm = model_cfg_bank()(cfgm_name)
    net = get_model()(cfgm)
    sd = torch.load(pth, map_location='cpu')
    net.load_state_dict(sd, strict=False)    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.clip = net.clip.to(device)
    
    train_caps = np.load(train_path) #path to train captions,
    test_caps = np.load(test_path) #path to test captions,

    num_embed, num_features, num_test, num_train = 77, 768, len(test_caps), len(train_caps)

    train_clip = np.zeros((num_train, num_embed, num_features))
    test_clip = np.zeros((num_test, num_embed, num_features))

    print("Content of first few train captions:")
    print(train_caps[:5])
    print("\nType of train_caps:", type(train_caps))
    print("Shape of train_caps:", train_caps.shape)

    with torch.no_grad():
        # Process test captions
        for i, annots in enumerate(test_caps):
            # Filter out empty strings and convert to list of strings
            cin = [str(annots)] # WARNING: original code
            #cin = list(annots[annots!=''])
            
            if not cin:  # Skip if no valid captions
                print(f"Warning: No valid captions for test sample {i}")
                continue
            print(f"Processing test sample {i}")
            try:
                # print("NEED DAT uncomment foo")
                c = net.clip_encode_text(cin)
                test_clip[i] = c.to('cpu').numpy().mean(0)
            except Exception as e:
                print(f"Error processing test sample {i}: {str(e)}")
                print(f"Captions: {cin}")
                continue
        
        # np.save('data/extracted_features/subj{:02d}/nsd_cliptext_test.npy'.format(sub), test_clip)
        
        # Process train captions
        for i, annots in enumerate(train_caps):
            # Filter out empty strings and convert to list of strings
            cin = [str(annots)]
            if not cin:  # Skip if no valid captions
                print(f"Warning: No valid captions for train sample {i}")
                continue
            print(f"Processing train sample {i}")
            try:
                # print("NEED DAT uncomment foo")
                c = net.clip_encode_text(cin)
                train_clip[i] = c.to('cpu').numpy().mean(0)
            except Exception as e:
                print(f"Error processing train sample {i}: {str(e)}")
                print(f"Captions: {cin}")
                continue
        #name the clip text features as 'nsd_cliptext_train.npy'
        np.save(save_path, train_clip)

