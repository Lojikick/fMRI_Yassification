import sys
sys.path.append('versatile_diffusion')
import os
import PIL
from PIL import Image
import numpy as np
import torch
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import argparse
import math

def save_features_in_chunks(features, output_path, chunk_size=1000):
    """Save large arrays in chunks to avoid memory issues"""
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        
    total_samples = features.shape[0]
    num_chunks = math.ceil(total_samples / chunk_size)
    
    # Create an empty file with the final shape
    merged_array = np.lib.format.open_memmap(
        output_path, mode='w+',
        shape=features.shape,
        dtype=features.dtype
    )
    
    # Save in chunks
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_samples)
        merged_array[start_idx:end_idx] = features[start_idx:end_idx]
    
    del merged_array  # Flush to disk

class BatchGeneratorExternalImages(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        # Load data in chunks using memmap
        self.im = np.load(data_path, mmap_mode='r')
        
    def __getitem__(self, idx):
        img = Image.fromarray(self.im[idx].astype(np.uint8))
        img = T.functional.resize(img, (512, 512))
        img = T.functional.to_tensor(img).float()
        img = img * 2 - 1
        return img
    
    def __len__(self):
        return len(self.im)

def process_dataset(net, dataloader, num_samples, batch_size, output_path, device):
    num_embed, num_features = 257, 768
    # Process in smaller chunks to save memory
    chunk_size = min(1000, num_samples)
    current_chunk = np.zeros((chunk_size, num_embed, num_features), dtype=np.float32)
    chunk_idx = 0
    sample_in_chunk = 0
    
    with torch.no_grad():
        for i, cin in enumerate(dataloader):
            print(f"Processing batch {i}/{len(dataloader)}")
            cin = cin.to(device)
            c = net.clip_encode_vision(cin)
            current_batch = c[0].cpu().numpy()
            
            # Add to current chunk
            batch_size_actual = current_batch.shape[0]
            if sample_in_chunk + batch_size_actual > chunk_size:
                # Save current chunk and start new one
                save_features_in_chunks(current_chunk[:sample_in_chunk], 
                                     output_path + f'.part{chunk_idx}',
                                     chunk_size=100)
                chunk_idx += 1
                sample_in_chunk = 0
                
            current_chunk[sample_in_chunk:sample_in_chunk + batch_size_actual] = current_batch
            sample_in_chunk += batch_size_actual
            
        # Save final chunk
        if sample_in_chunk > 0:
            save_features_in_chunks(current_chunk[:sample_in_chunk],
                                  output_path + f'.part{chunk_idx}',
                                  chunk_size=100)

def main():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument("-sub", "--sub", help="Subject Number", default=1)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing")
    args = parser.parse_args()
    
    sub = int(args.sub)
    assert sub in [1, 2, 5, 7]
    
    # Model setup
    cfgm_name = 'vd_noema'
    pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
    cfgm = model_cfg_bank()(cfgm_name)
    net = get_model()(cfgm)
    sd = torch.load(pth, map_location='cpu')
    net.load_state_dict(sd, strict=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.clip = net.clip.to(device)
    
    # Data loading
    batch_size = args.batch_size
    
    # Process training data
    train_path = f'data/processed_data/subj{sub:02d}/nsd_train_stim_sub{sub}.npy'
    train_images = BatchGeneratorExternalImages(data_path=train_path)
    trainloader = DataLoader(train_images, batch_size, shuffle=False)
    
    train_output = f'data/extracted_features/subj{sub:02d}/nsd_clipvision_train'
    process_dataset(net, trainloader, len(train_images), batch_size, train_output, device)
    
    # Process test data
    test_path = f'data/processed_data/subj{sub:02d}/nsd_test_stim_sub{sub}.npy'
    test_images = BatchGeneratorExternalImages(data_path=test_path)
    testloader = DataLoader(test_images, batch_size, shuffle=False)
    
    test_output = f'data/extracted_features/subj{sub:02d}/nsd_clipvision_test'
    process_dataset(net, testloader, len(test_images), batch_size, test_output, device)

if __name__ == "__main__":
    main()

    
