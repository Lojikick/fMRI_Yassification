import os
import sys
import numpy as np
import h5py
import scipy.io as spio
import nibabel as nib
import gc  # Added for garbage collection
import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]


def loadmat(filename):
   '''
   this function should be called instead of direct spio.loadmat
   as it cures the problem of not properly recovering python dictionaries
   from mat files. It calls the function check keys to cure all entries
   which are still mat-objects
   '''
   def _check_keys(d):
       '''
       checks if entries in dictionary are mat-objects. If yes
       todict is called to change them to nested dictionaries
       '''
       for key in d:
           if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
               d[key] = _todict(d[key])
       return d


   def _todict(matobj):
       '''
       A recursive function which constructs from matobjects nested dictionaries
       '''
       d = {}
       for strg in matobj._fieldnames:
           elem = matobj.__dict__[strg]
           if isinstance(elem, spio.matlab.mio5_params.mat_struct):
               d[strg] = _todict(elem)
           elif isinstance(elem, np.ndarray):
               d[strg] = _tolist(elem)
           else:
               d[strg] = elem
       return d


   def _tolist(ndarray):
       '''
       A recursive function which constructs lists from cellarrays
       (which are loaded as numpy ndarrays), recursing into the elements
       if they contain matobjects.
       '''
       elem_list = []
       for sub_elem in ndarray:
           if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
               elem_list.append(_todict(sub_elem))
           elif isinstance(sub_elem, np.ndarray):
               elem_list.append(_tolist(sub_elem))
           else:
               elem_list.append(sub_elem)
       return elem_list
   data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
   return _check_keys(data)


def process_batch(indices, sig_dict, fmri_data, stim_data, batch_size=100):
   """Process data in smaller batches to manage memory"""
   num_samples = len(indices)
   vox_dim, im_dim, im_c = fmri_data.shape[1], 425, 3
  
   # Pre-allocate arrays for the full batch
   fmri_array = np.zeros((batch_size, vox_dim), dtype=np.float32)
   stim_array = np.zeros((batch_size, im_dim, im_dim, im_c), dtype=np.float32)
  
   processed_samples = 0
   results_fmri = []
   results_stim = []
  
   for i in range(0, num_samples, batch_size):
       batch_indices = indices[i:min(i+batch_size, num_samples)]
       batch_size_current = len(batch_indices)
      
       # Reset arrays for current batch if smaller than batch_size
       if batch_size_current < batch_size:
           fmri_array = np.zeros((batch_size_current, vox_dim), dtype=np.float32)
           stim_array = np.zeros((batch_size_current, im_dim, im_dim, im_c), dtype=np.float32)
      
       for j, idx in enumerate(batch_indices):
           stim_array[j] = stim_data[idx]
           fmri_array[j] = fmri_data[sorted(sig_dict[idx])].mean(0)
           processed_samples += 1
           if processed_samples % 100 == 0:
               print(f"Processed {processed_samples}/{num_samples} samples")
      
       results_fmri.append(fmri_array.copy())
       results_stim.append(stim_array.copy())
      
       # Force garbage collection
       gc.collect()
  
   # Concatenate all batches
   final_fmri = np.concatenate(results_fmri, axis=0)
   final_stim = np.concatenate(results_stim, axis=0)
  
   return final_fmri, final_stim


def save_in_batches(data, filename, batch_size=1000):
   """Save large arrays to disk in batches"""
   with open(filename, 'wb') as f:
       np.save(f, data.shape)
       for i in range(0, len(data), batch_size):
           batch = data[i:i+batch_size]
           np.save(f, batch)


def process_captions(indices, annots, output_file):
   """Process and save captions in batches"""
   batch_size = 1000
   num_samples = len(indices)
  
   with open(output_file, 'wb') as f:
       # Save shape information
       shape = (num_samples, 5)
       np.save(f, shape)
      
       for i in range(0, num_samples, batch_size):
           batch_indices = indices[i:min(i+batch_size, num_samples)]
           captions_batch = np.empty((len(batch_indices), 5), dtype=annots.dtype)
          
           for j, idx in enumerate(batch_indices):
               captions_batch[j,:] = annots[idx,:]
          
           np.save(f, captions_batch)
           print(f"Processed captions: {i+len(batch_indices)}/{num_samples}")




# Load experiment design
print("Loading experiment design...")
stim_order_f = 'nsddata/experiments/nsd/nsd_expdesign.mat'
stim_order = loadmat(stim_order_f)


# Process trial information
print("Processing trial information...")
num_trials = 37*750
sig_train = {}
sig_test = {}
for idx in range(num_trials):
   nsdId = stim_order['subjectim'][sub-1, stim_order['masterordering'][idx] - 1] - 1
   if stim_order['masterordering'][idx]>1000:
       if nsdId not in sig_train:
           sig_train[nsdId] = []
       sig_train[nsdId].append(idx)
   else:
       if nsdId not in sig_test:
           sig_test[nsdId] = []
       sig_test[nsdId].append(idx)


train_im_idx = list(sig_train.keys())
test_im_idx = list(sig_test.keys())


# Load mask and prepare fMRI data
print("Loading mask and fMRI data...")
roi_dir = f'nsddata/ppdata/subj{sub:02d}/func1pt8mm/roi/'
betas_dir = f'nsddata_betas/ppdata/subj{sub:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'


mask = nib.load(roi_dir+'nsdgeneral.nii.gz').get_fdata()
num_voxel = mask[mask>0].shape[0]


# Load fMRI data in chunks
fmri = np.zeros((num_trials, num_voxel), dtype=np.float32)
chunk_size = 5  # Process 5 sessions at a time
for chunk_start in range(0, 37, chunk_size):
   chunk_end = min(chunk_start + chunk_size, 37)
   print(f"Loading fMRI sessions {chunk_start+1} to {chunk_end}...")
  
   for i in range(chunk_start, chunk_end):
       beta_filename = f"betas_session{i+1:02d}.nii.gz"
       beta_f = nib.load(betas_dir+beta_filename).get_fdata().astype(np.float32)
       fmri[i*750:(i+1)*750] = beta_f[mask>0].transpose()
       del beta_f
       gc.collect()


print("Loading stimuli...")
f_stim = h5py.File('nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5', 'r')
stim = f_stim['imgBrick']


# Process training data
print("Processing training data...")
fmri_train, stim_train = process_batch(train_im_idx, sig_train, fmri, stim, batch_size=100)


print("Saving training data...")
np.save(f'processed_data/subj{sub:02d}/nsd_train_fmriavg_nsdgeneral_sub{sub}.npy', fmri_train)
np.save(f'processed_data/subj{sub:02d}/nsd_train_stim_sub{sub}.npy', stim_train)
del fmri_train, stim_train
gc.collect()


# Process test data
print("Processing test data...")
fmri_test, stim_test = process_batch(test_im_idx, sig_test, fmri, stim, batch_size=100)


print("Saving test data...")
np.save(f'processed_data/subj{sub:02d}/nsd_test_fmriavg_nsdgeneral_sub{sub}.npy', fmri_test)
np.save(f'processed_data/subj{sub:02d}/nsd_test_stim_sub{sub}.npy', stim_test)
del fmri_test, stim_test
gc.collect()


# Process captions
print("Loading and processing captions...")
annots_cur = np.load('annots/COCO_73k_annots_curated.npy')


print("Processing training captions...")
process_captions(train_im_idx, annots_cur,
               f'processed_data/subj{sub:02d}/nsd_train_cap_sub{sub}.npy')


print("Processing test captions...")
process_captions(test_im_idx, annots_cur,
               f'processed_data/subj{sub:02d}/nsd_test_cap_sub{sub}.npy')


print("All data processing completed successfully!")
