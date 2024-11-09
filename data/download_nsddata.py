import os
#os.system('ls -l')
# Download Experiment Infos
#nsddata/experiments/nsd/
os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_expdesign.mat nsddata/experiments/nsd/')
#nsddata/experiments/nsd/
os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.pkl nsddata/experiments/nsd/')
#nsddata_stimuli/stimuli/nsd/
# Download Stimuli
os.system('aws s3 cp s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 nsddata_stimuli/stimuli/nsd/')
# nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/
# Download Betas
for sub in [1,2,5,7]:
    for sess in range(1,38):
        os.system('aws s3 cp s3://natural-scenes-dataset/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session{:02d}.nii.gz nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'.format(sub,sess,sub))
# nsddata/ppdata/subj{:02d}/func1pt8mm/roi/
for sub in [1,2,5,7]:
    os.system('aws s3 sync s3://natural-scenes-dataset/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/ nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub,sub))
