import numpy as np 
import torch

landmark_00 = np.genfromtxt('Case1_300_T00_xyz.txt', dtype = np.int64) - 1 # change to 0-based indexing
landmark_50 = np.genfromtxt('Case1_300_T50_xyz.txt', dtype = np.int64) - 1 # (n, 3), (w, h, d) order in the last dimension
disp_00_50 = (landmark_50 - landmark_00).astype(np.float32) # (n, 3)

landmark  = {'landmark_00':landmark_00, 'landmark_50':landmark_50, 'disp_00_50':disp_00_50}
torch.save(landmark, 'Case1_300_00_50.pt')
