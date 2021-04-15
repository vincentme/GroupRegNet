import numpy as np 
import torch, os
from scipy import io


mat_file_list = [file for file in os.listdir('.') if file.endswith('mat')]

for mat_file in mat_file_list:
    case = mat_file.split('.')[0]
    dirqa = io.loadmat(mat_file)
    landmark_00 = dirqa['landmark_EI'].astype(np.float32) - 1. # change to 0-based indexing
    landmark_50 = dirqa['landmark_EE'].astype(np.float32) - 1.
    landmark_00[:, [0, 1]] = landmark_00[:, [1, 0]]
    landmark_50[:, [0, 1]] = landmark_50[:, [1, 0]]
    disp_00_50 = landmark_50 - landmark_00 # (n, 3)
    landmark  = {'landmark_00':landmark_00, 'landmark_50':landmark_50, 'disp_00_50':disp_00_50}
    torch.save(landmark, f'{case}_00_50.pt')
