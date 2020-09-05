from . import regnet
import numpy as np
import torch

class StopCriterion(object):
    def __init__(self, stop_std = 0.001, query_len = 100, num_min_iter = 200):
        self.query_len = query_len
        self.stop_std = stop_std
        self.loss_list = []
        self.loss_min = 1.
        self.num_min_iter = num_min_iter
        
    def add(self, loss):
        self.loss_list.append(loss)
        if loss < self.loss_min:
            self.loss_min = loss
            self.loss_min_i = len(self.loss_list)
    
    def stop(self):
        # return True if the stop creteria are met
        query_list = self.loss_list[-self.query_len:]
        query_std = np.std(query_list)
        if query_std < self.stop_std and self.loss_list[-1] - self.loss_min < self.stop_std/3. and len(self.loss_list) > self.loss_min_i and len(self.loss_list) > self.num_min_iter:
            return True
        else:
            return False
    
class CalcDisp(object):
    def __init__(self, dim, calc_device = 'cuda'):
        self.device = torch.device(calc_device)
        self.dim = dim
        self.spatial_transformer = regnet.SpatialTransformer(dim = dim)
        
    def inverse_disp(self, disp, threshold = 0.01, max_iteration = 20):
        '''
        compute the inverse field. implementationof "A simple fixed‐point approach to invert a deformation field"

        disp : (n, 2, h, w) or (n, 3, d, h, w) or (2, h, w) or (3, d, h, w)
            displacement field
        '''
        forward_disp = disp.detach().to(device = self.device)
        if disp.ndim < self.dim + 2:
            forward_disp = torch.unsqueeze(forward_disp, 0)
        backward_disp = torch.zeros_like(forward_disp)
        backward_disp_old = backward_disp.clone()
        for i in range(max_iteration):
            backward_disp = -self.spatial_transformer(forward_disp, backward_disp)
            diff = torch.max(torch.abs(backward_disp - backward_disp_old)).item()
            if diff < threshold:
                break
            backward_disp_old = backward_disp.clone()
        if disp.ndim < self.dim + 2:
            backward_disp = torch.squeeze(backward_disp, 0)

        return backward_disp
        
    def compose_disp(self, disp_i2t, disp_t2i, mode = 'corr'):
        '''
        compute the composition field
        
        disp_i2t: (n, 3, d, h, w)
            displacement field from the input image to the template
            
        disp_t2i: (n, 3, d, h, w)
            displacement field from the template to the input image
            
        mode: string, default 'corr'
            'corr' means generate composition of corresponding displacement field in the batch dimension only, the result shape is the same as input (n, 3, d, h, w)
            'all' means generate all pairs of composition displacement field. The result shape is (n, n, 3, d, h, w)
        '''
        disp_i2t_t = disp_i2t.detach().to(device = self.device)
        disp_t2i_t = disp_t2i.detach().to(device = self.device)
        if disp_i2t.ndim < self.dim + 2:
            disp_i2t_t = torch.unsqueeze(disp_i2t_t, 0)
        if disp_t2i.ndim < self.dim + 2:
            disp_t2i_t = torch.unsqueeze(disp_t2i_t, 0)
        
        if mode == 'corr':
            composed_disp = self.spatial_transformer(disp_t2i_t, disp_i2t_t) + disp_i2t_t # (n, 2, h, w) or (n, 3, d, h, w)
        elif mode == 'all':
            assert len(disp_i2t_t) == len(disp_t2i_t)
            n, _, *image_shape = disp_i2t.shape
            disp_i2t_nxn = torch.repeat_interleave(torch.unsqueeze(disp_i2t_t, 1), n, 1) # (n, n, 2, h, w) or (n, n, 3, d, h, w)
            disp_i2t_nn = disp_i2t_nxn.reshape(n*n, self.dim, *image_shape) # (n*n, 2, h, w) or (n*n, 3, d, h, w), the order in the first dimension is [0_T, 0_T, ..., 0_T, 1_T, 1_T, ..., 1_T, ..., n_T, n_T, ..., n_T]
            disp_t2i_nn = torch.repeat_interleave(torch.unsqueeze(disp_t2i_t, 0), n, 0).reshape(n*n, self.dim, *image_shape) # (n*n, 2, h, w) or (n*n, 3, d, h, w), the order in the first dimension is [0_T, 1_T, ..., n_T, 0_T, 1_T, ..., n_T, ..., 0_T, 1_T, ..., n_T]
            composed_disp = self.spatial_transformer(disp_t2i_nn, disp_i2t_nn).reshape(n, n, self.dim, *image_shape) + disp_i2t_nxn # (n, n, 2, h, w) or (n, n, 3, d, h, w) + disp_i2t_nxn
        else:
            raise
        if disp_i2t.ndim < self.dim + 2 and disp_t2i.ndim < self.dim + 2:
            composed_disp = torch.squeeze(composed_disp)
        return composed_disp
        
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)