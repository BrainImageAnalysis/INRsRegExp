import torch

import torch.nn.functional as F
import numpy as np
import nibabel as nib

from torch.utils.data import Dataset
from collections import defaultdict

def load_mindboggle(path, normalization=True):
    
    '''
    Description:
    
    Load mindboggle data -> crop -> scale -> return image and rois
    
    Params: 
    
    path: path to the folder with the mindboggle mri images, for example Mindboggle101_volumes/Extra-18_volumes/Colin27-1/ 
    
    normalization: true indicates scaling of the mri image from [0,1] to [-1,1]. 
    
    '''        
    
    image = nib.load(path+'/t1weighted_brain.MNI152.nii.gz')
    image = image.get_fdata()
    image = image[3:-3, 13:-13, 3:-3]
    
    image = image / np.max(image)
    
    if normalization==True:
        
        image = image*2 - 1
        
    segmentations = nib.load(path+'/labels.DKT31.manual.MNI152.nii.gz')
    segmentations = segmentations.get_fdata()
    segmentations = segmentations[3:-3, 13:-13, 3:-3]    
    unique_ind = np.unique(segmentations)[1:]
    
    rois = np.zeros((len(unique_ind), *segmentations.shape))    
    
    for i, ind in enumerate(unique_ind):
        
        rois[i, :, :, :] = 1.0*(segmentations==ind)       
        
    return image, rois

def make_coords_tensor(dims=(256, 256, 64), is_vector=True):

    '''
    modification, from https://proceedings.mlr.press/v172/wolterink22a.html
    '''        
    
    n_dims = len(dims)
    
    coords = [torch.linspace(-1, 1, dims[i]) for i in range(n_dims)]
    coords = torch.meshgrid(*coords, indexing=None)
    coords = torch.stack(coords, dim=n_dims)
    
    if is_vector==True:
    
        coords = coords.view([np.prod(dims), n_dims])
        
    return coords

def fast_trilinear_color_interpolation(input_array, x_indices, y_indices, z_indices):
    
    '''
    
    from https://proceedings.mlr.press/v172/wolterink22a.html
    
    '''        
    
    n_dim = input_array.shape[0]
    x_indices = (x_indices + 1) * (input_array.shape[1] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[2] - 1) * 0.5
    z_indices = (z_indices + 1) * (input_array.shape[3] - 1) * 0.5

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[1] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[2] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[3] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[1] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[2] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[3] - 1)

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    n = len(x)
    output = torch.zeros(n_dim, np.prod(input_array.shape[1::])).to(input_array.device)
    
    for i in range(n_dim):
        
        temp =           (input_array[i, x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
                        + input_array[i, x1, y0, z0] * x * (1 - y) * (1 - z)
                        + input_array[i, x0, y1, z0] * (1 - x) * y * (1 - z)
                        + input_array[i, x0, y0, z1] * (1 - x) * (1 - y) * z
                        + input_array[i, x1, y0, z1] * x * (1 - y) * z
                        + input_array[i, x0, y1, z1] * (1 - x) * y * z
                        + input_array[i, x1, y1, z0] * x * y * (1 - z)
                        + input_array[i, x1, y1, z1] * x * y * z)
        
        output[i,:] = temp
    
    return output    

def fast_trilinear_interpolation(input_array, x_indices, y_indices, z_indices):
    
    '''
    
    from https://proceedings.mlr.press/v172/wolterink22a.html
    
    '''        
    
    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5
    z_indices = (z_indices + 1) * (input_array.shape[2] - 1) * 0.5

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[2] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[2] - 1)

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    output = (
        input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
        + input_array[x1, y0, z0] * x * (1 - y) * (1 - z)
        + input_array[x0, y1, z0] * (1 - x) * y * (1 - z)
        + input_array[x0, y0, z1] * (1 - x) * (1 - y) * z
        + input_array[x1, y0, z1] * x * (1 - y) * z
        + input_array[x0, y1, z1] * (1 - x) * y * z
        + input_array[x1, y1, z0] * x * y * (1 - z)
        + input_array[x1, y1, z1] * x * y * z
    )
    
    return output


def fast_bilinear_interpolation(input_array, x_indices, y_indices):
    
    '''
    
    '''        
    
    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)

    x1 = x0 + 1
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)

    x = x_indices - x0
    y = y_indices - y0

    output = (
              input_array[x0, y0] * (1 - x) * (1 - y) 
            + input_array[x1, y0] * x * (1 - y) 
            + input_array[x0, y1] * (1 - x) * y
            + input_array[x1, y1] * x * y
             ) 
    
    return output


class MetricMonitor:
    
    '''
    
    
    '''        
    
    def __init__(self, float_precision=4):
        
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        
        self.metrics = defaultdict(lambda: {'value': 0, 'count': 0, 'average': 0})

    def update(self, metric_name, value):
        
        metric = self.metrics[metric_name]

        metric['value'] += value
        metric['count'] += 1
        metric['average'] = metric['value'] / metric['count']

    def __str__(self):
        
        return ' | '.join(
            ['{metric_name}: {temp:.{float_precision}f}'.format(metric_name=metric_name, temp=metric['average'], float_precision=self.float_precision) for (metric_name, metric) in self.metrics.items()]
        )    

class VectorCoords(Dataset):
    
    def __init__(self, 
                 coords=None, # [H, W, D, 3] or [H, W, 2] torch tensor
                 scale_factor=1):
        
        self.scale_factor = scale_factor
        self.dims = coords.shape[-1]
        
        if self.scale_factor!=1:
            
            if self.dims==3:
            
                coords = F.interpolate(coords.permute(3, 2, 0, 1).unsqueeze(0), scale_factor=self.scale_factor, mode='trilinear', align_corners=True)
                coords = coords.squeeze().permute(2, 3, 1, 0)
                   
            elif self.dims==2:
                
                coords = F.interpolate(coords.permute(2, 0, 1).unsqueeze(0), scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
                coords = coords.squeeze().permute(1, 2, 0)  

        self.spatial_size = coords.shape[:-1]
                
        self.coords = coords.contiguous().view([np.prod(self.spatial_size), self.dims])
        
    def __len__(self):
        
        return self.coords.shape[0]

    def __getitem__(self, ind):
        
        coords = self.coords[ind, :]
        
        return coords

    
class BlockCoords(Dataset):
    
    def __init__(self, 
                 coords=None, # [H, W, D, 3] torch tensor
                 scale_factor=1, # [H, W, D]
                 block_size=[32, 32, 32],
                 steps=500):
        
        self.scale_factor = scale_factor
        self.dims = coords.shape[-1] 
        self.dx = torch.div(2, torch.tensor(coords.shape[:3]) )
        self.steps = steps
        
        self.block_size = np.ceil(np.array(block_size)/2).astype(np.int16)    
        
        block_dx_dims = torch.tensor(self.block_size) * self.dx
        
        block_coords = [torch.linspace(-block_dx_dims[i], block_dx_dims[i], 2*self.block_size[i]) for i in range(self.dims)]
        block_coords = torch.meshgrid(*block_coords, indexing=None)
        self.block_coords = torch.stack(block_coords, dim=self.dims)        
        
        coords = coords[self.block_size[0]:-self.block_size[0], self.block_size[1]:-self.block_size[1], self.block_size[2]:-self.block_size[2], ...]
        
        if self.scale_factor!=1:
                        
            coords = F.interpolate(coords.permute(3, 2, 0, 1).unsqueeze(0), scale_factor=self.scale_factor, mode='trilinear', align_corners=True)
            coords = coords.squeeze().permute(2, 3, 1, 0)
            
        self.spatial_size = coords.shape[:-1]                                            
        self.coords = coords
        
    def __len__(self):
        
        return self.steps

    def __getitem__(self, dummy):
        
        indx = np.random.randint(0, np.prod(self.spatial_size))
        
        inds = np.unravel_index(indx, self.spatial_size)
        
        center = self.coords[inds[0], inds[1], inds[2], :]
        coords = torch.clone(self.block_coords)
        
        coords[..., 0] = coords[..., 0] + center[0] 
        coords[..., 1] = coords[..., 1] + center[1]
        coords[..., 2] = coords[..., 2] + center[2]
        
        return coords



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    