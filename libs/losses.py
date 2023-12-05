import math

import torch
import torch.nn.functional as F

import numpy as np

class MSELoss(torch.nn.Module):    
    
    '''
    mean squared error loss function
    '''
    
    def __init__(self, 
                 is_tensor=False):
        
        super().__init__()
        
        self.is_tensor = is_tensor
        
    def forward(self, y_true, y_pred):

        if self.is_tensor==True:

            return (y_true - y_pred)**2

        else:

            return torch.mean((y_true - y_pred)**2)
            
class LNCCLoss(torch.nn.Module):
    
    '''
    local normalized cross-correlation loss function
    '''

    def __init__(self, 
                 win=(9, 9, 9),
                 n_channels=1,
                 is_tensor=False):
        
        super().__init__()
        
        self.is_tensor = is_tensor
        self.win = win
        self.n_channels = n_channels
        self.win_size = np.prod(win)*self.n_channels
        self.ndims = len(win)
        
        self.conv = getattr(torch.nn, 'Conv%dd' % self.ndims)(n_channels, 1, self.win, stride=1, padding='same', padding_mode='replicate', bias=False)
        
        with torch.no_grad():
            
            torch.nn.init.ones_(self.conv.weight)
                    
        for param in self.conv.parameters():
            
            param.requires_grad = False
            
    def forward(self, y_true, y_pred):
        
        true_sum = self.conv(y_true) / self.win_size
        pred_sum = self.conv(y_pred) / self.win_size    
                
        true_cent = y_true - true_sum
        pred_cent = y_pred - pred_sum
        
        nominator = self.conv(true_cent * pred_cent)
        nominator = nominator * nominator
                
        denominator = self.conv(true_cent * true_cent) * self.conv(pred_cent * pred_cent)
        
        cc = (nominator + 1e-6) / (denominator + 1e-6)  
        cc = torch.clamp(cc, 0, 1)  
        
        if self.is_tensor==True:
        
            return -cc
            
        else:

            return -torch.mean(cc)    
        
class NCCLoss(torch.nn.Module):
    
    '''
    normalized cross-correlation loss function
    '''    
    
    def __init__(self, is_tensor=False):
        
        super().__init__()

        self.is_tensor = is_tensor
        
    def forward(self, y_true, y_pred):
        
        nominator = ((y_true - y_true.mean()) * (y_pred - y_pred.mean())).mean()
        
        denominator = y_true.std() * y_pred.std()
        
        cc = (nominator + 1e-6) / (denominator + 1e-6)  
        cc = torch.clamp(cc, 0, 1)  
        
        if self.is_tensor==True:
        
            return -cc
            
        else:

            return -torch.mean(cc)    
        

class JacobianLossCoords(torch.nn.Module):
    
    '''
    
    '''    
    
    def __init__(self, 
                 add_identity=True,
                 is_tensor=False):
        
        super().__init__()
            
        self.add_identity = add_identity
        self.is_tensor = is_tensor
            
    def forward(self, input_coords, output):
        
        jac = self.compute_jacobian_matrix(input_coords, output, add_identity=self.add_identity)
        
        loss = 1 - torch.det(jac)
        
        if self.is_tensor==True:
        
            return torch.abs(loss)
            
        else:

            return torch.mean(torch.abs(loss))
        

    def compute_jacobian_matrix(self, input_coords, output, add_identity=True):

        dim = input_coords.shape[1]

        jacobian_matrix = torch.zeros(input_coords.shape[0], dim, dim)

        for i in range(dim):

            jacobian_matrix[:, i, :] = self.gradient(input_coords, output[:, i])

            if add_identity:

                jacobian_matrix[:, i, i] += torch.ones_like(jacobian_matrix[:, i, i])

        return jacobian_matrix        
    
        
    def gradient(self, input_coords, output, grad_outputs=None):

        grad_outputs = torch.ones_like(output)

        grad = torch.autograd.grad(output, [input_coords], grad_outputs=grad_outputs, create_graph=True)[0]

        return grad         
            
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    