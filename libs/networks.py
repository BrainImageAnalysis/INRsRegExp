import torch

import numpy as np
import torch.nn.functional as F

class ReluLayer(torch.nn.Module):
    
    '''
    
    Based on https://arxiv.org/pdf/2006.08195.pdf
    
    '''
    
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 bias=True):
        
        super().__init__()
        
        self.input_dim = input_dim       
        self.output_dim = output_dim
        
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=bias)
              
    def forward(self, x):
        
        x = self.linear(x)

        return F.relu(x)
    

class SineLayer(torch.nn.Module):
    
    '''
    
    Based on https://arxiv.org/abs/2006.09661, https://github.com/vsitzmann/siren
    
    '''
    
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 bias=True, 
                 is_first_layer=False, 
                 omega_0=30.0,
                 trainable_omega_0=False):
        
        super().__init__()
        
        self.input_dim = input_dim       
        self.output_dim = output_dim
        self.omega_0 = omega_0
        self.omega_0_tensor = torch.nn.Parameter(omega_0*torch.ones(1), requires_grad=trainable_omega_0)
        self.is_first_layer = is_first_layer
        
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=bias)
        
        self.init_weights()         
        
    def forward(self, x):
        
        return torch.sin(self.omega_0_tensor * self.linear(x))
    
    def init_weights(self):
        
        with torch.no_grad():
            
            if self.is_first_layer:
                
                self.linear.weight.uniform_(-1 / self.input_dim, 
                                             1 / self.input_dim)      
            else:
                
                self.linear.weight.uniform_(-np.sqrt(6 / self.input_dim) / self.omega_0, 
                                             np.sqrt(6 / self.input_dim) / self.omega_0)         
    
class AffineLayer(torch.nn.Module):
    
    '''
    
    
    
    '''
    
    def __init__(self, 
                 input_dim=3):
        
        super().__init__()
        
        self.input_dim = input_dim
                         
        self.affine = torch.nn.Parameter(torch.eye(input_dim))
        self.translation = torch.nn.Parameter(torch.zeros(input_dim))
                        
    def forward(self, x):
                 
        '''
        
        x ~ batch x dim
        
        '''
        
        return torch.matmul(x, self.affine) + self.translation
    
    
class AttentionSineLayer(torch.nn.Module):
    
    '''
    
    Based on https://arxiv.org/abs/2006.09661, https://github.com/vsitzmann/siren
    
    '''
    
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 bias=True, 
                 is_first_layer=False, 
                 omega_0=30.0,
                 delta_omega_0=20.0):
        
        super().__init__()
        
        self.input_dim = input_dim       
        self.output_dim = output_dim
        self.omega_0 = omega_0
        self.delta_omega = delta_omega_0
        self.is_first_layer = is_first_layer
        self.tanh = torch.nn.Tanh()
        
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=bias)
        self.attention = torch.nn.Linear(1, 1, bias=True)

        self.init_weights()         
        
    def forward(self, x):
        
        x = self.linear(x)
        
        attention = self.attention(torch.mean(torch.abs(x), 1, keepdim=True))
        delta_omega = self.delta_omega*self.tanh(attention)

        return torch.sin((self.omega_0+delta_omega)*x)
    
    def init_weights(self):
        
        with torch.no_grad():
            
            if self.is_first_layer:
                
                self.linear.weight.uniform_(-1 / self.input_dim, 
                                             1 / self.input_dim)      
            else:
                
                self.linear.weight.uniform_(-np.sqrt(6 / self.input_dim) / self.omega_0, 
                                             np.sqrt(6 / self.input_dim) / self.omega_0)     
                
class SnakeLayer(torch.nn.Module):
    
    '''
    
   
    
    '''
    
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 bias=True, 
                 is_first_layer=False, 
                 omega_0=30.0):
        
        super().__init__()
        
        self.input_dim = input_dim       
        self.output_dim = output_dim
        self.omega_0 = omega_0
        self.is_first_layer = is_first_layer
        
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=bias)
        
        self.init_weights()         
        
    def forward(self, x):
        
        x = self.linear(x)

        return x - torch.cos(self.omega_0*x)/self.omega_0 + 1/(self.omega_0)
    
    def init_weights(self):
        
        with torch.no_grad():
            
            if self.is_first_layer:
                
                self.linear.weight.uniform_(-1 / self.input_dim, 
                                             1 / self.input_dim)      
            else:
                
                self.linear.weight.uniform_(-np.sqrt(3 / self.input_dim) / self.omega_0, 
                                             np.sqrt(3 / self.input_dim) / self.omega_0)     
                
                
class SineplusLayer(torch.nn.Module):
    
    '''
    
    Based on https://arxiv.org/abs/2006.09661, https://github.com/vsitzmann/siren
    
    '''
    
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 bias=True, 
                 is_first_layer=False, 
                 omega_0=30.0):
        
        super().__init__()
        
        self.input_dim = input_dim       
        self.output_dim = output_dim
        self.omega_0 = omega_0
        self.is_first_layer = is_first_layer
        
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=bias)
        
        self.init_weights()         
        
    def forward(self, x):
        
        x = self.linear(x)
        return x + torch.sin(self.omega_0 * x)
    
    def init_weights(self):
        
        with torch.no_grad():
            
            if self.is_first_layer:
                
                self.linear.weight.uniform_(-1 / self.input_dim, 
                                             1 / self.input_dim)      
            else:
                
                self.linear.weight.uniform_(-np.sqrt(6 / self.input_dim) / self.omega_0, 
                                             np.sqrt(6 / self.input_dim) / self.omega_0)                     
                  
class ChirpLayer(torch.nn.Module):
    
    '''
    

    
    '''
    
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 bias=True, 
                 is_first_layer=False, 
                 omega_0=30.0,
                 delta_omega_0=10.0):
        
        super().__init__()
        
        self.input_dim = input_dim       
        self.output_dim = output_dim
        self.omega_0 = omega_0
        self.delta_omega_0 = delta_omega_0
        self.is_first_layer = is_first_layer
        
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=bias)
        
        self.init_weights()         
        
    def forward(self, x):
        
        x = self.linear(x)

        return torch.sin((self.omega_0 + self.delta_omega_0*torch.tanh(self.omega_0*x))*x)
    
    def init_weights(self):
        
        with torch.no_grad():
            
            if self.is_first_layer:
                
                self.linear.weight.uniform_(-1 / self.input_dim, 
                                             1 / self.input_dim)      
            else:
                
                self.linear.weight.uniform_(-np.sqrt(6 / self.input_dim) / self.omega_0, 
                                             np.sqrt(6 / self.input_dim) / self.omega_0)   
                
                
                
class MorletLayer(torch.nn.Module):
    
    '''
    
    
    
    '''
    
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 bias=True, 
                 is_first_layer=False, 
                 omega_0=30.0):
        
        super().__init__()
        
        self.input_dim = input_dim       
        self.output_dim = output_dim
        self.omega_0 = omega_0
        
        self.c = 3.1416**(-0.25)*(1+np.exp(-omega_0**2)-2*np.exp(-3*omega_0**2/4))**-0.5
        self.k = np.exp(-0.5*omega_0**2)
        
        # self.a = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        # self.b = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
                
        self.is_first_layer = is_first_layer
        
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=bias)
        
        self.init_weights()         
        
    def forward(self, x):
        
        x = self.linear(x)
        # x = self.a*x-self.b
                        
        return self.c*torch.exp(-0.5*x**2)*(torch.sin(self.omega_0*x) - self.k)
    
    def init_weights(self):
        
        with torch.no_grad():
            
            if self.is_first_layer:
                
                self.linear.weight.uniform_(-1 / self.input_dim, 
                                             1 / self.input_dim)      
            else:
                
                self.linear.weight.uniform_(-np.sqrt(6 / self.input_dim) / self.omega_0, 
                                             np.sqrt(6 / self.input_dim) / self.omega_0)                 
                
                
class InputEncoding(torch.nn.Module):
    
    '''
    
    Harmonic encoding of the spatial coordinates
        
    '''
    
    def __init__(self, 
                 n_functions=6, 
                 base_omega_0=1.0, 
                 append_coords=False):
        
        super().__init__()
                
        frequencies = 2.0 ** torch.arange(n_functions, dtype=torch.float32)
        
        self.register_buffer('frequencies', frequencies*base_omega_0, persistent=True)
        self.append_coords = append_coords

    def forward(self, x):
        
        '''
        
        Arguments:
            x: tensor of shape [batch, dim] with values in [-1, 1]
            
        Returns:
            x_encoded: a harmonic embedding of x with shape [batch, (n_functions * 2 + int(append_coords)) * dim]
            
        '''
        
        x = (x + 1) / 2 + 1 # from [-1, 1] to [1, 2]

        x_encoded = (x[..., None] * self.frequencies).reshape(*x.shape[:-1], -1)
        
        x_encoded = torch.cat((x_encoded.sin(), x_encoded.cos(), x) if self.append_coords else (x_encoded.sin(), x_encoded.cos()), dim=-1)
        
        return x_encoded
    
class ReluNet(torch.nn.Module):
    
    '''
    
    ReluLayer network
    
    '''
    
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 hidden_n_layers, 
                 output_dim, 
                 is_last_linear=True, 
                 mid_skip=True, 
                 input_n_encoding_functions=0,
                 last_layer_weights_small=True):
        
        super().__init__()
        
        self.mid_skip = mid_skip
        self.n_skip = -1
        
        self.input_encoding = input_n_encoding_functions>0
        
        self.layers = torch.nn.ModuleList()
        
        if self.input_encoding==True:
            
            self.encoding = InputEncoding(n_functions=input_n_encoding_functions, append_coords=True)
            self.layers.append(ReluLayer(2*input_n_encoding_functions*input_dim+input_dim, hidden_dim))
            self.skip_dim = 2*input_n_encoding_functions*input_dim+input_dim
        
        else:
            
            self.layers.append(ReluLayer(input_dim, hidden_dim))
            self.skip_dim = input_dim
            
        ##################
                
        for i in range(hidden_n_layers):
            
            if i+1 == np.ceil(hidden_n_layers/2) and self.mid_skip:
            
                self.layers.append(ReluLayer(hidden_dim+self.skip_dim, hidden_dim))  
                
                self.n_skip = len(self.layers)
                
            else:
            
                self.layers.append(ReluLayer(hidden_dim, hidden_dim))
                
        ##################
            
        if is_last_linear==True:

            last_linear = torch.nn.Linear(hidden_dim, output_dim)

            with torch.no_grad():

                if last_layer_weights_small:
                    
                    last_linear.weight.uniform_(-0.0001, 0.0001) # weights are set to small values to provide small deformations at the first epochs

                self.layers.append(last_linear)

        else:

            self.layers.append(ReluLayer(hidden_dim, output_dim))
           
    def forward(self, coords, clone=True):
        
        if clone==True:
        
            coords = coords.clone().detach().requires_grad_(True)
            x = coords
            
            if self.input_encoding==True:
                
                x = self.encoding(x)
                
            if self.mid_skip==True:    
            
                add_to_skip = x
            
            for i, layer in enumerate(self.layers):
                                
                if (i+1)==self.n_skip and self.mid_skip==True:
                    
                    x = torch.cat([x, add_to_skip], dim=-1)

                x = layer(x)
            
            return x, coords
                    
        else:
            
            x = coords
            
            if self.input_encoding==True:
                
                x = self.encoding(x)
                
            if self.mid_skip==True:    
            
                add_to_skip = x
            
            for i, layer in enumerate(self.layers):
                                
                if (i+1)==self.n_skip and self.mid_skip==True:
                    
                    x = torch.cat([x, add_to_skip], dim=-1)

                x = layer(x)
                        
            return x    
        
        
        
class SirenNet(torch.nn.Module):
    
    '''
    
    Siren network
    
    '''
    
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 hidden_n_layers, 
                 output_dim, 
                 is_last_linear=False, 
                 mid_skip=True, 
                 input_omega_0=30.0, 
                 hidden_omega_0=30.0,
                 trainable_omega_0=False,
                 input_n_encoding_functions=0,
                 last_layer_weights_small=False,
                 ):
        
        super().__init__()
        
        self.mid_skip = mid_skip
        self.n_skip = -1
        
        self.input_encoding = input_n_encoding_functions>0
        
        self.layers = torch.nn.ModuleList()
        
        if self.input_encoding==True:
            
            self.encoding = InputEncoding(n_functions=input_n_encoding_functions, append_coords=True)
            self.layers.append(SineLayer(2*input_n_encoding_functions*input_dim+input_dim, hidden_dim, is_first_layer=True, omega_0=input_omega_0, trainable_omega_0=trainable_omega_0))
            self.skip_dim = 2*input_n_encoding_functions*input_dim+input_dim
        
        else:
            
            self.layers.append(SineLayer(input_dim, hidden_dim, is_first_layer=True, omega_0=input_omega_0, trainable_omega_0=trainable_omega_0))
            self.skip_dim = input_dim
            
        ##################
                
        for i in range(hidden_n_layers):
            
            if i+1 == np.ceil(hidden_n_layers/2) and self.mid_skip:
            
                self.layers.append(SineLayer(hidden_dim+self.skip_dim, hidden_dim, is_first_layer=False, omega_0=hidden_omega_0, trainable_omega_0=trainable_omega_0))  
                
                self.n_skip = len(self.layers)
                
            else:
            
                self.layers.append(SineLayer(hidden_dim, hidden_dim, is_first_layer=False, omega_0=hidden_omega_0, trainable_omega_0=trainable_omega_0))
                
        ##################
            
        if is_last_linear==True:

            last_linear = torch.nn.Linear(hidden_dim, output_dim)

            with torch.no_grad():

                if last_layer_weights_small:
                    
                    last_linear.weight.uniform_(-0.0001, 0.0001) # weights are set to small values to provide small deformations at the first epochs
                        
                else:
                
                    last_linear.weight.uniform_(-np.sqrt(6 / hidden_dim) / hidden_omega_0, 
                                                 np.sqrt(6 / hidden_dim) / hidden_omega_0)

                self.layers.append(last_linear)

        else:

            self.layers.append(SineLayer(hidden_dim, output_dim, is_first_layer=False, omega_0=hidden_omega_0, trainable_omega_0=trainable_omega_0))
           
    def forward(self, coords, clone=True):
        
        if clone==True:
        
            coords = coords.clone().detach().requires_grad_(True)
            x = coords
            
            if self.input_encoding==True:
                
                x = self.encoding(x)
                
            if self.mid_skip==True:    
            
                add_to_skip = x
            
            for i, layer in enumerate(self.layers):
                                
                if (i+1)==self.n_skip and self.mid_skip==True:
                    
                    x = torch.cat([x, add_to_skip], dim=-1)

                x = layer(x)
            
            return x, coords
                    
        else:
            
            x = coords
            
            if self.input_encoding==True:
                
                x = self.encoding(x)
                
            if self.mid_skip==True:    
            
                add_to_skip = x
            
            for i, layer in enumerate(self.layers):
                                
                if (i+1)==self.n_skip and self.mid_skip==True:
                    
                    x = torch.cat([x, add_to_skip], dim=-1)

                x = layer(x)
                        
            return x    
                
class DecomNet(torch.nn.Module):
    
    '''
    
    
    
    '''
    
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 hidden_n_layers, 
                 output_dim, 
                 is_last_linear=False, 
                 mid_skip=True, 
                 input_omega_0=30.0, 
                 hidden_omega_0=30.0,
                 input_n_encoding_functions=0,
                 last_layer_weights_small=False):
        
        super().__init__()
        
        self.mid_skip = mid_skip
        self.n_skip = -1
        
        self.input_encoding = input_n_encoding_functions>0
        
        self.soft = torch.nn.Softmax(dim=1)
        
        self.layers = torch.nn.ModuleList()
        
        if self.input_encoding==True:
            
            self.encoding = InputEncoding(n_functions=input_n_encoding_functions, append_coords=True)
            self.layers.append(SineLayer(2*input_n_encoding_functions*input_dim+input_dim, hidden_dim, is_first_layer=True, omega_0=input_omega_0))
            self.skip_dim = 2*input_n_encoding_functions*input_dim+input_dim
        
        else:
            
            self.layers.append(SineLayer(input_dim, hidden_dim, is_first_layer=True, omega_0=input_omega_0))
            self.skip_dim = input_dim
            
        ##################
                
        for i in range(hidden_n_layers):
            
            if i+1 == np.ceil(hidden_n_layers/2) and self.mid_skip:
            
                self.layers.append(SineLayer(hidden_dim+self.skip_dim, hidden_dim, is_first_layer=False, omega_0=hidden_omega_0))  
                
                self.n_skip = len(self.layers)
                
            else:
            
                self.layers.append(SineLayer(hidden_dim, hidden_dim, is_first_layer=False, omega_0=hidden_omega_0))
                
        ##################
            
        if is_last_linear==True:

            last_linear = torch.nn.Linear(hidden_dim, output_dim)

            with torch.no_grad():

                if last_layer_weights_small:
                    
                    last_linear.weight.uniform_(-0.0001, 0.0001) # weights are set to small values to provide small deformations at the first epochs

                self.layers.append(last_linear)

        else:

            self.layers.append(SineLayer(hidden_dim, output_dim, is_first_layer=False, omega_0=hidden_omega_0))
           
    def forward(self, coords, clone=True):
        
        if clone==True:
        
            coords = coords.clone().detach().requires_grad_(True)
            x = coords
            
            if self.input_encoding==True:
                
                x = self.encoding(x)
                
            if self.mid_skip==True:    
            
                add_to_skip = x
            
            for i, layer in enumerate(self.layers):
                                
                if (i+1)==self.n_skip and self.mid_skip==True:
                    
                    x = torch.cat([x, add_to_skip], dim=-1)

                x = layer(x)
                
            x = self.soft(x)
            
            return x, coords
                    
        else:
            
            x = coords
            
            if self.input_encoding==True:
                
                x = self.encoding(x)
                
            if self.mid_skip==True:    
            
                add_to_skip = x
            
            for i, layer in enumerate(self.layers):
                                
                if (i+1)==self.n_skip and self.mid_skip==True:
                    
                    x = torch.cat([x, add_to_skip], dim=-1)

                x = layer(x)
                
            x = self.soft(x)
                        
            return x            
        
        
class AttentionSirenNet(torch.nn.Module):
    
    '''
    
    Siren network
    
    '''
    
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 hidden_n_layers, 
                 output_dim, 
                 is_last_linear=False, 
                 mid_skip=True, 
                 input_omega_0=30.0, 
                 hidden_omega_0=30.0,
                 delta_omega_0=20.0,
                 input_n_encoding_functions=0,
                 last_layer_weights_small=False):
        
        super().__init__()
        
        self.mid_skip = mid_skip
        self.n_skip = -1
        
        self.input_encoding = input_n_encoding_functions>0
        
        self.layers = torch.nn.ModuleList()
        
        if self.input_encoding==True:
            
            self.encoding = InputEncoding(n_functions=input_n_encoding_functions, append_coords=True)
            self.layers.append(AttentionSineLayer(2*input_n_encoding_functions*input_dim+input_dim, hidden_dim, is_first_layer=True, omega_0=input_omega_0, delta_omega_0=delta_omega_0))
            self.skip_dim = 2*input_n_encoding_functions*input_dim+input_dim
        
        else:
            
            self.layers.append(AttentionSineLayer(input_dim, hidden_dim, is_first_layer=True, omega_0=input_omega_0, delta_omega_0=delta_omega_0))
            self.skip_dim = input_dim
            
        ##################
                
        for i in range(hidden_n_layers):
            
            if i+1 == np.ceil(hidden_n_layers/2) and self.mid_skip:
            
                self.layers.append(AttentionSineLayer(hidden_dim+self.skip_dim, hidden_dim, is_first_layer=False, omega_0=hidden_omega_0, delta_omega_0=delta_omega_0))  
                
                self.n_skip = len(self.layers)
                
            else:
            
                self.layers.append(AttentionSineLayer(hidden_dim, hidden_dim, is_first_layer=False, omega_0=hidden_omega_0, delta_omega_0=delta_omega_0))
                
        ##################
            
        if is_last_linear==True:

            last_linear = torch.nn.Linear(hidden_dim, output_dim)

            with torch.no_grad():

                if last_layer_weights_small:
                    
                    last_linear.weight.uniform_(-0.0001, 0.0001) # weights are set to small values to provide small deformations at the first epochs
                        
                else:
                
                    last_linear.weight.uniform_(-np.sqrt(6 / hidden_dim) / hidden_omega_0, 
                                                 np.sqrt(6 / hidden_dim) / hidden_omega_0)

                self.layers.append(last_linear)

        else:

            self.layers.append(AttentionSineLayer(hidden_dim, output_dim, is_first_layer=False, omega_0=hidden_omega_0, delta_omega_0=delta_omega_0))
           
    def forward(self, coords, clone=True):
        
        if clone==True:
        
            coords = coords.clone().detach().requires_grad_(True)
            x = coords
            
            if self.input_encoding==True:
                
                x = self.encoding(x)
                
            if self.mid_skip==True:    
            
                add_to_skip = x
            
            for i, layer in enumerate(self.layers):
                                
                if (i+1)==self.n_skip and self.mid_skip==True:
                    
                    x = torch.cat([x, add_to_skip], dim=-1)

                x = layer(x)
            
            return x, coords
                    
        else:
            
            x = coords
            
            if self.input_encoding==True:
                
                x = self.encoding(x)
                
            if self.mid_skip==True:    
            
                add_to_skip = x
            
            for i, layer in enumerate(self.layers):
                                
                if (i+1)==self.n_skip and self.mid_skip==True:
                    
                    x = torch.cat([x, add_to_skip], dim=-1)

                x = layer(x)
                        
            return x    
        
        
class SnakeNet(torch.nn.Module):
    
    '''
    
    Snake network
    
    '''
    
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 hidden_n_layers, 
                 output_dim, 
                 is_last_linear=False, 
                 mid_skip=True, 
                 input_omega_0=30.0, 
                 hidden_omega_0=30.0,
                 input_n_encoding_functions=0,
                 last_layer_weights_small=False):
        
        super().__init__()
        
        self.mid_skip = mid_skip
        self.n_skip = -1
        
        self.input_encoding = input_n_encoding_functions>0
        
        self.layers = torch.nn.ModuleList()
        
        if self.input_encoding==True:
            
            self.encoding = InputEncoding(n_functions=input_n_encoding_functions, append_coords=True)
            self.layers.append(SnakeLayer(2*input_n_encoding_functions*input_dim+input_dim, hidden_dim, is_first_layer=True, omega_0=input_omega_0))
            self.skip_dim = 2*input_n_encoding_functions*input_dim+input_dim
        
        else:
            
            self.layers.append(SnakeLayer(input_dim, hidden_dim, is_first_layer=True, omega_0=input_omega_0))
            self.skip_dim = input_dim
            
        ##################
                
        for i in range(hidden_n_layers):
            
            if i+1 == np.ceil(hidden_n_layers/2) and self.mid_skip:
            
                self.layers.append(SnakeLayer(hidden_dim+self.skip_dim, hidden_dim, is_first_layer=False, omega_0=hidden_omega_0))  
                
                self.n_skip = len(self.layers)
                
            else:
            
                self.layers.append(SnakeLayer(hidden_dim, hidden_dim, is_first_layer=False, omega_0=hidden_omega_0))
                
        ##################
            
        if is_last_linear==True:

            last_linear = torch.nn.Linear(hidden_dim, output_dim)

            with torch.no_grad():

                if last_layer_weights_small:
                    
                    last_linear.weight.uniform_(-0.0001, 0.0001) # weights are set to small values to provide small deformations at the first epochs
                        
                else:
                
                    last_linear.weight.uniform_(-np.sqrt(6 / hidden_dim) / hidden_omega_0, 
                                                 np.sqrt(6 / hidden_dim) / hidden_omega_0)

                self.layers.append(last_linear)

        else:

            self.layers.append(SnakeLayer(hidden_dim, output_dim, is_first_layer=False, omega_0=hidden_omega_0))
           
    def forward(self, coords, clone=True):
        
        if clone==True:
        
            coords = coords.clone().detach().requires_grad_(True)
            x = coords
            
            if self.input_encoding==True:
                
                x = self.encoding(x)
                
            if self.mid_skip==True:    
            
                add_to_skip = x
            
            for i, layer in enumerate(self.layers):
                                
                if (i+1)==self.n_skip and self.mid_skip==True:
                    
                    x = torch.cat([x, add_to_skip], dim=-1)

                x = layer(x)
            
            return x, coords
                    
        else:
            
            x = coords
            
            if self.input_encoding==True:
                
                x = self.encoding(x)
                
            if self.mid_skip==True:    
            
                add_to_skip = x
            
            for i, layer in enumerate(self.layers):
                                
                if (i+1)==self.n_skip and self.mid_skip==True:
                    
                    x = torch.cat([x, add_to_skip], dim=-1)

                x = layer(x)
                        
            return x            
        
        
        
class SineplusNet(torch.nn.Module):
    
    '''
    
    Sine+ network
    
    '''
    
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 hidden_n_layers, 
                 output_dim, 
                 is_last_linear=False, 
                 mid_skip=True, 
                 input_omega_0=30.0, 
                 hidden_omega_0=30.0,
                 input_n_encoding_functions=0,
                 last_layer_weights_small=False):
        
        super().__init__()
        
        self.mid_skip = mid_skip
        self.n_skip = -1
        
        self.input_encoding = input_n_encoding_functions>0
        
        self.layers = torch.nn.ModuleList()
        
        if self.input_encoding==True:
            
            self.encoding = InputEncoding(n_functions=input_n_encoding_functions, append_coords=True)
            self.layers.append(SineplusLayer(2*input_n_encoding_functions*input_dim+input_dim, hidden_dim, is_first_layer=True, omega_0=input_omega_0))
            self.skip_dim = 2*input_n_encoding_functions*input_dim+input_dim
        
        else:
            
            self.layers.append(SineplusLayer(input_dim, hidden_dim, is_first_layer=True, omega_0=input_omega_0))
            self.skip_dim = input_dim
            
        ##################
                
        for i in range(hidden_n_layers):
            
            if i+1 == np.ceil(hidden_n_layers/2) and self.mid_skip:
            
                self.layers.append(SineplusLayer(hidden_dim+self.skip_dim, hidden_dim, is_first_layer=False, omega_0=hidden_omega_0))  
                
                self.n_skip = len(self.layers)
                
            else:
            
                self.layers.append(SineplusLayer(hidden_dim, hidden_dim, is_first_layer=False, omega_0=hidden_omega_0))
                
        ##################
            
        if is_last_linear==True:

            last_linear = torch.nn.Linear(hidden_dim, output_dim)

            with torch.no_grad():

                if last_layer_weights_small:
                    
                    last_linear.weight.uniform_(-0.0001, 0.0001) # weights are set to small values to provide small deformations at the first epochs
                        
                else:
                
                    last_linear.weight.uniform_(-np.sqrt(6 / hidden_dim) / hidden_omega_0, 
                                                 np.sqrt(6 / hidden_dim) / hidden_omega_0)

                self.layers.append(last_linear)

        else:

            self.layers.append(SineplusLayer(hidden_dim, output_dim, is_first_layer=False, omega_0=hidden_omega_0))
           
    def forward(self, coords, clone=True):
        
        if clone==True:
        
            coords = coords.clone().detach().requires_grad_(True)
            x = coords
            
            if self.input_encoding==True:
                
                x = self.encoding(x)
                
            if self.mid_skip==True:    
            
                add_to_skip = x
            
            for i, layer in enumerate(self.layers):
                                
                if (i+1)==self.n_skip and self.mid_skip==True:
                    
                    x = torch.cat([x, add_to_skip], dim=-1)

                x = layer(x)
            
            return x, coords
                    
        else:
            
            x = coords
            
            if self.input_encoding==True:
                
                x = self.encoding(x)
                
            if self.mid_skip==True:    
            
                add_to_skip = x
            
            for i, layer in enumerate(self.layers):
                                
                if (i+1)==self.n_skip and self.mid_skip==True:
                    
                    x = torch.cat([x, add_to_skip], dim=-1)

                x = layer(x)
                        
            return x            
        
        
                
        
class ChirpNet(torch.nn.Module):
    
    '''
    
    Chirp network
    
    '''
    
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 hidden_n_layers, 
                 output_dim, 
                 is_last_linear=False, 
                 mid_skip=True, 
                 input_omega_0=30.0, 
                 hidden_omega_0=30.0,
                 delta_omega_0=10.0,
                 input_n_encoding_functions=0,
                 last_layer_weights_small=False):
        
        super().__init__()
        
        self.mid_skip = mid_skip
        self.n_skip = -1
        
        self.input_encoding = input_n_encoding_functions>0
        
        self.layers = torch.nn.ModuleList()
        
        if self.input_encoding==True:
            
            self.encoding = InputEncoding(n_functions=input_n_encoding_functions, append_coords=True)
            self.layers.append(ChirpLayer(2*input_n_encoding_functions*input_dim+input_dim, hidden_dim, is_first_layer=True, omega_0=input_omega_0, delta_omega_0=delta_omega_0))
            self.skip_dim = 2*input_n_encoding_functions*input_dim+input_dim
        
        else:
            
            self.layers.append(ChirpLayer(input_dim, hidden_dim, is_first_layer=True, omega_0=input_omega_0, delta_omega_0=delta_omega_0))
            self.skip_dim = input_dim
            
        ##################
                
        for i in range(hidden_n_layers):
            
            if i+1 == np.ceil(hidden_n_layers/2) and self.mid_skip:
            
                self.layers.append(ChirpLayer(hidden_dim+self.skip_dim, hidden_dim, is_first_layer=False, omega_0=hidden_omega_0, delta_omega_0=delta_omega_0))  
                
                self.n_skip = len(self.layers)
                
            else:
            
                self.layers.append(ChirpLayer(hidden_dim, hidden_dim, is_first_layer=False, omega_0=hidden_omega_0, delta_omega_0=delta_omega_0))
                
        ##################
            
        if is_last_linear==True:

            last_linear = torch.nn.Linear(hidden_dim, output_dim)

            with torch.no_grad():

                if last_layer_weights_small:
                    
                    last_linear.weight.uniform_(-0.0001, 0.0001) # weights are set to small values to provide small deformations at the first epochs
                        
                else:
                
                    last_linear.weight.uniform_(-np.sqrt(6 / hidden_dim) / hidden_omega_0, 
                                                 np.sqrt(6 / hidden_dim) / hidden_omega_0)

                self.layers.append(last_linear)

        else:

            self.layers.append(ChirpLayer(hidden_dim, output_dim, is_first_layer=False, omega_0=hidden_omega_0, delta_omega_0=delta_omega_0))
           
    def forward(self, coords, clone=True):
        
        if clone==True:
        
            coords = coords.clone().detach().requires_grad_(True)
            x = coords
            
            if self.input_encoding==True:
                
                x = self.encoding(x)
                
            if self.mid_skip==True:    
            
                add_to_skip = x
            
            for i, layer in enumerate(self.layers):
                                
                if (i+1)==self.n_skip and self.mid_skip==True:
                    
                    x = torch.cat([x, add_to_skip], dim=-1)

                x = layer(x)
            
            return x, coords
                    
        else:
            
            x = coords
            
            if self.input_encoding==True:
                
                x = self.encoding(x)
                
            if self.mid_skip==True:    
            
                add_to_skip = x
            
            for i, layer in enumerate(self.layers):
                                
                if (i+1)==self.n_skip and self.mid_skip==True:
                    
                    x = torch.cat([x, add_to_skip], dim=-1)

                x = layer(x)
                        
            return x            
                
        
        
class MorletNet(torch.nn.Module):
    
    '''
    
    Morley wavelet network
    
    '''
    
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 hidden_n_layers, 
                 output_dim, 
                 is_last_linear=False, 
                 mid_skip=True, 
                 input_omega_0=30.0, 
                 hidden_omega_0=30.0,
                 input_n_encoding_functions=0,
                 last_layer_weights_small=False):
        
        super().__init__()
        
        self.mid_skip = mid_skip
        self.n_skip = -1
        
        self.input_encoding = input_n_encoding_functions>0
        
        self.layers = torch.nn.ModuleList()
        
        if self.input_encoding==True:
            
            self.encoding = InputEncoding(n_functions=input_n_encoding_functions, append_coords=True)
            self.layers.append(MorletLayer(2*input_n_encoding_functions*input_dim+input_dim, hidden_dim, is_first_layer=True, omega_0=input_omega_0))
            self.skip_dim = 2*input_n_encoding_functions*input_dim+input_dim
        
        else:
            
            self.layers.append(MorletLayer(input_dim, hidden_dim, is_first_layer=True, omega_0=input_omega_0))
            self.skip_dim = input_dim
            
        ##################
                
        for i in range(hidden_n_layers):
            
            if i+1 == np.ceil(hidden_n_layers/2) and self.mid_skip:
            
                self.layers.append(MorletLayer(hidden_dim+self.skip_dim, hidden_dim, is_first_layer=False, omega_0=hidden_omega_0))  
                
                self.n_skip = len(self.layers)
                
            else:
            
                self.layers.append(MorletLayer(hidden_dim, hidden_dim, is_first_layer=False, omega_0=hidden_omega_0))
                
        ##################
            
        if is_last_linear==True:

            last_linear = torch.nn.Linear(hidden_dim, output_dim)

            with torch.no_grad():

                if last_layer_weights_small:
                    
                    last_linear.weight.uniform_(-0.0001, 0.0001) # weights are set to small values to provide small deformations at the first epochs
                        
                else:
                
                    last_linear.weight.uniform_(-np.sqrt(6 / hidden_dim) / hidden_omega_0, 
                                                 np.sqrt(6 / hidden_dim) / hidden_omega_0)

                self.layers.append(last_linear)

        else:

            self.layers.append(MorletLayer(hidden_dim, output_dim, is_first_layer=False, omega_0=hidden_omega_0))
           
    def forward(self, coords, clone=True):
        
        if clone==True:
        
            coords = coords.clone().detach().requires_grad_(True)
            x = coords
            
            if self.input_encoding==True:
                
                x = self.encoding(x)
                
            if self.mid_skip==True:    
            
                add_to_skip = x
            
            for i, layer in enumerate(self.layers):
                                
                if (i+1)==self.n_skip and self.mid_skip==True:
                    
                    x = torch.cat([x, add_to_skip], dim=-1)

                x = layer(x)
            
            return x, coords
                    
        else:
            
            x = coords
            
            if self.input_encoding==True:
                
                x = self.encoding(x)
                
            if self.mid_skip==True:    
            
                add_to_skip = x
            
            for i, layer in enumerate(self.layers):
                                
                if (i+1)==self.n_skip and self.mid_skip==True:
                    
                    x = torch.cat([x, add_to_skip], dim=-1)

                x = layer(x)
                        
            return x            
        
        
class ModulationNet(torch.nn.Module):
    
    '''
    
    Siren network with layaer-wise omage_0 modulation
    
    '''
    
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 hidden_n_layers, 
                 output_dim, 
                 is_last_linear=False, 
                 mid_skip=True, 
                 input_omega_0=10.0, 
                 hidden_omega_0=10.0,
                 input_n_encoding_functions=0,
                 last_layer_weights_small=False):
        
        super().__init__()
        
        self.mid_skip = mid_skip
        self.n_skip = -1
        
        self.input_encoding = input_n_encoding_functions>0
        
        self.layers = torch.nn.ModuleList()
        
        if self.input_encoding==True:
            
            self.encoding = InputEncoding(n_functions=input_n_encoding_functions, append_coords=True)
            self.layers.append(SineLayer(2*input_n_encoding_functions*input_dim+input_dim, hidden_dim, is_first_layer=True, omega_0=input_omega_0))
            self.skip_dim = 2*input_n_encoding_functions*input_dim+input_dim
        
        else:
            
            self.layers.append(SineLayer(input_dim, hidden_dim, is_first_layer=True, omega_0=input_omega_0))
            self.skip_dim = input_dim
            
        ##################
                
        for i in range(hidden_n_layers):
            
            if i+1 == np.ceil(hidden_n_layers/2) and self.mid_skip:
            
                self.layers.append(SineLayer(hidden_dim+self.skip_dim, hidden_dim, is_first_layer=False, omega_0=hidden_omega_0))  
                
                self.n_skip = len(self.layers)
                
            else:
            
                self.layers.append(SineLayer(hidden_dim, hidden_dim, is_first_layer=False, omega_0=hidden_omega_0))

            hidden_omega_0 = hidden_omega_0 + 10
                
        ##################
            
        if is_last_linear==True:

            last_linear = torch.nn.Linear(hidden_dim, output_dim)

            with torch.no_grad():

                if last_layer_weights_small:
                    
                    last_linear.weight.uniform_(-0.0001, 0.0001) # weights are set to small values to provide small deformations at the first epochs
                        
                else:
                
                    last_linear.weight.uniform_(-np.sqrt(6 / hidden_dim) / hidden_omega_0, 
                                                 np.sqrt(6 / hidden_dim) / hidden_omega_0)

                self.layers.append(last_linear)

        else:

            self.layers.append(SineLayer(hidden_dim, output_dim, is_first_layer=False, omega_0=hidden_omega_0))
           
    def forward(self, coords, clone=True):
        
        if clone==True:
        
            coords = coords.clone().detach().requires_grad_(True)
            x = coords
            
            if self.input_encoding==True:
                
                x = self.encoding(x)
                
            if self.mid_skip==True:    
            
                add_to_skip = x
            
            for i, layer in enumerate(self.layers):
                                
                if (i+1)==self.n_skip and self.mid_skip==True:
                    
                    x = torch.cat([x, add_to_skip], dim=-1)

                x = layer(x)
            
            return x, coords
                    
        else:
            
            x = coords
            
            if self.input_encoding==True:
                
                x = self.encoding(x)
                
            if self.mid_skip==True:    
            
                add_to_skip = x
            
            for i, layer in enumerate(self.layers):
                                
                if (i+1)==self.n_skip and self.mid_skip==True:
                    
                    x = torch.cat([x, add_to_skip], dim=-1)

                x = layer(x)
                        
            return x                            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        