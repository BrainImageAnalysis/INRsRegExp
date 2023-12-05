import torch
import os

import numpy as np
import matplotlib.pyplot as plt

from itertools import chain
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.ndimage import laplace, gaussian_filter

from .networks import ReluNet, SirenNet, DecomNet, AffineLayer, AttentionSirenNet, SnakeNet, SineplusNet, ChirpNet, MorletNet, ModulationNet
from .data import load_mindboggle, make_coords_tensor, fast_trilinear_interpolation, MetricMonitor, VectorCoords, BlockCoords
from .losses import NCCLoss, LNCCLoss, JacobianLossCoords, MSELoss


def relu_registration_mindboggle(moving_path=None,
                                 fixed_path=None, 
                                 results_path=None,
                                 epochs=50,
                                 epoch_save_visualization=5,
                                 batch_size=64**3,
                                 block_size=[32,32,32],
                                 alpha_ncc=1,
                                 alpha_jac=0.1,
                                 steps=500,
                                 learning_rate=0.00001,
                                 device='cuda',
                                 ):
    
    moving, moving_segmentations = load_mindboggle(moving_path)
    fixed, fixed_segmentations = load_mindboggle(fixed_path)
    
    n_dims = len(block_size)

    coords_init = make_coords_tensor(moving.shape, is_vector=False)
   
    registration_network = ReluNet(input_dim=3, output_dim=3, hidden_dim=256, hidden_n_layers=5, last_layer_weights_small=True, is_last_linear=True, input_n_encoding_functions=6).to(device)

    optimizer = torch.optim.AdamW(lr=learning_rate, params=registration_network.parameters())

    ncc = NCCLoss(is_tensor=False).to(device)
    lncc = LNCCLoss(is_tensor=False).to(device)
    jac = JacobianLossCoords(add_identity=False, is_tensor=False).to(device)

    pair_val = VectorCoords(coords=coords_init, scale_factor=1)
    val_loader = DataLoader(pair_val, batch_size=batch_size, shuffle=False)

    image_as_blocks = BlockCoords(coords=coords_init, scale_factor=0.5, block_size=block_size, steps=steps)
    block_loader = DataLoader(image_as_blocks, batch_size=1, shuffle=True)
    
    os.makedirs(results_path, exist_ok=True)    

    moving = torch.from_numpy(moving)
    moving_image = moving.to(device, dtype=torch.float32)
    fixed = torch.from_numpy(fixed)
    fixed_image = fixed.to(device, dtype=torch.float32)

    for epoch in range(epochs):

        stream = tqdm(block_loader)
        loop_monitor = MetricMonitor()
        
        for i, image_coords in enumerate(stream, start=0):

            image_coords = image_coords.squeeze().to(device, dtype=torch.float32)

            flow, coords = registration_network(image_coords.view([np.prod(block_size), n_dims]))
            flow_add = torch.add(flow, coords)

            moved_image_pixels = fast_trilinear_interpolation(moving_image, flow_add[:,0], flow_add[:,1], flow_add[:,2]).view(block_size)
            fixed_image_pixels = fast_trilinear_interpolation(fixed_image, coords[:,0], coords[:,1], coords[:,2]).view(block_size)

            moved_image_pixels = moved_image_pixels.unsqueeze(0).unsqueeze(0)
            fixed_image_pixels = fixed_image_pixels.unsqueeze(0).unsqueeze(0)

            ################################

            loss_all = torch.tensor(0)

            loss_ncc = (ncc(moved_image_pixels, fixed_image_pixels) + lncc(moved_image_pixels, fixed_image_pixels)) / 2
            loss_jac = jac(coords, flow_add)

            if alpha_ncc>0:

                loss_all = loss_all + alpha_ncc*loss_ncc        

            if alpha_jac>0:

                loss_all = loss_all + alpha_jac*loss_jac        

            loop_monitor.update('jac', loss_jac.item())
            loop_monitor.update('ncc', loss_ncc.item())
            loop_monitor.update('loss', loss_all.item())
            stream.set_description('Epoch: {epoch}. Train. {metric_monitor}'.format(epoch=epoch, metric_monitor=loop_monitor))

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

        if not (epoch+1) % epoch_save_visualization:

            with torch.no_grad():

                for k, image_coords in enumerate(val_loader):

                    image_coords = image_coords.squeeze().to(device, dtype=torch.float32)

                    flow, coords = registration_network(image_coords)
                    flow_add = torch.add(flow, coords)

                    flow_add = flow_add.cpu().detach().squeeze()

                    if k==0:

                        total_flow = flow_add

                    else:

                        total_flow = torch.cat((total_flow, flow_add), 0)

                temp_moved = fast_trilinear_interpolation(moving, total_flow[:,0], total_flow[:,1], total_flow[:,2]).view(moving.shape)  
                temp_moved = temp_moved.numpy().squeeze()

            fig, axes = plt.subplots(1,3, figsize=(18,6))
            axes[0].imshow(moving[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            axes[1].imshow(temp_moved[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            axes[2].imshow(fixed[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            
            axes[0].set_title('Moving')
            axes[1].set_title('Moved')
            axes[2].set_title('Fixed')            
            
            plt.show()
            
            np.save(results_path+str(epoch+1)+'_deformation_field.npy', total_flow.view(*moving.shape, 3).numpy())        

            save_state = {'state_dict': registration_network.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'loss_all': loss_all.item()}         
            
            torch.save(save_state, results_path + str(epoch+1) + '.pth.tar')   
            
            
            
def siren_registration_mindboggle(moving_path=None,
                                  fixed_path=None, 
                                  results_path=None,
                                  epochs=50,
                                  epoch_save_visualization=5,
                                  batch_size=64**3,
                                  block_size=[32,32,32],
                                  alpha_ncc=1,
                                  alpha_jac=0.1,
                                  steps=500,
                                  learning_rate=0.00001,
                                  device='cuda',
                                  omega_0=30,
                                  trainable_omega_0=False,
                                  ):
    
    moving, moving_segmentations = load_mindboggle(moving_path)
    fixed, fixed_segmentations = load_mindboggle(fixed_path)
    
    n_dims = len(block_size)

    coords_init = make_coords_tensor(moving.shape, is_vector=False)
   
    registration_network = SirenNet(input_dim=3, output_dim=3, hidden_dim=256, hidden_n_layers=5, last_layer_weights_small=True, is_last_linear=True, input_n_encoding_functions=6, input_omega_0=omega_0, hidden_omega_0=omega_0, trainable_omega_0=trainable_omega_0).to(device)

    optimizer = torch.optim.AdamW(lr=learning_rate, params=registration_network.parameters())

    ncc = NCCLoss(is_tensor=False).to(device)
    lncc = LNCCLoss(is_tensor=False).to(device)
    jac = JacobianLossCoords(add_identity=False, is_tensor=False).to(device)

    pair_val = VectorCoords(coords=coords_init, scale_factor=1)
    val_loader = DataLoader(pair_val, batch_size=batch_size, shuffle=False)

    image_as_blocks = BlockCoords(coords=coords_init, scale_factor=0.5, block_size=block_size, steps=steps)
    block_loader = DataLoader(image_as_blocks, batch_size=1, shuffle=True)
    
    os.makedirs(results_path, exist_ok=True)    

    moving = torch.from_numpy(moving)
    moving_image = moving.to(device, dtype=torch.float32)
    fixed = torch.from_numpy(fixed)
    fixed_image = fixed.to(device, dtype=torch.float32)

    for epoch in range(epochs):

        stream = tqdm(block_loader)
        loop_monitor = MetricMonitor()
        
        for i, image_coords in enumerate(stream, start=0):

            image_coords = image_coords.squeeze().to(device, dtype=torch.float32)

            flow, coords = registration_network(image_coords.view([np.prod(block_size), n_dims]))
            flow_add = torch.add(flow, coords)

            moved_image_pixels = fast_trilinear_interpolation(moving_image, flow_add[:,0], flow_add[:,1], flow_add[:,2]).view(block_size)
            fixed_image_pixels = fast_trilinear_interpolation(fixed_image, coords[:,0], coords[:,1], coords[:,2]).view(block_size)

            moved_image_pixels = moved_image_pixels.unsqueeze(0).unsqueeze(0)
            fixed_image_pixels = fixed_image_pixels.unsqueeze(0).unsqueeze(0)

            ################################

            loss_all = torch.tensor(0)

            loss_ncc = (ncc(moved_image_pixels, fixed_image_pixels) + lncc(moved_image_pixels, fixed_image_pixels)) / 2
            loss_jac = jac(coords, flow_add)

            if alpha_ncc>0:

                loss_all = loss_all + alpha_ncc*loss_ncc        

            if alpha_jac>0:

                loss_all = loss_all + alpha_jac*loss_jac        

            loop_monitor.update('jac', loss_jac.item())
            loop_monitor.update('ncc', loss_ncc.item())
            loop_monitor.update('loss', loss_all.item())
            stream.set_description('Epoch: {epoch}. Train. {metric_monitor}'.format(epoch=epoch, metric_monitor=loop_monitor))

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

        if not (epoch+1) % epoch_save_visualization:

            with torch.no_grad():

                for k, image_coords in enumerate(val_loader):

                    image_coords = image_coords.squeeze().to(device, dtype=torch.float32)

                    flow, coords = registration_network(image_coords)
                    flow_add = torch.add(flow, coords)

                    flow_add = flow_add.cpu().detach().squeeze()

                    if k==0:

                        total_flow = flow_add

                    else:

                        total_flow = torch.cat((total_flow, flow_add), 0)

                temp_moved = fast_trilinear_interpolation(moving, total_flow[:,0], total_flow[:,1], total_flow[:,2]).view(moving.shape)  
                temp_moved = temp_moved.numpy().squeeze()

            fig, axes = plt.subplots(1,3, figsize=(18,6))
            axes[0].imshow(moving[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            axes[1].imshow(temp_moved[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            axes[2].imshow(fixed[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            
            axes[0].set_title('Moving')
            axes[1].set_title('Moved')
            axes[2].set_title('Fixed')            
            
            plt.show()
            
            np.save(results_path+str(epoch+1)+'_deformation_field.npy', total_flow.view(*moving.shape, 3).numpy())        

            save_state = {'state_dict': registration_network.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'loss_all': loss_all.item()}         
            
            torch.save(save_state, results_path + str(epoch+1) + '.pth.tar')   
            
            
            
def cycle_registration_mindboggle(moving_path=None,
                                  fixed_path=None, 
                                  results_path=None,
                                  epochs=50,
                                  epoch_save_visualization=5,
                                  batch_size=64**3,
                                  block_size=[32,32,32],
                                  alpha_ncc=1,
                                  alpha_jac=0.1,
                                  alpha_cycle = 100,
                                  steps=500,
                                  learning_rate=0.00001,
                                  device='cuda',
                                  omega_0=30,
                                  ):
    
    moving, moving_segmentations = load_mindboggle(moving_path)
    fixed, fixed_segmentations = load_mindboggle(fixed_path)
    
    n_dims = len(block_size)

    coords_init = make_coords_tensor(moving.shape, is_vector=False)
    
    registration_network_moving = SirenNet(input_dim=3, output_dim=3, hidden_dim=256, hidden_n_layers=5, last_layer_weights_small=True, is_last_linear=True, input_n_encoding_functions=6, input_omega_0=omega_0, hidden_omega_0=omega_0).to(device)
    
    registration_network_fixed = SirenNet(input_dim=3, output_dim=3, hidden_dim=256, hidden_n_layers=5, last_layer_weights_small=True, is_last_linear=True, input_n_encoding_functions=6, input_omega_0=omega_0, hidden_omega_0=omega_0).to(device) 

    params = chain(registration_network_moving.parameters(), registration_network_fixed.parameters())
    optimizer = torch.optim.AdamW(lr=learning_rate, params=params)    
    
    ncc = NCCLoss(is_tensor=False).to(device)
    lncc = LNCCLoss(is_tensor=False).to(device)
    jac = JacobianLossCoords(add_identity=False, is_tensor=False).to(device)
    mse = MSELoss()
    
    pair_val = VectorCoords(coords=coords_init, scale_factor=1)
    val_loader = DataLoader(pair_val, batch_size=batch_size, shuffle=False)

    image_as_blocks = BlockCoords(coords=coords_init, scale_factor=0.5, block_size=block_size, steps=steps)
    block_loader = DataLoader(image_as_blocks, batch_size=1, shuffle=True)
    
    os.makedirs(results_path, exist_ok=True)    

    moving = torch.from_numpy(moving)
    moving_image = moving.to(device, dtype=torch.float32)
    fixed = torch.from_numpy(fixed)
    fixed_image = fixed.to(device, dtype=torch.float32)

    for epoch in range(epochs):

        stream = tqdm(block_loader)
        loop_monitor = MetricMonitor()
        
        for i, image_coords in enumerate(stream, start=0):
            
            image_coords = image_coords.squeeze().to(device, dtype=torch.float32)

            flow_1, coords_1 = registration_network_moving(image_coords.view([np.prod(block_size), n_dims]))
            flow_add_1 = flow_1 + coords_1

            flow_2, coords_2 = registration_network_fixed(image_coords.view([np.prod(block_size), n_dims]))
            flow_add_2 = flow_2 + coords_2            

            flow_3 = registration_network_fixed(flow_add_1, clone=False)
            flow_add_3 = flow_3 + flow_add_1   
            
            flow_4 = registration_network_moving(flow_add_2, clone=False)
            flow_add_4 = flow_4 + flow_add_2             

            moved_image_pixels_1 = fast_trilinear_interpolation(moving_image, flow_add_1[:,0], flow_add_1[:,1], flow_add_1[:,2]).view(block_size)
            fixed_image_pixels_1 = fast_trilinear_interpolation(fixed_image, coords_1[:,0], coords_1[:,1], coords_1[:,2]).view(block_size)

            moved_image_pixels_1 = moved_image_pixels_1.unsqueeze(0).unsqueeze(0)
            fixed_image_pixels_1 = fixed_image_pixels_1.unsqueeze(0).unsqueeze(0)
            
            moved_image_pixels_2 = fast_trilinear_interpolation(fixed_image, flow_add_2[:,0], flow_add_2[:,1], flow_add_2[:,2]).view(block_size)
            fixed_image_pixels_2 = fast_trilinear_interpolation(moving_image, coords_2[:,0], coords_2[:,1], coords_2[:,2]).view(block_size)

            moved_image_pixels_2 = moved_image_pixels_2.unsqueeze(0).unsqueeze(0)
            fixed_image_pixels_2 = fixed_image_pixels_2.unsqueeze(0).unsqueeze(0)        

            ################################

            loss_all = torch.tensor(0)

            loss_ncc_1 = (ncc(moved_image_pixels_1, fixed_image_pixels_1) + lncc(moved_image_pixels_1, fixed_image_pixels_1)) / 2
            loss_jac_1 = jac(coords_1, flow_add_1)
            
            loss_ncc_2 = (ncc(moved_image_pixels_2, fixed_image_pixels_2) + lncc(moved_image_pixels_2, fixed_image_pixels_2)) / 2
            loss_jac_2 = jac(coords_2, flow_add_2)   
            
            loss_ncc = (loss_ncc_1 + loss_ncc_2)/2 
            loss_jac = (loss_jac_1 + loss_jac_2)/2     

            loss_cycle = (mse(coords_1, flow_add_3) + mse(coords_2, flow_add_4))/2
            
            if alpha_ncc>0:

                loss_all = loss_all + alpha_ncc*loss_ncc 

            if alpha_jac>0:

                loss_all = loss_all + alpha_jac*loss_jac 
                
            if alpha_cycle>0:

                loss_all = loss_all + alpha_cycle*loss_cycle        

            loop_monitor.update('jac', loss_jac.item())
            loop_monitor.update('ncc', loss_ncc.item())
            loop_monitor.update('cycle', loss_cycle.item())
            loop_monitor.update('loss', loss_all.item())
            stream.set_description('Epoch: {epoch}. Train. {metric_monitor}'.format(epoch=epoch, metric_monitor=loop_monitor))

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

        if not (epoch+1) % epoch_save_visualization:

            with torch.no_grad():

                for k, image_coords in enumerate(val_loader):

                    image_coords = image_coords.squeeze().to(device, dtype=torch.float32)

                    flow, coords = registration_network_moving(image_coords)
                    flow_add = torch.add(flow, coords)

                    flow_add = flow_add.cpu().detach().squeeze()

                    if k==0:

                        total_flow = flow_add

                    else:

                        total_flow = torch.cat((total_flow, flow_add), 0)

                temp_moved = fast_trilinear_interpolation(moving, total_flow[:,0], total_flow[:,1], total_flow[:,2]).view(moving.shape)  
                temp_moved = temp_moved.numpy().squeeze()

            fig, axes = plt.subplots(1,3, figsize=(18,6))
            axes[0].imshow(moving[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            axes[1].imshow(temp_moved[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            axes[2].imshow(fixed[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            
            axes[0].set_title('Moving')
            axes[1].set_title('Moved')
            axes[2].set_title('Fixed')            
            
            plt.show()
            
            np.save(results_path+str(epoch+1)+'_deformation_field.npy', total_flow.view(*moving.shape, 3).numpy())        

            save_state = {'state_dict_moving': registration_network_moving.state_dict(),
                          'state_dict_fixed': registration_network_fixed.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'loss_all': loss_all.item()}         
            
            torch.save(save_state, results_path + str(epoch+1) + '.pth.tar')               
            
            
            
            
def fusion_registration_mindboggle(moving_path=None,
                                   fixed_path=None, 
                                   results_path=None,
                                   epochs=50,
                                   epoch_save_visualization=5,
                                   batch_size=64**3,
                                   block_size=[32,32,32],
                                   alpha_ncc=1,
                                   alpha_jac=0.1,
                                   steps=500,
                                   learning_rate=0.00001,
                                   device='cuda',
                                   omega_0=30,
                                   ):
    
    moving, moving_segmentations = load_mindboggle(moving_path)
    fixed, fixed_segmentations = load_mindboggle(fixed_path)
    
    n_dims = len(block_size)

    coords_init = make_coords_tensor(moving.shape, is_vector=False)
   
    registration_network = SirenNet(input_dim=3, output_dim=3, hidden_dim=256, hidden_n_layers=5, last_layer_weights_small=True, is_last_linear=True, input_n_encoding_functions=6, input_omega_0=omega_0, hidden_omega_0=omega_0).to(device)
    
    decomposition_network = DecomNet(input_dim=3, output_dim=3, hidden_dim=256, hidden_n_layers=5, last_layer_weights_small=True,
                          is_last_linear=True, input_omega_0=30.0, hidden_omega_0=30., input_n_encoding_functions=6).to(device)        

    params = chain(registration_network.parameters(), decomposition_network.parameters())
    optimizer = torch.optim.AdamW(lr=learning_rate, params=params)    

    ncc = NCCLoss(is_tensor=False).to(device)
    lncc = LNCCLoss(is_tensor=False).to(device)
    jac = JacobianLossCoords(add_identity=False, is_tensor=False).to(device)

    pair_val = VectorCoords(coords=coords_init, scale_factor=1)
    val_loader = DataLoader(pair_val, batch_size=batch_size, shuffle=False)

    image_as_blocks = BlockCoords(coords=coords_init, scale_factor=0.5, block_size=block_size, steps=steps)
    block_loader = DataLoader(image_as_blocks, batch_size=1, shuffle=True)
    
    os.makedirs(results_path, exist_ok=True)    
    
    smooth = torch.from_numpy(gaussian_filter(moving, sigma=2))
    edge = torch.from_numpy(laplace(moving))    
    smooth_image = smooth.to(device, dtype=torch.float32)
    edge_image = edge.to(device, dtype=torch.float32)
    
    moving = torch.from_numpy(moving)
    moving_image = moving.to(device, dtype=torch.float32)
    fixed = torch.from_numpy(fixed)
    fixed_image = fixed.to(device, dtype=torch.float32)

    for epoch in range(epochs):

        stream = tqdm(block_loader)
        loop_monitor = MetricMonitor()
        
        for i, image_coords in enumerate(stream, start=0):

            image_coords.shape

            image_coords = image_coords.squeeze().to(device, dtype=torch.float32)

            flow, coords = registration_network(image_coords.view([np.prod(block_size), n_dims]))
            flow_add = torch.add(flow, coords)
            
            selection, coords_selection = decomposition_network(image_coords.view([np.prod(block_size), n_dims]))

            moved_image_pixels = fast_trilinear_interpolation(moving_image, flow_add[:,0], flow_add[:,1], flow_add[:,2])
            edge_image_pixels = fast_trilinear_interpolation(edge_image, flow_add[:,0], flow_add[:,1], flow_add[:,2])
            smooth_image_pixels = fast_trilinear_interpolation(smooth_image, flow_add[:,0], flow_add[:,1], flow_add[:,2])
            
            joint_image_pixels = torch.cat((moved_image_pixels.unsqueeze(1), edge_image_pixels.unsqueeze(1), smooth_image_pixels.unsqueeze(1)), dim=1)
            joint_image_pixels = torch.sum(joint_image_pixels*selection, dim=1).squeeze().view(block_size)
            
            moved_image_pixels = moved_image_pixels.view(block_size)
            
            fixed_image_pixels = fast_trilinear_interpolation(fixed_image, coords[:,0], coords[:,1], coords[:,2]).view(block_size)

            joint_image_pixels = joint_image_pixels.unsqueeze(0).unsqueeze(0)
            moved_image_pixels = moved_image_pixels.unsqueeze(0).unsqueeze(0)
            fixed_image_pixels = fixed_image_pixels.unsqueeze(0).unsqueeze(0)

            ################################

            loss_all = torch.tensor(0)

            loss_ncc_moving = (ncc(moved_image_pixels, fixed_image_pixels) + lncc(moved_image_pixels, fixed_image_pixels)) / 2
            loss_ncc_join = (ncc(joint_image_pixels, fixed_image_pixels) + lncc(joint_image_pixels, fixed_image_pixels)) / 2
            
            loss_ncc = (loss_ncc_moving + loss_ncc_join) / 2
            loss_jac = jac(coords, flow_add)

            if alpha_ncc>0:

                loss_all = loss_all + alpha_ncc*loss_ncc        

            if alpha_jac>0:

                loss_all = loss_all + alpha_jac*loss_jac        

            loop_monitor.update('jac', loss_jac.item())
            loop_monitor.update('ncc', loss_ncc.item())
            loop_monitor.update('loss', loss_all.item())
            stream.set_description('Epoch: {epoch}. Train. {metric_monitor}'.format(epoch=epoch, metric_monitor=loop_monitor))

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

        if not (epoch+1) % epoch_save_visualization:

            with torch.no_grad():

                for k, image_coords in enumerate(val_loader):

                    image_coords = image_coords.squeeze().to(device, dtype=torch.float32)

                    flow, coords = registration_network(image_coords)
                    flow_add = torch.add(flow, coords)
                    flow_add = flow_add.cpu().detach().squeeze()
                    
                    selection, coords_selection = decomposition_network(image_coords)
                    selection = selection.cpu().detach().squeeze()

                    if k==0:

                        total_flow = flow_add
                        total_selection = selection

                    else:

                        total_flow = torch.cat((total_flow, flow_add), 0)
                        total_selection = torch.cat((total_selection, selection), 0)

                temp_smooth = fast_trilinear_interpolation(smooth, total_flow[:,0], total_flow[:,1], total_flow[:,2])
                temp_edge = fast_trilinear_interpolation(edge, total_flow[:,0], total_flow[:,1], total_flow[:,2])           
                temp_moved = fast_trilinear_interpolation(moving, total_flow[:,0], total_flow[:,1], total_flow[:,2])
                
                joint_image = torch.cat((temp_moved.unsqueeze(1), temp_edge.unsqueeze(1), temp_smooth.unsqueeze(1)), dim=1)
                joint_image = torch.sum(joint_image*total_selection, dim=1).squeeze().view(moving.shape).numpy().squeeze()                
                
                temp_moved = temp_moved.view(moving.shape)  
                temp_moved = temp_moved.numpy().squeeze()

            fig, axes = plt.subplots(1,4, figsize=(18,6))
            axes[0].imshow(moving[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            axes[1].imshow(temp_moved[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            axes[2].imshow(fixed[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            axes[3].imshow(joint_image[:, :, 100], cmap='gray')
            
            axes[0].set_title('Moving')
            axes[1].set_title('Moved')
            axes[2].set_title('Fixed')            
            axes[3].set_title('Fusion')   
            plt.show()
            
            np.save(results_path+str(epoch+1)+'_deformation_field.npy', total_flow.view(*moving.shape, 3).numpy())        

            save_state = {'state_dict_registration': registration_network.state_dict(),
                          'state_dict_decomposition': decomposition_network.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'loss_all': loss_all.item()}         
            
            torch.save(save_state, results_path + str(epoch+1) + '.pth.tar')               
            
            
            
def ensemble_registration_mindboggle(moving_path=None,
                                     fixed_path=None, 
                                     results_path=None,
                                     epochs=50,
                                     epoch_save_visualization=5,
                                     batch_size=64**3,
                                     block_size=[32,32,32],
                                     alpha_ncc=1,
                                     alpha_jac=0.1,
                                     steps=500,
                                     learning_rate=0.00001,
                                     device='cuda',
                                     omega_0_low=10,
                                     omega_0_medium=30,
                                     omega_0_high=50,
                                     ):
    
    moving, moving_segmentations = load_mindboggle(moving_path)
    fixed, fixed_segmentations = load_mindboggle(fixed_path)
    
    n_dims = len(block_size)

    coords_init = make_coords_tensor(moving.shape, is_vector=False)
   
    registration_network_low = SirenNet(input_dim=3, output_dim=3, hidden_dim=256, hidden_n_layers=5, last_layer_weights_small=True, is_last_linear=True, input_n_encoding_functions=6, input_omega_0=omega_0_low, hidden_omega_0=omega_0_low).to(device)
    
    registration_network_medium = SirenNet(input_dim=3, output_dim=3, hidden_dim=256, hidden_n_layers=5, last_layer_weights_small=True, is_last_linear=True, input_n_encoding_functions=6, input_omega_0=omega_0_medium, hidden_omega_0=omega_0_medium).to(device)    
    
    registration_network_high = SirenNet(input_dim=3, output_dim=3, hidden_dim=256, hidden_n_layers=5, last_layer_weights_small=True, is_last_linear=True, input_n_encoding_functions=6, input_omega_0=omega_0_high, hidden_omega_0=omega_0_high).to(device)    

    params = chain(registration_network_low.parameters(), registration_network_medium.parameters(), registration_network_high.parameters())
    optimizer = torch.optim.AdamW(lr=learning_rate, params=params)

    ncc = NCCLoss(is_tensor=False).to(device)
    lncc = LNCCLoss(is_tensor=False).to(device)
    jac = JacobianLossCoords(add_identity=False, is_tensor=False).to(device)

    pair_val = VectorCoords(coords=coords_init, scale_factor=1)
    val_loader = DataLoader(pair_val, batch_size=batch_size, shuffle=False)

    image_as_blocks = BlockCoords(coords=coords_init, scale_factor=0.5, block_size=block_size, steps=steps)
    block_loader = DataLoader(image_as_blocks, batch_size=1, shuffle=True)
    
    os.makedirs(results_path, exist_ok=True)    

    moving = torch.from_numpy(moving)
    moving_image = moving.to(device, dtype=torch.float32)
    fixed = torch.from_numpy(fixed)
    fixed_image = fixed.to(device, dtype=torch.float32)

    for epoch in range(epochs):

        stream = tqdm(block_loader)
        loop_monitor = MetricMonitor()
        
        for i, image_coords in enumerate(stream, start=0):

            image_coords.shape

            image_coords = image_coords.squeeze().to(device, dtype=torch.float32)

            flow_low, coords = registration_network_low(image_coords.view([np.prod(block_size), n_dims])) 

            flow_medium = registration_network_medium(coords, clone=False)
            
            flow_high = registration_network_high(coords, clone=False)
            
            flow_add = torch.add((flow_low + flow_medium + flow_high)/3, coords)

            moved_image_pixels = fast_trilinear_interpolation(moving_image, flow_add[:,0], flow_add[:,1], flow_add[:,2]).view(block_size)
            fixed_image_pixels = fast_trilinear_interpolation(fixed_image, coords[:,0], coords[:,1], coords[:,2]).view(block_size)

            moved_image_pixels = moved_image_pixels.unsqueeze(0).unsqueeze(0)
            fixed_image_pixels = fixed_image_pixels.unsqueeze(0).unsqueeze(0)

            ################################

            loss_all = torch.tensor(0)

            loss_ncc = (ncc(moved_image_pixels, fixed_image_pixels) + lncc(moved_image_pixels, fixed_image_pixels)) / 2
            loss_jac = jac(coords, flow_add)

            if alpha_ncc>0:

                loss_all = loss_all + alpha_ncc*loss_ncc        

            if alpha_jac>0:

                loss_all = loss_all + alpha_jac*loss_jac        

            loop_monitor.update('jac', loss_jac.item())
            loop_monitor.update('ncc', loss_ncc.item())
            loop_monitor.update('loss', loss_all.item())
            stream.set_description('Epoch: {epoch}. Train. {metric_monitor}'.format(epoch=epoch, metric_monitor=loop_monitor))

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

        if not (epoch+1) % epoch_save_visualization:

            with torch.no_grad():

                for k, image_coords in enumerate(val_loader):

                    image_coords = image_coords.squeeze().to(device, dtype=torch.float32)
                    
                    flow_low, coords = registration_network_low(image_coords) 

                    flow_medium = registration_network_medium(coords, clone=False)
            
                    flow_high = registration_network_high(coords, clone=False)

                    flow_add = torch.add((flow_low + flow_medium + flow_high)/3, coords)
                
                    flow_add = flow_add.cpu().detach().squeeze()

                    if k==0:

                        total_flow = flow_add

                    else:

                        total_flow = torch.cat((total_flow, flow_add), 0)

                temp_moved = fast_trilinear_interpolation(moving, total_flow[:,0], total_flow[:,1], total_flow[:,2]).view(moving.shape)  
                temp_moved = temp_moved.numpy().squeeze()

            fig, axes = plt.subplots(1,3, figsize=(18,6))
            axes[0].imshow(moving[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            axes[1].imshow(temp_moved[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            axes[2].imshow(fixed[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            
            axes[0].set_title('Moving')
            axes[1].set_title('Moved')
            axes[2].set_title('Fixed')            
            
            plt.show()
            
            np.save(results_path+str(epoch+1)+'_deformation_field.npy', total_flow.view(*moving.shape, 3).numpy())        

            save_state = {'state_dict_low': registration_network_low.state_dict(),
                          'state_dict_medium': registration_network_medium.state_dict(),
                          'state_dict_high': registration_network_high.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'loss_all': loss_all.item()}         
            
            torch.save(save_state, results_path + str(epoch+1) + '.pth.tar')               
            
            
            
def affine_registration_mindboggle(moving_path=None,
                                   fixed_path=None, 
                                   results_path=None,
                                   epochs=50,
                                   epoch_save_visualization=5,
                                   batch_size=64**3,
                                   block_size=[32,32,32],
                                   alpha_ncc=1,
                                   alpha_jac=0.1,
                                   steps=500,
                                   learning_rate=0.00001,
                                   device='cuda',
                                   omega_0=30,
                                   ):
    
    moving, moving_segmentations = load_mindboggle(moving_path)
    fixed, fixed_segmentations = load_mindboggle(fixed_path)
    
    n_dims = len(block_size)

    coords_init = make_coords_tensor(moving.shape, is_vector=False)
   
    registration_network = SirenNet(input_dim=3, output_dim=3, hidden_dim=256, hidden_n_layers=5, last_layer_weights_small=True, is_last_linear=True, input_n_encoding_functions=6, input_omega_0=omega_0, hidden_omega_0=omega_0).to(device)

    affine = AffineLayer().to(device)
    
    params = chain(affine.parameters(), registration_network.parameters())
    optimizer = torch.optim.AdamW(lr=learning_rate, params=params)

    ncc = NCCLoss(is_tensor=False).to(device)
    lncc = LNCCLoss(is_tensor=False).to(device)
    jac = JacobianLossCoords(add_identity=False, is_tensor=False).to(device)

    pair_val = VectorCoords(coords=coords_init, scale_factor=1)
    val_loader = DataLoader(pair_val, batch_size=batch_size, shuffle=False)

    image_as_blocks = BlockCoords(coords=coords_init, scale_factor=0.5, block_size=block_size, steps=steps)
    block_loader = DataLoader(image_as_blocks, batch_size=1, shuffle=True)
    
    os.makedirs(results_path, exist_ok=True)    

    moving = torch.from_numpy(moving)
    moving_image = moving.to(device, dtype=torch.float32)
    fixed = torch.from_numpy(fixed)
    fixed_image = fixed.to(device, dtype=torch.float32)

    for epoch in range(epochs):

        stream = tqdm(block_loader)
        loop_monitor = MetricMonitor()
        
        for i, image_coords in enumerate(stream, start=0):

            image_coords = image_coords.squeeze().to(device, dtype=torch.float32)

            flow, coords = registration_network(image_coords.view([np.prod(block_size), n_dims]))
            flow_add_affine = affine(coords)
            
            flow_add = flow + flow_add_affine

            moved_image_pixels = fast_trilinear_interpolation(moving_image, flow_add[:,0], flow_add[:,1], flow_add[:,2]).view(block_size)
            fixed_image_pixels = fast_trilinear_interpolation(fixed_image, coords[:,0], coords[:,1], coords[:,2]).view(block_size)

            moved_image_pixels = moved_image_pixels.unsqueeze(0).unsqueeze(0)
            fixed_image_pixels = fixed_image_pixels.unsqueeze(0).unsqueeze(0)

            ################################

            loss_all = torch.tensor(0)

            loss_ncc = (ncc(moved_image_pixels, fixed_image_pixels) + lncc(moved_image_pixels, fixed_image_pixels)) / 2
            loss_jac = jac(coords, flow_add)

            if alpha_ncc>0:

                loss_all = loss_all + alpha_ncc*loss_ncc        

            if alpha_jac>0:

                loss_all = loss_all + alpha_jac*loss_jac        

            loop_monitor.update('jac', loss_jac.item())
            loop_monitor.update('ncc', loss_ncc.item())
            loop_monitor.update('loss', loss_all.item())
            stream.set_description('Epoch: {epoch}. Train. {metric_monitor}'.format(epoch=epoch, metric_monitor=loop_monitor))

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

        if not (epoch+1) % epoch_save_visualization:

            with torch.no_grad():

                for k, image_coords in enumerate(val_loader):

                    image_coords = image_coords.squeeze().to(device, dtype=torch.float32)

                    flow, coords = registration_network(image_coords)
            
                    flow_add_affine = affine(coords)
                    flow_add = flow + flow_add_affine

                    flow_add = flow_add.cpu().detach().squeeze()

                    if k==0:

                        total_flow = flow_add

                    else:

                        total_flow = torch.cat((total_flow, flow_add), 0)

                temp_moved = fast_trilinear_interpolation(moving, total_flow[:,0], total_flow[:,1], total_flow[:,2]).view(moving.shape)  
                temp_moved = temp_moved.numpy().squeeze()

            fig, axes = plt.subplots(1,3, figsize=(18,6))
            axes[0].imshow(moving[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            axes[1].imshow(temp_moved[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            axes[2].imshow(fixed[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            
            axes[0].set_title('Moving')
            axes[1].set_title('Moved')
            axes[2].set_title('Fixed')            
            
            plt.show()
            
            np.save(results_path+str(epoch+1)+'_deformation_field.npy', total_flow.view(*moving.shape, 3).numpy())        

            save_state = {'state_dict': registration_network.state_dict(),
                          'state_affine': affine.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'loss_all': loss_all.item()}         
            
            torch.save(save_state, results_path + str(epoch+1) + '.pth.tar')               
            
            
def attention_siren_registration_mindboggle(moving_path=None,
                                           fixed_path=None, 
                                           results_path=None,
                                           epochs=50,
                                           epoch_save_visualization=5,
                                           batch_size=64**3,
                                           block_size=[32,32,32],
                                           alpha_ncc=1,
                                           alpha_jac=0.1,
                                           steps=500,
                                           learning_rate=0.00001,
                                           device='cuda',
                                           omega_0=30,
                                           delta_omega_0=20,
                                           ):
    
    moving, moving_segmentations = load_mindboggle(moving_path)
    fixed, fixed_segmentations = load_mindboggle(fixed_path)
    
    n_dims = len(block_size)

    coords_init = make_coords_tensor(moving.shape, is_vector=False)
   
    registration_network = AttentionSirenNet(input_dim=3, output_dim=3, hidden_dim=256, hidden_n_layers=5, last_layer_weights_small=True, is_last_linear=True, input_n_encoding_functions=6, input_omega_0=omega_0, hidden_omega_0=omega_0, delta_omega_0=delta_omega_0).to(device)

    optimizer = torch.optim.AdamW(lr=learning_rate, params=registration_network.parameters())

    ncc = NCCLoss(is_tensor=False).to(device)
    lncc = LNCCLoss(is_tensor=False).to(device)
    jac = JacobianLossCoords(add_identity=False, is_tensor=False).to(device)

    pair_val = VectorCoords(coords=coords_init, scale_factor=1)
    val_loader = DataLoader(pair_val, batch_size=batch_size, shuffle=False)

    image_as_blocks = BlockCoords(coords=coords_init, scale_factor=0.5, block_size=block_size, steps=steps)
    block_loader = DataLoader(image_as_blocks, batch_size=1, shuffle=True)
    
    os.makedirs(results_path, exist_ok=True)    

    moving = torch.from_numpy(moving)
    moving_image = moving.to(device, dtype=torch.float32)
    fixed = torch.from_numpy(fixed)
    fixed_image = fixed.to(device, dtype=torch.float32)

    for epoch in range(epochs):

        stream = tqdm(block_loader)
        loop_monitor = MetricMonitor()
        
        for i, image_coords in enumerate(stream, start=0):

            image_coords = image_coords.squeeze().to(device, dtype=torch.float32)

            flow, coords = registration_network(image_coords.view([np.prod(block_size), n_dims]))
            flow_add = torch.add(flow, coords)

            moved_image_pixels = fast_trilinear_interpolation(moving_image, flow_add[:,0], flow_add[:,1], flow_add[:,2]).view(block_size)
            fixed_image_pixels = fast_trilinear_interpolation(fixed_image, coords[:,0], coords[:,1], coords[:,2]).view(block_size)

            moved_image_pixels = moved_image_pixels.unsqueeze(0).unsqueeze(0)
            fixed_image_pixels = fixed_image_pixels.unsqueeze(0).unsqueeze(0)

            ################################

            loss_all = torch.tensor(0)

            loss_ncc = (ncc(moved_image_pixels, fixed_image_pixels) + lncc(moved_image_pixels, fixed_image_pixels)) / 2
            loss_jac = jac(coords, flow_add)

            if alpha_ncc>0:

                loss_all = loss_all + alpha_ncc*loss_ncc        

            if alpha_jac>0:

                loss_all = loss_all + alpha_jac*loss_jac        

            loop_monitor.update('jac', loss_jac.item())
            loop_monitor.update('ncc', loss_ncc.item())
            loop_monitor.update('loss', loss_all.item())
            stream.set_description('Epoch: {epoch}. Train. {metric_monitor}'.format(epoch=epoch, metric_monitor=loop_monitor))

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

        if not (epoch+1) % epoch_save_visualization:

            with torch.no_grad():

                for k, image_coords in enumerate(val_loader):

                    image_coords = image_coords.squeeze().to(device, dtype=torch.float32)

                    flow, coords = registration_network(image_coords)
                    flow_add = torch.add(flow, coords)

                    flow_add = flow_add.cpu().detach().squeeze()

                    if k==0:

                        total_flow = flow_add

                    else:

                        total_flow = torch.cat((total_flow, flow_add), 0)

                temp_moved = fast_trilinear_interpolation(moving, total_flow[:,0], total_flow[:,1], total_flow[:,2]).view(moving.shape)  
                temp_moved = temp_moved.numpy().squeeze()

            fig, axes = plt.subplots(1,3, figsize=(18,6))
            axes[0].imshow(moving[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            axes[1].imshow(temp_moved[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            axes[2].imshow(fixed[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            
            axes[0].set_title('Moving')
            axes[1].set_title('Moved')
            axes[2].set_title('Fixed')            
            
            plt.show()
            
            np.save(results_path+str(epoch+1)+'_deformation_field.npy', total_flow.view(*moving.shape, 3).numpy())        

            save_state = {'state_dict': registration_network.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'loss_all': loss_all.item()}         
            
            torch.save(save_state, results_path + str(epoch+1) + '.pth.tar')               
            
            
            
            
def snake_registration_mindboggle(moving_path=None,
                                  fixed_path=None, 
                                  results_path=None,
                                  epochs=50,
                                  epoch_save_visualization=5,
                                  batch_size=64**3,
                                  block_size=[32,32,32],
                                  alpha_ncc=1,
                                  alpha_jac=0.1,
                                  steps=500,
                                  learning_rate=0.00001,
                                  device='cuda',
                                  omega_0=30,
                                  ):
    
    moving, moving_segmentations = load_mindboggle(moving_path)
    fixed, fixed_segmentations = load_mindboggle(fixed_path)
    
    n_dims = len(block_size)

    coords_init = make_coords_tensor(moving.shape, is_vector=False)
   
    registration_network = SnakeNet(input_dim=3, output_dim=3, hidden_dim=256, hidden_n_layers=5, last_layer_weights_small=True, is_last_linear=True, input_n_encoding_functions=6, input_omega_0=omega_0, hidden_omega_0=omega_0).to(device)

    optimizer = torch.optim.AdamW(lr=learning_rate, params=registration_network.parameters())

    ncc = NCCLoss(is_tensor=False).to(device)
    lncc = LNCCLoss(is_tensor=False).to(device)
    jac = JacobianLossCoords(add_identity=False, is_tensor=False).to(device)

    pair_val = VectorCoords(coords=coords_init, scale_factor=1)
    val_loader = DataLoader(pair_val, batch_size=batch_size, shuffle=False)

    image_as_blocks = BlockCoords(coords=coords_init, scale_factor=0.5, block_size=block_size, steps=steps)
    block_loader = DataLoader(image_as_blocks, batch_size=1, shuffle=True)
    
    os.makedirs(results_path, exist_ok=True)    

    moving = torch.from_numpy(moving)
    moving_image = moving.to(device, dtype=torch.float32)
    fixed = torch.from_numpy(fixed)
    fixed_image = fixed.to(device, dtype=torch.float32)

    for epoch in range(epochs):

        stream = tqdm(block_loader)
        loop_monitor = MetricMonitor()
        
        for i, image_coords in enumerate(stream, start=0):

            image_coords = image_coords.squeeze().to(device, dtype=torch.float32)

            flow, coords = registration_network(image_coords.view([np.prod(block_size), n_dims]))
            flow_add = torch.add(flow, coords)

            moved_image_pixels = fast_trilinear_interpolation(moving_image, flow_add[:,0], flow_add[:,1], flow_add[:,2]).view(block_size)
            fixed_image_pixels = fast_trilinear_interpolation(fixed_image, coords[:,0], coords[:,1], coords[:,2]).view(block_size)

            moved_image_pixels = moved_image_pixels.unsqueeze(0).unsqueeze(0)
            fixed_image_pixels = fixed_image_pixels.unsqueeze(0).unsqueeze(0)

            ################################

            loss_all = torch.tensor(0)

            loss_ncc = (ncc(moved_image_pixels, fixed_image_pixels) + lncc(moved_image_pixels, fixed_image_pixels)) / 2
            loss_jac = jac(coords, flow_add)

            if alpha_ncc>0:

                loss_all = loss_all + alpha_ncc*loss_ncc        

            if alpha_jac>0:

                loss_all = loss_all + alpha_jac*loss_jac        

            loop_monitor.update('jac', loss_jac.item())
            loop_monitor.update('ncc', loss_ncc.item())
            loop_monitor.update('loss', loss_all.item())
            stream.set_description('Epoch: {epoch}. Train. {metric_monitor}'.format(epoch=epoch, metric_monitor=loop_monitor))

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

        if not (epoch+1) % epoch_save_visualization:

            with torch.no_grad():

                for k, image_coords in enumerate(val_loader):

                    image_coords = image_coords.squeeze().to(device, dtype=torch.float32)

                    flow, coords = registration_network(image_coords)
                    flow_add = torch.add(flow, coords)

                    flow_add = flow_add.cpu().detach().squeeze()

                    if k==0:

                        total_flow = flow_add

                    else:

                        total_flow = torch.cat((total_flow, flow_add), 0)

                temp_moved = fast_trilinear_interpolation(moving, total_flow[:,0], total_flow[:,1], total_flow[:,2]).view(moving.shape)  
                temp_moved = temp_moved.numpy().squeeze()

            fig, axes = plt.subplots(1,3, figsize=(18,6))
            axes[0].imshow(moving[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            axes[1].imshow(temp_moved[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            axes[2].imshow(fixed[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            
            axes[0].set_title('Moving')
            axes[1].set_title('Moved')
            axes[2].set_title('Fixed')            
            
            plt.show()
            
            np.save(results_path+str(epoch+1)+'_deformation_field.npy', total_flow.view(*moving.shape, 3).numpy())        

            save_state = {'state_dict': registration_network.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'loss_all': loss_all.item()}         
            
            torch.save(save_state, results_path + str(epoch+1) + '.pth.tar')               
            
            
            
def sineplus_registration_mindboggle(moving_path=None,
                                     fixed_path=None, 
                                     results_path=None,
                                     epochs=50,
                                     epoch_save_visualization=5,
                                     batch_size=64**3,
                                     block_size=[32,32,32],
                                     alpha_ncc=1,
                                     alpha_jac=0.1,
                                     steps=500,
                                     learning_rate=0.00001,
                                     device='cuda',
                                     omega_0=30,
                                     ):
    
    moving, moving_segmentations = load_mindboggle(moving_path)
    fixed, fixed_segmentations = load_mindboggle(fixed_path)
    
    n_dims = len(block_size)

    coords_init = make_coords_tensor(moving.shape, is_vector=False)
   
    registration_network = SineplusNet(input_dim=3, output_dim=3, hidden_dim=256, hidden_n_layers=5, last_layer_weights_small=True, is_last_linear=True, input_n_encoding_functions=6, input_omega_0=omega_0, hidden_omega_0=omega_0).to(device)

    optimizer = torch.optim.AdamW(lr=learning_rate, params=registration_network.parameters())

    ncc = NCCLoss(is_tensor=False).to(device)
    lncc = LNCCLoss(is_tensor=False).to(device)
    jac = JacobianLossCoords(add_identity=False, is_tensor=False).to(device)

    pair_val = VectorCoords(coords=coords_init, scale_factor=1)
    val_loader = DataLoader(pair_val, batch_size=batch_size, shuffle=False)

    image_as_blocks = BlockCoords(coords=coords_init, scale_factor=0.5, block_size=block_size, steps=steps)
    block_loader = DataLoader(image_as_blocks, batch_size=1, shuffle=True)
    
    os.makedirs(results_path, exist_ok=True)    

    moving = torch.from_numpy(moving)
    moving_image = moving.to(device, dtype=torch.float32)
    fixed = torch.from_numpy(fixed)
    fixed_image = fixed.to(device, dtype=torch.float32)

    for epoch in range(epochs):

        stream = tqdm(block_loader)
        loop_monitor = MetricMonitor()
        
        for i, image_coords in enumerate(stream, start=0):

            image_coords = image_coords.squeeze().to(device, dtype=torch.float32)

            flow, coords = registration_network(image_coords.view([np.prod(block_size), n_dims]))
            flow_add = torch.add(flow, coords)

            moved_image_pixels = fast_trilinear_interpolation(moving_image, flow_add[:,0], flow_add[:,1], flow_add[:,2]).view(block_size)
            fixed_image_pixels = fast_trilinear_interpolation(fixed_image, coords[:,0], coords[:,1], coords[:,2]).view(block_size)

            moved_image_pixels = moved_image_pixels.unsqueeze(0).unsqueeze(0)
            fixed_image_pixels = fixed_image_pixels.unsqueeze(0).unsqueeze(0)

            ################################

            loss_all = torch.tensor(0)

            loss_ncc = (ncc(moved_image_pixels, fixed_image_pixels) + lncc(moved_image_pixels, fixed_image_pixels)) / 2
            loss_jac = jac(coords, flow_add)

            if alpha_ncc>0:

                loss_all = loss_all + alpha_ncc*loss_ncc        

            if alpha_jac>0:

                loss_all = loss_all + alpha_jac*loss_jac        

            loop_monitor.update('jac', loss_jac.item())
            loop_monitor.update('ncc', loss_ncc.item())
            loop_monitor.update('loss', loss_all.item())
            stream.set_description('Epoch: {epoch}. Train. {metric_monitor}'.format(epoch=epoch, metric_monitor=loop_monitor))

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

        if not (epoch+1) % epoch_save_visualization:

            with torch.no_grad():

                for k, image_coords in enumerate(val_loader):

                    image_coords = image_coords.squeeze().to(device, dtype=torch.float32)

                    flow, coords = registration_network(image_coords)
                    flow_add = torch.add(flow, coords)

                    flow_add = flow_add.cpu().detach().squeeze()

                    if k==0:

                        total_flow = flow_add

                    else:

                        total_flow = torch.cat((total_flow, flow_add), 0)

                temp_moved = fast_trilinear_interpolation(moving, total_flow[:,0], total_flow[:,1], total_flow[:,2]).view(moving.shape)  
                temp_moved = temp_moved.numpy().squeeze()

            fig, axes = plt.subplots(1,3, figsize=(18,6))
            axes[0].imshow(moving[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            axes[1].imshow(temp_moved[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            axes[2].imshow(fixed[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            
            axes[0].set_title('Moving')
            axes[1].set_title('Moved')
            axes[2].set_title('Fixed')            
            
            plt.show()
            
            np.save(results_path+str(epoch+1)+'_deformation_field.npy', total_flow.view(*moving.shape, 3).numpy())        

            save_state = {'state_dict': registration_network.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'loss_all': loss_all.item()}         
            
            torch.save(save_state, results_path + str(epoch+1) + '.pth.tar')                 
            
            
            
            
def chirp_registration_mindboggle(moving_path=None,
                                  fixed_path=None, 
                                  results_path=None,
                                  epochs=50,
                                  epoch_save_visualization=5,
                                  batch_size=64**3,
                                  block_size=[32,32,32],
                                  alpha_ncc=1,
                                  alpha_jac=0.1,
                                  steps=500,
                                  learning_rate=0.00001,
                                  device='cuda',
                                  omega_0=30,
                                  ):
    
    moving, moving_segmentations = load_mindboggle(moving_path)
    fixed, fixed_segmentations = load_mindboggle(fixed_path)
    
    n_dims = len(block_size)

    coords_init = make_coords_tensor(moving.shape, is_vector=False)
   
    registration_network = ChirpNet(input_dim=3, output_dim=3, hidden_dim=256, hidden_n_layers=5, last_layer_weights_small=True, is_last_linear=True, input_n_encoding_functions=6, input_omega_0=omega_0, hidden_omega_0=omega_0).to(device)

    optimizer = torch.optim.AdamW(lr=learning_rate, params=registration_network.parameters())

    ncc = NCCLoss(is_tensor=False).to(device)
    lncc = LNCCLoss(is_tensor=False).to(device)
    jac = JacobianLossCoords(add_identity=False, is_tensor=False).to(device)

    pair_val = VectorCoords(coords=coords_init, scale_factor=1)
    val_loader = DataLoader(pair_val, batch_size=batch_size, shuffle=False)

    image_as_blocks = BlockCoords(coords=coords_init, scale_factor=0.5, block_size=block_size, steps=steps)
    block_loader = DataLoader(image_as_blocks, batch_size=1, shuffle=True)
    
    os.makedirs(results_path, exist_ok=True)    

    moving = torch.from_numpy(moving)
    moving_image = moving.to(device, dtype=torch.float32)
    fixed = torch.from_numpy(fixed)
    fixed_image = fixed.to(device, dtype=torch.float32)

    for epoch in range(epochs):

        stream = tqdm(block_loader)
        loop_monitor = MetricMonitor()
        
        for i, image_coords in enumerate(stream, start=0):

            image_coords = image_coords.squeeze().to(device, dtype=torch.float32)

            flow, coords = registration_network(image_coords.view([np.prod(block_size), n_dims]))
            flow_add = torch.add(flow, coords)

            moved_image_pixels = fast_trilinear_interpolation(moving_image, flow_add[:,0], flow_add[:,1], flow_add[:,2]).view(block_size)
            fixed_image_pixels = fast_trilinear_interpolation(fixed_image, coords[:,0], coords[:,1], coords[:,2]).view(block_size)

            moved_image_pixels = moved_image_pixels.unsqueeze(0).unsqueeze(0)
            fixed_image_pixels = fixed_image_pixels.unsqueeze(0).unsqueeze(0)

            ################################

            loss_all = torch.tensor(0)

            loss_ncc = (ncc(moved_image_pixels, fixed_image_pixels) + lncc(moved_image_pixels, fixed_image_pixels)) / 2
            loss_jac = jac(coords, flow_add)

            if alpha_ncc>0:

                loss_all = loss_all + alpha_ncc*loss_ncc        

            if alpha_jac>0:

                loss_all = loss_all + alpha_jac*loss_jac        

            loop_monitor.update('jac', loss_jac.item())
            loop_monitor.update('ncc', loss_ncc.item())
            loop_monitor.update('loss', loss_all.item())
            stream.set_description('Epoch: {epoch}. Train. {metric_monitor}'.format(epoch=epoch, metric_monitor=loop_monitor))

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

        if not (epoch+1) % epoch_save_visualization:

            with torch.no_grad():

                for k, image_coords in enumerate(val_loader):

                    image_coords = image_coords.squeeze().to(device, dtype=torch.float32)

                    flow, coords = registration_network(image_coords)
                    flow_add = torch.add(flow, coords)

                    flow_add = flow_add.cpu().detach().squeeze()

                    if k==0:

                        total_flow = flow_add

                    else:

                        total_flow = torch.cat((total_flow, flow_add), 0)

                temp_moved = fast_trilinear_interpolation(moving, total_flow[:,0], total_flow[:,1], total_flow[:,2]).view(moving.shape)  
                temp_moved = temp_moved.numpy().squeeze()

            fig, axes = plt.subplots(1,3, figsize=(18,6))
            axes[0].imshow(moving[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            axes[1].imshow(temp_moved[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            axes[2].imshow(fixed[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            
            axes[0].set_title('Moving')
            axes[1].set_title('Moved')
            axes[2].set_title('Fixed')            
            
            plt.show()
            
            np.save(results_path+str(epoch+1)+'_deformation_field.npy', total_flow.view(*moving.shape, 3).numpy())        

            save_state = {'state_dict': registration_network.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'loss_all': loss_all.item()}         
            
            torch.save(save_state, results_path + str(epoch+1) + '.pth.tar')                 
                        
            
            
def morlet_registration_mindboggle(moving_path=None,
                                     fixed_path=None, 
                                     results_path=None,
                                     epochs=50,
                                     epoch_save_visualization=5,
                                     batch_size=64**3,
                                     block_size=[32,32,32],
                                     alpha_ncc=1,
                                     alpha_jac=0.1,
                                     steps=500,
                                     learning_rate=0.00001,
                                     device='cuda',
                                     omega_0=30,
                                     ):
    
    moving, moving_segmentations = load_mindboggle(moving_path)
    fixed, fixed_segmentations = load_mindboggle(fixed_path)
    
    n_dims = len(block_size)

    coords_init = make_coords_tensor(moving.shape, is_vector=False)
   
    registration_network = MorletNet(input_dim=3, output_dim=3, hidden_dim=256, hidden_n_layers=5, last_layer_weights_small=True, is_last_linear=True, input_n_encoding_functions=6, input_omega_0=omega_0, hidden_omega_0=omega_0).to(device)

    optimizer = torch.optim.AdamW(lr=learning_rate, params=registration_network.parameters())

    ncc = NCCLoss(is_tensor=False).to(device)
    lncc = LNCCLoss(is_tensor=False).to(device)
    jac = JacobianLossCoords(add_identity=False, is_tensor=False).to(device)

    pair_val = VectorCoords(coords=coords_init, scale_factor=1)
    val_loader = DataLoader(pair_val, batch_size=batch_size, shuffle=False)

    image_as_blocks = BlockCoords(coords=coords_init, scale_factor=0.5, block_size=block_size, steps=steps)
    block_loader = DataLoader(image_as_blocks, batch_size=1, shuffle=True)
    
    os.makedirs(results_path, exist_ok=True)    

    moving = torch.from_numpy(moving)
    moving_image = moving.to(device, dtype=torch.float32)
    fixed = torch.from_numpy(fixed)
    fixed_image = fixed.to(device, dtype=torch.float32)

    for epoch in range(epochs):

        stream = tqdm(block_loader)
        loop_monitor = MetricMonitor()
        
        for i, image_coords in enumerate(stream, start=0):

            image_coords = image_coords.squeeze().to(device, dtype=torch.float32)

            flow, coords = registration_network(image_coords.view([np.prod(block_size), n_dims]))
            flow_add = torch.add(flow, coords)

            moved_image_pixels = fast_trilinear_interpolation(moving_image, flow_add[:,0], flow_add[:,1], flow_add[:,2]).view(block_size)
            fixed_image_pixels = fast_trilinear_interpolation(fixed_image, coords[:,0], coords[:,1], coords[:,2]).view(block_size)

            moved_image_pixels = moved_image_pixels.unsqueeze(0).unsqueeze(0)
            fixed_image_pixels = fixed_image_pixels.unsqueeze(0).unsqueeze(0)

            ################################

            loss_all = torch.tensor(0)

            loss_ncc = (ncc(moved_image_pixels, fixed_image_pixels) + lncc(moved_image_pixels, fixed_image_pixels)) / 2
            loss_jac = jac(coords, flow_add)

            if alpha_ncc>0:

                loss_all = loss_all + alpha_ncc*loss_ncc        

            if alpha_jac>0:

                loss_all = loss_all + alpha_jac*loss_jac        

            loop_monitor.update('jac', loss_jac.item())
            loop_monitor.update('ncc', loss_ncc.item())
            loop_monitor.update('loss', loss_all.item())
            stream.set_description('Epoch: {epoch}. Train. {metric_monitor}'.format(epoch=epoch, metric_monitor=loop_monitor))

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

        if not (epoch+1) % epoch_save_visualization:

            with torch.no_grad():

                for k, image_coords in enumerate(val_loader):

                    image_coords = image_coords.squeeze().to(device, dtype=torch.float32)

                    flow, coords = registration_network(image_coords)
                    flow_add = torch.add(flow, coords)

                    flow_add = flow_add.cpu().detach().squeeze()

                    if k==0:

                        total_flow = flow_add

                    else:

                        total_flow = torch.cat((total_flow, flow_add), 0)

                temp_moved = fast_trilinear_interpolation(moving, total_flow[:,0], total_flow[:,1], total_flow[:,2]).view(moving.shape)  
                temp_moved = temp_moved.numpy().squeeze()

            fig, axes = plt.subplots(1,3, figsize=(18,6))
            axes[0].imshow(moving[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            axes[1].imshow(temp_moved[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            axes[2].imshow(fixed[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            
            axes[0].set_title('Moving')
            axes[1].set_title('Moved')
            axes[2].set_title('Fixed')            
            
            plt.show()
            
            np.save(results_path+str(epoch+1)+'_deformation_field.npy', total_flow.view(*moving.shape, 3).numpy())        

            save_state = {'state_dict': registration_network.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'loss_all': loss_all.item()}         
            
            torch.save(save_state, results_path + str(epoch+1) + '.pth.tar')                 
                        
            
            
def modulation_registration_mindboggle(moving_path=None,
                                       fixed_path=None, 
                                       results_path=None,
                                       epochs=50,
                                       epoch_save_visualization=5,
                                       batch_size=64**3,
                                       block_size=[32,32,32],
                                       alpha_ncc=1,
                                       alpha_jac=0.1,
                                       steps=500,
                                       learning_rate=0.00001,
                                       device='cuda',
                                       omega_0=30,
                                       trainable_omega_0=False,
                                       ):
    
    moving, moving_segmentations = load_mindboggle(moving_path)
    fixed, fixed_segmentations = load_mindboggle(fixed_path)
    
    n_dims = len(block_size)

    coords_init = make_coords_tensor(moving.shape, is_vector=False)
   
    registration_network = ModulationNet(input_dim=3, output_dim=3, hidden_dim=256, hidden_n_layers=5, last_layer_weights_small=True, is_last_linear=True, input_n_encoding_functions=6, input_omega_0=omega_0, hidden_omega_0=omega_0).to(device)

    optimizer = torch.optim.AdamW(lr=learning_rate, params=registration_network.parameters())

    ncc = NCCLoss(is_tensor=False).to(device)
    lncc = LNCCLoss(is_tensor=False).to(device)
    jac = JacobianLossCoords(add_identity=False, is_tensor=False).to(device)

    pair_val = VectorCoords(coords=coords_init, scale_factor=1)
    val_loader = DataLoader(pair_val, batch_size=batch_size, shuffle=False)

    image_as_blocks = BlockCoords(coords=coords_init, scale_factor=0.5, block_size=block_size, steps=steps)
    block_loader = DataLoader(image_as_blocks, batch_size=1, shuffle=True)
    
    os.makedirs(results_path, exist_ok=True)    

    moving = torch.from_numpy(moving)
    moving_image = moving.to(device, dtype=torch.float32)
    fixed = torch.from_numpy(fixed)
    fixed_image = fixed.to(device, dtype=torch.float32)

    for epoch in range(epochs):

        stream = tqdm(block_loader)
        loop_monitor = MetricMonitor()
        
        for i, image_coords in enumerate(stream, start=0):

            image_coords = image_coords.squeeze().to(device, dtype=torch.float32)

            flow, coords = registration_network(image_coords.view([np.prod(block_size), n_dims]))
            flow_add = torch.add(flow, coords)

            moved_image_pixels = fast_trilinear_interpolation(moving_image, flow_add[:,0], flow_add[:,1], flow_add[:,2]).view(block_size)
            fixed_image_pixels = fast_trilinear_interpolation(fixed_image, coords[:,0], coords[:,1], coords[:,2]).view(block_size)

            moved_image_pixels = moved_image_pixels.unsqueeze(0).unsqueeze(0)
            fixed_image_pixels = fixed_image_pixels.unsqueeze(0).unsqueeze(0)

            ################################

            loss_all = torch.tensor(0)

            loss_ncc = (ncc(moved_image_pixels, fixed_image_pixels) + lncc(moved_image_pixels, fixed_image_pixels)) / 2
            loss_jac = jac(coords, flow_add)

            if alpha_ncc>0:

                loss_all = loss_all + alpha_ncc*loss_ncc        

            if alpha_jac>0:

                loss_all = loss_all + alpha_jac*loss_jac        

            loop_monitor.update('jac', loss_jac.item())
            loop_monitor.update('ncc', loss_ncc.item())
            loop_monitor.update('loss', loss_all.item())
            stream.set_description('Epoch: {epoch}. Train. {metric_monitor}'.format(epoch=epoch, metric_monitor=loop_monitor))

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

        if not (epoch+1) % epoch_save_visualization:

            with torch.no_grad():

                for k, image_coords in enumerate(val_loader):

                    image_coords = image_coords.squeeze().to(device, dtype=torch.float32)

                    flow, coords = registration_network(image_coords)
                    flow_add = torch.add(flow, coords)

                    flow_add = flow_add.cpu().detach().squeeze()

                    if k==0:

                        total_flow = flow_add

                    else:

                        total_flow = torch.cat((total_flow, flow_add), 0)

                temp_moved = fast_trilinear_interpolation(moving, total_flow[:,0], total_flow[:,1], total_flow[:,2]).view(moving.shape)  
                temp_moved = temp_moved.numpy().squeeze()

            fig, axes = plt.subplots(1,3, figsize=(18,6))
            axes[0].imshow(moving[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            axes[1].imshow(temp_moved[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            axes[2].imshow(fixed[:, :, 100], cmap='gray', vmin=-1, vmax=1)
            
            axes[0].set_title('Moving')
            axes[1].set_title('Moved')
            axes[2].set_title('Fixed')            
            
            plt.show()
            
            np.save(results_path+str(epoch+1)+'_deformation_field.npy', total_flow.view(*moving.shape, 3).numpy())        

            save_state = {'state_dict': registration_network.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'loss_all': loss_all.item()}         
            
            torch.save(save_state, results_path + str(epoch+1) + '.pth.tar')   
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            