from clifford.g3 import *
import open3d as o3d
from CGENNs.algebra.cliffordalgebra import CliffordAlgebra
from cliffordlayers.cliffordlayers.models.utils import partialclass
from cliffordlayers.cliffordlayers.nn.modules.cliffordconv import CliffordConv1d
from cliffordlayers.cliffordlayers.models.basic.threed import (
    CliffordMaxwellNet3d,
    CliffordConv3d,
    CliffordFourierBasicBlock3d
)
from cliffordlayers.cliffordlayers.nn.modules.cliffordfourier import CliffordSpectralConv3d
from typing import Callable, Union
from cliffordlayers.cliffordlayers.nn.modules.batchnorm import CliffordBatchNorm3d
from cliffordlayers.cliffordlayers.nn.modules.groupnorm import CliffordGroupNorm3d
import torch.nn.functional as F
import vtk
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import random
import meshio
from neuralop.utils import UnitGaussianNormalizer
from neuralop.models import TFNO
from prettytable import PrettyTable
from typing import Tuple
import os
import sys
import gc
import math



import sys


SEED  = 1
random.seed(SEED)

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            params = 0
        else:
            params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table, flush = True)
    print(f"Total Trainable Params: {total_params}", flush = True)
    return total_params


#This function is used to map our input point cloud N x 3, with N about 100k points
#into a regular grid M x M x M x 3, with resolution M. This is done in analogy with the GINO pipeline and because
#the fourier neural operator needs a spatial grid in input

def encode_point_cloud(point_cloud, grid_size, press, velo, biv):
    # Initialize the grids with fixed resolution
    grid = np.zeros((grid_size, grid_size, grid_size, 3))
    velocity = np.zeros((grid_size, grid_size, grid_size))
    mask = np.zeros((grid_size, grid_size, grid_size))
    pressure = np.zeros((grid_size, grid_size, grid_size, 1))
    bivectors = np.zeros((grid_size, grid_size, grid_size, 3))

    # Calculate the bounding box of the point cloud
    max_coords = np.max(point_cloud, axis=0)
    min_coords = np.min(point_cloud, axis=0)
    cloud_range = max_coords - min_coords

    # Scale points to fit within the grid
    scale = (grid_size - 1) / cloud_range
    scaled_points = (point_cloud - min_coords) * scale

    # Calculate the integer indices for the scaled points
    indices = np.floor(scaled_points).astype(int)

    # Ensure indices are within grid bounds
    indices = np.clip(indices, 0, grid_size - 1)


    # Assign points to the grid
    for idx, point, p, b in zip(indices, point_cloud, press, biv):
        if mask[idx[0], idx[1], idx[2]] == 0:  # Only assign if the cell is empty
            grid[idx[0], idx[1], idx[2]] = point
            pressure[idx[0], idx[1], idx[2]] = p
            velocity[idx[0], idx[1], idx[2]] = velo
            mask[idx[0], idx[1], idx[2]] = 1
            bivectors[idx[0], idx[1], idx[2]] = b

            

    return velo, grid, bivectors, mask, pressure.squeeze()


#used for plotting 3D point clouds with equal x - y - z axes.

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/3
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)




def calculate_dual(points):

    points = points.reshape((-1, 3))
    #nonzeroidx = np.where((points[:, 0] != 0) | (points[:, 1] != 0) | (points[:, 2] != 0))[0]

    dual = np.zeros((len(points), 3))

    for idx in range(len(points)):
        p = points[idx,0]*e1 + points[idx,1]*e2 + points[idx,2]*e3
        pdual = p.dual()
        dual[idx,0] =pdual[4]
        dual[idx,1] =pdual[5]
        dual[idx,2] =pdual[6]

    
    dual = dual.reshape((-1, 3))

    return dual

#this is a function in Geometric Algebra, might be a bit difficult to understand. We use it to compute the dual to the point cloud. 
#If a point cloud is composed by vectors P = p1e1 + p2e2 + p3e3, their dual will be a bivector, which is the perpendicular oriented plane with respect to P.
#We do it for 2 reasons: 1 to have a bivector component in the network (see below) and 2 to have some sort of "surface" information.



#Function to check if the meshes are watertight. In the GINO paper we have that for the Shapenet Car dataset (the one we are using) there 
#are 889 car shapes. In GINO, they only consider the watertight meshes (why is this relevant if the input is a point cloud?), 
#which they say to be 611. They use 500 for training and 111 for testing (no validation, suspicious). 
#I tried different approaches, with pyvista, trimesh, etc. Most of the times it says that *all* the meshes are watertight. In the 
#code for GINO, they don't specify how they check that, but they have a file .txt with the list of watertight meshes which is not
# included. I might email them about it.




M = 80
spatial_resolution = (M, M, M)

reader = vtk.vtkXMLPolyDataReader()

Xtrain = np.zeros((500, M, M, M, 8))
Ytrain = np.zeros((500, M, M, M, 1))


traindatapath = "ahmed/train"

i = 0

point_clouds = []
pressures = []
velocities = []
bivectors = []
rnumbers = []

for root, dirs, files in os.walk(traindatapath):
    for name in files:
        if name[:4] == 'mesh':
            

            print(root + '/' + name)
            pc = o3d.io.read_triangle_mesh(root + '/' + name)

            pc.compute_triangle_normals()
            normals = np.asarray(pc.triangle_normals)
            bivectors.append(calculate_dual(normals))

            
            points = np.asarray(pc.vertices)
            triangles = np.asarray(pc.triangles)
            
            # Initialize an array to store the centroids
            centroids = np.zeros((len(triangles), 3))
            
            # Calculate the centroid for each triangle
            for k, triangle in enumerate(triangles):
                v1, v2, v3 = triangle
                centroid = (points[v1] + points[v2] + points[v3]) / 3.0
                centroids[k] = centroid

            #print(root + '/' + name)
            #print(root + '/press' + name[4:-3] + 'npy')
            point_clouds.append(centroids)

            press = np.load(root + '/press' + name[4:-3] + 'npy')
            #print(press)

            pressures.append(press)
    

            RN = torch.load(root + '/info' + name[4:-3] + 'pt')



            tensor = torch.load(root + '/info' + name[4:-3] + 'pt')

            # Print the tensor
            velo = tensor['velocity']

            velocities.append(velo)

            i += 1
            

def normalize_point_clouds(point_clouds):
    # Concatenate all point clouds into a single array

    
    centered_point_clouds = []
    for pc in point_clouds:
        centroid = np.mean(pc, axis=0)
        centered_pc = pc - centroid
        centered_point_clouds.append(centered_pc)
    
    all_points = np.concatenate(point_clouds, axis=0)

    print(all_points.shape)
    
    # Compute global min and max values
    global_min = np.min(all_points, axis=0)
    global_max = np.max(all_points, axis=0)
    
    # Normalize each point cloud
    normalized_point_clouds = [(pc - np.min(pc)) / (np.max(pc) - np.min(pc)) for pc in point_clouds]
    
    # Scale to [-1, 1]
    normalized_point_clouds = [(2 * pc) - 1 for pc in normalized_point_clouds]
    
    return normalized_point_clouds




def normalize_pressure(point_clouds):
    # Concatenate all point clouds into a single array
    all_points = np.concatenate(point_clouds, axis=0)
    
    # Compute global min and max values
    global_mean = np.mean(all_points, axis=0)
    global_std = np.std(all_points, axis=0)
    
    # Normalize each point cloud
    normalized_point_clouds = [(pc - global_mean) / global_std for pc in point_clouds]
    
    
    return normalized_point_clouds, global_mean, global_std



normalized_point_clouds = np.array(normalize_point_clouds(point_clouds))
normalized_pressures, MU, STD = np.array(normalize_pressure(pressures))


velocities = np.array(velocities)
vel_min = np.min(velocities)
vel_max = np.max(velocities)
velocities = (velocities - vel_min)/(vel_max - vel_min)


for i in range(len(normalized_point_clouds)):
    Xtrain[i,:,:,:,0], Xtrain[i,:,:,:,1:4], Xtrain[i,:,:,:,4:7], Xtrain[i,:,:,:,7], Ytrain[i,:,:,:,0]  = encode_point_cloud(normalized_point_clouds[i], M, normalized_pressures[i], velocities[i], bivectors[i])



for IDX in range(3):

    volume_points = Xtrain[IDX, :, :, :, 1:4].reshape(-1, 3)
    #print(volume_points.shape, flush = True)
    press = Ytrain[IDX, :, :, :, 0].reshape(-1, 1)

    #print(volume_points)

    fig=plt.figure(figsize=(5,5))
    ax = fig.add_subplot(projection = "3d")
    # Plot the points
    ss = ax.scatter(volume_points[:, 2], volume_points[:, 0], volume_points[:, 1], c = press[:,0], marker='o', s=8)
    ax.set_title('Sanity Check - Grid must be equal to PC')
    axisEqual3D(ax)
    plt.colorbar(ss)
    plt.show()
    plt.savefig("PLOT_PRESS_TRAIN_ahmed" + str(IDX) + ".png")


Xtest = np.zeros((51, M, M, M, 8))
Ytest = np.zeros((51, M, M, M, 1))


traindatapath = "ahmed/test"

i = 0

point_clouds = []
pressures = []
velocities = []
bivectors = []

for root, dirs, files in os.walk(traindatapath):
    for name in files:
        if name[:4] == 'mesh':
            

            #print(root + '/' + name)
            pc = o3d.io.read_triangle_mesh(root + '/' + name)

            pc.compute_triangle_normals()
            normals = np.asarray(pc.triangle_normals)
            bivectors.append(calculate_dual(normals))

            
            points = np.asarray(pc.vertices)
            triangles = np.asarray(pc.triangles)
            
            # Initialize an array to store the centroids
            centroids = np.zeros((len(triangles), 3))
            
            # Calculate the centroid for each triangle
            for k, triangle in enumerate(triangles):
                v1, v2, v3 = triangle
                centroid = (points[v1] + points[v2] + points[v3]) / 3.0
                centroids[k] = centroid

            point_clouds.append(centroids)

            press = np.load(root + '/press' + name[4:-3] + 'npy')
            #print(press)

            pressures.append(press)

            tensor = torch.load(root + '/info' + name[4:-3] + 'pt')

            # Print the tensor
            velo = tensor['velocity']

            velocities.append(velo)

            i += 1


normalized_point_clouds = np.array(normalize_point_clouds(point_clouds))
normalized_pressures = [(pc - MU) / STD for pc in pressures]
velocities = (velocities - vel_min)/(vel_max - vel_min)

for i in range(len(normalized_point_clouds)):
    Xtest[i,:,:,:,0], Xtest[i,:,:,:,1:4], Xtest[i,:,:,:,4:7], Xtest[i,:,:,:,7], Ytest[i,:,:,:,0]  = encode_point_cloud(normalized_point_clouds[i], M, normalized_pressures[i], velocities[i], bivectors[i])


for IDX in range(3):

    volume_points = Xtest[IDX, :, :, :, 1:4].reshape(-1, 3)
    #print(volume_points.shape, flush = True)
    press = Ytest[IDX, :, :, :, 0].reshape(-1, 1)

    #print(volume_points)

    fig=plt.figure(figsize=(5,5))
    ax = fig.add_subplot(projection = "3d")
    # Plot the points
    ss = ax.scatter(volume_points[:, 2], volume_points[:, 0], volume_points[:, 1], c = press[:,0], marker='o', s=8)
    ax.set_title('Sanity Check - Grid must be equal to PC')
    axisEqual3D(ax)
    plt.colorbar(ss)
    plt.show()
    plt.savefig("PLOT_PRESS_TEST_ahmed" + str(IDX) + ".png")

print("DONE")


#defining the algebra, we work in 3D with signature 1 1 1 (i.e. 3 basis vectors, e1, e2 and e3 that all square to 1)
algebra = CliffordAlgebra((1., 1., 1.))


#using the GPU
gc.collect()
torch.cuda.empty_cache() 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}', flush = True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
num_gpus = 3

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      

      x = torch.linalg.cholesky(torch.ones((1, 1), device=device))
      del x

      #this module was took off the shelf from the paper we worked on October.
      #it expects input of the type B x M x M x M x 6
      #with B = batch size, M = spatial resolution and 3 vectors + 3 bivector coefficients.
      #The fourier neural operator captures global interactions. Outputs also have shape B x M x M x M x 6

      self.FNO1 = CliffordMaxwellNet3d(
        g=[1, 1, 1],
        block=partialclass("CliffordFourierBasicBlock3d", CliffordFourierBasicBlock3d, modes1=8, modes2=8, modes3 =8),
        num_blocks=[1, 1],
        in_channels=4,
        out_channels=4,
        hidden_channels=25,
        activation=F.gelu, 
        norm=True,
        num_groups=1)
      
   

      #Convolutions grasp local interactions. We use them to downsample the input volume. They work just like regular convolution with a
      #sliding kernel over the volume, and they expect input of form B x M x M x M x 8, in which 8 are the
      #possible coefficients in 3D GA, (1 scalar, 3 vectors, 3 bivectors and 1 trivector).

      self.act = F.gelu


      self.norm01 = CliffordGroupNorm3d(g = [1, 1, 1], num_groups=1,  channels = 1)
      self.norm02 = CliffordGroupNorm3d(g = [1, 1, 1], num_groups=1,  channels = 2)
      self.norm03 = CliffordGroupNorm3d(g = [1, 1, 1], num_groups=1,  channels = 4)


      self.norm1 = CliffordGroupNorm3d(g = [1, 1, 1],  num_groups=1, channels = 2)
      self.norm2 = CliffordGroupNorm3d(g = [1, 1, 1],  num_groups=1, channels = 1)

      self.conv01 = CliffordConv3d(g=[1, 1, 1], in_channels=1, out_channels=1, kernel_size=5, padding=2, stride = 1)
      self.conv02 = CliffordConv3d(g=[1, 1, 1], in_channels=1, out_channels=2, kernel_size=5, padding=2, stride = 1)
      self.conv03 = CliffordConv3d(g=[1, 1, 1], in_channels=2, out_channels= 4, kernel_size=5, padding=2, stride = 1)



      self.conv1 = CliffordConv3d(g=[1, 1, 1], in_channels=4, out_channels=2, kernel_size=5, padding=2, stride = 1)
      self.conv2 = CliffordConv3d(g=[1, 1, 1], in_channels=2, out_channels=1, kernel_size=5, padding=2, stride = 1)
      self.conv3 = CliffordConv3d(g=[1, 1, 1], in_channels=1, out_channels=1, kernel_size=5, padding=2, stride = 1)

    def forward(self, x):
      
 
      x_v = algebra.embed_grade(x[:,:,:,:,1:4].unsqueeze(1), grade= 1)
      x_b = algebra.embed_grade(x[:,:,:,:,4:7].unsqueeze(1), grade= 2)
      x_s1 = algebra.embed_grade(x[:,:,:,:,0].unsqueeze(4), grade = 3)
      x_s1 = x_s1.unsqueeze(1)

      x_mask = algebra.embed_grade(x[:,:,:,:,7].unsqueeze(4), grade = 0)
      x_mask = x_mask.unsqueeze(1)

      x_press =  x_mask +  x_v + x_b + x_s1

      x = self.conv01(x_press)
      x = self.norm01(x)
      x = self.act(x)
  
      x = self.conv02(x)
      x = self.norm02(x)
      x = self.act(x)

      x = self.conv03(x)
      x = self.norm03(x)

      x_press = x
      
          
      x_first = x_press
      x1 = self.FNO1(x_first)


      x0 = x1 
      
      

      x = self.conv1(x0)
      x = self.norm1(x)
      x = self.act(x)
    
      x = self.conv2(x)
      x = self.norm2(x)
      x = self.act(x) 

      x = self.conv3(x)
      out_s = x 
      
      out_s = out_s * x_mask

    
      out_s = algebra.get_grade(out_s, 0) 

      out = out_s.reshape((-1, M, M, M, 1))


      return out


model1 = Net()



if num_gpus >= 1:
    model = torch.nn.DataParallel(model1).cuda()



#counting the number of trainable parameters
count_parameters(model)


#defining hyperparameters
batchsize = 1 * num_gpus
epochs = 100

#initial learning rate
lr = 1e-4

#number of epochs to pass without improvement on the validation error for the training to stop
patience = 50
strike = 0

gc.collect()
torch.cuda.empty_cache() 


torch.manual_seed(SEED)




model = model.to(device)



def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]




tensorX = torch.Tensor(Xtrain)
tensorY = torch.Tensor(Ytrain)


print(tensorX.shape, tensorY.shape, flush = True)

dataset = TensorDataset(tensorX,tensorY) # create your datset
traindataloader = DataLoader(dataset, batch_size=int(batchsize), shuffle=True)


tensorvX = torch.Tensor(Xtest)
tensorvY = torch.Tensor(Ytest)


print(tensorvX.shape, tensorvY.shape, flush = True)


vdataset = TensorDataset(tensorvX,tensorvY) # create your datset
valdataloader = DataLoader(vdataset, batch_size = int(batchsize), shuffle=False)



for IDX in range(3):

    volume_points = tensorvX[IDX, :, :, :, 1:4].reshape(-1, 3)
    print(volume_points.shape, flush = True)
    press = tensorvY[IDX, :, :, :, 0].reshape(-1, 1)

    fig=plt.figure(figsize=(5,5))
    ax = fig.add_subplot(projection = "3d")
    # Plot the points
    ss = ax.scatter(volume_points[:, 2], volume_points[:, 0], volume_points[:, 1], c = press[:,0], marker='o', s=8)
    ax.set_title('Sanity Check - Grid must be equal to PC')
    axisEqual3D(ax)
    plt.colorbar(ss)
    plt.show()
    plt.savefig("PLOT_PRESS_Ahmed_VAL" + str(IDX) + ".png")



lambda_l1 = 1e-5

optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=0)
total_steps = epochs * len(traindataloader)

#define how the learning rate decreases
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold=0.1, factor=0.5, patience=20)

iters = len(traindataloader)



#defining the loss function, just as in GINO. Same metric is used for measuring the error.

loss_fn0 = torch.nn.L1Loss()

def loss_fn(output, target):
 
    output_p = output[:,:,:,:,0].reshape((-1, M**3))
    target_p = target[:,:,:,:,0].reshape((-1, M**3))

    return torch.mean(loss_fn0(output_p, target_p))


def loss_angle(output, target):

    tensor1 = output[:,:,:,:,1:]
    tensor2 = target[:,:,:,:,1:]

    # Reshape tensors to perform dot product along the last dimension
    tensor1_flat = tensor1.view(tensor1.shape[0], -1, 3)  # Reshape to (B, M*M*M, 3)
    tensor2_flat = tensor2.view(tensor1.shape[0], -1, 3)  # Reshape to (B, M*M*M, 3)

    # Normalize the vectors
    tensor1_norm = tensor1_flat / (torch.norm(tensor1_flat, dim=-1, keepdim=True) + 1e-6)
    tensor2_norm = tensor2_flat / (torch.norm(tensor2_flat, dim=-1, keepdim=True) + 1e-6)

    # Compute dot product using torch.einsum()
    dot_product = torch.einsum('bik,bik->bi', tensor1_norm, tensor2_norm)
    dot_product = torch.mean(dot_product, dim = [1])
    return (1 - dot_product)

def loss_pressure(output, target):
 
    output_p = output[:,:,:,:,0].reshape((-1, M**3))
    target_p = target[:,:,:,:,0].reshape((-1, M**3))

    diff_norms = torch.linalg.norm(output_p - target_p, ord = 2, dim = [1])
    y_norms = torch.linalg.norm(target_p, ord = 2, dim = [1])

    return diff_norms / y_norms

def loss_pressure_analytic(output, target, mu, stand, mask):
 
    output_p = output[:,:,:,:,0].reshape((-1, M**3))
    target_p = target[:,:,:,:,0].reshape((-1, M**3))
    mask = mask.reshape((-1, M**3))

    diff_norms = torch.linalg.norm(output_p - target_p, ord = 2, dim = 1)
    y_norms = torch.linalg.norm(target_p + mask*(mu/stand) , ord = 2,  dim = 1)

    return diff_norms / y_norms




pressure = 10e9



#training loop
timestamp = datetime.now().strftime('%Y%m%d')
    
model.train(True)

train_loss = np.zeros(epochs)
validation_loss = np.zeros(epochs)
press_loss= np.zeros(epochs)
press_loss_v= np.zeros(epochs)

best_pressureloss = 10e9



for epoch in range(epochs):
        
    running_loss = 0 
    avg_loss = 0

    avg_pressureloss = 0

    running_vloss = 0
    avg_vloss = 0

    running_pressureloss = 0

    running_pressureloss_v = 0

    start = time.time()
    model.train()
    for i, data in enumerate(traindataloader):
        
        x, y = data

        x = x.to(device)


        y = y.to(device)
            
        torch.cuda.empty_cache()

        pred = model(x)

        LP = loss_pressure(pred, y)

       
        beta = 1
        alpha = 5

        l1_norm = sum(p.abs().sum() for p in model.parameters())

        
        loss = torch.mean(LP) + lambda_l1*l1_norm + loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
  
        optimizer.step()

        running_loss += loss.item()
        running_pressureloss += torch.mean(LP).item()

        if i % 10  == 9:
            avg_loss = running_loss / 10 # loss per batch
            avg_pressureloss = running_pressureloss / 10

            print('  batch {} loss: {}'.format(i + 1, avg_loss), flush = True)
            tb_x = epoch * len(traindataloader) + i + 1
            print(f"LOSS train' {avg_loss}, {tb_x}", flush = True)
            print("loss press: ", avg_pressureloss)
            #print(f"Gradient l2_norm and max_norm' {grad_l2norm}, {grad_maxnorm}")
            running_loss = 0
            running_pressureloss = 0
            running_velloss = 0
            
    
    with torch.no_grad():
        model.eval()
        for j, vdata in enumerate(valdataloader):
            vx, vy = vdata

            vx = vx.to(device)
            vy = vy.to(device)

            vpred = model(vx)

            LP = loss_pressure(vpred, vy)

            l1_norm = sum(p.abs().sum() for p in model.parameters())

      
            vloss = torch.mean(LP) + lambda_l1*l1_norm + loss_fn(vpred, vy)

            #vloss = loss_fn(vpred, vy)
            running_vloss += vloss
            running_pressureloss_v += torch.mean(LP).item()


        
        avg_vloss = running_vloss / (j + 1)
        avg_pressureloss_v = running_pressureloss_v / (j + 1)

    
    scheduler.step(avg_vloss)
    
    for param_group in optimizer.param_groups:
        print(param_group['lr'], flush = True)
    
    
    end = time.time()
    print(f"Epoch: {epoch} - LOSS train: {avg_loss} LOSS val: {avg_vloss} PRESS ERROR val: {avg_pressureloss_v} - Elapsed time: {end-start} s", flush = True)

    train_loss[epoch] = avg_loss
    validation_loss[epoch] = avg_vloss

    press_loss[epoch] = avg_pressureloss

    press_loss_v[epoch] = avg_pressureloss_v


        
    gc.collect()
    torch.cuda.empty_cache()
        
    if avg_pressureloss_v < best_pressureloss:
        best_pressureloss = avg_pressureloss_v
        strike = 0
        model_path = 'trainedmodels/model_' + str(M) + '_ahmed'
        torch.save(model.state_dict(), model_path)
        
    else:
        strike += 1
        
    if strike == patience:
        break


fig=plt.figure(figsize=(8,8))
plt.plot(train_loss[:epoch], c = "b", linewidth = 3, label = "Train Loss")
plt.plot(press_loss[:epoch], c = "r", linewidth = 3, label = "Pressure Train Loss")

plt.plot(validation_loss[:epoch], c = "b", linestyle = '--',  linewidth = 3,label = "Validation Loss")
plt.plot(press_loss_v[:epoch], c = "r" , linestyle = '--', linewidth = 3, label = "Pressure Validation Loss")

plt.legend()
plt.show()
plt.savefig('losses_ahmed.png')
plt.savefig('losses_ahmed.pdf')


#testing
model_path = 'trainedmodels/model_' +str(M)+ '_ahmed'
model.load_state_dict(torch.load(model_path))


testdataloader = valdataloader
    
print('*************')
print('             ')
print("Starting to test....")
print('             ')

model.eval() 
totloss = 0
totloss_V = 0
totloss_an = 0

alpha_list = []
beta_list = []




with torch.no_grad():
    for i, data in enumerate(testdataloader):
        x, y = data

        x = x.to(device)
        y = y.to(device)

        predY = model(x)

        print(y.shape)
        print(predY.shape)
        print(M)
    
        
        totloss +=torch.mean(loss_pressure(predY, y))


         

print(f"total error PRESSURE: {100*totloss/(i+1)} %")

totloss = 0
totloss_V = 0

with torch.no_grad():
    for i, data in enumerate(testdataloader):
        x, y = data

        x = x.to(device)
        y = y.to(device)

        predY = model(x)

        scalar = x[:,:,:,:,4]
        
        predY[:,:,:,:,0] = (predY[:,:,:,:,0] * (scalar*STD)) + (MU * scalar)
        y[:,:,:,:,0] = (y[:,:,:,:,0] * (scalar*STD)) + (MU * scalar)


        
        totloss +=torch.mean(loss_pressure(predY, y))



print(f"total error PRESSURE - denorm: {100*totloss/(i+1)} %")


totloss = 0

with torch.no_grad():
    for i, data in enumerate(traindataloader):
        x, y = data

        x = x.to(device)
        y = y.to(device)

        predY = model(x)

        totloss += torch.mean(loss_pressure(predY, y))
         
       
print(f"TRAIN total error - norm: {100*totloss/(i+1)} %")



totloss = 0



with torch.no_grad():
    for i, data in enumerate(traindataloader):
        x, y = data

        x = x.to(device)
        y = y.to(device)

        predY = model(x)

        scalar = x[:,:,:,:,4]

        print(STD, MU)
        #print(scalar)

        predY[:,:,:,:,0] = (predY[:,:,:,:,0] * (scalar*STD)) + (MU * scalar)
        y[:,:,:,:,0] = (y[:,:,:,:,0] * (scalar*STD)) + (MU * scalar)
       



        totloss += torch.mean(loss_pressure(predY, y))
         
       
print(f"TRAIN total error - denorm: {100*totloss/(i+1)} %")


import matplotlib.pyplot as plt


cnt = 0
with torch.no_grad():
    for i, data in enumerate(testdataloader):
        x, y = data

        x = x.to(device)
        y = y.to(device)
        predY = model(x)

        #predY = (predY*std) + mu
        #y = (y*std) + mu


        if cnt < 3:

            

            scalar = x[:,:,:,:,4]

        
            predY[:,:,:,:,0] = (predY[:,:,:,:,0] * (scalar*STD)) + (MU * scalar)
            y[:,:,:,:,0] = (y[:,:,:,:,0] * (scalar*STD)) + (MU * scalar)

            x = x.detach().cpu()
            y = y.detach().cpu()
            predY = predY.detach().cpu()

            volume_points = x[i,:,:,:,1:4].reshape((-1, 3))
            press = y[i, :, :, :, 0].reshape(-1, 1)
            press_pred = predY[i, :, :, :, 0].reshape(-1, 1)

            
            cnt += 1
            
            fig=plt.figure(figsize=(8,8))
            # Plot the points
            ax = fig.add_subplot(projection = "3d")

            ss = ax.scatter(volume_points[:, 2], volume_points[:, 0], volume_points[:, 1], c = press[:,0], marker='o', s=15, alpha = 0.9)
            #ax.set_title('Sanity Check - Grid must be equal to PC')
            axisEqual3D(ax)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            plt.colorbar(ss)

            plt.show()
            plt.savefig("PLOT_PRESS_GT_ahmed" + str(cnt) + ".png")
            #plt.savefig("PLOT_PRESS_GT" + str(cnt) + ".pdf")


            fig=plt.figure(figsize=(8,8))
            # Plot the points
            ax = fig.add_subplot(projection = "3d")

            ss = ax.scatter(volume_points[:, 2], volume_points[:, 0], volume_points[:, 1], c = press_pred[:,0], marker='o', s=15, alpha = 0.9)
            #ax.set_title('Sanity Check - Grid must be equal to PC')
            axisEqual3D(ax)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            plt.colorbar(ss)
            plt.show()
            plt.savefig("PLOT_PRESS_PREDICTION_ahmed" + str(cnt) + ".png")
            #plt.savefig("PLOT_PRESS_PREDICTION" + str(cnt) + ".pdf")

            fig=plt.figure(figsize=(8,8))
            ax = fig.add_subplot(projection = "3d")

            # Plot the points
            ss = ax.scatter(volume_points[:, 2], volume_points[:, 0], volume_points[:, 1], c = (press[:,0] - press_pred[:,0])/press[:,0], marker='o', s=15, alpha = 0.9, vmin = 0, vmax = 1.5)
            #ax.set_title('Sanity Check - Grid must be equal to PC')
            axisEqual3D(ax)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            plt.colorbar(ss)
            plt.show()
            plt.savefig("PLOT_PRESS_DIFFERENCE_ahmed" + str(cnt) + ".png")
            #plt.savefig("PLOT_PRESS_DIFFERENCE" + str(cnt) + ".pdf")
