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


class ScaledUnitGaussianNormalizer:
    def __init__(self, x, eps=0.00001, reduce_dim=[0], verbose=True, scale = 8):
        super().__init__()

        
        n_samples, *shape = x.shape
        self.sample_shape = shape
        self.verbose = verbose
        self.reduce_dim = reduce_dim

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, reduce_dim, keepdim=True).squeeze(0)
        self.std = torch.std(x, reduce_dim, keepdim=True).squeeze(0) * scale
        self.eps = eps

        if verbose:
            print(
                f"UnitGaussianNormalizer init on {n_samples}, reducing over {reduce_dim}, samples of shape {shape}."
            )
            print(f"   Mean and std of shape {self.mean.shape}, eps={eps}")

    def encode(self, x):
        x -= self.mean
        x /= self.std + self.eps
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps  # T*batch*n
                mean = self.mean[:, sample_idx]

        x *= std
        x += mean

        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        return self

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        return self

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self




SEED  = 24
random.seed(SEED)


def grad_norm(model: torch.nn.Module) -> Tuple[float, float]:

    parameters = [p for p in model.parameters() if p.grad is not None]
    if len(parameters) == 0:
        return 0.0, 0.0
    max_norm = max(p.grad.detach().abs().max() for p in parameters)
    norm_type = 2.0
    l2_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
        norm_type,
    )
    return float(l2_norm), float(max_norm)


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


#This function is used to map our input point cloud N x 3, with N about 3.6k points
#into a regular grid M x M x M x 3, with resolution M. This is done in analogy with the GINO pipeline and because
#the fourier neural operator needs a spatial grid in input

def encode_point_cloud(point_cloud, grid_size, press, norm):
    # Initialize the grids with fixed resolution
    grid = np.zeros((grid_size, grid_size, grid_size, 3))
    mask = np.zeros((grid_size, grid_size, grid_size))
    pressure = np.zeros((grid_size, grid_size, grid_size, 1))
    normals = np.zeros((grid_size, grid_size, grid_size, 3))

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
    print(len(indices))
    print(len(scaled_points))
    print("***")

    # Assign points to the grid
    for idx, point, p, n in zip(indices, point_cloud, press, norm):
        if mask[idx[0], idx[1], idx[2]] == 0:  # Only assign if the cell is empty
            grid[idx[0], idx[1], idx[2]] = point
            pressure[idx[0], idx[1], idx[2]] = p
            mask[idx[0], idx[1], idx[2]] = 1
            normals[idx[0], idx[1], idx[2]] = n

    return grid, pressure.squeeze(), mask, normals

def encode_vel(point_cloud, grid_size, vel):
    grid = np.zeros((grid_size, grid_size, grid_size))
    points = np.zeros((grid_size, grid_size, grid_size, 3))
    velo = np.zeros((grid_size, grid_size, grid_size, 3))
    
    max_coords = np.max(point_cloud, axis=0)
    min_coords = np.min(point_cloud, axis=0)
    cloud_range = max_coords - min_coords

    scale = (grid_size - 1) / cloud_range
    scaled_points = (point_cloud - min_coords) * scale

    # Calculate the integer indices for the scaled points
    indices = np.floor(scaled_points).astype(int)

    # Ensure indices are within grid bounds
    indices = np.clip(indices, 0, grid_size - 1)
    print(len(indices))



    # Assign points to the grid
    for idx, point, v in zip(indices, point_cloud, vel):
        #if grid[idx[0], idx[1], idx[2]].all() == 0:
        grid[idx[0], idx[1], idx[2]] = 1
        points[idx[0], idx[1], idx[2]] = point
        velo[idx[0], idx[1], idx[2]] = v
    
    return velo, grid, points



#used for plotting 3D point clouds with equal x - y - z axes.

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/3
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)



def get_random_rotation_matrix():

    angle_x = random.uniform(0, 2 * np.pi)
    angle_y = random.uniform(0, 2 * np.pi)
    angle_z = random.uniform(0, 2 * np.pi)

    # Create rotation matrices from Euler angles
    rotation_x = np.array([[1, 0, 0],
                       [0, np.cos(angle_x), -np.sin(angle_x)],
                       [0, np.sin(angle_x), np.cos(angle_x)]])

    rotation_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                       [0, 1, 0],
                       [-np.sin(angle_y), 0, np.cos(angle_y)]])

    rotation_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                       [np.sin(angle_z), np.cos(angle_z), 0],
                       [0, 0, 1]])

    # Combine the rotation matrices
    rotation_matrix = rotation_x.dot(rotation_y).dot(rotation_z)

    return rotation_matrix


#this is a function in Geometric Algebra, might be a bit difficult to understand. We use it to compute the dual to the point cloud. 
#If a point cloud is composed by vectors P = p1e1 + p2e2 + p3e3, their dual will be a bivector, which is the perpendicular oriented plane with respect to P.
#We do it for 2 reasons: 1 to have a bivector component in the network (see below) and 2 to have some sort of "surface" information.

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


#Function to check if the meshes are watertight. In the GINO paper we have that for the Shapenet Car dataset (the one we are using) there 
#are 889 car shapes. In GINO, they only consider the watertight meshes (why is this relevant if the input is a point cloud?), 
#which they say to be 611. They use 500 for training and 111 for testing (no validation, suspicious). 
#I tried different approaches, with pyvista, trimesh, etc. Most of the times it says that *all* the meshes are watertight. In the 
#code for GINO, they don't specify how they check that, but they have a file .txt with the list of watertight meshes which is not
# included. I might email them about it.



M = 80
spatial_resolution = (M, M, M)

#We store the 889 x N x 3 point clouds here 
Points = np.zeros((683, 3586, 3))
PointsV = np.zeros((683, 29498, 3))
#We store the 889 x N x 3 point clouds here 
Press = np.zeros((683, 3586, 1))
Velo = np.zeros((683, 29498, 3))


X = np.zeros((1700, M, M, M, 12))
Ygrid = np.zeros((1700, M, M, M, 4))
Y = np.zeros((1700, 3586, 1))

with open('mesh_is_watertight.txt', 'r') as output_file:
    data = output_file.readlines()

data =sorted(set(data))



print(data)
random.shuffle(data)
print(data)


i = 0

for elem in data:

    mesh1 = meshio.read(str(elem)[:-4] + 'vtk')
    mesh2 = meshio.read(str(elem)[:-19] + 'hexvelo_smpl.vtk')

    points = mesh1.points
    points2 = mesh2.points

    #kdtree = cKDTree(points2)
    # Find nearest neighbors in points2 for each point in points1
    #distances, indices = kdtree.query(np.concatenate((points[0:16], points[112:]), axis=0).astype(np.float32))
   
    vel = np.load(str(elem)[:-19] + 'velo.npy').reshape(-1, 3)
    press = np.load(str(elem)[:-19] + 'press.npy').reshape(-1, 1)
    
            
    #the points between 16 and 112 don't belong to the structure and are discarded, as in GINO
    Points[i] = np.concatenate((points[0:16], points[112:]), axis=0).astype(np.float64)
    Press[i] = np.concatenate((press[0:16], press[112:]), axis=0).astype(np.float64)
    PointsV[i] = points2.astype(np.float64)
    Velo[i] = vel.astype(np.float64)

    i+= 1


print(np.isnan(PointsV.any()))

Points[:500] = (2*(Points[:500] - np.min(PointsV[:500])) / (np.max(PointsV[:500] - np.min(PointsV[:500])))) - 1
Points[500:611] = (2*(Points[500:611] - np.min(PointsV[500:611])) / (np.max(PointsV[500:611] - np.min(PointsV[500:611])))) - 1

PointsV[:500] = (2*(PointsV[:500] - np.min(PointsV[:500])) / (np.max(PointsV[:500] - np.min(PointsV[:500])))) - 1
PointsV[500:611] = (2*(PointsV[500:611] - np.min(PointsV[500:611])) / (np.max(PointsV[500:611] - np.min(PointsV[500:611])))) - 1
i = 0
j = 0




Press = torch.tensor(Press)
Velo = torch.tensor(Velo)

encoder = ScaledUnitGaussianNormalizer(
          Press[:500], eps=1e-6, reduce_dim=[0, 1], scale = 1, verbose=True
        )

encoderV = UnitGaussianNormalizer(
           Velo[:500], eps=1e-6, reduce_dim=[0, 1, 2], verbose=True
        )


Press[:500] = encoder.encode(Press[:500])
Velo[:500] = encoderV.encode(Velo[:500])


Press[500:611] = encoder.encode(Press[500:611])
Velo[500:611] = encoderV.encode(Velo[500:611])




for elem in data:

    mesh = o3d.io.read_triangle_mesh(str(elem)[:-1])
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
    normals = np.concatenate((normals[0:16], normals[112:]), axis=0).astype(np.float64)
            
                
    tx = np.linspace(np.min(Points[i,0]), np.max(Points[i,0]), M)
    ty = np.linspace(np.min(Points[i,1]), np.max(Points[i,1]), M)
    tz = np.linspace(np.min(Points[i,2]), np.max(Points[i,2]), M)
    #tx = tz = ty = np.linspace(-1, 1, M)
    query_points = np.stack(np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1).astype(
        np.float32
        )
                
    #scene = o3d.t.geometry.RaycastingScene()
    #_ = scene.add_triangles(mesh)
    #signed_distance = scene.compute_signed_distance(query_points).numpy()


    points = Points[i]
    press = Press[i]

    #X[j,:,:,:,0] = signed_distance
    X[j,:,:,:,1:4], Ygrid[j,:,:,:,0], X[j,:,:,:,4], normals_grid = encode_point_cloud(points, M, press, calculate_dual(normals))
    X[j,:,:,:,5:8] = normals_grid
                
    #Y[i,:,0:3] = Points[i]
    #Ygrid[j,:,:,:,0] = Press[i]
    Ygrid[j,:,:,:,1:], X[j,:,:,:,8], X[j,:,:,:,9:] = encode_vel(PointsV[i], M, Velo[i])

    
    i += 1
    j += 1
    
    if j < 0:
        R = get_random_rotation_matrix()
        mesh = o3d.io.read_triangle_mesh(str(elem)[:-1])
        mesh = mesh.rotate(R)
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)


        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        signed_distance = scene.compute_signed_distance(query_points).numpy()

        X[j,:,:,:,0] = signed_distance
        X[j,:,:,:,1:4], Ygrid[j] = encode_point_cloud(points.dot(R.T), M, press)
        X[j,:,:,:,4:] = calculate_dual(X[i,:,:,:,1:4])

        j+=1

        R = get_random_rotation_matrix()
        mesh = o3d.io.read_triangle_mesh(str(elem)[:-1])
        mesh = mesh.rotate(R)
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)


        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        signed_distance = scene.compute_signed_distance(query_points).numpy()

        X[j,:,:,:,0] = signed_distance
        X[j,:,:,:,1:4], Ygrid[j] = encode_point_cloud(points.dot(R.T), M, press)
        X[j,:,:,:,4:] = calculate_dual(X[i,:,:,:,1:4])

        j+=1



Y = Ygrid
algebra = CliffordAlgebra((1., 1., 1.))



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

      self.act = F.tanh
      self.act_phys = F.gelu


      self.norm01 = CliffordGroupNorm3d(g = [1, 1, 1], num_groups=1,  channels = 1)
      self.norm02 = CliffordGroupNorm3d(g = [1, 1, 1], num_groups=1,  channels = 2)
      self.norm03 = CliffordGroupNorm3d(g = [1, 1, 1], num_groups=1,  channels = 4)


      self.norm001 = CliffordGroupNorm3d(g = [1, 1, 1], num_groups=1,  channels = 1)      
      self.norm002 = CliffordGroupNorm3d(g = [1, 1, 1], num_groups=1,  channels = 2)
      self.norm003 = CliffordGroupNorm3d(g = [1, 1, 1], num_groups=1,  channels = 4)

      self.norm1 = CliffordGroupNorm3d(g = [1, 1, 1],  num_groups=1, channels = 2)
      self.norm2 = CliffordGroupNorm3d(g = [1, 1, 1],  num_groups=1, channels = 1)

      self.norm11 = CliffordGroupNorm3d(g = [1, 1, 1],  num_groups=1, channels = 2)
      self.norm22 = CliffordGroupNorm3d(g = [1, 1, 1],  num_groups=1, channels = 1)

      self.conv01 = CliffordConv3d(g=[1, 1, 1], in_channels=1, out_channels=1, kernel_size=5, padding=2, stride = 1)
      self.conv02 = CliffordConv3d(g=[1, 1, 1], in_channels=1, out_channels=2, kernel_size=5, padding=2, stride = 1)
      self.conv03 = CliffordConv3d(g=[1, 1, 1], in_channels=2, out_channels= 4, kernel_size=5, padding=2, stride = 1)


      self.conv001 = CliffordConv3d(g=[1, 1, 1], in_channels=1, out_channels=1, kernel_size=5, padding=2, stride = 1)
      self.conv002 = CliffordConv3d(g=[1, 1, 1], in_channels=1, out_channels=2, kernel_size=5, padding=2, stride = 1)
      self.conv003 = CliffordConv3d(g=[1, 1, 1], in_channels = 2, out_channels= 4, kernel_size=5, padding=2, stride = 1)


      self.conv1 = CliffordConv3d(g=[1, 1, 1], in_channels=4, out_channels=2, kernel_size=5, padding=2, stride = 1)
      self.conv2 = CliffordConv3d(g=[1, 1, 1], in_channels=2, out_channels=1, kernel_size=5, padding=2, stride = 1)
      self.conv3 = CliffordConv3d(g=[1, 1, 1], in_channels=1, out_channels=1, kernel_size=5, padding=2, stride = 1)


      self.conv11 = CliffordConv3d(g=[1, 1, 1], in_channels=4, out_channels=2, kernel_size=5, padding=2, stride = 1)
      self.conv22 = CliffordConv3d(g=[1, 1, 1], in_channels=2, out_channels=1, kernel_size=5, padding=2, stride = 1)
      self.conv33 = CliffordConv3d(g=[1, 1, 1], in_channels=1, out_channels=1, kernel_size=5, padding=2, stride = 1)






    def forward(self, x):
      
      #scalar part of the input, we embed it in the 3D GA with grade 0 (i.e. scalar). It's the binary mask.
      #bivector part of the input, we embed it in the 3D GA with grade 2 (i.e. bivector). It's the dual of the point cloud.
      #x_b = algebra.embed_grade(x[:,:,:,:,:,3:6], grade= 2)

      #first pair of convolution + grouphnorm. We downsample the volume and normalize the output.
      #we consider only vector + bivector part to pass it through the FNO

      x_v = algebra.embed_grade(x[:,:,:,:,1:4].unsqueeze(1), grade= 1)
      x_b = algebra.embed_grade(x[:,:,:,:,5:8].unsqueeze(1), grade= 2)

      x_vel_v = algebra.embed_grade(x[:,:,:,:,9:].unsqueeze(1), grade= 1)

      x_s20 = algebra.embed_grade(x[:,:,:,:,8].unsqueeze(4), grade= 0)
      x_s20 = x_s20.unsqueeze(1)

      x_s2 = algebra.embed_grade(x[:,:,:,:,8].unsqueeze(4), grade= 0)
      x_s2 = x_s2.unsqueeze(1)

      x_s2[:,:,:,:,:,1] = x_s20[:,:,:,:,:,0]
      x_s2[:,:,:,:,:,2] = x_s20[:,:,:,:,:,0]
      x_s2[:,:,:,:,:,3] = x_s20[:,:,:,:,:,0]

      x_s1 = algebra.embed_grade(x[:,:,:,:,4].unsqueeze(4), grade = 0)
      x_s1 = x_s1.unsqueeze(1)

      x_press =  x_s1 +  x_v  + x_b

      x_vel = x_s20 + x_vel_v 

      
      x = self.conv01(x_press)
      x = self.norm01(x)
      x = self.act(x)
  
      x = self.conv02(x)
      x = self.norm02(x)
      x = self.act(x)

      x = self.conv03(x)
      x = self.norm03(x)

      x_press = x
      
     
      x = self.conv001(x_vel)
      x = self.norm001(x)
      x = self.act(x)
  
    
      x = self.conv002(x)
      x = self.norm002(x)
      x = self.act(x)

      x = self.conv003(x)
      x = self.norm003(x)
      x_vel = x

          
      x_first = x_vel + x_press
      x = self.FNO1(x_first)

      x0 = x

      x = self.conv1(x0)
      x = self.norm1(x)
      x = self.act_phys(x)
    
      x = self.conv2(x)
      x = self.norm2(x)
      x_int = self.act_phys(x) 

      x = self.conv3(x_int)
      out_s = x 
      
      x = self.conv11(x0)
      x = self.norm11(x)
      x = self.act_phys(x) 
    
      x = self.conv22(x)
      x = self.norm22(x)
      x = self.act_phys(x) 

      out_v = self.conv33(x)
      
      out_s = out_s * x_s1
      out_v = out_v * x_s2
    
      out_s = algebra.get_grade(out_s, 0) 
      out_v = algebra.get_grade(out_v, 1)

      out = torch.concat((out_s, out_v), dim = 5)
      out = out.reshape((-1, M, M, M, 4))


      return out, out_s





model1 = Net()




if num_gpus >= 1:
    model = torch.nn.DataParallel(model1).cuda()



count_parameters(model)


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



tensorX = torch.Tensor(X[:500])
tensorY = torch.Tensor(Y[:500])


print(tensorX.shape, tensorY.shape, flush = True)

print(torch.mean(tensorY[:,:,:,:,0]), torch.std(tensorY[:,:,:,:,0]), flush=True)
print(torch.mean(tensorY[:,:,:,:,1:]), torch.std(tensorY[:,:,:,:,1:]), flush=True)

dataset = TensorDataset(tensorX,tensorY) # create your datset
traindataloader = DataLoader(dataset, batch_size=int(batchsize), shuffle=True)


tensorvX = torch.Tensor(X[500:611])
tensorvY = torch.Tensor(Y[500:611])


print(tensorvX.shape, tensorvY.shape, flush = True)
print(torch.mean(tensorvY[:,:,:,:,0]), torch.std(tensorvY[:,:,:,:,0]), flush=True)
print(torch.mean(tensorvY[:,:,:,:,1:]), torch.std(tensorvY[:,:,:,:,1:]), flush=True)

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
    plt.savefig("PLOT_PRESS_VAL" + str(IDX) + ".png")

for IDX in range(3):

    volume_points = tensorvX[IDX, :, :, :, 9:].reshape(-1, 3)
    print(volume_points.shape, flush = True)
    velo = tensorvY[IDX, :, :, :, 1:].reshape(-1, 3)

    fig=plt.figure(figsize=(5,5))
    ax = fig.add_subplot(projection = "3d")
    # Plot the points
    ss = ax.scatter(volume_points[:, 2], volume_points[:, 0], volume_points[:, 1], c = velo[:,0], marker='o', alpha = 0.5, s=8)
    ax.set_title('Sanity Check - Grid must be equal to PC')
    axisEqual3D(ax)
    plt.colorbar(ss)
    plt.show()
    plt.savefig("PLOT_VELO_VAL" + str(IDX) + ".png")



lambda_l1 = 1e-5

optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=0)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold=0.1, factor=0.9, patience=30)
total_steps = epochs * len(traindataloader)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold=0.1, factor=0.5, patience=20)

iters = len(traindataloader)


#defining the loss function, just as in GINO. Same metric is used for measuring the error.
loss_fn0 = torch.nn.L1Loss()

def loss_fn(output, target):
 
    output_p = output[:,:,:,:,0].reshape((-1, M**3))
    target_p = target[:,:,:,:,0].reshape((-1, M**3))

    return torch.mean(loss_fn0(output_p, target_p))



def loss_velocity(output, target):

    output_v = output[:,:,:,:,1:].reshape((-1, M**3, 3))
    target_v = target[:,:,:,:,1:].reshape((-1, M**3, 3))

    diff_norms = torch.linalg.norm(output_v - target_v, ord = 2, dim = [1, 2])
    y_norms = torch.linalg.norm(target_v, ord = 2,  dim = [1, 2])

    return torch.mean(diff_norms / y_norms)

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
vel_loss= np.zeros(epochs)
press_loss_v= np.zeros(epochs)
vel_loss_v= np.zeros(epochs)

best_pressureloss = 10e9



for epoch in range(epochs):
        
    running_loss = 0 
    avg_loss = 0

    avg_pressureloss = 0
    avg_velloss = 0

    running_vloss = 0
    avg_vloss = 0

    running_pressureloss = 0
    running_velloss = 0

    running_pressureloss_v = 0
    running_velloss_v = 0

    start = time.time()
    model.train()
    #torch.set_grad_enabled(True)
    for i, data in enumerate(traindataloader):
        
        x, y = data

        x = x.to(device)

        #print(x.shape)

        #print(torch.max(x[:,:,:,:,:,0:3]), torch.min(x[:,:,:,:,:,0:3]))
        #print(torch.max(x[:,:,:,:,:,6]), torch.min(x[:,:,:,:,:,6]))
        y = y.to(device)
            
        torch.cuda.empty_cache()

        pred = model(x)

        LP = loss_pressure(pred, y)
        LV = loss_velocity(pred, y)

       
        beta = 1
        alpha = 5

        l1_norm = sum(p.abs().sum() for p in model.parameters())

        
        loss = torch.mean(alpha*LP + beta*LV) + lambda_l1*l1_norm + loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      
  
        running_loss += loss.item()
        running_pressureloss += torch.mean(LP).item()
        running_velloss += torch.mean(LV).item()


    

        if i % 10  == 9:
            avg_loss = running_loss / 10 # loss per batch
            avg_pressureloss = running_pressureloss / 10
            avg_velloss = running_velloss / 10

            print('  batch {} loss: {}'.format(i + 1, avg_loss), flush = True)
            tb_x = epoch * len(traindataloader) + i + 1
            print(f"LOSS train' {avg_loss}, {tb_x}", flush = True)
            print("loss press: ", avg_pressureloss, "; loss velo: ", avg_velloss)
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
            LV = loss_velocity(vpred, vy)

            l1_norm = sum(p.abs().sum() for p in model.parameters())

      
            vloss = torch.mean(alpha*LP + beta*(LV)) + lambda_l1*l1_norm + loss_fn(vpred, vy)

            running_vloss += vloss
            running_pressureloss_v += torch.mean(LP).item()
            running_velloss_v += torch.mean(LV).item()


        
        avg_vloss = running_vloss / (j + 1)
        avg_pressureloss_v = running_pressureloss_v / (j + 1)
        avg_velloss_v = running_velloss_v / (j + 1)

    
    scheduler.step(avg_vloss)
    
    for param_group in optimizer.param_groups:
        print(param_group['lr'], flush = True)
    
    
    end = time.time()
    print(f"Epoch: {epoch} - LOSS train: {avg_loss} LOSS val: {avg_vloss} PRESS ERROR val: {avg_pressureloss_v} VEL ERROR val: {avg_velloss_v} - Elapsed time: {end-start} s", flush = True)

    train_loss[epoch] = avg_loss
    validation_loss[epoch] = avg_vloss

    press_loss[epoch] = avg_pressureloss
    vel_loss[epoch] = avg_velloss

    press_loss_v[epoch] = avg_pressureloss_v
    vel_loss_v[epoch] = avg_velloss_v


        
    gc.collect()
    torch.cuda.empty_cache()
        
    if avg_pressureloss_v < best_pressureloss:
        best_pressureloss = avg_pressureloss_v
        strike = 0
        #model_path = 'trainedmodels/model_{}_{}_{}_{}'.format(batchsize, SEED, epoch, timestamp)
        model_path = 'trainedmodels/fengbo_shapenet'
        torch.save(model.state_dict(), model_path)
        
    else:
        strike += 1
        
    if strike == patience:
        break



fig=plt.figure(figsize=(8,8))
plt.plot(train_loss[:epoch], c = "b", linewidth = 3, label = "Train Loss")
plt.plot(press_loss[:epoch], c = "r", linewidth = 3, label = "Pressure Train Loss")
plt.plot(vel_loss[:epoch], c = "g",  linewidth = 3,label = "Velocity Train Loss")

plt.plot(validation_loss[:epoch], c = "b", linestyle = '--',  linewidth = 3,label = "Validation Loss")
plt.plot(press_loss_v[:epoch], c = "r" , linestyle = '--', linewidth = 3, label = "Pressure Validation Loss")
plt.plot(vel_loss_v[:epoch], c = "g" , linestyle = '--', linewidth = 3,label = "Velocity Validation Loss")

plt.legend()
plt.show()
plt.savefig('losses.png')
plt.savefig('losses.pdf')




#Testing
model_path = 'trainedmodels/fengbo_shapenet'
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



encoder = encoder.to(device)
encoderV = encoderV.to(device)

with torch.no_grad():
    for i, data in enumerate(testdataloader):
        x, y = data

        x = x.to(device)
        y = y.to(device)

        predY = model(x)
        
        totloss +=torch.mean(loss_pressure(predY, y))
        totloss_V +=torch.mean(loss_velocity(predY, y))

        totloss_an += torch.mean(loss_pressure_analytic(predY, y, encoder.mean, encoder.std, x[:,:,:,:,4]))

         

print(f"total error PRESSURE: {100*totloss/(i+1)} %")
print(f"total error VELOCITY: {100*totloss_V/(i+1)} %")
#print(f"total error PRESSURE - denorm analytically: {100*totloss_an/(i+1)} %")


#print("PRESSURE ENCODER mean and std: ", encoder.mean, encoder.std)
#print("VELOCITY ENCODER mean and std: ", encoderV.mean, encoderV.std)

totloss = 0
totloss_V = 0

losses = []


#denorm loss
with torch.no_grad():
    for i, data in enumerate(testdataloader):
        x, y = data

        x = x.to(device)
        y = y.to(device)

        predY = model(x)
        
        predY[:,:,:,:,0] = (predY[:,:,:,:,0] * (x[:,:,:,:,4]*encoder.std)) + (encoder.mean *x[:,:,:,:,4])
        y[:,:,:,:,0] = (y[:,:,:,:,0] * (x[:,:,:,:,4]*encoder.std)) + (encoder.mean *x[:,:,:,:,4])

        predY[:,:,:,:,1] = (predY[:,:,:,:,1] * (x[:,:,:,:,8]*encoderV.std)) + (encoderV.mean *x[:,:,:,:,8])
        y[:,:,:,:,1] = (y[:,:,:,:,1] * (x[:,:,:,:,8]*encoderV.std)) + (encoderV.mean *x[:,:,:,:,8])

        predY[:,:,:,:,2] = (predY[:,:,:,:,2] * (x[:,:,:,:,8]*encoderV.std)) + (encoderV.mean *x[:,:,:,:,8])
        y[:,:,:,:,2] = (y[:,:,:,:,2] * (x[:,:,:,:,8]*encoderV.std)) + (encoderV.mean *x[:,:,:,:,8])

        predY[:,:,:,:,3] = (predY[:,:,:,:,3] * (x[:,:,:,:,8]*encoderV.std)) + (encoderV.mean *x[:,:,:,:,8])
        y[:,:,:,:,3] = (y[:,:,:,:,3] * (x[:,:,:,:,8]*encoderV.std)) + (encoderV.mean *x[:,:,:,:,8])
    
    
        
        totloss +=torch.mean(loss_pressure(predY, y))
        totloss_V +=torch.mean(loss_velocity(predY, y))

        losses.append((torch.mean(loss_pressure(predY, y))).detach().cpu())


         

print(f"total error PRESSURE - denorm: {100*totloss/(i+1)} %")
print(f"total error VELOCITY - denorm: {100*totloss_V/(i+1)} %")

print(f"total error PRESSURE - denorm, median: {np.median(np.asarray(losses))} %")
print(f"total error PRESSURE - denorm, mean: {np.mean(np.asarray(losses))} %")



totloss = 0

losses = []
with torch.no_grad():
    for i, data in enumerate(traindataloader):
        x, y = data

        x = x.to(device)
        y = y.to(device)

        predY = model(x)
  

        totloss += torch.mean(loss_pressure(predY, y))
        losses.append((torch.mean(loss_pressure(predY, y))).detach().cpu())

         
       
print(f"TRAIN total error - norm: {100*totloss/(i+1)} %")
print(f"total error PRESSURE - denorm, median: {np.median(np.asarray(losses))} %")
#print(f"total error PRESSURE - denorm, mean - should agree with above: {np.mean(np.asarray(losses))} %")


#plotting

totloss = 0

with torch.no_grad():
    for i, data in enumerate(traindataloader):
        x, y = data

        x = x.to(device)
        y = y.to(device)

        predY = model(x)

        predY[:,:,:,:,0] = (predY[:,:,:,:,0] * (x[:,:,:,:,4]*encoder.std)) + (encoder.mean *x[:,:,:,:,4])
        y[:,:,:,:,0] = (y[:,:,:,:,0] * (x[:,:,:,:,4]*encoder.std)) + (encoder.mean *x[:,:,:,:,4])

        predY[:,:,:,:,1] = (predY[:,:,:,:,1] * (x[:,:,:,:,8]*encoderV.std)) + (encoderV.mean *x[:,:,:,:,8])
        y[:,:,:,:,1] = (y[:,:,:,:,1] * (x[:,:,:,:,8]*encoderV.std)) + (encoderV.mean *x[:,:,:,:,8])

        predY[:,:,:,:,2] = (predY[:,:,:,:,2] * (x[:,:,:,:,8]*encoderV.std)) + (encoderV.mean *x[:,:,:,:,8])
        y[:,:,:,:,2] = (y[:,:,:,:,2] * (x[:,:,:,:,8]*encoderV.std)) + (encoderV.mean *x[:,:,:,:,8])

        predY[:,:,:,:,3] = (predY[:,:,:,:,3] * (x[:,:,:,:,8]*encoderV.std)) + (encoderV.mean *x[:,:,:,:,8])
        y[:,:,:,:,3] = (y[:,:,:,:,3] * (x[:,:,:,:,8]*encoderV.std)) + (encoderV.mean *x[:,:,:,:,8])
       
   

        #predY = (predY*std) + mu
        #y = (y*std) + mu


        #print(y[0])
        #print(predY[0])
        #print("****")


        totloss += torch.mean(loss_pressure(predY, y))
         
       
print(f"TRAIN total error - denorm: {100*totloss/(i+1)} %")



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

            

            predY[:,:,:,:,0] = (predY[:,:,:,:,0] * (x[:,:,:,:,4]*encoder.std)) + (encoder.mean *x[:,:,:,:,4])
            y[:,:,:,:,0] = (y[:,:,:,:,0] * (x[:,:,:,:,4]*encoder.std)) + (encoder.mean *x[:,:,:,:,4])

            predY[:,:,:,:,1] = (predY[:,:,:,:,1] * (x[:,:,:,:,8]*encoderV.std)) + (encoderV.mean *x[:,:,:,:,8])
            y[:,:,:,:,1] = (y[:,:,:,:,1] * (x[:,:,:,:,8]*encoderV.std)) + (encoderV.mean *x[:,:,:,:,8])

            predY[:,:,:,:,2] = (predY[:,:,:,:,2] * (x[:,:,:,:,8]*encoderV.std)) + (encoderV.mean *x[:,:,:,:,8])
            y[:,:,:,:,2] = (y[:,:,:,:,2] * (x[:,:,:,:,8]*encoderV.std)) + (encoderV.mean *x[:,:,:,:,8])

            predY[:,:,:,:,3] = (predY[:,:,:,:,3] * (x[:,:,:,:,8]*encoderV.std)) + (encoderV.mean *x[:,:,:,:,8])
            y[:,:,:,:,3] = (y[:,:,:,:,3] * (x[:,:,:,:,8]*encoderV.std)) + (encoderV.mean *x[:,:,:,:,8])

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

            ss = ax.scatter(volume_points[:, 0], volume_points[:, 2], volume_points[:, 1], c = press[:,0], marker='o', s=15, alpha = 0.9)
            #ax.set_title('Sanity Check - Grid must be equal to PC')
            axisEqual3D(ax)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_axis_off()
            plt.colorbar(ss)

            plt.show()
            plt.savefig("PLOT_PRESS_GT" + str(cnt) + ".png")
            #plt.savefig("PLOT_PRESS_GT" + str(cnt) + ".pdf")


            fig=plt.figure(figsize=(8,8))
            # Plot the points
            ax = fig.add_subplot(projection = "3d")

            ss = ax.scatter(volume_points[:, 0], volume_points[:, 2], volume_points[:, 1], c = press_pred[:,0], marker='o', s=15, alpha = 0.9)
            #ax.set_title('Sanity Check - Grid must be equal to PC')
            axisEqual3D(ax)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_axis_off()
            plt.colorbar(ss)
            plt.show()
            plt.savefig("PLOT_PRESS_PREDICTION" + str(cnt) + ".png")
            #plt.savefig("PLOT_PRESS_PREDICTION" + str(cnt) + ".pdf")

            fig=plt.figure(figsize=(8,8))
            ax = fig.add_subplot(projection = "3d")

            # Plot the points
            ss = ax.scatter(volume_points[:, 0], volume_points[:, 2], volume_points[:, 1], c = (press[:,0] - press_pred[:,0])/press[:,0], marker='o', s=15, alpha = 0.9, vmin = 0, vmax = 1.5)
            #ax.set_title('Sanity Check - Grid must be equal to PC')
            axisEqual3D(ax)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_axis_off()
            plt.colorbar(ss)
            plt.show()
            plt.savefig("PLOT_PRESS_DIFFERENCE" + str(cnt) + ".png")
            #plt.savefig("PLOT_PRESS_DIFFERENCE" + str(cnt) + ".pdf")
