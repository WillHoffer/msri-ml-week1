# all imports here
import numpy as np
import gudhi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from sklearn import datasets
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from tqdm import tqdm, trange

import os

def load_pcd(filename, show=False):
    ''' Loads 2D/3D point cloud data from *.npy files

    Parameters:
        filename: name of *.npy file.
        show: option to visualize point clouds in 3D

    Returns: 
        np.array formatted as [point_1, point_2, ..., point_n]
    
    '''
    array = np.load(filename)
    dim = array.shape[1]
    if show:
        fig = plt.figure()
        if dim == 3:
            ax = fig.add_subplot(projection='3d')
            ax.set_xlim3d(-1, 1)
            ax.set_ylim3d(-1, 1)
            ax.set_zlim3d(-1, 1)
            ax.scatter(array[:, 0],array[:, 1], array[:, 2])
        else:
            ax = fig.add_subplot()
            ax.scatter(array[:, 0], array[:, 1])
    return array 


def complex_visualizer_PCD(points, complex, ax, animation=False):
    ''' Visualizes complex generated from 2D or 3D point cloud data.

    Parameters:
        points: original point cloud data
        complex: simplicial complex formatted as [simplex_1, simplex_2, .... simplex_nn]. 
        Each simplex_i is a list of indices.

    '''
    dim = points.shape[1]

    filtration = complex.get_filtration()
    if dim == 3:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='k')
    for filtered_value in filtration:
        if filtered_value[1] == 0.0:
            continue
        indices = filtered_value[0]
        sdim = len(indices) - 1
        if sdim > 1:
            alpha=0.3
        else:
            alpha = 1.0
        splex_verts = points[indices]
        color = [(1 - 1/sdim)*0.5, (1 - 1/sdim)*0.5, (1 - 1/sdim)]
        if dim == 2:
            ax.fill(splex_verts[:, 0], splex_verts[:, 1], c=color, alpha=alpha)
        else:
            verts = splex_verts.tolist()
            ax.add_collection3d(Poly3DCollection([verts], color=color, alpha=alpha))
    if animation:
        assert dim == 2
        container = ax.get_children()
        return container
    return ax


import time
def rips_gudhi(points, radius, max_dimension=2, show=False):
    ''' TODO: use Gudhi to create a Rips complex of some fixed radius parameter. 

    Parameters:
        points: original point cloud data
        radius: radius parameter
        max_dimension: maximum dimension of simplices in resulting complex
        show: option to visualize complex 

    Returns:
        Gudhi SimplexTree object

    '''
    cplex = gudhi.RipsComplex(points=points, max_edge_length=radius).create_simplex_tree(max_dimension=max_dimension)
    if show:
        fig = plt.figure()
        if points.shape[1] == 3:
            ax=fig.add_subplot(projection='3d')
        else:
            ax = fig.add_subplot()
        ax = complex_visualizer_PCD(points, cplex, ax)
    return cplex

def plot_persistence_diagram(ax, persistence_points, dimension=0):
    dgm = []
    for pt in persistence_points:
        if pt[0] == dimension:
            dgm.append(pt[1])
    dgm = np.array(dgm)
    ax.scatter(dgm[:, 0], dgm[:, 1])

def make_persist_diagram(rel_file_path, radius=1, max_dim=2):
    points = load_pcd(os.getcwd()+rel_file_path, show=True)
    complex = rips_gudhi(points, radius, max_dimension=max_dim, show=False)
    persistence_points = complex.persistence()
    return gudhi.plot_persistence_diagram(persistence_points)

def make_persist_points(rel_file_path, radius=1, max_dim=2):
    points = load_pcd(os.getcwd()+rel_file_path, show=True)
    complex = rips_gudhi(points, radius, max_dimension=max_dim, show=False)
    persistence_points = complex.persistence()
    return persistence_points


