#  Copyright (c) 2019. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.


import matplotlib.cm as cmx
import matplotlib.patches as patches
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import cv2
import matplotlib.colors as clr


def plot_distribuition(data1, data2, data3=0, data4=0):
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], marker='.')
    for i in range(len(data1)):
        ax.text(data1[i, 0], data1[i, 1], data1[i, 2], '%s' % (data2[i].image_id), size=10, zorder=1, color='k')
    ax.scatter(data3[:, 0], data3[:, 1], data3[:, 2], marker='o')
    for i in range(len(data3)):
        ax.text(data3[i, 0], data3[i, 1], data3[i, 2], '%s' % (data4[i].image_id), size=10, zorder=1, color='m')

    plt.show()


def visualize_3d_gmm(points, w, mu, stdev, export=True):
    '''
    plots points and their corresponding gmm model in 3D
    Input:
        points: N X 3, sampled points
        w: n_gaussians, gmm weights
        mu: 3 X n_gaussians, gmm means
        stdev: 3 X n_gaussians, gmm standard deviation (assuming diagonal covariance matrix)
    Output:
        None
    '''

    n_gaussians = mu.shape[1]
    N = int(np.round(points.shape[0] / n_gaussians))
    # Visualize data
    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    axes.set_xlim([-1, 1])
    axes.set_ylim([-1, 1])
    axes.set_zlim([-1, 1])
    plt.set_cmap('Set1')
    colors = cmx.Set1(np.linspace(0, 1, n_gaussians))
    for i in range(n_gaussians):
        idx = range(i * N, (i + 1) * N)
        axes.scatter(points[idx, 0], points[idx, 1], points[idx, 2], alpha=0.3, c=colors[i])
        plot_sphere(w=w[i], c=mu[:, i], r=stdev[:, i], ax=axes)

    plt.title('3D GMM')
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    axes.view_init(35.246, 45)
    if export:
        if not os.path.exists('images/'): os.mkdir('images/')
        plt.savefig('images/3D_GMM_demonstration.png', dpi=100, format='png')
    plt.show()


def plot_sphere(w=0, c=[0,0,0], r=[1, 1, 1], subdev=10, ax=None, sigma_multiplier=3):
    '''
        plot a sphere surface
        Input:
            c: 3 elements list, sphere center
            r: 3 element list, sphere original scale in each axis ( allowing to draw elipsoids)
            subdiv: scalar, number of subdivisions (subdivision^2 points sampled on the surface)
            ax: optional pyplot axis object to plot the sphere in.
            sigma_multiplier: sphere additional scale (choosing an std value when plotting gaussians)
        Output:
            ax: pyplot axis object
    '''

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:complex(0, subdev), 0.0:2.0 * pi:complex(0, subdev)]
    x = sigma_multiplier*r[0] * sin(phi) * cos(theta) + c[0]
    y = sigma_multiplier*r[1] * sin(phi) * sin(theta) + c[1]
    z = sigma_multiplier*r[2] * cos(phi) + c[2]
    cmap = cmx.ScalarMappable()
    cmap.set_cmap('jet')
    c = cmap.to_rgba(w)

    ax.plot_surface(x, y, z, color=c, alpha=0.2, linewidth=1)

    return ax

def visualize_2D_gmm(points, w, mu, stdev, export=False):
    '''
    plots points and their corresponding gmm model in 2D
    Input:
        points: N X 2, sampled points
        w: n_gaussians, gmm weights
        mu: 2 X n_gaussians, gmm means
        stdev: 2 X n_gaussians, gmm standard deviation (assuming diagonal covariance matrix)
    Output:
        None
    '''
    n_gaussians = mu.shape[1]
    N = int(np.round(points.shape[0] / n_gaussians))
    # Visualize data
    fig = plt.figure()
    axes = plt.gca()

    plt.set_cmap('Set1')
    colors = cmx.Set1(np.linspace(0, 1, n_gaussians))
    for i in range(n_gaussians):
        idx = range(i * N, (i + 1) * N)
        plt.scatter(points[idx, 0], points[idx, 1], alpha=0.3, c=colors[i])
        for j in range(8):
            axes.add_patch(
                patches.Ellipse(mu[:, i], width=(j+1) * stdev[0, i], height=(j+1) *  stdev[1, i], fill=False, color=[0.0, 0.0, 1.0, 1.0/(0.5*j+1)]))
        plt.title('GMM')
    plt.xlabel('X')
    plt.ylabel('Y')

    if export:
        if not os.path.exists('images/'): os.mkdir('images/')
        plt.savefig('images/2D_GMM_demonstration.png', dpi=100, format='png')

    plt.show()



def plot_boxes(path_image, bb_detect, bb_gt):
    image = cv2.imread(path_image)
    cv2.rectangle(image, bb_detect.parse_int(bb_detect.top_left), bb_detect.parse_int(bb_detect.get_bottom_right()), (0, 255, 0))
    cv2.rectangle(image, bb_detect.parse_int(bb_gt.top_left), bb_detect.parse_int(bb_gt.get_bottom_right()), (255, 0, 0))
    cv2.imshow("Boundig Box (Green Detect, Red GT)", image)
    cv2.waitKey(1)


def plot_color_gradients(matrix, name: str):

    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, interpolation='nearest')
    ax.tick_params(axis='both', which='major', labelsize=20)

    for (i, j), z in np.ndenumerate(matrix*10):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', color="black", rotation="30")

    plt.xlabel("Model Clusters", fontsize=20)
    plt.ylabel("Number Dimensions", fontsize=20)
    plt.title(name, fontsize=25)

    divider = make_axes_locatable(ax)
    cax_size = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(cax, cax= cax_size)
    cbar.ax.tick_params(labelsize=15)

    plt.savefig(name+'.png', pad_inches=0)
    plt.show()


def plot_color_3dmap(matrix, name: str):

    ax = plt.axes(projection='3d')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.plot_surface(matrix[:,:,0], matrix[:,:,1], matrix[:,:,2], rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none');

    plt.xlabel("Model Clusters", fontsize=20)
    plt.ylabel("Number Dimensions", fontsize=20)
    plt.title(name, fontsize=25)

    plt.savefig(name+'.png', pad_inches=0)
    plt.show()