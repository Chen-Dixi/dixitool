import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from typing import Tuple

def plot_2d(cartesian_CG, out_CG, filename:str, cartesian_color = 'r', out_color = 'b'):
    """
    Visualize Gaussian curve line
    """
    x_cartesian = cartesian_CG[0].numpy()
    y_cartesian = cartesian_CG[1].numpy()

    x_out = out_CG[0].numpy()
    y_out = out_CG[1].numpy()

    plt.clf() # clear current figure
    plt.plot(x_cartesian, y_cartesian, c=cartesian_color)
    plt.plot(x_out, y_out, c=out_color)
    plt.legend(['Cartesian Target', 'Gaussian Predict'])
    plt.title(filename.split('/')[-1])
    # set different color
    
    # output image
    plt.savefig(filename)

def plot_3d_wireframe(data: Tuple, title: Tuple[str, ...], color: Tuple[str, ...], savefile:str):
    """
    3D Plot, 用matlibplot画3D wirefram图
    Args:
        data: 都是CPU上的 Numpy array
    """
    sz = len(title)
    assert len(data) == len(title) == len(color)
    aspect = 1 / sz
    fig = plt.figure(figsize=plt.figaspect(aspect), facecolor=(1., 1., 1.))
    for i in range(1,sz+1):

        wireframe = fig.add_subplot(1,sz,i, projection='3d')
        x1,y1,v1 = data[i-1]
        wireframe.plot_wireframe(x1, y1, v1, colors=color[i-1])
        # title
        wireframe.set_title(title[i-1])
        # label
        wireframe.set_xlabel('x')
        wireframe.set_ylabel('y')
        wireframe.set_zlabel('value')

    plt.savefig(savefile)
    plt.clf()
    # will remove a specific figure instance from the pylab state machine
    # allow it to be garbage collected
    plt.close(fig)