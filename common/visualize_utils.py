import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from typing import List

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

def plot_hist(datas, bins: int, labels: List[str],
             colors: List[str], title:str,
             savefig_file: str,
             x_label: str = None,
             y_label: str = None,
             log: bool = False,
             density: bool = True,):
    r"""在一张图片里画出多组数据的直方图
    Args:
    - bins: Defines the number of equal-width bins in the range.
    """
    assert len(labels) == len(colors)

    fig, ax = plt.subplots()
    n, bin_edges, patches = ax.hist(datas, bins=bins, color=colors, label=labels, density=density, log=log)
    ax.set_title(title)
    ax.legend()
    
    if x_label is not None:
        ax.set_xlabel(x_label)
    
    if y_label is not None:
        ax.set_ylabel(y_label)

    plt.savefig(savefig_file)

    plt.clf()
    plt.close(fig)
    
def plot_3d_wireframe(data: List, title: List[str], color: List[str], savefile:str):
    """
    3D Plot, 用matlibplot画3D wirefram图
    Args:
        data: 都是CPU上的 Numpy array
    """
    sz = len(title)
    assert len(data) == len(title) == len(color)
    row = (sz // 2) + (sz % 2)
    col = 2
    fig, axs = plt.subplots(row, col, subplot_kw={'projection':'3d'},figsize=[col*6.4, row*4.8])
    for index in range(sz):
        i = index // 2
        j = index % 2
        x1,y1,v1 = data[index]
        axs[i,j].plot_wireframe(x1, y1, v1, colors=color[index])
        # title
        axs[i,j].set_title(title[index])
        # label
        axs[i,j].set_xlabel('x')
        axs[i,j].set_ylabel('y')
        axs[i,j].set_zlabel('value')
    
    plt.tight_layout()
    plt.savefig(savefile)
    plt.clf()
    # will remove a specific figure instance from the pylab state machine
    # allow it to be garbage collected
    plt.close(fig)