# import packages
from matplotlib.colors import ListedColormap
import numpy as np

# generate my colors
from matplotlib.colors import ListedColormap
# red
Red_colors = np.ones([256,4])
Red_colors[:,1] = np.linspace(1,0,256)
Red_colors[:,2] = np.linspace(1,0,256)
myReds = ListedColormap(Red_colors)
myReds_r = ListedColormap(np.flipud(Red_colors))
# blue
Blue_colors = np.ones([256,4])
Blue_colors[:,0] = np.linspace(1,0,256)
Blue_colors[:,1] = np.linspace(1,0,256)
myBlues = ListedColormap(Blue_colors)
myBlues_r = ListedColormap(np.flipud(Blue_colors))
# green
Green_colors = np.ones([256,4])
Green_colors[:,0] = np.linspace(1,0,256)
Green_colors[:,2] = np.linspace(1,0,256)
myGreens = ListedColormap(Green_colors)
myGreens_r = ListedColormap(np.flipud(Green_colors))
# cmaps
_myCmaps = [myReds, myReds_r, 
            myBlues, myBlues_r, 
            myGreens, myGreens_r]

def transparent_cmap(cmap, increasing_alpha=True, N=255, max_alpha=1):
    "Copy colormap and set a gradually changing alpha values"
    mycmap = cmap
    mycmap._init()
    if increasing_alpha:
        mycmap._lut[:,-1] = np.linspace(0, max_alpha, N+4)
    else:
        mycmap._lut[:,-1] = np.linspace(max_alpha, 0, N+4)
    return mycmap

def black_gradient(color, num_colors=256, max_alpha=1, transparent=False):
    """Generate a black backed linear color map"""
    color =  np.array(color[:3])
    # initialzie colors
    _colors = np.zeros([num_colors,4])
    for _i, _c in enumerate(color):
        _colors[:,_i] = np.linspace(0, _c, num_colors)
    if transparent:
        _colors[:, -1] = np.linspace(0, max_alpha, num_colors)
    else:
        _colors[:, -1] = max_alpha
    return ListedColormap(_colors)

def transparent_gradient(color, num_colors=256, max_alpha=1):
    """Generate a black backed linear color map"""
    color =  np.array(color[:3])
    # initialzie colors
    _colors = np.zeros([num_colors,4])
    _colors[:,:3] = color[np.newaxis,:]
    # set alpha
    _colors[:, -1] = np.linspace(0, max_alpha, num_colors)

    return ListedColormap(_colors)

def normlize_color(mat, vmin=None, vmax=None):
    """linearly Normalize extreme colors"""
    _mat = np.array(mat).copy()
    if vmin is None:
        vmin = np.nanmin(_mat)
    if vmax is None:
        vmax = np.nanmax(_mat)
    # based on vmin vmax, do thresholding
    _mat[_mat < vmin] = vmin
    _mat[_mat > vmax] = vmax
    _mat = (_mat - np.nanmin(_mat)) / (np.nanmax(_mat) - np.nanmin(_mat))
    return _mat

