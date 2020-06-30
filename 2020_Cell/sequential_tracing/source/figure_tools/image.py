# required packages
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
import os, glob, sys, time
# 3d plotting
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes_lewiner # 3d cloud calculation

# load common parameters
from .. import visual_tools
from .. import _distance_zxy
from ..spot_tools.translating import normalize_center_spots
from .color import transparent_gradient
from . import _dpi,_single_col_width,_double_col_width,_single_row_height,_ref_bar_length, _ticklabel_size,_ticklabel_width,_font_size

_color_dict={
    'A':[1.,0.,0.],
    'B':[0.,0.,1.],
}

def visualize_2d_projection(im, kept_axes=(1,2), ax=None, 
                            crop=None, cmap='gray', color_limits=None, figure_alpha=1, 
                            add_colorbar=False, add_reference_bar=True, 
                            reference_bar_length=_ref_bar_length, reference_bar_color=[1,1,1], 
                            figure_width=_single_col_width, figure_dpi=_dpi, imshow_kwargs={},
                            ticklabel_size=_ticklabel_size, ticklabel_width=_ticklabel_width, font_size=_font_size,
                            save=False, save_folder='.', save_basename='2d-projection.png', verbose=True):
    """Function to draw a 2d-projection of a representatitve image
    Inputs:
        im: imput image, np.ndarray
        kept_axes: kept axies to project this image, tuple of 2 (default: (1,2))
        ax: input axis to plot image, if not given then create a new one, matplotlib.axis (default: None)
        crop: crop for 2d projected image, None or np.ndarray (defualt: None, no cropping)
        cmap: color-map used for the map, string or matplotlib.cm (default: 'gray')
        color_limits: color map limits for image, array-like of 2 (default: None,0-max)
        add_colorbar: whether add colorbar to image, bool (defualt: False)
        add_reference_bar: whether add refernce bar to image, bool (default: True)
        reference_bar_length: length of reference bar, float (default: bar length representing 1um)
        figure_width: figure width in inch, float (default: _single_col_width=2.25) 
        figure_dpi: figure dpi, float (default: _dpi=300)
        save: whether save image into file, bool (default: False)
        save_folder: where to save image, str of path (default: '.', local)
        save_basename: file basename to save, str (default: '2dprojection.png')
        verbose: say something during plotting, bool (default: True)
    Output:
        ax: matplotlib.axis handle containing 2d image
    """
    ## check inputs
    _im = np.array(im)
    if len(_im.shape) < 2:
        raise IndexError(f"Wrong shape of image:{_im.shape}, should have at least 2-axes")
    if len(kept_axes) != 2:
        raise IndexError(f"Wrong length of kept_axis:{len(kept_axes)}, should be 2")
    for _a in kept_axes:
        if _a >= len(_im.shape):
            raise ValueError(f"Wrong kept axis value {_a}, should be less than image dimension:{len(_im.shape)}")
    # project image
    _proj_axes = tuple([_i for _i in range(len(_im.shape)) if _i not in kept_axes])
    _im = np.mean(_im, axis=_proj_axes)
    # crops
    if crop is None:
        crop_slice = tuple([slice(0, _im.shape[0]), slice(0, _im.shape[1])])
    else:
        crop = np.array(crop, dtype=np.int)
        if crop.shape[0] != 2 or crop.shape[1] != 2:
            raise ValueError(f"crop should be 2x2 array telling 2d crop of the image")
        crop_slice = tuple([slice(max(0, crop[0,0]), min(_im.shape[0], crop[0,1])),
                           slice(max(0, crop[1,0]), min(_im.shape[1], crop[1,1])) ])
    _im = _im[crop_slice] # crop image
    # color_limits
    if color_limits is None:
        color_limits = [0, np.max(_im)]
    elif len(color_limits) < 2:
        raise IndexError(f"Wrong input length of color_limits:{color_limits}, should have at least 2 elements.")
    ## create axis if necessary
    if ax is None:
        fig, ax = plt.subplots(figsize=(figure_width, figure_width*_im.shape[0]/im.shape[1]*(5/6)**add_colorbar),
                               dpi=figure_dpi)
        #grid = plt.GridSpec(3, 1, height_ratios=[5,1,1], hspace=0., wspace=0.2)
        _im_obj = ax.imshow(_im, cmap=cmap, 
                            vmin=min(color_limits), vmax=max(color_limits), 
                            alpha=figure_alpha, **imshow_kwargs)
        ax.axis('off')
        #ax.tick_params('both', width=ticklabel_width, length=0, labelsize=ticklabel_size)
        #ax.get_xaxis().set_ticklabels([])
        #ax.get_yaxis().set_ticklabels([])
        if add_colorbar:
            cbar = plt.colorbar(_im_obj, ax=ax, shrink=0.9)
        if add_reference_bar:
            if isinstance(reference_bar_color, list) or isinstance(reference_bar_color, np.ndarray):
                _ref_color = reference_bar_color[:3]
            else:
                _ref_color = matplotlib.cm.get_cmap(cmap)(255)
            ax.hlines(y=_im.shape[0]-10, xmin=_im.shape[1]-10-reference_bar_length, 
                      xmax=_im.shape[1]-10, color=_ref_color, linewidth=2, visible=True)
    
    if save:
        save_basename = f"axis-{kept_axes}"+'_'+save_basename
        if '.png' not in save_basename:
            save_basename += '.png'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder, save_basename), dpi=figure_dpi, transparent=True)
    
    return ax


def visualize_2d_gaussian(im, spot, color=[0,0,0], kept_axes=(1,2), ax=None, crop=None, 
                          spot_alpha=0.8, spot_sigma_scale=1, 
                          plot_im=True, im_cmap='gray', im_background='black', color_limits=None, figure_alpha=1, 
                          add_colorbar=False, add_reference_bar=True, 
                          reference_bar_length=_ref_bar_length,
                          figure_width=_single_col_width, figure_dpi=_dpi, 
                          figure_title='',
                          projection_kwargs={},
                          ticklabel_size=_ticklabel_size, ticklabel_width=_ticklabel_width, font_size=_font_size,
                          save=False, save_folder='.', save_basename='2d-projection.png', verbose=True):
    """Function to visualize a 2d-gaussian in given spot object
    Inputs:
    
    Outputs:
        ax: matplotlib.axis including plotted gaussian feature"""

    ## check inputs
    _im = np.array(im)
    if len(_im.shape) < 2:
        raise IndexError(f"Wrong shape of image:{_im.shape}, should have at least 2-axes")
    if len(kept_axes) != 2:
        raise IndexError(f"Wrong length of kept_axis:{len(kept_axes)}, should be 2")
    for _a in kept_axes:
        if _a >= len(_im.shape):
            raise ValueError(f"Wrong kept axis value {_a}, should be less than image dimension:{len(_im.shape)}")
    # project image
    _proj_axes = tuple([_i for _i in range(len(_im.shape)) if _i not in kept_axes])
    _im = np.mean(_im, axis=_proj_axes)
    # crops
    if crop is None:
        crop_slice = tuple([slice(0, _im.shape[0]), slice(0, _im.shape[1])])
    else:
        crop = np.array(crop, dtype=np.int)
        if crop.shape[0] != 2 or crop.shape[1] != 2:
            raise ValueError(f"crop should be 2x2 array telling 2d crop of the image")
        crop_slice = tuple([slice(max(0, crop[0,0]), min(_im.shape[0], crop[0,1])),
                           slice(max(0, crop[1,0]), min(_im.shape[1], crop[1,1])) ])
    _im = _im[crop_slice] # crop image
    
    # generate image axis
    if plot_im:
        ax = visualize_2d_projection(im, kept_axes=kept_axes, ax=ax, crop=crop, cmap=im_cmap,
                                     color_limits=color_limits, figure_alpha=figure_alpha,
                                     add_colorbar=add_colorbar, add_reference_bar=add_reference_bar,
                                     reference_bar_length=reference_bar_length, figure_width=figure_width,
                                     figure_dpi=figure_dpi, 
                                     imshow_kwargs=projection_kwargs, 
                                     ticklabel_size=ticklabel_size, ticklabel_width=ticklabel_width,
                                     save=False, save_folder=save_folder, save_basename=save_basename,
                                     verbose=verbose)
    elif ax is None:
        fig, ax = plt.subplots(figsize=(figure_width, figure_width*_im.shape[0]/im.shape[1]*(5/6)**add_colorbar),
                               dpi=figure_dpi)
    # extract spot parameters
    _spot_center = [spot[1:4][_i] for _i in kept_axes]
    _spot_sigma = [spot[5:8][_i] for _i in kept_axes]
    # get transparent profile
    _spot_im = visual_tools.add_source(np.zeros(np.shape(_im)), h=1, pos=_spot_center, sig=np.array(_spot_sigma)*spot_sigma_scale)
    # plot image
    ax.imshow(_spot_im, cmap=transparent_gradient(color, max_alpha=spot_alpha), vmin=0, vmax=1,)

    # save
    if save:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if '.png' not in save_basename:
            save_basename += '.png'
        save_filename = os.path.join(save_folder, save_basename)
        plt.savefig(save_filename, transparent=True)
    
    return ax


def chromosome_structure_3d_rendering(spots, ax3d=None, cmap='Spectral', 
                                      distance_zxy=_distance_zxy, 
                                      center=True, pca_align=False, image_radius=2000,
                                      marker_size=6, marker_alpha=1, 
                                      marker_edge=False, 
                                      background_color=[0,0,0], 
                                      line_search_dist=3, 
                                      line_width=1, line_alpha=1, depthshade=False,
                                      view_elev_angle=0, view_azim_angle=90, 
                                      add_reference_bar=True, reference_bar_length=1000, 
                                      reference_bar_width=2, reference_bar_color=[1,1,1],
                                      add_colorbar=True, cbar_shrink=1.0,
                                      cbar_label=None, cbar_tick_labels=None,
                                      tick_label_length=_ticklabel_size, tick_label_width=_ticklabel_width, 
                                      font_size=_font_size, figure_width=_single_col_width, figure_dpi=_dpi,
                                      figure_title='',
                                      save=False, save_folder='.', save_basename='3d-projection.png', verbose=True):
    """Function to visualize 3d rendering of chromosome structure
    Inputs:
    
    
    Outputs:
        ax: matplotlib axes object containing chromosome structures
    """
    ## check inputs
    # spots
    _spots = np.array(spots)
    if len(np.shape(_spots)) != 2:
        raise IndexError(f"Wrong _spots dimension, should be 2d array but {np.shape(_spots)} is given")
    # prepare spots
    if np.shape(_spots)[1] == 3:
        _zxy = _spots 
    else:
        _zxy = _spots[:,1:4] * distance_zxy[np.newaxis,:]
    _n_zxy = normalize_center_spots(_zxy, distance_zxy=distance_zxy,
                                    center_zero=center, scale_variance=False, 
                                    pca_align=pca_align, scaling=1)
    _valid_inds = (np.isnan(_n_zxy).sum(1) == 0)
    # set dimension
    if image_radius is None:
        _radius = np.nanmax(np.abs(_n_zxy)) + reference_bar_length
    else:
        _radius = image_radius + reference_bar_length 
    # cmap
    if isinstance(cmap, str):
        _cmap = matplotlib.cm.get_cmap(cmap)
        _colors = np.array([np.array(_cmap(_i)[:4]) for _i in np.linspace(0,1,len(_spots))])
    elif isinstance(cmap, np.ndarray) or isinstance(cmap, list):
        if len(cmap) == len(_spots):
            # create cmap
            _cmap_mat = np.ones([len(cmap), 4])
            _cmap_mat[:,:len(cmap[0])] = np.array(cmap)
            _cmap = ListedColormap(_cmap_mat)
            # color
            _colors = np.array(cmap)
        else:
            raise IndexError(f"length of cmap doesnt match number of spots")
    elif isinstance(cmap, matplotlib.colors.LinearSegmentedColormap): 
        _cmap = cmap
        _colors = np.array([np.array(_cmap(_i)[:3]) for _i in np.linspace(0,1,len(_spots))])

    elif isinstance(cmap, matplotlib.colors.ListedColormap):
        _cmap = cmap
        _colors = np.array([np.array(_cmap(_i)[:3]) for _i in range(len(_spots))])
    else:
        raise TypeError(f"Wrong input type for cmap:{type(cmap)}, should be str or matplotlib.colors.LinearSegmentedColormap")
    # extract colors from cmaps
    
    # background and reference bar color
    _back_color = np.array(background_color[:3])
    if isinstance(reference_bar_color, np.ndarray) or isinstance(reference_bar_color, list):
        _ref_bar_color = np.array(reference_bar_color[:3])
    elif reference_bar_color is None:
        _ref_bar_color = 1 - _back_color
    else:
        raise TypeError(f"Wrong input type for reference_bar_color:{type(reference_bar_color)}")

    ## plot
    # initialize figure
    if ax3d is None:
        fig = plt.figure(figsize=(figure_width, figure_width), dpi=figure_dpi)
        ax3d = fig.gca(projection='3d')
        try:
            ax3d.set_aspect('equal')
        except:
            pass
    else:
        if ax3d.__module__ != 'matplotlib.axes._subplots':
            raise TypeError(f"Wrong input type for ax3d:{type(ax3d)}, it should be Axec3DsSubplot object.")
    # background color
    ax3d.set_facecolor(_back_color)
    if marker_edge:
        if _colors.shape[1] == 3:
            _edge_colors = [[0,0,0, marker_alpha] for _c in _colors[_valid_inds]]
        else:
            _edge_colors = _colors[_valid_inds].copy()
            _edge_colors[:,:3] = 0
    else:
        _edge_colors = 'none'
    # scatter plot
    _sc = ax3d.scatter(_n_zxy[_valid_inds,1], _n_zxy[_valid_inds,2], _n_zxy                              [_valid_inds,0],
                       c=_colors[_valid_inds], s=marker_size, depthshade=depthshade, #alpha=marker_alpha,
                       edgecolors=_edge_colors, 
                       linewidth=0.1)
    # plot lines between spots
    if _colors.shape[1] == 3:
        _line_alphas = [line_alpha for _c in _colors]
    else:
        _line_alphas = _colors[:,-1]
    for _i,_coord in enumerate(_n_zxy[:-1]):
        for _j in range(line_search_dist):
            if len(_n_zxy) > _i+_j+1:
                _n_coord = _n_zxy[_i+_j+1]
                # if both coordinates are valid:
                if _valid_inds[_i] and _valid_inds[_i+_j+1]:
                    # calculate midpoint
                    _mid_coord = (_coord + _n_coord) / 2
                    # plot the connecting lines in two halves:
                    ax3d.plot([_coord[1],_mid_coord[1]],
                            [_coord[2],_mid_coord[2]],
                            [_coord[0],_mid_coord[0]],
                            color = _colors[_i], alpha=_line_alphas[_i], linewidth=line_width)
                    ax3d.plot([_mid_coord[1],_n_coord[1]],
                            [_mid_coord[2],_n_coord[2]],
                            [_mid_coord[0],_n_coord[0]],
                            color = _colors[_i+_j+1], alpha=_line_alphas[_i+_j+1], linewidth=line_width)
                    break
    # plot reference bar
    if add_reference_bar:
        # standardize angles
        _azim = (view_azim_angle%360) / 180 * np.pi
        _elev = (view_elev_angle%360) / 180 * np.pi
        # start coordinate for colorbar
        _bar_starts = np.array([-np.cos(_elev),
                                -np.sin(_azim) +np.sin(_elev)*np.cos(_azim),
                                np.cos(_azim) +np.sin(_elev)*np.sin(_azim),
                                ]) * _radius
        # ongoing vector for colorbar:
        _bar_vector = np.array([0,
                                -np.sin(_azim),
                                np.cos(_azim),
                                ]) * reference_bar_length
        # therefore, end of colorbar
        _bar_ends = _bar_starts + _bar_vector
        _ref_line = ax3d.plot([_bar_starts[1], _bar_ends[1]],
                              [_bar_starts[2], _bar_ends[2]], 
                              [_bar_starts[0], _bar_ends[0]], 
                              color=_ref_bar_color, 
                              linewidth=reference_bar_width)
                            
    # colorbar    
    if add_colorbar:
        import matplotlib.cm as cm
        _color_inds = np.where(_valid_inds)[0]
        norm = cm.colors.Normalize(vmax=_color_inds.max(), vmin=_color_inds.min())
        if verbose:
            print(f"-- add colorbar with colornorm: {norm}")
        m = cm.ScalarMappable(cmap=_cmap, norm=norm)
        m.set_array(_color_inds)
        #divider = make_axes_locatable(ax3d)
        #cax = divider.append_axes('bottom', size='6%', pad="2%")
        cb = plt.colorbar(m, ax=ax3d, orientation='horizontal', pad=0.01, shrink=cbar_shrink)
        cb.ax.tick_params(labelsize=font_size, width=tick_label_width, length=tick_label_length-1,pad=1)
        # set ticklabel
        if cbar_tick_labels is not None:
            _tick_coords = np.arange(0, len(_spots), int(len(_spots)/5))
            cb.set_ticks(_tick_coords)
            cb.set_ticklabels(np.array(cbar_tick_labels)[_tick_coords])
        if cbar_label is not None:
            cb.set_label(cbar_label, fontsize=font_size, labelpad=1)
        # border
        cb.outline.set_linewidth(tick_label_width)
    else:
        cb = None
    # axis view angle
    ax3d.grid(False)
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.axis('off')
    # title
    if figure_title is not None and figure_title != '':
        ax3d.set_title(figure_title, fontsize=font_size, pad=font_size*1.5)
    # view angle
    ax3d.view_init(elev=view_elev_angle, azim=view_azim_angle)
    # set limits
    ax3d.set_xlim([-_radius, _radius])
    ax3d.set_ylim([-_radius, _radius])
    ax3d.set_zlim([-_radius, _radius])
    # save
    if save:
        matplotlib.rcParams['pdf.fonttype'] = 42
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if '.png' not in save_basename and '.pdf' not in save_basename:
            save_basename += '.png'
        save_filename = os.path.join(save_folder, save_basename)
        if verbose:
            print(f"-- save 3d-rendering into file:{save_filename}")
        plt.savefig(save_filename, transparent=False)

    return ax3d, cb


def visualize_chromosome_3d_cloud(_spots, comp_dict, density_dict=None, 
                                  color_dict=_color_dict, 
                                  ax3d=None,
                                  im_radius=30, distance_zxy=_distance_zxy,
                                  center=True, pca_align=False,
                                  spot_variance=[1.,1.,1.], scaling=1., 
                                  cloud_thres = 1., cloud_alpha=0.6,
                                  max_projection_alpha=0.5, 
                                  elev_angle=30, azim_angle=120,
                                  pane_color=[1., 1., 1., 1.],
                                  axis_color=[0., 0., 0., 1.],
                                  figure_width=_single_col_width, figure_dpi=_dpi,
                                  tick_label_length=_ticklabel_size, 
                                  tick_label_width=_ticklabel_width, 
                                  tick_param_kwargs={}, show_outer_box=True,
                                  set_title=None, 
                                  save=False, save_folder='.', 
                                  save_basename='3d_surface', 
                                  return_density=False, 
                                  verbose=True):
    """Function to visualize chromosomal 3d cloud and its 2d projection"""
    ## check inputs
    
    # calculate density if necessary
    if density_dict is None:
        if verbose:
            print(f"-- calculating density_dict for chromosome")
        from ..compartment_tools.scoring import convert_spots_to_cloud
        _den = convert_spots_to_cloud(_spots, comp_dict=comp_dict,
                                      im_radius=im_radius, distance_zxy=distance_zxy,
                                      scaling=scaling, center=center, pca_align=pca_align,
                                      spot_variance=spot_variance, 
                                      return_scores=False, verbose=verbose)
    else:
        _den = {_k:_v  for _k,_v in density_dict.items()}
        if verbose:
            print(f"--* note: please make sure parameter matches for input density_dict and parameter matches")
    if len(_den) == 0:
        print(f"No compartment specified, exit!")
        return ax3d
    
    # check if color_dict matches
    for _k in _den:
        if _k not in color_dict:
            raise KeyError(f"key:{_k} exists in density dict, but color not given in color_dict!")

    ## plot surface
    # create image
    if ax3d is None:
        fig = plt.figure(figsize=(figure_width, figure_width), dpi=figure_dpi)
        ax3d = fig.gca(projection='3d')
        ax3d.set_aspect('equal')
    else:
        if ax3d.__module__ != 'matplotlib.axes._subplots':
            raise TypeError(f"Wrong input type for ax3d:{type(ax3d)}, it should be Axec3DsSubplot object.")
    # get image size        
    dz,dx,dy = np.shape(list(_den.values())[0])
    # for each density, plot3d
    if verbose:
        print(f"-- plotting thresholded density, threshold={cloud_thres}")
    for _c, _d in _den.items():
        # find 3d surface
        _verts, _faces, _normals, _values = marching_cubes_lewiner(_d, level=cloud_thres)
        # plot
        _sf = ax3d.plot_trisurf(_verts[:, 1], _verts[:,2], _faces, _verts[:, 0], # go as x, y, face, z
                                color=color_dict[_c], lw=1, alpha=cloud_alpha, shade=False)
    ## determine view axis settings
    # determine view angle
    ax3d.view_init(elev=elev_angle, azim=azim_angle)
    _elev = (elev_angle%360) /180 * np.pi
    _azim = (azim_angle%360) /180 * np.pi
    
    ## plot 2d porjection of surface
    if verbose:
        print(f"-- plotting 2d projection of densities")
    # manually thresholded density for later projection
    _thres_den = {_k:np.array(_v >= cloud_thres, dtype=np.float) for _k,_v in _den.items()}
    # get colors
    cmap_dict = {_k:transparent_gradient(_c) for _k,_c in color_dict.items()}
    # init a trim-edge slice
    _s = slice(1,-1)
    # generate 2d projection 
    from matplotlib.colors import Normalize
    Z, Y = np.mgrid[0:dz, 0:dy]
    for _k, _td in _thres_den.items():
        _zy_proj = np.mean(_td, axis=1)[_s,_s]
        _zy_norm = Normalize(vmin=0, vmax=_zy_proj.max()/max_projection_alpha)
        _zy_offset = int(-np.cos(_azim) * np.cos(_elev) > 0) * dx
        ax3d.plot_surface(_zy_offset * np.ones((dy, dz))[_s,_s], 
                          Y[_s,_s], 
                          Z[_s,_s],
                          rstride=1, cstride=1, 
                          facecolors=cmap_dict[_k](_zy_norm(_zy_proj)),
                          linewidth=0, shade=False)
    Z, X = np.mgrid[0:dz, 0:dx]
    for _k, _td in _thres_den.items():
        _zx_proj = np.mean(_td, axis=2)[_s,_s]
        _zx_norm = Normalize(vmin=0, vmax=_zx_proj.max()/max_projection_alpha)
        _zx_offset = int(-np.sin(_azim) * np.cos(_elev) > 0) * dy
        ax3d.plot_surface(X[_s,_s],
                          _zx_offset * np.ones((dz, dx))[_s,_s],
                          Z[_s,_s], 
                          rstride=1, cstride=1, 
                          facecolors=cmap_dict[_k](_zx_norm(_zx_proj)),
                          linewidth=0, shade=False)

    X, Y = np.mgrid[0:dx, 0:dy]
    for _k, _td in _thres_den.items():
        _xy_proj = np.mean(_td, axis=0)[_s,_s]
        _xy_norm = Normalize(vmin=0, vmax=_xy_proj.max()/max_projection_alpha)
        _xy_offset = int(-np.sin(_elev)>0) * dz
        ax3d.plot_surface(X[_s,_s],
                          Y[_s,_s],
                          _xy_offset * np.ones((dx, dy))[_s,_s], 
                          rstride=1, cstride=1, 
                          facecolors=cmap_dict[_k](_xy_norm(_xy_proj)),
                          linewidth=0, shade=False)
    if set_title is not None:
        ax3d.set_title(set_title, pad=2, fontsize=_font_size)

    ## grid
    ax3d.grid(False)
    # limits
    ax3d.set_xlim([0, dx])
    ax3d.set_ylim([0, dy])
    ax3d.set_zlim([0, dz])
    # labels
    ax3d.set_xlabel('X ($\mu$m)', fontsize=_font_size, labelpad=-10)
    ax3d.set_ylabel('Y ($\mu$m)', fontsize=_font_size, labelpad=-10)
    ax3d.set_zlabel('Z ($\mu$m)', fontsize=_font_size, labelpad=-10)
    # ticklabels
    ax3d.set_xticks(np.arange(0, dx, 10))
    ax3d.set_xticklabels(np.round((np.arange(0, dx, 10)-int(dx/2))\
                                  /(1000/np.min(distance_zxy)*scaling),1))
    ax3d.set_yticks(np.arange(0, dy, 10))
    ax3d.set_yticklabels(np.round((np.arange(0, dy, 10)-int(dy/2))\
                                  /(1000/np.min(distance_zxy)*scaling),1))
    ax3d.set_zticks(np.arange(0, dz, 10))
    ax3d.set_zticklabels(np.round((np.arange(0, dz, 10)-int(dz/2))\
                                  /(1000/np.min(distance_zxy)*scaling),1))
    # ticks
    ax3d.tick_params('both', labelsize=_font_size-1, 
                    width=tick_label_width-1, length=tick_label_length,
                    pad=-4, **tick_param_kwargs) # remove bottom ticklabels for ax3d

        
    # background grid-plane pane color
    ax3d.w_xaxis.set_pane_color(pane_color)
    ax3d.w_yaxis.set_pane_color(pane_color)
    ax3d.w_zaxis.set_pane_color(pane_color)
    ## axis color
    # axes
    ax3d.w_xaxis.line.set_color(axis_color) 
    ax3d.w_yaxis.line.set_color(axis_color) 
    ax3d.w_zaxis.line.set_color(axis_color)
    if show_outer_box:
        ax3d.set_frame_on(True)
        # other frames
        ax3d.w_xaxis.pane.set_edgecolor(axis_color)
        ax3d.w_yaxis.pane.set_edgecolor(axis_color)
        ax3d.w_zaxis.pane.set_edgecolor(axis_color)
    # set width
    # axes
    ax3d.w_xaxis.line.set_linewidth(tick_label_width)
    ax3d.w_yaxis.line.set_linewidth(tick_label_width)
    ax3d.w_zaxis.line.set_linewidth(tick_label_width)
    if show_outer_box:
        # other outer frames
        ax3d.w_xaxis.pane.set_linewidth(tick_label_width)
        ax3d.w_yaxis.pane.set_linewidth(tick_label_width)
        ax3d.w_zaxis.pane.set_linewidth(tick_label_width)

    # adjust plot size to include all notes
    plt.gcf().subplots_adjust(right=0.8, bottom=0.15)
    if save:
        if '.png' in save_basename:
            save_basename = save_basename.split('.png')[0]+f"_thres{cloud_thres:.2f}.png"
        else:
            save_basename = save_basename + f"_thres{cloud_thres:.2f}.png"
        save_filename = os.path.join(save_folder, save_basename)
        # do saving
        if verbose:
            print(f"-- saving 3d surface plot to file:{save_filename}")
        plt.savefig(save_filename, transparent=True)
        
    # return
    if return_density:
        return ax3d, _den
    else:
        return ax3d

