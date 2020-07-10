import numpy as np 


## define zigzag
def find_loopout_regions(zxys, region_ids=None, 
                         method='neighbor', dist_th=1500, 
                         neighbor_region_num=5):
    """Function to find loopout, or zig-zag features within chromosomes.
    Inputs:

    Outputs:

    """

    # convert inputs
    from ..spot_tools.scoring import _local_distance
    from ..spot_tools.scoring import _neighboring_distance
    _zxys = np.array(zxys)

    # if region ids not specified, presume it is continuous
    if region_ids is None:
        region_ids = np.arange(len(_zxys))
    else:
        region_ids = np.array(region_ids)
    
    # identify distance to neighbors
    if method == 'neighbor':
        _nb_dists = _neighboring_distance(zxys, spot_ids=region_ids, neighbor_step=1)[:-1]
        _loopout_flags = np.zeros(len(zxys))
        _loopout_flags[1:] += (_nb_dists >= dist_th) * (1-np.isnan(_nb_dists))
        _loopout_flags[:-1] += (_nb_dists >= dist_th) * (1-np.isnan(_nb_dists))
        
        return _loopout_flags == 2

    elif method == 'local':
        _lc_dists = _local_distance(zxys, spot_ids=region_ids,
                                    sel_zxys=zxys, sel_ids=region_ids, local_size=neighbor_region_num)
        return _lc_dists >= dist_th

    else:
        raise ValueError(f"wrong input method:{method}, exit.")