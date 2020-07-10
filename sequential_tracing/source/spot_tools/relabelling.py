# This is functions to repick
import numpy as np 


# generate recombined spots
def generate_recombined_spots(repeat_cand_spots, repeat_ids, original_cand_spots, original_ids):
    """Function to re-assemble fitted original candidate spots and repeat candidate spots 
        to perform spot repick to determine relabeling spots
    """
    ## check inputs
    if len(repeat_cand_spots) != len(repeat_ids):
        raise IndexError(f"Wrong length of repeat candidate spots")
    if len(original_cand_spots) != len(original_ids):
        raise IndexError(f"Wrong length of original candidate spots")
    
    # initialize recombined spots
    recombined_cand_spots = [_pts for _pts in original_cand_spots]
    # loop through repeated spots
    for _id, _spots in zip(repeat_ids, repeat_cand_spots):
        _ind = np.where(np.array(original_ids)==_id)[0]
        if len(_ind) == 1:
            _ind = _ind[0]
        else:
            raise ValueError(f"index for region {_id} has {_ind} matches, not unique!")
        recombined_cand_spots[_ind] = _spots
    
    return recombined_cand_spots