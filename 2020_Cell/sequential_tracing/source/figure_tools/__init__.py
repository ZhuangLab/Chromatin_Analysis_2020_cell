# import shared parameters from ImageAnalysis3
from .. import _distance_zxy, _sigma_zxy, _allowed_colors
from .. import visual_tools

## Define some global settings
_dpi = 600 # dpi required by figure
_single_col_width = 2.25 # figure width in inch if occupy 1 colomn
_double_col_width = 4.75 # figure width in inch if occupy 1 colomn
_single_row_height= 2 # comparable height to match single-colomn-width
_ref_bar_length = 1000 / _distance_zxy[-1]
_ticklabel_size=2
_ticklabel_width=0.5
_font_size=7.5

## import sub packages
# domain related plots
from . import domain
# raw image related plots
from . import image
# colors and color maps
from . import color 
# distmap etc
from . import distmap
# correlation between variables
from . import correlation
