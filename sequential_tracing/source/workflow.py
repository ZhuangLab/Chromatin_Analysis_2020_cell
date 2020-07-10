import os,sys
os.system(r'python E:\Users\puzheng\Documents\Startup_py3.py')
sys.path.append(r"E:\Users\puzheng\Documents")
from ImageAnalysis3 import get_img_info, visual_tools, corrections
import ImageAnalysis3 as ia

print(os.getpid())


fov_id = 57
cell_id = 0

@profile
def pre_process_cell():
    # specify initialization parameters:
    param = {'data_folder': 'Z:/20180911-IMR90_whole-chr21',
             'fov_id': fov_id,
             'cell_id': cell_id,
             'temp_folder': r'I:\Pu_temp',
             'save_folder': r'Z:\20180911-IMR90_whole-chr21\Analysis\dense-gpu',
             'map_folder': r'Z:\20180911-IMR90_whole-chr21\Analysis\dense-gpu\distmap',
             'num_threads': 15,
            }
    # initialize cell_data class:
    a = ia.classes.Cell_Data(param)

    # load color_usage
    a._load_color_info()
    # load encoding_scheme
    a._load_encoding_scheme()
    # load existing cell_info, dont run for the first time
    a._load_from_file('cell_info')
    # load segmentation
    a._load_segmentation(_denoise_window=0)
    # load drift info
    drift = a._load_drift_sequential(_size=650, _dynamic=True)


if __name__ == '__main__':
    pre_process_cell()

