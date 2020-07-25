# Sequential tracing for chromosome-wide imaging

## ImageAnalysis

by Pu Zheng

2020.07.25

The jupyter notebook: 
1. [Example_workflow_sequential_DNA.ipynb ](https://github.com/ZhuangLab/Chromatin_Analysis_2020_cell/blob/master/sequential_tracing/ImageAnalysis/Example_workflow_sequential_DNA.ipynb) in this folder provide example scripts to analyze raw z-stack images into spatial coordinates. 


    1. Initialize Cell_List class
    
    2. Segmentation for all field-of-views
    
    3. Create cell objects
    
    4. Crop sequential images for each cell
    
    5. Pick chromosomes
    
    6. Multi-fitting candidate spots for chromosomes
    
    7. Pick spots to acquire region coordinates
    
    8. Generate population median distance map / contact probability map
    
    9. Save processed data into single file for further analysis