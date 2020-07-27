# Sequential tracing for chromosome-wide imaging

## source code

by Pu Zheng

2020.07.21

This is a python module that contains all required functions to perform image analysis, post analysis and library design. 

This package contains functions to support the following analysis:

1. Basic imaging processing, including loading, cropping and imaging correction functions to pre-process raw z-stack 3D images. 

2. Spot finding, including spot fitting, and spot picking among candidates. 

3. Characterization of chromosomal structures including domains and compartments. 

4. Useful basic visualization tools for chromosome 3D structure, distance matrix and single-cell domains. 

5. Design primary probe library. 

To properly use functions for this module, function part 1-4 requires:

1. basic modules from Anaconda dsitribution

2. opencv-python

If you hope to use function 5, it further requires:

3. biopython

4. proper compile of seqint.pyd in [library_tools](https://github.com/ZhuangLab/Chromatin_Analysis_2020_cell/tree/master/sequential_tracing/source/library_tools) folder. To compile this cython unit, go to folder: ./library_tools/C_tools and type in: (tested in Windows 10)

```console
    cd .\library_tools\C_Tools
    python setup.py install
```
