# Analysis for sequential tracing

by Pu Zheng

2020.07.09

This folder provides source codes and example jupyter scripts for sequential DNA-FISH tracing of the whole chromosome structure in single cells.

Components:

1. [source](https://github.com/ZhuangLab/Chromatin_Analysis_2020_cell/tree/master/sequential_tracing/source) - contains a python module (could be directly imported if it is added to sys.path) contains all the basic functions required for sequential DNA-FISH analysis.

2. [ImageAnalysis](https://github.com/ZhuangLab/Chromatin_Analysis_2020_cell/tree/master/sequential_tracing/ImageAnalysis) - contains an example jupyter notebook to demonstrate analysis from raw microscopy images to 3D coordinates of designed genomic loci in individual cells.

3. [PostAnalysis](https://github.com/ZhuangLab/Chromatin_Analysis_2020_cell/tree/master/sequential_tracing/PostAnalysis) - contains three jupyter notebooks to perform further analysis and generate figures based on previously obtained 3D coordinates and transcription information.

4. [LibraryDesign](https://github.com/ZhuangLab/Chromatin_Analysis_2020_cell/tree/master/sequential_tracing/LibraryDesign) - contains a jupyter notebook to demonstrate our workflow to design a primary oligonucleotide probe library. 

These functions are compatible with Python 3.7 ([Anaconda distribution, release for 2020.02](https://repo.anaconda.com/archive/Anaconda3-2020.02-Windows-x86_64.exe)). Please see more details in [installation_guide](https://github.com/ZhuangLab/Chromatin_Analysis_2020_cell/blob/master/sequential_tracing/installation_guide.md)

