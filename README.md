# Chromatin_Tracing_Analysis

Repository for codes analyzing sequential and combinatorial choromatin imaging data

This repository contains two main sections, as following:

1. Combinatorial_tracing - contains the code and examples scripts used for analysing to combinatorial genome-scale chromatin tracing data. Code is separated into seaparate categories, each contained in an individual sub-folder. Each of these sub-folders contains a "functions" folder, in which all functions and classses are defined in .py files. In addition, it contains Jupyter notebook (.ipynb) files, demonstrating the use of the functions:

   a. BarcodeGeneration - contains the code and example scripts for generating barcodes and assigning them to genomic loci of interest

   b. ImageAnalysis - contains all code used to analyse the raw microscopy images and obtain a set of 3D positions of the genomic loci in each individual cell (as well as, where available, the transcriptional state and location of nuclear bodies).
   
   c. PostAnalysis - contains the code and example scripts used to perform statistical analysis on the 3D single-cell positions.

These functions are intended to be used with Phyton 2.7

2. Sequential_tracing - contains the code and examples scripts for analysing the sequential, chromosome-wide chromatin tracing data. This section is organized in the following sections: 

    a. Source - contains all function and classes within .py files which are organized as a module.

    b. ImageAnalysis - contains a Jupyter notebook (.ipynb) file with example scripts used to analyse the raw microscopy images and obtain a set of 3D postions of the genomic loci in chromosomes of each individual cells.

    c. PostAnalysis - contains example scripts to statistically analyze the 3D single-cell positions.

These functions are compatible with Python 3.7 (Anaconda distribution, newest release). Please see further detailed installation guide in sequential_tracing folder. 

3. Data - contains tab delimited files with the 3D positions of the chromatin loci imaged in each cell and, where available the transcriptional activity, the distance from the nuclear landmarks measured and the cell-cycle state.


The algoritms writen by Bogdan Bintu, Pu Zheng, Seon Kinrot and Jun-Han Su.

Correspondence: Bogdan Bintu (bbintu -at- g.harvard.edu) and Xiaowei Zhuang.

June 6th, 2020
