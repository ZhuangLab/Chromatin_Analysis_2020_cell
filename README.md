# Chromatin_Analysis for 2020 Cell

Repository for codes analyzing sequential and combinatorial choromatin imaging data

This repository contains two main sections:

1. Combinatorial_tracing - contains the code and examples scripts used for analysing to combinatorial genome-scale chromatin tracing data. Code is separated into seaparate categories, each contained in an individual sub-folder. Each of these sub-folders contains a "functions" folder, in which all functions and classses are defined in .py files. In addition, it contains Jupyter notebook (.ipynb) files, demonstrating the use of the functions:

   a. BarcodeGeneration - contains the code and example scripts for generating barcodes and assigning them to genomic loci of interest

   b. ImageAnalysis - contains all code used to analyse the raw microscopy images and obtain a set of 3D positions of the genomic loci in each individual cell (as well as, where available, the transcriptional state and location of nuclear bodies).
   
   c. PostAnalysis - contains the code and example scripts used to perform statistical analysis on the 3D single-cell positions.

These functions are intended to be used with Phyton 2.7

2. Sequential_tracing - contains the code and examples scripts for analysing the sequential, chromosome-wide chromatin tracing data. This section is organized in the following sections: 

    a. Source - contains all function and classes within .py files which are organized as a module.

    b. ImageAnalysis - contains a Jupyter notebook (.ipynb) file with example scripts used to analyse the raw microscopy images and obtain a set of 3D postions of the genomic loci in chromosomes of each individual cells.

    c. PostAnalysis - contains example scripts to statistically analyze the 3D single-cell positions to reproduce reported results in the paper. 
    
    d. LibraryDesign - contains example scripts to generate encoding/primary probe libraries for sequential tracing experiment. 

These functions are compatible with Python 3.7 (Anaconda distribution, release for 2020.02). Please see further detailed installation guide in sequential_tracing folder. 


* Data download: The corresponding datasets could be found by DOI: 10.5281/zenodo.3928890

* Citation: If you happened to use our codes or dataset, please cite: [Link to the paper]()

    [This is a place holder for paper citation]

* Contributors: Bogdan Bintu, Pu Zheng, Seon Kinrot, Jun-Han Su and Xiaowei Zhuang.

* Correspondence: Bogdan Bintu (bbintu -at- g.harvard.edu), Pu Zheng (pu_zheng -at- g.harvard.edu) and Xiaowei Zhuang (zhuang -at- chemistry.harvard.edu).

July 9th, 2020
