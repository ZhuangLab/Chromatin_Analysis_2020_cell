# Chromatin_Analysis for Cell 2020 (https://doi.org/10.1016/j.cell.2020.07.032)

Repository for codes analyzing choromatin imaging data obtained using sequential hybridization or using DNA-MERFISH

This repository contains two main sections:
1. Combinatorial_tracing - contains the code and examples scripts used for analyzing genome-scale chromatin tracing data obtained using DNA-MERFISH, a combinatorial imaging method to massively multiplex FISH. Code is separated into separate categories, each contained in an individual sub-folder. Each of these sub-folders contains a "functions" folder, in which all functions and classes are defined in .py files. In addition, it contains Jupyter notebook (.ipynb) files, demonstrating the use of the functions:

   a. BarcodeGeneration - contains the code and example scripts for generating barcodes and assigning them to genomic loci of interest

   b. ImageAnalysis - contains all code used to analyse the raw microscopy images and obtain a set of 3D positions of the genomic loci in each individual cell (as well as, where available, the transcriptional state and location of nuclear bodies).
   
   c. PostAnalysis - contains the code and example scripts used to perform statistical analysis on the 3D single-cell positions.

These functions are intended to be used with Phyton 2.7

2. Sequential_tracing - contains the code and examples scripts for analyzing the high-resolution chromatin tracing data obtained using sequential hybridization. This section is organized in the following sections: 

    a. Source - contains all function and classes within .py files which are organized as a module.

    b. ImageAnalysis - contains a Jupyter notebook (.ipynb) file with example scripts used to analyse the raw microscopy images and obtain a set of 3D postions of the genomic loci in chromosomes of each individual cells.

    c. PostAnalysis - contains example scripts to statistically analyze the 3D single-cell positions to reproduce reported results in the paper. 
    
    d. LibraryDesign - contains example scripts to generate encoding/primary probe libraries for sequential tracing experiment. 

These functions are compatible with Python 3.7 (Anaconda distribution, release for 2020.02). Please see further detailed installation guide in sequential_tracing folder. 


* Data download: The corresponding datasets could be found by DOI: 10.5281/zenodo.3928890

* Citation: If you happened to use our codes or dataset, please cite: [Jun-Han Su, Pu Zheng, Seon S. Kinrot, Bogdan Bintu, and Xiaowei Zhuang. Genome-Scale Imaging of the 3D Organization and Transcriptional Activity of Chromatin. Cell 2020](https://doi.org/10.1016/j.cell.2020.07.032)

* Contributors: Bogdan Bintu, Pu Zheng, Seon Kinrot, Jun-Han Su and Xiaowei Zhuang.

* Correspondence: Bogdan Bintu (bbintu -at- g.harvard.edu), Pu Zheng (pu_zheng -at- g.harvard.edu) and Xiaowei Zhuang (zhuang -at- chemistry.harvard.edu).

July 9th, 2020
