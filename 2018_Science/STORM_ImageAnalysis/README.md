# ChromatinImaging/STORM Image Analysis

This contains scripts and functions used for analysing the STORM raw data.

The main script is StandardAnalysis.ipynb

The functions for STORM fitting (in the storm-analysis subfolder) are originally written by Hazen Babcock.
Minor modifications and wrappers have been added by Bogdan Bintu.
To run the fitting functions associated with this file on a windows machine add ...\storm-analysis\windows_dll to environmental variables.

The output of this analysis is a list of coordinates for the STORM-localizations corresponding to 30kb segments comprising the genomic region labelled.
