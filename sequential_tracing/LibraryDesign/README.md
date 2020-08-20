# Example code for library design

by Pu Zheng

2020.07.21

This folder contains a jupyter notebook describing the workflow to design oligo pool library for chromatin tracing by sequential hybridization, using Ch2 probe library design as an example:

1. [Design_library_chr2.ipynb](https://github.com/ZhuangLab/Chromatin_Analysis_2020_cell/blob/master/sequential_tracing/LibraryDesign/Design_library_chr2.ipynb)
    This jupyter provides scripts to design primary probe library for chr2 including first 50kb segments in every 250kb across the whole chr2, and a 50kb consecutive sequential library for 10Mb segment (chr2:27500000-38000000) with the following steps:

    1. Extract region sequences

    2. Design probe targeting sequences by probe_designer

    3. Assemble probes with primers and readout sequences

    4. Check quality

    5. Save probes
