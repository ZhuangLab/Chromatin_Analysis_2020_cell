# Sequential tracing for chromosome-wide imaging

## PostAnalysis

by Pu Zheng

2020.07.21

The jupyter notebooks in this folder provide example scripts to further analyze provided 3D coordinates of targeted chromosomal regions and transcription information from data deposit: **DOI:10.5281/zenodo.3928890**

1. [Part1_chr21_Domain_Analysis.ipynb](https://github.com/ZhuangLab/Chromatin_Analysis_2020_cell/blob/master/sequential_tracing/PostAnalysis/Part1_chr21_Domain_Analysis.ipynb)
    This jupyter provides scripts to perform population-averaged and single-cell based domain analysis for chr21 sequential tracing dataset:

    1. Load data from deposited datset (two replicates for chr21, with/without cell-cycle information)

    2. Population-averaged description of chr21 (FigS1F-K, M)

    3. Analysis for single-cell domains (Fig1C-D, I-M, and FigS1N)

    4. Characterization single-cell domains in G1/G2-S cells

2. [Part2_chr21_Compartment_Analysis_with_transcription.ipynb](https://github.com/ZhuangLab/Chromatin_Analysis_2020_cell/blob/master/sequential_tracing/PostAnalysis/Part2_chr21_Compartment_Analysis_with_transcription.ipynb)
    This jupyter provides scripts to perform A/B compartment and transcription related analysis for chr21:

    1. Load data from deposited datset (two replicates for chr21, with/without cell-cycle information)

    2. Compartment analysis of chr21 (Fig2A, FigS2A)

    3. Analysis of A/B density in single-cells (Fig2C,H-I, FigS2B-D)

    4. Characterization of compartments in G1, G2/S cells (FigS2D)


3. [Part3_chr2_Compartment_Analysis_and_Domain_Interaction.ipynb](https://github.com/ZhuangLab/Chromatin_Analysis_2020_cell/blob/master/sequential_tracing/PostAnalysis/Part3_chr2_Compartment_Analysis_and_Domain_Interaction.ipynb)
    This jupyter provides scripts to perform A/B compartment and domain-domain interaction  analysis for chr2.

    1. Load data from deposited datset (two replicates for chr2)

    2. Population-averaged description of chr2 (Fig2D, FigS2E-F)

    3. Analysis of A/B density in single-cells (Fig2F, Fig3D, FigS2G)

    4. Domain-domain interaction Analysis and relationship with A/B compartments (Fig3C, E-H)

    5. Compare p-arm of two replicates