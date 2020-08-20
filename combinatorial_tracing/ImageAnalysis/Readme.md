### Table of contents for the provided Jupyter notebook files

__Part1__DataStructure-BeadFitting__ contents:

1. Raw imaging data structure description

2. Organize the data and flatten the illumination profile

3. Run the rough alignment and fiducial bead fitting across all fields of view and all imaging rounds

__Part2__FittingSignalAndDecoding__ contents:

1. Segment nuclei based on the DAPI signal

2. Fit the signal of chromatin loci of each cell in each field of view and use the fitted fiducial data (see part 1) to align fitted loci

3. Run decoding analysis for each cell

4. Construct a parameter file to correct for chromatic aberrations

__Part3__Nuclear_Bodies__ contents:

1. For each cell determine the 3D locations of the nuclear bodies in the coordinate space of the drift corrected chromatin loci
