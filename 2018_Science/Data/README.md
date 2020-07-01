### Example use of the imaging data:

https://github.com/BogdanBintu/ChromatinImaging/blob/master/PostAnalysis/DistanceMaps.ipynb

### Hi-C comparison

The Hi-C data used for the comparison with the imaging was downloaded from the following studies:

Rao et al. 2014 (IMR90, K562)
https://www.ncbi.nlm.nih.gov/pubmed/25497547

Rao et al. 2017 (HCT116 +/- AUXIN)
https://www.cell.com/cell/pdf/S0092-8674(17)31120-0.pdf


### Example for IMR90 on how to obtain the Hi-C data:

In our manuscript the "balanced" nomalization was performed at 5kb for a genomic region encompassing the region imaged in Juicebox.js. 
The Hi-C data was then exported and rebinned at 30kb by summation

Alternatively, with essentially equivalent results, we recommend:

Downloading the .hic data from: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525
and then rebinning from GSE63525_IMR90_combined.hic.gz the 5kb binned data at 30kb by summation.
(We found the normalization had minimal effect on the correlation coefficients at the 30kb resolution.)

### The genomic coordinates of the imaged regions in hg38 (from the first probe to the last probe):

chr21:28000071-29949939 (IMR90,K562,A549,HCT116+/-AUXIN - diffraction limited)

chr21:28000071-29229892  (IMR90 STORM) 

chr21:18627714-20577518 (IMR90 - diffraction limited)

chr21:34628096-37117534 (IMR90,K562,A549,HCT116+/-AUXIN - diffraction limited)  

### and in hg19:

chr21:29372390-31322257 (IMR90,K562,HCT116+/-AUXIN - diffraction limited)

IMR90: https://www.aidenlab.org/juicebox/?juiceboxURL=http://bit.ly/2yGCdNW

K562: https://www.aidenlab.org/juicebox/?juiceboxURL=http://bit.ly/2ObCld7

HCT116+/-AUXIN: https://www.aidenlab.org/juicebox/?juiceboxURL=http://bit.ly/2OgYneh

chr21:29372390-30602213 (IMR90 STORM) 

https://www.aidenlab.org/juicebox/?juiceboxURL=http://bit.ly/2OhYEOm

chr21:20000032-21949831 (IMR90 - diffraction limited)

https://www.aidenlab.org/juicebox/?juiceboxURL=http://bit.ly/2Ofskf3

chr21:36000395-38489834 (HCT116+/-AUXIN - diffraction limited)  

https://www.aidenlab.org/juicebox/?juiceboxURL=http://bit.ly/2OgYsyB


All the imaging experiments are perforemd at 30kb resolution.
Thus for the first region (chr21:28Mb-30Mb ) the 1st,2nd... segments are approximately: chr21:28000000-28030000, chr21:28030000-28060000...


### Note on imaging data:

The nan's in the imaging data (~5-10% of the positions in a given cell) correspond to "dim" spots that are of comparable brightness with the background fluctuations and were consequently removed. 
We found in practice that removing this percentage of low confidence/precision which occurs stochastically does not affect computing the average or the median inter-distances or even the domain boundary calling in which the nan's were simply not included in the analyis.
Similarly the "contact fractions" were computed relative to the number of total observations of each pair of chromatin segments, not relative to the total number of chromosomes.

(See the example images and example analysis code on the sample data for a better understanding of the raw data: https://github.com/BogdanBintu/ChromatinImaging/blob/master/Difraction-limited_ImageAnalysis/StandardAnalysis.ipynb)
