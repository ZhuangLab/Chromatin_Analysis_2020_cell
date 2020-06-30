# LibraryDesigner3

Adopted from Bogdan's LibraryDesigner to make it work in python3
[Link to Original Repository](https://github.com/BogdanBintu/ChromatinImagingV2)

Created by:
Pu Zheng, 2019.06.04
Modified by:


## 1. Conversion from 2 to 3:

First of all you need to have 2to3 installed in your environment: [2to3 tutorial](https://docs.python.org/2/library/2to3.html)

Do conversion by typing the following into ternimal:

```console
2to3 -w -n -o LibraryDesign3 LibraryDesign
```

Then you may see something like this:

```python
@@ -508,7 +508,7 @@
             location+=1
             if location+2*pb_len>len(gene_seq):
                 break
-        print gene_name+" (pairs: "+str(num_pairs)+") done!"
+        print(gene_name+" (pairs: "+str(num_pairs)+") done!")
         fastawrite(folder_save+os.sep+gene_name+'_probes.fasta',pb_names,pb_seqs)
         pb_names_f.append(pb_names)
         pb_seqs_f.append(pb_seqs)
```

This means after line 508, there's a line of *"print"* function to be transformed from python2 format (which didn't use brackets) into python3 format. 

## 2. Build new seqint package in python3

To run the following commands you need to install *cython* in your environment.

```console
cd LibraryDesign3\C_Tools
python setup.py install
```

This will allow you to install seq2int packages into your environment.

## Test your installation by:

```python
import sys
sys.path.append(r'path/to/library-designer-folder/LibraryDesign3')
import LibraryDesigner as ld
import LibraryTools as lt
from seqint import seq2Int, seq2Int_rc
# some other packages may required
import Bio

```

If no error occurs then you are good to go!
