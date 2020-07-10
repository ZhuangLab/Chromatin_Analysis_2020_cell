# shared variables
_primer_folder = r'\\smirnov\Chromatin_NAS_2\Libraries\Primers'
_readout_folder = r'\\smirnov\Chromatin_NAS_2\Libraries\Readouts'
_genome_folder = r'\\SMIRNOV\Chromatin_NAS_2\Libraries\Genomes\human\hg38'

_adaptor_site_names = ['Stv_82', 'Stv_1', 'Stv_79']

# shared functions
def _rand_seq_generator(length=1, dna_alphabet=['A', 'C', 'G', 'T']):
    from random import choice
    from Bio.Seq import Seq
    from Bio.Alphabet import IUPAC
    _rand_seq = [choice(dna_alphabet) for _i in range(length)]
    _rand_seq = ''.join(_rand_seq)
    return SeqRecord(Seq(_rand_seq))


# import sub packages
from Bio.SeqRecord import SeqRecord
from . import LibraryDesigner
from . import LibraryTools
from . import readouts
from . import probes
from . import sequences