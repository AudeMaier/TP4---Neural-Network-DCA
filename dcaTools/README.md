# dcaTools

**dcaTools** are a collection of tools used in conjunction with the lbsDCA code.
The scripts in this repository contain sequence preparation tools to prepare input to lbsDCA, as well as tools used to pre- and post-process and analyse the DCA output.

A worked-out example using dcaTools can be found [here](https://link.springer.com/protocol/10.1007%2F978-1-4939-9608-7_16). Please consider citing it in case you make use of dcaTools.

# Compilation and Install 
The objective of this repository is to have a homogeneous set of scripts written in python3

To run the scripts, the user must have the following (standard) python packages installed:

* numpy
* scipy
* pandas
* sklearn
* matplotlib
* biopython

These can easily be installed by your favorite python package manager, for instance with:

```shell
$ pip install numpy scipy pandas sklearn matplotlib biopython

```
In the above command, make sure that the pip installation is configured to install Python 3 packages. If needed, the explicit pip3 installer can be used.

Furthermore, a working copy of the hmmer software is needed for some sequence handling scripts (see http://hmmer.org). In Ubuntu, this can be obtained by
```shell
$ sudo apt-get install hmmer
```
For local non-root installation, please refer to the hmmer documentation (http://hmmer.org/documentation.html).

To use the python code in your scripts, make sure that /path/to/dcaTools is in your PYTHONPATH, e.g.
```shell
echo 'export PYTHONPATH=$PYTHONPATH:/path/to/dcaTools/' >> .bashrc
```
To use the comamnd line version of the scripts, add the bin subfolder of dcaTools to your PATH variable, i.e.
```shell
echo 'export PATH=$PATH:/path/to/dcaTools/bin/' >> .bashrc
```

# License 
This code is distributed under the GPL 2+ license. Please feel free to fork it, modify it, re-use at will. Please cite the present repository in your redistributed code. All the code provided here is distributed as is, without any guarantee of functioning.

# Feedback
If you enjoy the use of this repository, you can contribute by reporting bugs in Issues section. If you would like to see any additions or improvements that might be useful to others, please post your suggestions in the Issues section.

# Contents and organization
dcaTools is organized around four main python modules:

* ***dca.py***: Set of functions for processing and analyzing DCA results, compatible with the output format of lbsDCA.

* ***sequenceHandler.py***: Set of functions for the pre-processing of sequence data in view of performing DCA or phylogenetic sequence analysis.

* ***plotting.py***: Plotting utilities for displaying DCA and sequence analysis results.

* ***seqStats.py***: Sequences and MSAs statistcs measures (Conservation, Shannon Entropy, ... ) 

The mose useful scripts have stand alone scripts (located in the bin/ folder) which can be directly executed from command line. Here is a short list of these stand alone scripts:

* ***binarizeMSA*** 
  Transforms the MSA into a binarized form by extending each amino acid-position to 21 binary positions
    * usage: binarizeMSA [-h] [--output OUTPUT] baseMSA
  
* ***combineMaps***
  Combines two pdb-generated contacts maps by taking the minimal distance for each residue-pair.
    * usage: combineMaps [-h] mapFiles [mapFiles ...] combinedMap

* ***compareMaps***
  Combines two aligned distance maps in one, putting them in the two different triangular parts of the resulting map.
    * usage: compareMaps [-h] mapFiles mapFiles combinedMap

* ***countSeqs***
  Prints the number of sequences in a fasta file
    * usage: countSeqs [-h] msa

* ***dummyMap***
  Creates a dummy distance map of size NxN, with 1.0 on the diagonal and off-diagonal elements set to 100.
    * usage: dummyMap [-h] N outFile
    
* ***filterMSAByGapContent***
  Filters sequences in an MSA, keeping only those having less than a user-specified fraction of gaps.
    * usage: filterSequencesByGapContent [-h] inMSA gapThreshold filteredMSA

* ***filterMSAOfInserts***
  Filters sequences in an MSA of inserts (. for gaps, lower case letters for amino-acids), which can be found if working with PFAM alignments.
    * usage: filterMSAOfInserts [-h] inMSA filteredMSA

* ***listInterContacts***
  List the predicted inter-protein interface contacts as defined by the eLIFE (Hopf2014) criterion.
  The contacts can be mapped to a reference sequence numbering.
    * listInterContacts [-h] [-hmm1 HMM1] [-hmm2 HMM2] [-seq1 SEQ1] [-seq2 SEQ2] [-cutoff CUTOFF] dcaFile N1 Beff
    
* ***listTopContacts***
  Returns the list of the N top ranked contacts. The contacts can be mapped to a reference sequence numbering.
    * usage: listTopContacts [-h] [-hmm HMM] [-seq SEQ] [-ik] dcaFile Ntop

* ***makeTaxaTags***
  Tags all sequences in an MSA, based on user-specified taxnonic queries.
    * usage: makeTaxaTags [-h] inMSA lineage taxaQuery [taxaQuery ...] tagList

* ***mapPDB***
  Computed the distance map of a PDB structure, and aligns it to an hmm defining a protein family.
    * usage: mapPDB [-h] [-hmm1 HMM1] [-hmm2 HMM2] pdbFile chainIds mapFile

* ***mergeSeqDomains***
  Merges different domains hits in the same MSA into single sequences.
    * usage: mergeSeqDomains [-h] inMSA outMSA
	 
* ***msaEnergies***
  Computes the energies of sequences in an MSA given an inferred Potts Model.
    * usage: msaEnergies [-h] [-o O] msa prm
	 
* ***pca***
  Performs principal component analysis on sequences and displays them as 2D histograms.
    * usage: pca [-h] [--over OVER] baseMSA axes

* ***pdbToFasta***
  Extracts the amino-acid sequence from a pdb file and saves it in fasta format.
    * usage: pdbToFasta [-h] pdbFile chainId fastaFile
	  
* ***plotDistDistribution***
  Plots the distribution of distances of DCA predicted contacts.
    * usage: plotDistDistribution [-h] pdbMap dcaFile Ntops [Ntops ...]
  
* ***plotFrequencies***
  Plots the single- and two-site amino acid frequencies in two MSAs.
    * usage: plotFrequencies [-h] [-cf] origMSA resampledMSA

* ***plotTPrates***
  Plots the precision curves for DCA predicted contacts.
    * usage: plotTPrates [-h] pdbMap dcaFiles [dcaFiles ...] Nmax contactThreshold

* ***plotTopContacts***
  Plots a user-defined number of highest-ranking DCA predicted contacts, overlaid on PDB contact map.
    * usage: plotTopContacts [-h] [-ik] pdbMap dcaFile Ntop contactThreshold

* ***prmtToIsingGauge***
  Concerts Potts model parameters in a prm file to the Ising gauge.
    * usage: prmToIsingGauge [-h] InPRM OutPRM

* ***prmtToScores***
  Computes the Frobenius norm DCA scores from Potts model parameters in a prm file.
    * usage: prmToScores [-h] [--gaps] [--raw] InPRM outFil
    
* ***splitSpeciesMSA***
  Splits sequences from an MSA based on taxonomic identifiers.
    * usage: splitSpeciesMSA [-h] inMSA lineage taxaQuery speciesMSA

* ***stockholm2fasta***
  Converts MSAs from stockholm to fasta format.
    * usage: stockholm2fasta [-h] stockholmFile fastaFile

