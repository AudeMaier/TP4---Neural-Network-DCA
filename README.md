# Neural Network DCA

## Description
The goal of Neural Network DCA is to enable DCA-based protein contact prediction using non-linear models. The scripts contained in this repository allow to train different neural network architectures on an MSA to predict the type of a residue given all other residues in a sequence and then to extract the knowledge learned by the network to do contact prediction.

## Installation
To run these scripts, the user must have the following Python packages installed :

* numpy
* pandas
* biopython
* numba
* torch
* itertools
* matplotlib

## Content and Organization
This repository is mostly articulated around 4 python modules :

* preprocessing.py
* weights.py
* model.py
* couplings.py

Each of them is accompanied by a main file that can be directly executed from command line. Here is a description of these files :


***preprocessing.py***

This module takes an MSA in fasta format (as it can be dowloaded on Pfam) and preprocess it to make it suitable for the module model.py. It removes inserts, discard sequences with more than 10% gaps, encodes the amino acid / gap symbols into numbers and write a csv file with the result.

Arguments needed by the main :
* input_name : name of the file containing the MSA in fasta format
* output_name : name that will be used to create the output file

Example of usage :

```shell
$ main_preprocessing.py PF00226_full.fasta PF00226_preprocessed.csv

```

***weights.py***

This module takes an MSA preprocessed using preprocessing.py and compute the weights of each sequence. The weights of a sequence is the inverse number of sequences in the MSA that have more than 80% similarity with it. Once computed, the weights will be written in an output file.

Arguments needed by the main :
* input_name : name of the file containing the preprocessed MSA (i.e. the output file of preprocessing.py)
* ouput_name : name for the output file containing the weights

Example of usage :

```shell
$ main_weights.py PF00226_preprocessed.csv PF00226_weights.txt

```

***model.py***

This module builds and trains a neural network on the MSA to be able to predict the value of a residue given all other residues of the sequence.
The user can choose between 3 architectures for the neural network :
* linear : this is simply a linear classifier (with softmax activation on the output layer and cross entropy loss) whose input and output are the residues and where the output residues are connected to every input residues exept from themselves (in order to avoid a trivial identity)
* non-linear : this model adds a hidden layer to the network, the architecture is designed such that the output residues are disconnected from the corresponding input residues. Two activation function are available for the hidden layer, a custom activation that squares the output of the hidden layer ("square") and a tanh activation ("tanh")
* mix : this model is a combination of the first two, the input and output neurons are connected both linearly and via a hidden layer. Both the square and tanh activations are possible for the hidden layer

After training, the learning curve will be plotted and the model will be saved as well as the error rate after each epoch and the final error rate per residue.

The hyper parameters can be changed in the function "execute" of the file model.py

Arguments needed by the main :
* MSA_name : name of the file containing the preprocessed MSA (i.e. the output file of preprocessing.py)
* weights_name : name of the file containing the weights of the MSA (i.e. the output file of weights.py)
* model_type : type of network that must be built and trained, can be "linear", "non-linear" or "mix"
* activation : activation function for the hidden layer if model_type is "non-linear" or "mix" (otherwise this parameter will be ignored), can be "square" or "tanh"
* output_name : name that will be used to create the 3 output files

Example of usage :

```shell
$ main_model.py PF00226_preprocessed.csv PF00226_weights.txt mix square mix_square

```

***couplings.py***

This module extracts from the trained model (the output of model.py) the coupling coefficients (which describe the interaction between any two residues for each categories) and applies to them a series of operations to make them suitable for contact prediction. It applies the Ising gauge to the coupling coefficients, makes
the matrix of couplings symetrical by averaging with its transpose, takes the Frobenius norm of over all the categories of the residues and apply average product correction.

Arguments needed by the main :
* model_name : name of the file containing the saved model from model.py
* model_type : type of the model, can be "linear", "non-linear" or "mix"
* L : length of the sequences (second dimension of the preprocessed MSA)
* K : number of categories for the residues (22 in general)
* output_name : name of the output file that will contain the coupling coefficients

Example of usage :

```shell
$ main_couplings.py mix_square_model mix 63 22 mix_square_couplings.txt

```

***
In addition to these 4 modules. The directory dcaTools contains 3 scripts downloaded from https://gitlab.com/ducciomalinverni/dcaTools.git that enable to make and evaluate contact prediction using the output of couplings.py. Details on their usage can be found here https://link.springer.com/protocol/10.1007%2F978-1-4939-9608-7_16.

***mapPDB***

Compute the distance map of a PDB structure for the purpose of evaluating the predictions made from the coupling coeffcicients.

***PlotTopContacts***

Plots a user-defined number of highest-ranking DCA predicted contacts, overlaid on PDB contact map. The output file format of couplings.py is adapted to make it suitable for this script

***PlotTPrates***

Plots the precision curves for DCA predicted contacts. The output file format of couplings.py is adapted to make it suitable for this script


***
***Data***

The directory data contains :
* PF00226.fasta : the MSA of the DnaJ domain downloaded in fasta format from Pfam
* PF00226_preprocessed.csv : the preprocessed MSA obtained using preprocessed.py
* weights.txt : the weights file obtained with weights.py
* DnaJ.hmm : the hidden markov model of the MSA
* 2qsa.pdb : the PDB file that can be used to generate a distance map of the DnaJ domain
* distance_map : the distance map generated by mapPDB using 2qsa.pdb and DnaJ.hmm

