import numpy as np
import pandas as pd
from Bio import SeqIO


def preprocessing(input_name, output_name) :
    """
    load the sequences from the input file, remove inserts, remove sequences with more than 10% gaps, encode the MSA into numbers
    and write the preprocessed MSA in the output_file

    :param input_name: name of the input file containing the sequences in fasta format
    :type input_name: string
    :param output_name: name of the output file which will contain the preprocessed MSA in csv format
    :type output_name: string
    :return: nothing
    :rtype: None
    """
    #load the sequences
    MSA = SeqIO.parse(input_name, "fasta")
    #remove inserts and sequences with more that 10% gaps
    MSA = filter_data(MSA)
    #encode the MSA into numbers in a pandas dataframe (the rows are the sequences, the columns are the
    #amino acid postions)
    MSA = amino_acids_to_numbers(MSA)

    print("MSA shape : ", MSA.shape)
    MSA.to_csv(output_name, header=None, index=None)


def filter_data(MSA) :
    """
    remove inserts and sequences with more that 10% gaps from the MSA given in parameter
    """
    output = []
    for sequence in MSA :
        #remove inserts
        sequence.seq = ''.join(res for res in sequence.seq if not (res.islower() or res == '.'))
        #keep only sequences with less that 10% gaps
        if sequence.seq.count('-') < 0.1 * len(sequence) :
            output.append(sequence.seq)
    return output


def amino_acids_to_numbers(MSA) :
    """
    takes an MSA in the form of a list of strings and return a panda DataFrame where the rows
    are the sequences and the columns the amino acid positions and the amino acid letters
    are encoded into numbers from 0 to 21
    """
    #put the MSA in the form of a panda DataFrame where the rows are the sequences and
    #the columns the amino acid positions
    MSA = pd.DataFrame(separate_residues(MSA))
    #create dictionnary to encode the amino acid letters into numbers from 0 to 21
    letters = list("-GALMFWKQESPVICYHRNDTX")
    numbers = np.arange(0, 22, 1, dtype=int)
    amino_acids_code = {letter: number for letter, number in zip(letters, numbers)}
    #encode the MSA using the dictionnary
    MSA = MSA.replace(amino_acids_code)

    return MSA


def separate_residues(sequences) :
    """
    takes a list of string sequences and returns an array where the row are the sequences
    a the columns are the amino acid positions
    """
    MSA = []
    for sequence in sequences :
        MSA.append(list(sequence))
    return MSA

