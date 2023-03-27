import os
import sys
import Bio.PDB
import Bio.SeqIO
import copy
import Bio.Seq
import subprocess
import numpy as np
from Bio import pairwise2
import scipy.spatial.distance
from sklearn.preprocessing import OneHotEncoder

aaDict={'-':0, 'X':0,'Z':0,'B':0,'*':0,'U':0,'O':0,'J':0, 'A':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8,'K':9,
        'L':10, 'M':11,'N':12, 'P':13, 'Q':14, 'R':15, 'S':16, 'T':17, 'V':18, 'W':19, 'Y':20}

invAaDict={0:'-', 1:'A', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'K', 10:'L',
        11:'M', 12:'N', 13:'P', 14:'Q', 15:'R', 16:'S', 17:'T', 18:'V', 19:'W', 20:'Y'}

def binarizeMSA(msa):
    """ Expands the categorical MSA in extended binary form.

    Keyword Arguments:
            msa (ndarray): A numpy array containing the MSA in numerical format

    Returns:
        binaryMSA (scipy.sparse.csr): A 2D scipy.sparse.csr.csr_matrix of dimensions (B,21*N), 
                                      where B is the number of sequences in the MSA and N
                                      the original width of the MSA.
    """
   
    enc=OneHotEncoder(categories=[range(21) for n in range(msa.shape[1])],sparse=False)
    binaryMSA=enc.fit_transform(msa)
    return binaryMSA

def alignSequenceToHMM(sequence,hmmFile):
    """ Aligns a sequence to a hmmer-generated HMM.

    Keyword Arguments:
        sequence (str): The sequence to be alignd as a simple string.

        hmmFile (str): The family hmm file.


    Returns:
        mapIndexes (ndarray): 1-D array containing the mapping indexes between the sequence and the hmmFile. 
                              The array is such that mapIndexes[i] maps sequence[i] to its position in the hmm.
    """
    
    # Align the pdbSequence to the HMM
    Bio.SeqIO.write(Bio.SeqRecord.SeqRecord(Bio.Seq.Seq(sequence),'tmp','tmp'),'__tmpFasta.fasta','fasta')
    out=subprocess.check_output(["hmmalign",hmmFile,'__tmpFasta.fasta']).splitlines()
    os.remove('__tmpFasta.fasta')
    rawMap=''
    gapMap=''
    for l in out:
        if l.startswith(b'#=GC RF'):
            rawMap+=str(l.split()[2])
        if l.startswith(b'#=GC PP_cons'):
            gapMap+=str(l.split()[2])

    mapIndexes=[]
    aaCounter=0
    for idx,c in enumerate(rawMap):
        if c == "x" and gapMap[idx] != '.' :
            mapIndexes.append(aaCounter)
            aaCounter+=1
        elif c=="x" and gapMap[idx] == '.':
            mapIndexes.append(-1)
        elif c==".":
            aaCounter+=1

    return np.asarray(mapIndexes)


def stockholm2fasta(stoFile,fastaFile,noFilterInserts=False):
    """ Converts an MSA in stockholm format to fasta format.

    Keyword Arguments:
        stoFile (str): The MSA file in stockholm format.

        fastaFile (str): The output fasta MSA filename.

        noFilterInserts(bool): If true, do not filter inserts (lower case and .)
    """

    # Extract all sequences from the Stockholm file
    seqs={}
    for line in open(stoFile,'r'):
        if line[0] not in ['#','/','\n']:
            seq=line.rstrip().split()
            if seq[0] in seqs:
                seqs[seq[0]]+=seq[1]
            else:
                seqs[seq[0]]=seq[1]

    # Build fasta MSA, removing inserts and delete symbols
    msa=[]
    for seqId,seq in seqs.items():
        # Filter inserts: delete '.' and lower case symbols
        if not noFilterInserts:
            seq=seq.replace('.','')
            seq=''.join([s for s in seq if not s.islower()])
        s=Bio.SeqRecord.SeqRecord(Bio.Seq.Seq(seq),id=seqId,description='')
        msa.append(s)
    Bio.SeqIO.write(msa,fastaFile,'fasta')


def filterSequenceByGapContent(inMSA,gapThreshold,filteredMSA,verbose=True):
    """ Filter MSA based on gap content.

    Keep only sequences having a maximum allowed gap fraction.
    Input sequences must be in fasta format.

    Keyword Arguments:
        inMSA (str): Input MSA file in fasta format.

        gapThreshold (float): Maximum allowed fraction of gaps. 

        filteredMSA (str): Output name for the filtered MSA.
    """
    
    sequences=Bio.SeqIO.parse(inMSA,"fasta")
    seq1=next(sequences)

    
    realLength=float(len(seq1.seq)-seq1.seq.count(".")-len([c for c in str(seq1.seq) if c.islower()]))
    sequences=Bio.SeqIO.parse(inMSA,"fasta")
    filteredSequences=[seq for seq in sequences if float(seq.seq.count("-"))/realLength <= float(gapThreshold)]
    Bio.SeqIO.write(filteredSequences,filteredMSA,"fasta")
    if verbose:
        print("Original number of sequences ",len(list(sequences)))
        print("Sequences after filtering : ",len(filteredSequences))
        print("Filtered sequences saved to ",filteredMSA)

def filterSequenceOfInserts(inMSA,filteredMSA):
    """ Filter MSA of inserts (. and lower case amino acids)

    Removes all non consensus characters from the MSA, i.e. removes inserts (gap . and AA in lower case)
    Input sequences must be in fasta format.

    Keyword Arguments:
        inMSA (str): Input MSA file in fasta format.
    
        filteredMSA (str): Output name for the filtered MSA.
    """
    
    sequences=Bio.SeqIO.parse(inMSA,"fasta")
    out=[]
    for sequence in sequences:
        sequence.seq=sequence.seq.ungap(".")
        sequence.seq=Bio.Seq.Seq(''.join([s for s in sequence.seq if (s.isupper() or s=="-")]))
        out.append(sequence)
    Bio.SeqIO.write(out,filteredMSA,"fasta")

def filterSequenceIdentity(inMSA,maxID,filteredMSA):
    """ Filter MSA by keeping only sequences with maximal pairwise identity of maxID.
        This routine relies on the hhfilter function of hhsuite, which needs to be installed.

    Keyword Arguments:
        inMSA (str): Input MSA file in fasta format.
        
        maxID (int): Max identity percentage in [0,100]

        filteredMSA (str): Output name for the filtered MSA.
    """
    subprocess.run(['hhfilter', '-maxseq', '500000', '-i', inMSA,'-id',str(maxID), '-o',filteredMSA], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def splitSpeciesMSA(fullMSA,lineage,taxaQuery,speciesMSA):
    """ Extracts all sequences from a specific species from an MSA
    
    Keyword Arguments:
        fullMSA (str): Complete MSA in fasta format.

        lineage (str): Lineage file. Each line corresponds to a sequence in fullMSA in the same order. 

        taxaQuery (str): The taxonomic query. All sequences belonging to the query are returned in the output speciesMSA.

        speciesMSA (str): The output MSA file name containing only sequences of fullMSA belonging to the taxa query.
    """

    msa=Bio.SeqIO.parse(fullMSA,'fasta')
    B=len(list(msa))
    
    with open(lineage,'r') as f:
        lineageList=list(f)
    lineageList=[[ll.lstrip().rstrip() for ll in l.split('\t')] for l in lineageList]

    if len(list(lineageList)) != B:
        print("Lineage file lenght inconsistent with MSA length.")
        sys.exit()


    msa=Bio.SeqIO.parse(fullMSA,'fasta')
    taxaHits=[]
    for idx,seq in enumerate(msa):
        if seq.id.split('_')[0].split('.')[0].split('/')[0]!=lineageList[idx][0]:
            # seq.id.split('|')[1]!=lineageList[idx][0]:
            print("Inconsistent lineage file")
            print(seq.id)
            sys.exit()

        if taxaQuery.rstrip().lstrip() in lineageList[idx]:
            taxaHits.append(seq)

    Bio.SeqIO.write(taxaHits,speciesMSA,'fasta')


def makeTaxaTags(fullMSA,lineage,taxaQueries,tagFile):
    """ Tags all sequences in fullMSA with specific  taxa queries.

    Keyword Arguments:
        fullMSA (str): Complete MSA in fasta format.

        lineage (str): Lineage file. Each line corresponds to a sequence in fullMSA in the same order. 

        taxaQueries (list): List of the taxonomic queries.

        tagFile (str): The output file containing the numeric tags associated to the taxaQueries.
    """
    
    msa=sys.argv[1]
    lineageFile=sys.argv[2]
    taxaQueries=[]
    queries=sys.argv[3:]
    
    with open(lineageFile,'r') as f:
        lineage=list(f)
        
    lineage=[l.split('\t') for l in lineage]
    lineage={l[0] : l for l in lineage}

    msa=Bio.SeqIO.parse(msa,'fasta')
    genericTag=0
    fout=open(tagFile,'w')
    fout.write("Taxa\tHeader\n")
    for seq in msa:
        id=seq.id.split('|')[1]
        tag=genericTag
        for idx,query in enumerate(queries):
            if query in lineage[id]:
                tag=idx+1
                break
        if tag != genericTag:
            fout.write(seq.id+'\t'+str(tag) + "\n")
    fout.close()


def mapPDBToHMM(pdbFile, chainIds, hmmFile1, hmmFile2,mapFile,distType='all'):
    """ Maps a PDB structure to a hmm to compare DCA predictions with structural contacts. 
        This can be used either to only compute the PDB distance map (without hmmFile1='None' and hmmFile2='None') 
        or used to futher map the distance map such to aligned it to the hmm models.

    Keyword arguments:
        pdbFile (str)  : The structure file to map in PDB format 
    
        chainIds (str) : The chain Id(s) of the chain(s) to be mapped. For multiple chains, pass a unique string (e.g. AB)

        hmmFile1 (str) : Optional hmm file onto which the pdb map is aligned.
 
        hmmFile2 (str) : An optional second hmm file for mapping of hetero-dimers. 

        mapFile (str)  : The output name for the mapped distance map

       distType (str)  : Type of distance to compute: 'all': minimal distance between any atoms, 'alpha': Carbon-alpha distance, 'beta': Carbon-beta distance
    Return:
        map (ndarray) : A NxN array containing the (possibly mapped) distance map.
    """

    # Parse the command line hmms
    hmmFiles=[]
    if(hmmFile1 != 'None'):
        hmmFiles.append(hmmFile1)
    if(len(chainIds)==2 and hmmFile1!='None' and hmmFile2 != 'None'):
        hmmFiles.append(hmmFile2)
    elif(len(chainIds)==2 and hmmFile1!='None' and hmmFile2=='None'):
        hmmFiles.append(hmmFile1)

    # Parse the PDB and extract the residues of the target chains
    chainIds=list(chainIds)
    structure = Bio.PDB.PDBParser().get_structure('void', pdbFile)
    
    chains = structure[0].get_chains()

    residues=[chain.get_list() for chain in chains if chain.id in chainIds]
    residues = [res for resList in residues for res in resList]

    # Remove waters and hetero_residues
    residues = [res for res in residues if (res.id[0][0] not in ["W", "H"])]

    # Compute the distance map between all residues (minimal distance between atoms of the chain)
    distanceMap=np.zeros([len(residues),len(residues)],dtype=float)
    catPdbSeq=''
    for idx1,res1 in enumerate(residues):
        if distType=='all':
            atoms1 = np.array([a.get_coord() for a in res1.get_atoms()])
        elif distType=='alpha':
            atoms1 = np.asarray([a.get_coord() for a in res1.get_atoms() if a.name=='CA'])
        elif distType=='beta':
            if res1.resname=='GLY':
                atoms1 = np.asarray([a.get_coord() for a in res1.get_atoms() if a.name=='CA'])
            else:
                atoms1 = np.asarray([a.get_coord() for a in res1.get_atoms() if a.name=='CB'])
                
        catPdbSeq+=Bio.PDB.Polypeptide.three_to_one(res1.resname)
        for idx2,res2 in enumerate(residues):
            if distType=='all':
                atoms2 = np.array([a.get_coord() for a in res2.get_atoms()])
            elif distType=='alpha':
                atoms2 = np.asarray([a.get_coord() for a in res2.get_atoms() if a.name=='CA'])
            elif distType=='beta':
                if res2.resname=='GLY':
                    atoms2 = np.asarray([a.get_coord() for a in res2.get_atoms() if a.name=='CA'])
                else:
                    atoms2 = np.asarray([a.get_coord() for a in res2.get_atoms() if a.name=='CB'])
            atomDist = scipy.spatial.distance.cdist(atoms1,atoms2)
            distanceMap[idx1,idx2]=atomDist.min()
    
    # Align the pdb sequences to the HMMs
    if hmmFiles:
        pdbSeqs=[]
        currentLength=0
        for chainId in chainIds:
            for chain in structure[0].get_chains():
                if chain.id==chainId:
                    d=len([res for res in chain.get_list() if res.id[0][0] not in ["W","H"]])
                    pdbSeqs.append(catPdbSeq[currentLength:(currentLength+d)])
                    currentLength+=d

        mapIndexes=np.array([],dtype=int)
        offset=0
        for i in range(len(chainIds)):
            mapIndexes=np.concatenate([mapIndexes,offset+np.array(alignSequenceToHMM(pdbSeqs[i],hmmFiles[i]))])
            offset+=len(pdbSeqs[i])
        
        # Align the PDB distance map onto the HMM
        mi=mapIndexes
        dm=np.zeros([len(mi), len(mi)])
        dm[:]=np.inf
        dm[np.ix_(mi>=0,mi>=0)]=distanceMap[np.ix_(mi[mi>=0],mi[mi>=0])]  
    else:
        dm=distanceMap

    # For homo-dimeric mappings, reduce the distance map to a NxN map containing also the homo-dimeric contacts.
    if(len(chainIds)==2 and hmmFile1!='None' and hmmFile2=='None'):
        dm=np.minimum(np.minimum(np.minimum(dm[0:(len(dm)//2),0:(len(dm)//2)],dm[(len(dm)//2):,0:(len(dm)//2)]),dm[(len(dm)//2):,(len(dm)//2):].T),dm[0:(len(dm)//2),(len(dm)//2):].T)

    # Save output map
    if mapFile!='None':
        np.savetxt(mapFile,dm,fmt='%.2f')

    return dm


def mapPDBToSequence(pdbFile, chainId, sequence1, seq1Mapping,mapFile):
    """ Maps a PDB structure to a reference mapped structure. 

    Keyword arguments:

    Return:
        map (ndarray) : A NxN array containing the (possibly mapped) distance map.

   """

    ### Compute the unaligned distance map
    # Parse the PDB and extract the residues of the target chains
    structure = Bio.PDB.PDBParser().get_structure('void', pdbFile)
    
    chains = structure[0].get_chains()

    residues=[chain.get_list() for chain in chains if chain.id== chainId]
    residues = [res for resList in residues for res in resList]

    # Remove waters and hetero_residues
    residues = [res for res in residues if (res.id[0][0] not in ["W", "H"])]

    # Compute the distance map between all residues (minimal distance between atoms of the chain)
    distanceMap=np.zeros([len(residues),len(residues)],dtype=float)
    catPdbSeq=''
    for idx1,res1 in enumerate(residues):
        atoms1 = np.array([r.get_coord() for r in res1.get_atoms()])
        catPdbSeq+=Bio.PDB.Polypeptide.three_to_one(res1.resname)
        for idx2,res2 in enumerate(residues):
            atoms2 = np.array([r.get_coord() for r in res2.get_atoms()])
            atomDist = scipy.spatial.distance.cdist(atoms1,atoms2)
            distanceMap[idx1,idx2]=atomDist.min()

    
    ### Align the pdb sequence to its reference mapped sequence
    pdbSeq = pdbToFasta(pdbFile,chainId,None)
    origSeq = list(Bio.SeqIO.parse(sequence1,'fasta'))[0].seq
    ali = pairwise2.align.localxx(origSeq,pdbSeq)[0]
    aliIndexes = getAlignmentIndexes(ali[1],ali[0])

    if seq1Mapping!="None":
        with open(seq1Mapping,'r') as f:
            refMapping = [int(l) for l in f]
    else:
        refMapping = np.arange(len(origSeq))
        
    ### Align the map to the refernce mapping
    dm1 = np.inf*np.ones(())

    # Save output map
    if mapFile!='None':
        np.savetxt(mapFile,dm,fmt='%.2f')

    return dm

        
def getAlignmentIndexes(seq1,seq2):
    """ To be commented

    """

    gap="-"
    
    if len(seq1)!=len(seq2):
        print("Error: Sequences must be of same length if aligned.")
        return

    seq2Map=-np.ones(len(seq2),dtype=int)   
    posCounter=0
    for i,c in enumerate(seq2):
        if c is not gap:
            seq2Map[i]=posCounter
            posCounter+=1
            
    indexes=[]
    for i,c in enumerate(seq1):
        if c is not gap:
            indexes.append(seq2Map[i])

    return np.asarray(indexes,dtype=int)


def mergeDomains(inFasta, outFasta):
    """ Combines domains hits into single sequences based on sequence identifiers. 
        If multiple domains (with the same sequence id) are presenent as different sequences in the MSA, stitch them together as a new sequence.

    Keyword Arguments:
        inFasta (str): The input MSA in fasta format. The sequence header must have the following form >XX|Id|YYYYY

        ouFasta (str): The output name for the recomposed MSA. 

    """

    msa=Bio.SeqIO.to_dict(Bio.SeqIO.parse(inFasta,'fasta'))
    
    N = len(msa[msa.keys()[0]].seq)

    uniqIds = set([key.split('|')[1] for key in msa.keys()])
    mergedMSA = []
    ff = open('pp','w')
    for n,uniqId in enumerate(uniqIds):
        ids = [fullId for fullId in msa.keys() if uniqId in fullId]
        seqs = [msa[i] for i in ids]

        tmpSeq= ["-"]*N
        for seq in seqs:
            tmpSeq = [c if tmpSeq[ind]=='-' else tmpSeq[ind] 
                 for ind,c in enumerate(seq.seq)]

        mergedSeq = Bio.SeqRecord.SeqRecord(Bio.Seq.Seq("".join(tmpSeq)),id=uniqId)
        mergedMSA.append(mergedSeq)
        print(n,"/",len(uniqIds))

    ff.close()
    with open(outFasta,"w") as f:
        Bio.SeqIO.write(mergedMSA,f,"fasta")

def pdbToFasta(pdbFile,chainID, fastaFile):
    """ Extracts the sequence of a PBD file and saves it to fasta format

    Keyword Arguments:
        pdbFile (str): Filename of PDB structure
        
        chainID (str): Chain ID in the PDB

        fastaFile(str): Name of the output fasta file 
    """
    
    structure = Bio.PDB.PDBParser().get_structure('void', pdbFile)
    chains = structure[0].get_chains()

    residues=[chain.get_list() for chain in chains if chain.id== chainID]
    residues = [res for resList in residues for res in resList]

    for i,r in enumerate(residues[:]):
        if r.id[0]=='W':
            residues.remove(r)
        elif r.id[0]=='H_ ZN':
            residues.remove(r)
        elif r.id[0]=='H_MSE':
            residues[i].id=('',r.id[1],'')
            residues[i].resname='MET'

    residues = [res.resname for res in residues ]
    sequence = "".join([Bio.PDB.Polypeptide.three_to_one(res) for res in residues])
    if fastaFile:
        seqRecord = Bio.SeqRecord.SeqRecord(Bio.Seq.Seq(sequence),id=pdbFile,description='')
        Bio.SeqIO.write(seqRecord,fastaFile,'fasta')

    return sequence

def fastaToMatrix(fastaFile):
    """ Loads a fasta file into a numerical matrix with amino-acids from 0 to 20.

    Keyword Arguments:
        fastaFile (str): Name of the input fasta file

    Returns:
        msa (ndarray): A numpy array containing the MSA in numerical format
    """
    
    msa=Bio.SeqIO.parse(fastaFile,'fasta')
    seqs=[list(str(s.seq)) for s in msa]   
    matrix= np.asarray([[aaDict[aa] for aa in seq] for seq in seqs])

    msa=Bio.SeqIO.parse(fastaFile,'fasta')
    ids=np.asarray([s.description for s in msa],dtype=None)
    return matrix,ids

def matrixToFasta(msa,fastaFile,ids=None):
    """ Saves a MSA in numerical matrix format to a fasta file

    Keyword Arguments:
        msa (ndarray): A numpy array containing the MSA in numerical format

        fastaFile (str): Name of the output fasta file

        ids (list): List of ids to be used in the fasta header of each sequence in msa.
    """

    tmp=[]
    for i,line in enumerate(msa):
        seq =  ''.join([invAaDict[int(aa)] for aa in line])
        if ids is not None:
            tmp.append(Bio.SeqRecord.SeqRecord(Bio.Seq.Seq(seq),id=ids[i],name="",description=""))
        else:
            tmp.append(Bio.SeqRecord.SeqRecord(Bio.Seq.Seq(seq),id=str(i),name=str(i),description=str(i)))
    Bio.SeqIO.write(tmp,fastaFile,"fasta-2line")
    
def shuffleMSA(msa):
    """ Shuffles the columns of the input MSA and returns a MSA conserving only the positional conservations.

    Keyword Arguments:
        msa (ndarray): A numpy array containing the MSA in numerical format

    Returns:
        shuffledMSA (ndarray): A numpy array containing the reshuffled MSA in numerical format

    """

    shuffledMSA = np.zeros(msa.shape,dtype=int)

    for i in range(msa.shape[1]):
        rndIs = np.random.permutation(msa.shape[0])
        shuffledMSA[:,i] = msa[rndIs,i]

    return shuffledMSA


def countSequencesInMSA(msaFile):
    """ Counts the number of sequences in a fasta file
    
    Keyword argumnts:
        msaFile (str): Fasta file containing the MSA

    Returns:
       N (int): The number of sequences in te MSA
    """
    N=0
    with open(msaFile,'r') as f:
        for l in f:
            if l[0]==">":
                N+=1
    return N

def compareMaps(mapFiles,combinedMap):
   """ Combines two aligned distance maps in one, putting them in the two different triangular parts of the resulting map.

   Keyword Arguments:
       mapFiles (list): List of aligned distance maps (as returned by pdbMap) to be combined.

       combinedMap (str): The output name where the combined maps is saved.
   """

   # Load all maps to be combined
   if len(mapFiles)!=2:
     raise("Error: you can only compare 2 maps")
   maps=[]
   for map in mapFiles:
       maps.append(np.loadtxt(map))

   compMap=np.tril(maps[1])+np.triu(maps[0 ])

   np.savetxt(combinedMap,compMap,fmt='%.2f')

def linearizeFasta(msaFile,linearMSA):
    ''' Linearizes a fasta file, writing each sequence on two lines: first line head >..., second line  the whole sequence.
    
    Keyword Arguments:
       msaFile (str): Fasta file containing the MSA

       linearFasta(str): Fasta file in which to save the linearized fasta.
    '''

    msa=Bio.SeqIO.parse(msaFile,'fasta')
    Bio.SeqIO.write(msa,linearMSA,'fasta-2line')

def protLength(fastaFile):
    ''' Return a list of protein lengths in a unaligned fasta file

    Keyword Arguments:
       fastaFile (str): Fasta file containing unaligned sequences.

    Return:
       lengths (list): List of protein lengths in input file.
    '''

    msa=Bio.SeqIO.parse(fastaFile,'fasta')
    lengths=[len(s.seq) for s in msa]

    return lengths

def trimToRefSequence(fastaFile,outFile):
    ''' Kepp only MSA sequences for which the first (reference) sequence contains AA.
     
    Keyword Arguments:
       fastaFile (str): Fasta file containing unaligned sequences.
    
       outFile (str): The output fasta MSA filename.
    '''
    msa,ids=fastaToMatrix(fastaFile)
    idx=msa[0,:]!=0
    msa=msa[:,idx]
    matrixToFasta(msa,outFile,ids)

def trimToRefSequenceGenerator(fastaFile,outFile,filterSymbols=True):
    ''' Kepp only MSA sequences for which the first (reference) sequence contains AA.
     
    Keyword Arguments:
       fastaFile (str): Fasta file containing unaligned sequences.
    
       outFile (str): The output fasta MSA filename.
    '''
    msa=Bio.SeqIO.parse(fastaFile,'fasta')
    ref=[aa!='-' for aa in str(next(msa).seq)]
    msa=Bio.SeqIO.parse(fastaFile,'fasta')
    with open(outFile,'w') as f:
        if filterSymbols:
            for s in msa:
                trimmed=''.join([invAaDict[aaDict[aa]] for aa,hit in zip(str(s.seq),ref) if hit])
                f.write(">"+s.description+'\n')
                f.write(trimmed+'\n')
        else:
            for s in msa:
                trimmed=''.join([aa for aa,hit in zip(str(s.seq),ref) if hit])
                f.write(">"+s.description+'\n')
                f.write(trimmed+'\n')
