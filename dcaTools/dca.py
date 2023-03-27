import numpy as np
import scipy.spatial.distance
import pandas as pd
import Bio.SeqIO
import matplotlib.pyplot as plt
import sequenceHandler as sh

def getSize(dcaFile):
    """Returns the size N of the NxN DCA map.

    Keyword arguments:
        dcaFile (str): The file containing the DCA contacts. The file must only contain the upper triangular part 
                           of the DCA score matrix, ordered in linear form, i.e. the contacts are ordered as 
                           S(1,2),S(1,3),...,S(1,N),S(2,3),S(2,4),...,S(N-1,N)        
    Returns:
        N (int): The size N of the NxN DCA map.
    """

    dca=scipy.spatial.distance.squareform(np.loadtxt(dcaFile))
    return dca.shape[0]

def extractTopContacts(dcaFile,numTopContacts,diagIgnore=4):
    """Extract the top N ranked DCA contacts.

    Keyword arguments:
        dcaFile (str): The file containing the DCA contacts. The file must only contain the upper triangular part 
                           of the DCA score matrix, ordered in linear form, i.e. the contacts are ordered as 
                           S(1,2),S(1,3),...,S(1,N),S(2,3),S(2,4),...,S(N-1,N)
    
        numTopContacts (int): The number of top ranked contacts to return. 

        diagIgnore (int): The number of diagonal bands to be ignored, starting from the center. 
                      

    Returns:
        topContacts (ndarray) : A 2D numpy array of dimension (numTopContacts,2), 
                                    containing the (i,j) pairs of the extracted contacts. 

        sortedScores (ndarray): A 1D numpy array containing the sorted scores of the numTopContacts.
    """
   
    # Load the data from file and put in symmetric matrix form
    dca=scipy.spatial.distance.squareform(np.loadtxt(dcaFile))

    # Ignore the diagIgnore first diagonal bands
    N=dca.shape[0]
    for i in range(1,diagIgnore+1):
        dca=dca-1e5*np.diag(np.ones(N-i),k=i)
        dca=dca-1e5*np.diag(np.ones(N-i),k=-i)
    # Extract the top ranked contacts
    sortedScores = -np.sort(scipy.spatial.distance.squareform(-dca))
    dcaContacts=np.argwhere(dca>sortedScores[numTopContacts])
    contactRanks=np.argsort(-dca[dcaContacts[:,0],dcaContacts[:,1]])
    dcaContacts = dcaContacts[contactRanks,:]
    return dcaContacts,sortedScores

def extractInterContacts(dcaFile,N1,Beff,cutoff=0.8):
    """ Extracts the inter-protein DCA contacts using the Hopf2014 criterion 
    
    Keyword arguments:
        dcaFile (str): The file containing the DCA contacts. The file must only contain the upper triangular part 
                           of the DCA score matrix, ordered in linear form, i.e. the contacts are ordered as 
                           S(1,2),S(1,3),...,S(1,N),S(2,3),S(2,4),...,S(N-1,N)

        N1 (int): The length of the first  protein in the alignmnet.

        Beff (double): The number of effective sequences after identity reweighting.

        cutoff (double): The cutoff on the score over which inter-protein contacts are considered.


    Returns:
            topContacts (ndarray) : A 2D numpy array of dimension (numTopContacts,2), 
                                    containing the (i,j) pairs of the extracted contacts. 

            sortedScores (ndarray): A 1D numpy array containing the sorted ***original*** DCA scores of the numTopContacts.
    """

    dca=scipy.spatial.distance.squareform(np.loadtxt(dcaFile))
    N=dca.shape[0]
    inter = dca[:N1,N1:]

    Qraw = inter/np.abs(inter.min())
    Qij = Qraw/(1.+np.sqrt(N/float(Beff)))
    contacts = np.argwhere(Qij>=cutoff)
    scores = Qij[contacts[:,0],contacts[:,1]]
    return contacts,scores
    
def combineMaps(mapFiles,combinedMap):
    """ Combines multiple mapped structural contact maps in a single one. The combination is based on an element-wise minimum.

    Keyword Arguments:
        mapFiles (list): List of aligned distance maps (as returned by pdbMap) to be combined. 

        combinedMap (str): The output name where the combined maps is saved.
    """

    # Load all maps to be combined
    maps=[]
    for map in mapFiles:
        maps.append(np.loadtxt(map))
       
    minMap=np.amin(np.array(maps),axis=0)
    np.savetxt(combinedMap,minMap,fmt='%.2f')

def computeMSAPottsEnergies(h,J,fastaFile):
    """ Computes the Potts energies for all sequences in the MSA.

    Keyword Arguments:
        h (ndarray): 1-D array containing the potts model local biases h_i(A) in the following order: h_0(0),h_0(1),...,h_0(q),h_1(0),...,h_N(q)

        J (ndarray): 1-D array containing the potts model couplings J_ij(A,B) in the following order J_01(0,0),J_01(0,1),...,J_01(0,q),J_01(1,0),...,J_02(0,0),...,J_N-1,N(q,q)

        fastaFile (str): The file containing the MSA in fasta format.
    
    Returns:
        energies (ndarray): Mx3 array, containing a energy triplet (E_tot, E_h, E_J) for each of the M sequences in the MSA.
    """

    msa=list(Bio.SeqIO.parse(fastaFile,'fasta'))

    energies = np.zeros((len(msa),3))
    N = len(msa[0].seq)
    q= 21
    for k,seq in enumerate(msa):
        numSeq=[sh.aaDict[p] for p in seq.seq]
        energies[k,1]=h[q*np.arange(N)+numSeq].sum()
        for i in range(N-1):
            Jind = (i*N -(i+2)*(i+1)/2+np.arange((i+1),N))*q*q + q*numSeq[i]+numSeq[(i+1):N]
            energies[k,2]+=J[Jind].sum()

    energies[:,0]= energies[:,1]+energies[:,2]
    return energies

def computeMSAPottsEnergiesDomains(h,J,msa,nSplit):
    """ Computes the Potts energies for all sequences in the MSA.

    Keyword Arguments:
        h (ndarray): 1-D array containing the potts model local biases h_i(A) in the following order: h_0(0),h_0(1),...,h_0(q),h_1(0),...,h_N(q)

        J (ndarray): 1-D array containing the potts model couplings J_ij(A,B) in the following order J_01(0,0),J_01(0,1),...,J_01(0,q),J_01(1,0),...,J_02(0,0),...,J_N-1,N(q,q)

        msa (ndarray): A numpy array containing the MSA in numerical format
    
        nSplit (int): The last position (zero-based) of the first domain/protein in the concatenated MSA, defining where the energy contributions are split.
    
    Returns:
        energies (ndarray): An Mx3 array containing the energy triplets (E_intra1, E_intra2, E_inter) for each of the M sequences in the MSA.
    """
    energies = np.zeros((msa.shape[0],3))
    N = msa.shape[1]
    q= 21
    Ns=nSplit+1
    
    for k in np.arange(msa.shape[0]):
        numSeq=msa[k,:]
        # Intra1

        energies[k,0]=h[q*np.arange(Ns)+numSeq[:Ns]].sum()
        for i in range(Ns-1):
            Jind = ((i*N -(i+2)*(i+1)/2+np.arange((i+1),Ns))*q*q + q*numSeq[i]+numSeq[(i+1):Ns]).astype(int)
            energies[k,0]+=J[Jind].sum()

        # Intra2
        energies[k,1]=h[q*np.arange(Ns,N)+numSeq[Ns:N]].sum()
        for i in range(Ns,N-1):
            Jind = ((i*N -(i+2)*(i+1)/2+np.arange((i+1),N))*q*q + q*numSeq[i]+numSeq[(i+1):N]).astype(int)
            energies[k,1]+=J[Jind].sum()

        # Inter
        for i in range(Ns):
            Jind = ((i*N -(i+2)*(i+1)/2+np.arange(Ns,N))*q*q + q*numSeq[i]+numSeq[Ns:N]).astype(int)
            energies[k,2]+=J[Jind].sum()

    return energies


def computeMutantPottsEnergiesDomains(h,J,sequence,mutations,nSplit):
    """ Computes the Potts energies for all mutants on the provided sequence

    Keyword Arguments:
        h (ndarray): 1-D array containing the potts model local biases h_i(A) in the following order: h_0(0),h_0(1),...,h_0(q),h_1(0),...,h_N(q)

        J (ndarray): 1-D array containing the potts model couplings J_ij(A,B) in the following order J_01(0,0),J_01(0,1),...,J_01(0,q),J_01(1,0),...,J_02(0,0),...,J_N-1,N(q,q)

        sequence (ndarray): A  numpy array containing the native sequence in numerical format
    
        mutations (list) : A list of mutations in string format in the following format S14T
    
        nSplit (int): The last position (zero-based) of the first domain/protein in the concatenated sequence, defining where the energy contributions are split.
    
    Returns:
        dEnergies (ndarray): An Mx3 array containing the delta Energy triplets (dE_intra1, dE_intra2, dE_inter) for each of the M mutants
    """
    mutations=np.asarray(mutations,dtype=str)
    energies = np.zeros((mutations.shape[0],3))
    N = len(sequence)
    q= 21
    Ns=nSplit+1

    
    for k,mutation in enumerate(mutations):
        iMut=int(mutation[1:-1])
        aaMut=sh.aaDict[mutation[-1]]

        dH = h[q*iMut+aaMut]-h[q*iMut+sequence[iMut]]
        if iMut<Ns:
            dJ_intra1=(J[(iMut*N -(iMut+2)*(iMut+1)/2+np.arange((iMut+1),Ns))*q*q + q*aaMut+sequence[(iMut+1):Ns]]-
                       J[(iMut*N -(iMut+2)*(iMut+1)/2+np.arange((iMut+1),Ns))*q*q + q*sequence[iMut]+sequence[(iMut+1):Ns]]).sum()
            dJ_intra1+=(J[(np.arange(iMut)*N -(np.arange(iMut)+2)*(np.arange(iMut)+1)/2+iMut)*q*q + q*sequence[:iMut] +aaMut]-
                        J[(np.arange(iMut)*N -(np.arange(iMut)+2)*(np.arange(iMut)+1)/2+iMut)*q*q + q*sequence[:iMut]+sequence[iMut]]).sum()
           
            dJ_inter=(J[(iMut*N -(iMut+2)*(iMut+1)/2+np.arange(Ns,N))*q*q + q*aaMut+sequence[Ns:N]]-
                      J[(iMut*N -(iMut+2)*(iMut+1)/2+np.arange(Ns,N))*q*q + q*sequence[iMut]+sequence[Ns:N]]).sum()
            energies[k,:]=[dH+dJ_intra1,0.,dJ_inter]
            
        else:
            dJ_intra2=(J[(iMut*N -(iMut+2)*(iMut+1)/2+np.arange(iMut+1,N))*q*q + q*aaMut+sequence[(iMut+1):N]]-
                       J[(iMut*N -(iMut+2)*(iMut+1)/2+np.arange(iMut+1,N))*q*q + q*sequence[iMut]+sequence[(iMut+1):N]]).sum()
            dJ_intra2+=(J[(np.arange(Ns+1,iMut)*N -(np.arange(Ns+1,iMut)+2)*(np.arange(Ns+1,iMut)+1)/2+iMut)*q*q + q*sequence[Ns+1:iMut] +aaMut]-
                        J[(np.arange(Ns+1,iMut)*N -(np.arange(Ns+1,iMut)+2)*(np.arange(Ns+1,iMut)+1)/2+iMut)*q*q + q*sequence[Ns+1:iMut]+sequence[iMut]]).sum()
       
            dJ_inter=(J[(np.arange(Ns)*N -(np.arange(Ns)+2)*(np.arange(Ns)+1)/2+iMut)*q*q + q*sequence[:Ns] +aaMut]-
                        J[(np.arange(Ns)*N -(np.arange(Ns)+2)*(np.arange(Ns)+1)/2+iMut)*q*q + q*sequence[:Ns]+sequence[iMut]]).sum()
            energies[k,:]=[0.,dH+dJ_intra2,dJ_inter]
 
    return energies


def readPRMfile(prmFile):
    """ Reads Potts model parameters in prm format and returns two numpy arrays containing the fields and couplings.
    
    Keyword Arguments:
         prmFile (str): The file containing the local biases h_i(A) and couplings J_ij(A,B) 
                        of the potts model in ASCII format.

    Returns:
        h (ndarray): 1-D array containing the potts model local biases h_i(A) in the following order: 
                     h_0(0),h_0(1),...,h_0(q),h_1(0),...,h_N(q)
    
        J (ndarray): 1-D array containing the potts model couplings J_ij(A,B) in the following order: 
                     J_01(0,0),J_01(0,1),...,J_01(0,q),J_01(1,0),...,J_02(0,0),...,J_N-1,N(q,q)
    """

    with open(prmFile,'r') as f:
        line = f.readline()
        splt = line.split()
        N = int(splt[1])
        q = int(splt[2])

    with open(prmFile,'r') as f:
        h = pd.read_csv(f,usecols=[2],delimiter='\s+',nrows=N*q,engine='python').values    
    with open(prmFile,'r') as f:
        J = pd.read_csv(f,usecols=[2],delimiter='\s+',nrows=N*(N-1)*q*q,skiprows=N*q,engine='python').values

    return h.ravel(),J.ravel()

def writePRMFile(h,J,prmFile):
    """ Saves the Potts Model parameters in h,J in prm format.

    Keyword Arguments:
        h (ndarray): 1-D array containing the potts model local biases h_i(A) in the following order: 
                     h_0(0),h_0(1),...,h_0(q),h_1(0),...,h_N(q)

        J (ndarray): 1-D array containing the potts model couplings J_ij(A,B) in the following order: 
                     J_01(0,0),J_01(0,1),...,J_01(0,q),J_01(1,0),...,J_02(0,0),...,J_N-1,N(q,q)

        prmFile (str): The output file containing the local biases h_i(A) and couplings J_ij(A,B) 
                       of the Potts model in ASCII format.
    """
    q = 21
    N = int(h.shape[0]/q)

    with open(prmFile,'w') as f:
        f.write('protein   '+str(N)+'   '+str(q)+'\n')
        for i in range(N):
            for A in range(q):
                f.write('    '+str(i+1)+'    '+str(A+1)+'    '+str(h[q*i+A])+'\n')
        c=0
        for i in range(N):
            for j in range(i+1,N):
                for A in range(q):
                    for B in range(q):
                        f.write('    '+str(i+1)+'    '+str(j+1)+'    '+str(A+1)+'    '+str(B+1)
                                +'    '+str(J[c])+'\n')
                        c+=1
                        
def prmToIsingGauge(h,J):
    """ Shifts the Potts model parameters h and J to the zero-sum (i.e. Ising) gauge such that the 
        following conditions are satisfied:
    
        \sum_A[h_i(A)] = 0    \forall i
  
        \sum_A[J_ij(A,B)] = 0 \forall i,j,B

        \sum_B[J_ij(A,B)] = 0 \forall i,j,A   
    
        Keyword Arguments:
            h (ndarray): 1-D array containing the potts model local biases h_i(A) in the following order:   
                     h_0(0),h_0(1),...,h_0(q),h_1(0),...,h_N(q)                                                   
                                                                                                                  
            J (ndarray): 1-D array containing the potts model couplings J_ij(A,B) in the following order:
                     J_01(0,0),J_01(0,1),...,J_01(0,q),J_01(1,0),...,J_02(0,0),...,J_N-1,N(q,q)
     
       Returns:
            h0 (ndarray): Array of same dimensions as h containing the zero-sum biases

            J0 (ndarray): Array of same dimensions as J containig the zero-sum couplings
    """

    q=21
    N=int(h.shape[0]/q)
    h0 = np.zeros(h.shape)
    J0 = np.zeros(J.shape)

    # Shift couplings to zero-sum gauge
    c=0
    for i in range(N):
        h0[i*q:(i+1)*q] += h[i*q:(i+1)*q]
        for j in range(i+1,N):
            jij = np.reshape(J[c:c+q*q],(q,q))
            j0 = jij - jij.mean(axis=0,keepdims=True) - jij.mean(axis=1,keepdims=True) + jij.mean()
            J0[c:c+q*q]=np.reshape(j0,(q*q))
            c+=q*q
            
            h0[i*q:(i+1)*q] += jij.mean(axis=1)
            h0[j*q:(j+1)*q] += jij.mean(axis=0)

    # Shift fields to zero-sum gauge
    for i in range(N):
        h0[i*q:(i+1)*q]-=h0[i*q:(i+1)*q].mean()

    return h0,J0

def dummyMap(N,outFile):
    """ Creates a NxN empty distance map for rapid visualization of DCA results. Only the diagonal has contacts (dist=1) while the rest has distances of 100.

        Keyword Arguments:
            N (int): The number of residues in the dummy map

            outFile (str): The name of the output file
    """

    dm = 100*np.ones((N,N))
    dm = dm - np.diag(np.diag(dm)) + np.eye(N)

    np.savetxt(outFile,dm,delimiter=' ',fmt='%.2f')

def prmToScores(J,countGaps=False,apcTransform=True):
    """ Compute Frobenius norm scores from couplings J_ij(A,B), optionally counting gaps and performing ACP transform
    
        Keyword Arguments:
            J (ndarray): 1-D array containing the potts model couplings J_ij(A,B) in the following order: 
                     J_01(0,0),J_01(0,1),...,J_01(0,q),J_01(1,0),...,J_02(0,0),...,J_N-1,N(q,q)
         
            countGaps (bool): If true, compute the Frobenius norm over the full 21x21 matrices, otherwise over the 20x20 ignoring couplings to gaps

            apcTransform (bool): Apply the Average Product Correction to the computed scores.
    """

    S=[]
    for i in np.arange(0,J.shape[0],21*21):
        m=np.reshape(J[i:i+21*21],(21,21))
        if countGaps:
            S.append((m*m).sum())
        else:
            S.append((m[1:,1:]*m[1:,1:]).sum())
            
    S=np.asarray(S)

    if apcTransform:
        S=scipy.spatial.distance.squareform(S)
        S = S-np.outer(S.mean(axis=0),S.mean(axis=0))/S.mean()
        S=scipy.spatial.distance.squareform(S-np.diag(np.diag(S)))

    return S
