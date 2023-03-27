import dca
import Bio
import scipy
import numpy as np
import sequenceHandler as sh
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.spatial.distance import squareform
from sklearn.decomposition import TruncatedSVD
from matplotlib.colors import LinearSegmentedColormap

def plotTopContacts(pdbMap,dcaFile,Ntop,contactThreshold,minSeqSeparation=4):
    """ Plots the top N ranked DCA predictions overlaip on the structual contact map

    Keword Arguments:
        pdbMap (str) : The aligned distance map file, as output by mapPDB.

        dcaFile (str): The file containing the DCA contacts. The file must only contain the upper triangular part 
                           of the DCA score matrix, ordered in linear form, i.e. the contacts are ordered as 
                           S(1,2),S(1,3),...,S(1,N),S(2,3),S(2,4),...,S(N-1,N)

        Ntop (int)   : The number of top ranked DCA contacts to plot. 

        contactThreshold (float): The threshold defining a structural contact. Contacts are computed between heavy-atoms.
        minSeqSeparation (int): Minimum number of separation along the sequence (in the alignment) below which contacts are not extracted (default 4)
    """

    # Plot structural contacts
    dm=np.loadtxt(pdbMap,dtype=float)
    pdbContacts=np.argwhere(dm<=contactThreshold)
    
    # Overlay DCA predictions
    dcaContacts,_=dca.extractTopContacts(dcaFile,Ntop,minSeqSeparation)
    dcaColors=dm[dcaContacts[:,0],dcaContacts[:,1]]<contactThreshold
    dcaColors=['lime' if col else 'r' for col in dcaColors]

    # Handle the figure
    plt.figure(figsize=(6,6))
    plt.scatter(pdbContacts[:,0],pdbContacts[:,1],s=15,color='0.55',alpha=0.4)
    plt.scatter(dcaContacts[:,0],dcaContacts[:,1],s=8,color=dcaColors)
    plt.xlim([-2,dm.shape[0]+2])
    plt.ylim([-2,dm.shape[1]+2])
    plt.grid()
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    plt.tight_layout(pad=0., w_pad=0., h_pad=0.)
    plt.show()


def plotTPrates(pdbMap,dcaFiles,Nmax,contactThreshold):
    """ Plots the precitions curves for DCA predictions

    Keyword Arguments:
        pdbMap (str): The aligned distance map file, as output by mapPDB.

        dcaFiles (list): List of file containing DCA contacts. Each file must only contain the upper triangular part 
                           of the DCA score matrix, ordered in linear form, i.e. the contacts are ordered as 
                           S(1,2),S(1,3),...,S(1,N),S(2,3),S(2,4),...,S(N-1,N)

        Nmax (int): Maximum number of predicted contacts to report for the precision curves

        contactThreshold (float): The threshold defining a structural contact. Contacts are computed between heavy-atoms.
    """

    # Load all DCA scores
    dcaScores=[]
    for dcaScore in dcaFiles:
        dcaScores.append(np.loadtxt(dcaScore))
   
    # Get structural contacts
    dm=np.loadtxt(pdbMap,dtype=float)
    pdbContacts=np.argwhere(dm<=contactThreshold)

    for score in dcaScores:
        # Compute precision for N in [1:Nmax]
        dca=scipy.spatial.distance.squareform(score)
        for i in range(5):
            dca=dca-np.diag(np.diag(dca,k=i),k=i)
            dca=dca-np.diag(np.diag(dca,k=-i),k=-i)
        sortedScores=-np.sort(scipy.spatial.distance.squareform(-dca),)
            
        precision=np.zeros(Nmax)
        for N in np.arange(Nmax):
            dcaContacts=np.argwhere(dca>sortedScores[N+1])
            precision[N]=(dm[dcaContacts[:,0],dcaContacts[:,1]]<contactThreshold).mean()
                
        # Plot the precision curve
        plt.plot(np.arange(Nmax)/float(dm.shape[0]),precision)

    # Handle the figure
    plt.ylim(0,1.05)
    plt.grid()
    plt.xlabel('$N_{\mathrm{pred}} /L$',fontsize=15)
    plt.ylabel('Precision',fontsize=15)
    plt.show()
    


def plotDistDistribution(pdbMap,dcaFile,Ntops):
    """ Plots the distribution of the distances of predicted DCA contacts.

    Keyword Arguments:
        pdbMap (str): The aligned distance map file, as output by mapPDB.

        dcaFile (str): The file containing the DCA contacts. The file must only contain the upper triangular part 
                           of the DCA score matrix, ordered in linear form, i.e. the contacts are ordered as 
                           S(1,2),S(1,3),...,S(1,N),S(2,3),S(2,4),...,S(N-1,N)

        Ntops (list): List of number of top ranked DCA predictions. A distance histogram is plotted for each value.
    """

    # Apply short-range filter (i.e. ignore i-i+4 contacts)
    dm=np.loadtxt(pdbMap,dtype=float)
    for i in range(5):
        dm=dm-np.diag(np.diag(dm,k=i),k=i)
        dm=dm-np.diag(np.diag(dm,k=-i),k=-i)
        
    # Compute PDB distances
    pdbDistances=np.triu(dm).flatten()
    pdbDistances=pdbDistances[np.isfinite(pdbDistances)]
    pdbDistances=pdbDistances[pdbDistances>0]
    pdbHist,pdbBins=np.histogram(pdbDistances,50,density=True)
    plt.plot(pdbBins[:-1],pdbHist/max(pdbHist),linewidth=3)

    legend=['PDB']
    
    # Compute DCA distances   
    for N in Ntops:
        dcaContacts,_=dca.extractTopContacts(dcaFile,N)
        dcaDistances=dm[dcaContacts[:,0],dcaContacts[:,1]]
        dcaDistances=dcaDistances[np.isfinite(dcaDistances)]
        dcaHist,dcaBins=np.histogram(dcaDistances,10,density=True)
        plt.plot(dcaBins[:-1],dcaHist/max(dcaHist),linewidth=3)
        legend.append('DCA, $N_{DCA}/N$=' + str(round(N/float(dm.shape[0]),2)))
        
    # Handle the figure
    plt.xlabel('Distance [A]',fontsize=18)
    plt.ylabel('Unnormalized Density',fontsize=18)
    plt.legend(legend)
    plt.ylim([0,1.1])
    plt.grid()
    plt.show()    



def plotFrequencyCorrelations(originalMSA, resampledMSA, computeCoFrequencies,save):
    """ Plots the frequencies and co-frequencies of the resampled sequences versus the original sequences.

    Keyword Arguments:
        originalMSA (str): The file containing the original MSA in fasta format

        resampledMSA (str): The file containig the aligned resampled sequences in fasta format

        computeCoFrequencies (bool): If false, only compute single-site frequencies

        save  (str): Optional filename to wich save the computed frequencies. If None, no frequencies are saved."
    """

    origMSA=sh.binarizeMSA(sh.fastaToMatrix(originalMSA)[0])
    resampledMSA=sh.binarizeMSA(sh.fastaToMatrix(resampledMSA)[0])

    fi=origMSA.mean(axis=0)
    fi_resampled=resampledMSA.mean(axis=0)

    plt.figure(1)
    plt.plot(fi,fi_resampled,'+',ms=3,c='b')
    plt.plot([0,1],[0,1],'r')
    plt.axis('equal')
    plt.xlabel('$f_i^{Data}$',fontsize=18)
    plt.ylabel('$f_i^{Model}$',fontsize=18)
    plt.xlim(0,np.max([np.max(fi),np.max(fi_resampled)]))
    plt.ylim(0,np.max([np.max(fi),np.max(fi_resampled)]))
    plt.grid()

    if save:
        
        np.savetxt(save+"_fi.dat",np.vstack((fi,fi_resampled)).T,delimiter=' ')
        
    if computeCoFrequencies:
        fij=(origMSA.T.dot(origMSA)/float(origMSA.shape[0])).todense()
        fij=squareform(fij-np.diag(np.diag(fij)))
        fij_resampled=(resampledMSA.T.dot(resampledMSA)/float(resampledMSA.shape[0])).todense()
        fij_resampled=squareform(fij_resampled-np.diag(np.diag(fij_resampled)))
        plt.figure(2)
        plt.plot(fij[::1],fij_resampled[::1],'+',ms=3,c='b')
        plt.plot([0,1],[0,1],'r')
        plt.axis('equal')
        plt.xlabel('$f_{ij}^{Data}$',fontsize=18)
        plt.ylabel('$f_{ij}^{Model}$',fontsize=18)
        plt.xlim(0,np.max([np.max(fij),np.max(fij_resampled)]))
        plt.ylim(0,np.max([np.max(fij),np.max(fij_resampled)]))
        plt.grid()

        if save:
            np.savetxt(save+"_fij.dat",np.vstack((fij,fij_resampled)).T,delimiter=' ')


    plt.show()


def pca(baseMSA,axes,overMSA,fast,save,mSize=7,plot=False):
    """ Performs PCA on the baseMSA, optionally overlaying sequences in overMSA.

        Keyword Arguments:
            baseMSA (str): The file containing the base MSA in fasta format.

            axes (2-tuple): The PCA axes on which to project the data

            overMSA (str): The file containing the MSA sequences to overaly. To ignore this option, pass None"

            fast   (bool): If true, consider only a maximum of 1000 sequences to perform the PCA.

            save  (str): Optional filename to wich save the projected coordinates. If None, no coordinates are saved."

            plot (bool): If False, does not plot the PCA.
    """
    
    axes=[int(axes.split(',')[0]),int(axes.split(',')[1])]
    categorical=sh.binarizeMSA(sh.fastaToMatrix(baseMSA)[0])

    if fast: 
        categorical = categorical[0:min(1000,categorical.shape[0]),:]

    meanSeq=categorical.mean(axis=0)
    categorical=categorical-meanSeq
    svd=TruncatedSVD(n_components=max(axes)+1,algorithm="randomized", n_iter=20)
    svd=svd.fit(categorical)
    projected=svd.transform(categorical)

    if plot:
        plt.hexbin(projected[:,axes[0]],projected[:,axes[1]],gridsize=60,cmap='Blues',norm=LogNorm(),mincnt=1)
    
    if overMSA:
        categoricalOver=sh.binarizeMSA(sh.fastaToMatrix(overMSA)[0])
        categoricalOver=categoricalOver-meanSeq
        projectedOver=svd.transform(categoricalOver)
        if plot:
            plt.scatter(projectedOver[:,axes[0]],projectedOver[:,axes[1]],s=mSize,color='r')
        if save:
            np.savetxt(save+"_over",projectedOver,delimiter=' ')

    if plot:
        plt.axis('equal')
        plt.xlabel('PC ' + str(axes[0]))
        plt.ylabel('PC ' + str(axes[1]))
        plt.grid()
        plt.show()

    if save:
        np.savetxt(save,projected,delimiter=' ')

    if overMSA:
        return projected,projectedOver
    else:
        return projected

def listTopContacts(dcaFile,Ntop,hmmFile,seqFile,minSeqSeparation=4):
    """ List the N top ranked DCA predicted contacts. If an optional reference sequence and the mapping hmm are passed,
        the predicted contacts are reported in the reference sequence indexing.

     Keyword Arguments:
        dcaFile (str): The file containing the DCA contacts. The file must only contain the upper triangular part 
                           of the DCA score matrix, ordered in linear form, i.e. the contacts are ordered as 
                           S(1,2),S(1,3),...,S(1,N),S(2,3),S(2,4),...,S(N-1,N)

        Ntop (int)   : Ntop (int)   : The number of top ranked DCA contacts to plot. 

        hmmFile (str): An optional hmmFile defining the family. If this is passed together with seqFile, the contacts indexes are 
                       both returned in MSA numbering and in the numbering of the reference sequence in seqFile.

        seqFile (str): Optional reference sequence file. If provided together with the family hmmFile, the contacts indexes are 
                       both returned in MSA numbering and in the numbering of the reference sequence in seqFile.

        minSeqSeparation (int): Minimum number of separation along the sequence (in the alignment) below which contacts are not extracted (default 4)
    """
    
    # Extract DCA predictions
    dcaContacts,sortedScores=dca.extractTopContacts(dcaFile,Ntop,minSeqSeparation)
    dcaMatrix=scipy.spatial.distance.squareform(np.loadtxt(dcaFile))
    N=dca.getSize(dcaFile)

    # Map contacts coordinates to reference sequence (Optional)
    if(hmmFile):
        refSequence=str(list(Bio.SeqIO.parse(seqFile,'fasta'))[0].seq)
        mapIndexes=sh.alignSequenceToHMM(refSequence,hmmFile)
    else:
        mapIndexes=range(N)

    for i,index in enumerate(mapIndexes):
        if index<0:
            mapIndexes[i]=-1

    
    # Display DCA predicted contacts
    print('#%7s %7s %2s %7s %7s %7s' % ('i','j','|','i_map','j_map','rank'))
    print("#"+'-'*42)
    for contact in dcaContacts:
        if contact[0]<contact[1]:
            rank=np.argwhere(sortedScores>=dcaMatrix[contact[0],contact[1]])[-1,0]+1
            if hmmFile:
                mapAA1=refSequence[mapIndexes[contact[0]]]+str(mapIndexes[contact[0]]+1),
                mapAA2=refSequence[mapIndexes[contact[1]]]+str(mapIndexes[contact[1]]+1),
                if mapIndexes[contact[0]]==-1:
                    mapAA1=['--']
                if mapIndexes[contact[1]]==-1:
                    mapAA2=['--']
                print('%7s %7s %2s %7s %7s %7s' % (contact[0]+1,contact[1]+1,'|',mapAA1[0],mapAA2[0],rank))
            else:
                print('%7s %7s %2s %7s %7s %7s' % (contact[0]+1,contact[1]+1,'|','--','--',rank))


def listInterfaceContacts(dcaFile,N1,Beff,hmmFile1,hmmFile2,seqFile1,seqFile2,cutoff=0.8):
    """ List the predicted inter-protein interface contacts as defined by the Hopf2014 criterion.
        If optional sequence files and hmm mapping files are provide, the predicted contacts are reported in the reference sequeces indexing.

     Keyword Arguments:
        dcaFile (str) : The file containing the DCA contacts. The file must only contain the upper triangular part 
                           of the DCA score matrix, ordered in linear form, i.e. the contacts are ordered as 
                           S(1,2),S(1,3),...,S(1,N),S(2,3),S(2,4),...,S(N-1,N)

        N1 (int)      : The length of the first  protein in the alignmnet.

        Beff (double) : The number of effective sequences after identity reweighting.s
        
        hmmFile1 (str): An optional hmmFile defining the family 1. If this is passed together with seqFile1, the contacts indexes are 
                       both returned in MSA numbering and in the numbering of the reference sequence in seqFile1.

        hmmFile2 (str): An optional hmmFile defining the family 2. If this is passed together with seqFile2, the contacts indexes are 
                       both returned in MSA numbering and in the numbering of the reference sequence in seqFile2.

        seqFile1 (str): Optional reference sequence file 1. If provided together with the family hmmFile1, the contacts indexes are 
                       both returned in MSA numbering and in the numbering of the reference sequence in seqFile1.

        seqFile2 (str): Optional reference sequence file 2. If provided together with the family hmmFile2, the contacts indexes are 
                       both returned in MSA numbering and in the numbering of the reference sequence in seqFile2.

        cutoff (double): The cutoff on the score over which inter-protein contacts are considered.
    """

    contacts = dca.extractInterContacts(dcaFile,N1,Beff,cutoff)

    Ntot = dca.getSize(dcaFile)
    N2 = Ntot-N1
    
    if (hmmFile1 and seqFile1):
        refSeq1=str(list(Bio.SeqIO.parse(seqFile1,'fasta'))[0].seq)
        mapIndexes1=sh.alignSequenceToHMM(refSeq1,hmmFile1)
    else:
        mapIndexes1=range(N1)

    for i,index in enumerate(mapIndexes1):
        if index<0:
            mapIndexes1[i]=-1

    if (hmmFile2 and seqFile2):
        refSeq2=str(list(Bio.SeqIO.parse(seqFile2,'fasta'))[0].seq)
        mapIndexes2=sh.alignSequenceToHMM(refSeq2,hmmFile2)
    else:
        mapIndexes2=range(N2)

    for i,index in enumerate(mapIndexes2):
        if index<0:
            mapIndexes2[i]=-1

    contacts=dca.extractInterContacts(dcaFile,N1,Beff,cutoff)

    print('#%7s %7s %2s %7s %7s ' % ('i','j','|','i_map','j_map'))
    print('#'+'-'*42)
    for contact in contacts[0]:
        if (hmmFile1 and seqFile1 and hmmFile2 and seqFile2):
            print('%7s %7s %2s %7s %7s' % (contact[0]+1,contact[1]+1,'|',
                  refSeq1[mapIndexes1[contact[0]]]+str(mapIndexes1[contact[0]]+1),
                  refSeq2[mapIndexes2[contact[1]]]+str(mapIndexes2[contact[1]]+1)))
        else:
            print('%7s %7s %2s %7s %7s' % (contact[0]+1,contact[1]+1,'|','--','--'))

            
def linCMap(values,colors):
    """ Return a colormap, linearly interpolated such that colors are mapped onto values.

    Keyword Arguments:
         values (list): The values onto which to map the colors

         color (list): The colors used for interpolatio in hex code.

    Returns:
         cmap (Colormap object): The colormap object in pyplot format

    """
    values=np.asarray(values)
    values=(values-values.min())/(values.max()-values.min())
    pairs=zip(values,colors)
    cmap = LinearSegmentedColormap.from_list('name',pairs)

    return cmap

