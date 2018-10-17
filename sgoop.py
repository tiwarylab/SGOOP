""" Script that evaluates reaction coordinates using the SGOOP method. 
Probabilites are calculated using MD trajectories. Transition rates are
found using the maximum caliber approach.  
For unbiased simulations use rc_eval().
For biased simulations calculate unbiased probabilities and analyze then with sgoop().

The original method was published by Tiwary and Berne, PNAS 2016, 113, 2839.

Author: Zachary Smith                   zsmith7@terpmail.umd.edu
Original Algorithm: Pratyush Tiwary     ptiwary@umd.edu 
Contributor: Pablo Bravo Collado        ptbravo@uc.cl"""

import numpy as np
import scipy.optimize as opt
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



"""User Defined Variables"""
in_file = 'ZS.traj' # Input file
nrc = 18            # Number of reaction coordinates
rc_bin = 20         # Bins over RC
wells = 2           # Expected number of wells with barriers > kT
d = 1               # Distance between indexes for transition
prob_cutoff = 1e-5  # Minimum nonzero probability

"""Auxiliary Variables"""
SG = []             # List of Spectral Gaps
RC = []             # List of Reaction Coordinates
P = []              # List of probabilites on RC
SEE = []            # SGOOP Eigen exp
SEV = []            # SGOOP Eigen values
SEVE = []           # SGOOP Eigen vectors

"""Load MD File"""
data_array = np.loadtxt(in_file)



def rei():
    # Reinitializes arrays for new runs
    global SG,RC,P,SEE,SEV,SEVE
    SG = []
    RC = []         
    P = []              
    SEE = []            
    SEV = []            
    SEVE = []       



def normalize_rc(rc):
    # Normalizes input RC
    squares=0
    for i in rc:
        squares+=i**2
    denom=np.sqrt(squares)
    return np.array(rc)/denom



def generate_rc(i):
    # Generates a unit vector with angle pi*i
    x=np.cos(np.pi*i)
    y=np.sin(np.pi*i)
    return (x,y)



def md_prob(rc):
    # Calculates probability along a given RC
    global binned
    proj=[]
    
    for v in data_array:
        proj.append(np.dot(v,rc))
    rc_min=np.min(proj)
    rc_max=np.max(proj)
    binned=(proj-rc_min)/(rc_max-rc_min)*(rc_bin-1)
    binned=np.array(binned).astype(int)
    
    prob=np.zeros(rc_bin)
    
    for point in binned:
        prob[point]+=1
        
    return prob/prob.sum()   # Normalize



def set_bins(rc,bins,rc_min,rc_max):  
    # Sets bins from an external source
    global binned, rc_bin
    rc_bin = bins
    proj = np.dot(data_array,rc)
    binned=(proj-rc_min)/(rc_max-rc_min)*(rc_bin-1)
    binned=np.array(binned).astype(int)



def clean_whitespace(p): 
    # Removes values of imported data that do not match MaxCal data
    global rc_bin, binned
    bmin = np.min(binned)
    bmax = np.max(binned)
    rc_bin = bmax - bmin + 1
    binned -= bmin
    return p[bmin:bmax+1]



def eigeneval(matrix):
    # Returns eigenvalues, eigenvectors, and negative exponents of eigenvalues
    eigenValues, eigenVectors = np.linalg.eig(matrix)
    idx = eigenValues.argsort()     # Sorting by eigenvalues
    eigenValues = eigenValues[idx]  # Order eigenvalues
    eigenVectors = eigenVectors[:,idx]  # Order eigenvectors
    eigenExp = np.exp(-eigenValues)     # Calculate exponentials
    return eigenValues, eigenExp, eigenVectors



def mu_factor(binned,p):
    # Calculates the prefactor on SGOOP for a given RC
    # Returns the mu factor associated with the RC
    # NOTE: mu factor depends on the choice of RC!
    # <N>, number of neighbouring transitions on each RC
    J = 0
    N_mean = 0
    D = 0
    for I in binned:
        N_mean += (np.abs(I-J) <= d)*1
        J = np.copy(I)
    N_mean = N_mean/len(binned)

    # Denominator
    for j in range(rc_bin):
        for i in range(rc_bin):
            if (np.abs(i-j) <= d) and (i != j):
                    D += np.sqrt(p[j]*p[i])
    MU = N_mean/D
    return MU



def transmat(MU,p):
    # Generates transition matrix
    S = np.zeros([rc_bin, rc_bin])
    # Non diagonal terms
    for j in range(rc_bin):
        for i in range(rc_bin):
            if (p[i] != 0) and (np.abs(i-j) <= d and (i != j)) :
                S[i, j] = MU*np.sqrt(p[j]/p[i])

    for i in range(rc_bin):
        S[i,i] = -S.sum(1)[i]  # Diagonal terms
    S = -np.transpose(S)      # Tranpose and fix 
    
    return S



def spectral():
    # Calculates spectral gap for appropriate number of wells
    SEE_pos=SEE[-1][SEV[-1]>-1e-10] # Removing negative eigenvalues
    SEE_pos=SEE_pos[SEE_pos>0] # Removing negative exponents
    gaps=SEE_pos[:-1]-SEE_pos[1:]
    if np.shape(gaps)[0]>=wells:
        return gaps[wells-1]
    else: 
        return 0



def sgoop(rc,p):
    # SGOOP for a given probability density on a given RC
    # Start here when using probability from an external source
    MU = mu_factor(binned,p) # Calculated with MaxCal approach

    S = transmat(MU,p)       # Generating the transition matrix
    
    sev, see, seve = eigeneval(S) # Calculating eigenvalues and vectors for the transition matrix
    SEV.append(sev)               # Recording values for later analysis
    SEE.append(see)
    SEVE.append(seve)
    
    sg = spectral() # Calculating the spectral gap
    SG.append(sg)
    
    return sg



def biased_prob(rc,old_rc):
    # Calculates probabilities while "forgetting" original RC
    global binned
    bias_prob=md_prob(old_rc)
    bias_bin=binned
    
    proj=[]
    for v in data_array:
        proj.append(np.dot(v,rc))
    rc_min=np.min(proj)
    rc_max=np.max(proj)
    binned=(proj-rc_min)/(rc_max-rc_min)*(rc_bin-1)
    binned=np.array(binned).astype(int)
    
    prob=np.zeros(rc_bin)
    
    for i in range(np.shape(binned)[0]):
        prob[binned[i]]+=1/bias_prob[bias_bin[i]] # Dividing by RAVE-like weights
        
    return prob/prob.sum()   # Normalize



def best_plot():
    # Displays the best RC for 2D data
    best_rc=np.ceil(np.arccos(RC[np.argmax(SG)][0])*180/np.pi)
    plt.figure()
    cmap=plt.cm.get_cmap("jet")
    hist = np.histogram2d(data_array[:,0],data_array[:,1],100)
    hist = hist[0]
    prob = hist/np.sum(hist)
    potE=-np.ma.log(prob)
    potE-=np.min(potE)
    np.ma.set_fill_value(potE,np.max(potE))
    plt.contourf(np.transpose(np.ma.filled(potE)),cmap=cmap)

    plt.title('Best RC = {0:.2f} Degrees'.format(best_rc))
    origin=[50,50]
    rcx=np.cos(np.pi*best_rc/180)
    rcy=np.sin(np.pi*best_rc/180)
    plt.quiver(*origin,rcx,rcy,scale=.1,color='grey');
    plt.quiver(*origin,-rcx,-rcy,scale=.1,color='grey');



def rc_eval(rc):
    # Unbiased SGOOP on a given RC
    # Input type: array of weights
    
    """Save RC for Calculations"""
    rc = normalize_rc(rc)
    RC.append(rc)

    """Probabilities and Index on RC"""
    prob=md_prob(rc)
    P.append(prob)

    """Main SGOOP Method"""
    sg = sgoop(rc,prob)
    
    return sg



def biased_eval(rc,bias_rc):
    # Biased SGOOP on a given RC with bias along a second RC
    # Input type: array of weights, probability from original RC
    
    """Save RC for Calculations"""
    rc = normalize_rc(rc)
    RC.append(rc)

    """Probabilities and Index on RC"""
    prob=biased_prob(rc,bias_rc)
    P.append(prob)

    """Main SGOOP Method"""
    sg = sgoop(rc,prob)
    
    return sg

