#!/usr/bin/env python
"""This reweighting code is based on the algorithm proposed by Tiwary
and Parrinello, JPCB 2014, 119 (3), 736-742. This is a modified version
of the reweighting code based on earlier version (v1.0 - 23/04/2015) 
available in GitHub which was originally written by L. Sutto and 
F.L. Gervasio, UCL.

Co-Author: Debabrata Pramanik       pramanik@umd.edu
Co-Author: Zachary Smith            zsmith7@terpmail.umd.edu """

import os.path
import argparse
import numpy as np
from math import log, exp, ceil



# Default Arguments
gamma = 15                    # Biasfactor in well-tempered metadynamics.
kT = 2.5                      # Temperature times Boltzmann constant.
fesfilename = "fes_"          # FES file name start.
numdat = 20                   # Number of FES files.
col_fe = 1                    # Column of free energy, indexing from 0.
datafile = "COLVAR_short5"    # COLVAR file name.
col_rewt = [2,3,5,6]          # COLVAR columns corresponding to RC variables, indexing from 0.
col_bias = [7]                # COLVAR bias column, indexing from 0.
ngrid = 50                    # Number of grid bins.


def weights():
    # compute reweighting weights from FES data and bias column
    global kT, numdat, col_bias, weights
    numcolv = np.shape(colvar)[0]
    weights = np.zeros(numcolv)

    # go through the CV(t) trajectory
    i = 0
    for row in colvar:
        i += 1
        indx = int(ceil(float(i)/numcolv*numdat))-1
        bias = sum([row[j] for j in col_bias])
        ebias = exp(bias/kT)/ebetac[indx]
        weights[i-1] = ebias
        
def ebc():
    global colvar,ebetac
    # File Inputs
    colvar = np.loadtxt(datafile)

    # Calculating c(t):
    # calculates ebetac = exp(beta c(t)), using eq. 12 in eq. 3 in the JPCB paper
    #
    ebetac = []

    for i in range(numdat):
        # set appropriate format for FES file names, NB: i starts from 0
        fname = '%s%d.dat' % (fesfilename,i)

        data = np.loadtxt(fname)
        s1, s2 = 0., 0.
        for p in data:
            exponent = -p[col_fe]/kT
            s1 += exp(exponent)
            s2 += exp(exponent/gamma)
        ebetac += s1 / s2,

def load():
    # Loads given files. Runs on import to prevent redundant loading, note this increases import time.
    ebc()   
    weights()


load()



def reweight(rc,sparse=False,size=50,data=None):
    # Reweighting biased MD trajectory to unbiased probabilty along a given RC.
    # By default (sparse=False) bins on the edge of the range with probabilities lower
    # than 1/N where N is number of data points will be removed.
    global kT, fesfilename, numdat, col_fe, datafile, col_rewt, numrewt, col_bias, ngrid, s_min, s_max
    
    if data != None:
        datafile = data
        load()

    rc_space = np.dot(colvar[:,col_rewt],rc)
    s_max = np.max(rc_space)
    s_min = np.min(rc_space)

    

    # initialize square array numrewt-dimensional
    
    
    hist = np.histogram(rc_space,size,weights=weights)[0]
    pnorm = hist/np.sum(hist)


        
    # Trimming off probability values less than one data point could provide
    if not sparse:
        cutoff = 1/np.shape(colvar)[0]
        trim = np.nonzero(pnorm >= cutoff)
        trimmed = pnorm[np.min(trim):np.max(trim)+1]
        if np.min(trimmed) < cutoff:
            cutoff = np.min(trimmed)
            trim = np.nonzero(pnorm >= cutoff)
            trimmed = pnorm[np.min(trim):np.max(trim)+1]
        return trimmed
    return pnorm



def rebias(rc,old_rc,old_p,sparse=False,old_size=50,new_size=50,data=None):
    # Reweighting biased MD trajectory to a probability along a RC with SGOOP-bias along a second RC.
    # By default (sparse=False) bins on the edge of the range with probabilities lower
    # than 1/N where N is number of data points will be removed.
    global kT, fesfilename, numdat, col_fe, datafile, col_rewt, numrewt, col_bias, ngrid, s_min, s_max
    
    if data != None:
        datafile = data
        load()

    rc_space = np.dot(colvar[:,col_rewt],rc)
    bias_space = np.dot(colvar[:,col_rewt],old_rc)
    
    s_max = np.max(rc_space)
    s_min = np.min(rc_space)
    
    bins = np.histogram(bias_space,old_size)[1]
    binned = np.digitize(bias_space,bins[:-1])-1
    
    hist = np.histogram(rc_space,new_size,weights=weights/old_p[binned])[0]
    pnorm = hist/np.sum(hist)

        
    # Trimming off probability values less than one data point could provide
    if not sparse:
        cutoff = 1/np.shape(colvar)[0]
        trim = np.nonzero(pnorm >= cutoff)
        trimmed = pnorm[np.min(trim):np.max(trim)+1]
        if np.min(trimmed) < cutoff:
            cutoff = np.min(trimmed)
            trim = np.nonzero(pnorm >= cutoff)
            trimmed = pnorm[np.min(trim):np.max(trim)+1]
        return trimmed
    return pnorm



def reweight2d(d1,d2,size=100,data=None):
    # Reweighting biased MD trajectory to a 2D probability.
    global kT, fesfilename, numdat, col_fe, datafile, col_rewt, numrewt, col_bias, ngrid, s_min, s_max,fes, weights
    if data != None:
        datafile = data
        load()

    hist = np.histogram2d(colvar[:,d1],colvar[:,d2],size,weights=weights)
    hist = hist[0]
    pnorm = hist/np.sum(hist)
    return pnorm



