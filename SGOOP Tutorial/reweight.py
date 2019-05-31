#!/usr/bin/env python
"""This reweighting code is based on the algorithm proposed by Tiwary
and Parrinello, JPCB 2014, 119 (3), 736-742. This is a modified version
of te reweighting code based on earlier version (v1.0 - 23/04/2015) 
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
col_fe = 1                    # Column of free energy.
datafile = "COLVAR_short5"    # COLVAR file name.
col_rewt = [2,3,5,6]          # COLVAR columns corresponding to RC variables.
numrewt = 1                   # Number of reweighting iterations.
col_bias = [7]                # COLVAR bias column.
ngrid = 50                    # Number of grid bins.



def load():
    # Loads given files. Runs on import to prevent redundant loading.
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



load()



def parse():
    # Used for import when running on the command line.
    d = """
    It is a reweighting code to reweight some RC which is linear combination of a set
    of order parameters which have the effect of biasing while the metadynamics run
    were performed along some CV. Here, RC=c1*a1+c2*a2+c3*a3+... (where RC is the 
    reaction coordinate to be reweighted, c1, c2,... are the coefficients, a1, a2,
    ... are the order parameters)
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=d, epilog=" ")

    parser.add_argument("-bsf", type=float, default=15.0, help="biasfactor for the well-tempered metadynamics")
    parser.add_argument("-kT", type=float, default=2.5, help="kT energy value in kJ/mol")
    parser.add_argument("-fpref", default="fes", help="free energy filenames from sum hills (default: %(default)s)")

    parser.add_argument("-nf", type=int, default=100, help="number of FES input files (default: %(default)s)")
    parser.add_argument("-fcol", type=int, default=2, help="free energy column in the FES input files (first column = 1) (default: %(default)s)")

    parser.add_argument("-colvar", default="COLVAR", help="filename containing original CVs, reweighting CVs and metadynamics bias")
    parser.add_argument("-rewcol", type=int, nargs='+', default=[ 2 ], help="column(s) in colvar file containing the CV to be reweighted (first column = 1) (default: %(default)s)")
    #parser.add_argument("-coef", type=float, nargs='+', default=[ 1 ], help="coefficients for each order parameters")

    parser.add_argument("-biascol", type=int, nargs='+', default=[ 4 ], help="column(s) in colvar file containing any energy bias (metadynamic bias, walls, external potentials..) (first column = 1) (default: %(default)s)")

    parser.add_argument("-min", type=float, nargs='+', help="minimum values of the CV")
    parser.add_argument("-max", type=float, nargs='+', help="maximum values of the CV")
    parser.add_argument("-bin", type=int, default=50, help="number of bins for the reweighted FES (default: %(default)s)")

    #parser.add_argument("-outfile", default="fes_rew", help="output FES filename (default: %(default)s)")

    parser.print_help()
    return parser.parse_args



def reweight(rc,commandline=False,sparse=False):
    # Reweighting biased MD trajectory to unbiased probabilty along a given RC.
    # By default (sparse=False) bins on the edge of the range with probabilities lower
    # than 1/N where N is number of data points will be removed.
    global gamma, kT, fesfilename, numdat, col_fe, datafile, col_rewt, numrewt, col_bias, ngrid, s_min, s_max
    #### INPUTS
    if commandline:
        args = parser.parse_args()
        gamma = args.bsf
        kT = args.kT
        fesfilename = args.fpref
        numdat = args.nf
        col_fe = args.fcol - 1
        datafile = args.colvar
        col_rewt = [ i-1 for i in args.rewcol ]
        numrewt = 1
        col_bias = [ i-1 for i in args.biascol ] 
        minz = args.min
        s_min = np.min(minz)
        s_min = np.reshape(s_min,newshape=(s_min.size,1))
        maxz = args.max
        s_max = np.max(maxz)
        s_max = np.reshape(s_max,newshape=(s_max.size,1))
        ngrid = args.bin



    #
    #Boltzmann-like sampling for reweighting
    #

    coeff = rc
    rc_space = np.dot(colvar[:,col_rewt],coeff)
    s_max = np.max(rc_space)
    s_min = np.min(rc_space)

    # build the new square grid for the reweighted FES
    s_grid = [[ ]] * numrewt

    #print(s_grid, numrewt)
    #print(s_max, s_min, ngrid)

    ds = (s_max - s_min)/(ngrid-1)
    s_grid = [ s_min + n*ds for n in range(ngrid) ]
    #print(s_max, s_min, ds)
    
    
    
    numcolv = np.shape(colvar)[0]

    # initialize square array numrewt-dimensional
    fes = np.zeros( [ ngrid ] * numrewt)

    # go through the CV(t) trajectory
    denom = 0.
    i = 0
    for row in colvar:
        i += 1

        # build the array of grid indeces locs corresponding to the point closest to current point
        locs = [[ ]] * numrewt
        for j in range(numrewt):
            col = col_rewt[j]
            #depending on the number of order parameters to be linearly added to get the RC to be reweighted, 
            #number of row[col]*q[0] terms will be added to the val below. 
            #val = (row[col]*q[0] + row[col+1]*q[1] + row[col+2]*q[2] + row[col+3]*q[3] + row[col+4]*q[4])/5
            val = np.dot(row[col_rewt],coeff) 
            #val = row[col]*0 + row[col+1]*1
            locs[j] = int((val-s_min)/ds) # find position of minimum in diff array

        #find closest c(t) for this point of time
        indx = int(ceil(float(i)/numcolv*numdat))-1
        bias = sum([row[j] for j in col_bias])
        ebias = exp(bias/kT)/ebetac[indx]
        fes[locs] += ebias
        denom += ebias

    # ignore warnings about log(0) and /0
    np.seterr(all='ignore')
    fes /= denom
    fes = -kT*np.log(fes)

    # set FES minimum to 0
    fes -= np.min(fes)
    z = np.sum(np.exp(-fes/kT))
    pavg = (np.exp(-fes/kT))/z
    total = np.sum(pavg)
    pnorm = pavg/total

    if commandline:
        #with open(out_fes_xy, 'w') as f:    
        with open("prob_rew.dat", 'w') as f:
            for nx,x in enumerate(s_grid):
                f.write('%20.12f %20.12f\n' % (x,pnorm[nx]))
        f.close()
        
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



def rebias(rc,old_rc,old_p,commandline=False,sparse=False):
    # Reweighting biased MD trajectory to a probability along a RC with SGOOP-bias along a second RC.
    # By default (sparse=False) bins on the edge of the range with probabilities lower
    # than 1/N where N is number of data points will be removed.
    global gamma, kT, fesfilename, numdat, col_fe, datafile, col_rewt, numrewt, col_bias, ngrid, s_min, s_max
    #### INPUTS
    if commandline:
        args = parser.parse_args()
        gamma = args.bsf
        kT = args.kT
        fesfilename = args.fpref
        numdat = args.nf
        col_fe = args.fcol - 1
        datafile = args.colvar
        col_rewt = [ i-1 for i in args.rewcol ]
        numrewt = 1
        col_bias = [ i-1 for i in args.biascol ] 
        minz = args.min
        s_min = np.min(minz)
        s_min = np.reshape(s_min,newshape=(s_min.size,1))
        maxz = args.max
        s_max = np.max(maxz)
        s_max = np.reshape(s_max,newshape=(s_max.size,1))
        ngrid = args.bin



    #
    #Boltzmann-like sampling for reweighting
    #

    coeff = rc
    rc_space = np.dot(colvar[:,col_rewt],coeff)
    bias_space = np.dot(colvar[:,col_rewt],old_rc)
    
    s_max = np.max(rc_space)
    s_min = np.min(rc_space)
    
    
    b_max = np.max(bias_space)
    b_min = np.min(bias_space)
    
    # build the new square grid for the reweighted FES
    s_grid = [[ ]] * numrewt

    #print(s_grid, numrewt)
    #print(s_max, s_min, ngrid)

    ds = (s_max - s_min)/(ngrid-1)
    db = (b_max - b_min)/(ngrid-1)
    s_grid = [ s_min + n*ds for n in range(ngrid) ]
    #print(s_max, s_min, ds)
    
    
    
    numcolv = np.shape(colvar)[0]

    # initialize square array numrewt-dimensional
    fes = np.zeros( [ ngrid ] * numrewt)

    # go through the CV(t) trajectory
    denom = 0.
    i = 0
    for row in colvar:
        i += 1

        # build the array of grid indeces locs corresponding to the point closest to current point
        locs = [[ ]] * numrewt
        blocs = [[ ]] * numrewt
        for j in range(numrewt):
            col = col_rewt[j]
            #depending on the number of order parameters to be linearly added to get the RC to be reweighted, 
            #number of row[col]*q[0] terms will be added to the val below. 
            #val = (row[col]*q[0] + row[col+1]*q[1] + row[col+2]*q[2] + row[col+3]*q[3] + row[col+4]*q[4])/5
            val = np.dot(row[col_rewt],coeff) 
            bval = np.dot(row[col_rewt],old_rc)
            locs[j] = int((val-s_min)/ds) # find position of minimum in diff array
            blocs[j] = int((bval-b_min)/db)

        #find closest c(t) for this point of time
        indx = int(ceil(float(i)/numcolv*numdat))-1
        bias = sum([row[j] for j in col_bias])
        ebias = exp(bias/kT)/(ebetac[indx]*old_p[blocs])
        fes[locs] += ebias
        denom += ebias

    # ignore warnings about log(0) and /0
    np.seterr(all='ignore')
    fes /= denom
    fes = -kT*np.log(fes)

    # set FES minimum to 0
    fes -= np.min(fes)
    z = np.sum(np.exp(-fes/kT))
    pavg = (np.exp(-fes/kT))/z
    total = np.sum(pavg)
    pnorm = pavg/total

    if commandline:
        #with open(out_fes_xy, 'w') as f:    
        with open("prob_rew.dat", 'w') as f:
            for nx,x in enumerate(s_grid):
                f.write('%20.12f %20.12f\n' % (x,pnorm[nx]))
        f.close()
        
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
    global gamma, kT, fesfilename, numdat, col_fe, datafile, col_rewt, numrewt, col_bias, ngrid, s_min, s_max,fes
    if data != None:
        datafile = data
        load()
    numcolv = np.shape(colvar)[0]

    # initialize square array numrewt-dimensional
    fes = np.zeros(numcolv)

    # go through the CV(t) trajectory
    denom = 0.
    i = 0
    for row in colvar:
        i += 1

        # build the array of grid indeces locs corresponding to the point closest to current point
        locs = [[ ]] * numrewt
        for j in range(numrewt):
            col = col_rewt[j]
        indx = int(ceil(float(i)/numcolv*numdat))-1
        bias = sum([row[j] for j in col_bias])
        ebias = exp(bias/kT)/ebetac[indx]
        fes[i-1] = ebias
        denom += ebias

    hist = np.histogram2d(colvar[:,d1],colvar[:,d2],size,weights=fes)
    hist = hist[0]
    pnorm = hist/np.sum(hist)
    return pnorm

