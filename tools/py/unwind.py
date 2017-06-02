#!/usr/bin/env python2 
"""

Relies on the infrastructure of i-pi, so the ipi package should
be installed in the Python module directory, or the i-pi
main directory must be added to the PYTHONPATH environment variable.

Cuts short the output of a previous i-pi simulation, up to the
step indicated in the <step> field of the input file.
This is useful to restart a simulation that crashed.

It should be run in the same dyrectory as where i-pi was (or is being)
run, and simply fetches all information from the simulation input file.
One should also specify a directory name in which the trimmed files
will be output.

Syntax:
   trimsim.py inputfile.xml
"""


import sys
import os
import argparse
import numpy as np
from ipi.engine.outputs import *
from ipi.engine.properties import getkey
from ipi.inputs.simulation import InputSimulation
from ipi.utils.io.inputs import io_xml
try:
    import scipy.linalg as sp
    __has_scipy = True
except:
    __has_scipy = False    

def input_vvac(path2inputfile, mrows, stride):
    """Imports the vvac file and extracts the ."""
    #TODO: make changes to the column numbers.
    dvvac=np.genfromtxt(path2inputfile, usecols=((2,3)))
    if( mrows == -1 ):
        mrows = len(dvvac)
    return dvvac[:mrows][::stride]

def output_vvac(xy,oprefix, refvvac):
    """Imports the vvac file and extracts the ."""
    xorg=refvvac[:,0]
    xred=xy[0]
    yred=xy[1]
    x=xred
    y=yred
    np.savetxt(oprefix + "-vv.data", np.vstack((x, y)).T)

def Aqp(omega_0, Ap):
    """Given the free particle Ap matrix and the frequency of the harmonic oscillator, computes the full drift matrix."""
    dAqp = np.zeros(np.asarray(Ap.shape) + 1)
    dAqp[0,1] = -np.power(omega_0,2)
    dAqp[1,0] = 1
    dAqp[1:,1:] = Ap.T
    return dAqp

def Dqp(omega_0, Dp):
    """Given the free particle Dp matrix and the frequency of the harmonic oscillator, computes the full D matrix."""
    dDqp = np.zeros(np.asarray(Dp.shape) + 1)
    dDqp[1:,1:] = Dp.T
    return dDqp

def Cqp(omega_0, Ap, Dp):
    """Given the free particle Ap and Dp matrices and the frequency of the harmonic oscillator, computes the full covariance matrix."""
    dAqp = Aqp(omega_0, Ap)
    dDqp = Dqp(omega_0, Dp)
    return sp.solve_continuous_are( -dAqp, np.zeros(dAqp.shape), dDqp, np.eye(dAqp.shape[-1]))

def Cqp(omega0, dAqp, dDqp):
    """Given the free particle Ap and Dp matrices and the frequency of the harmonic oscillator, computes the full covariance matrix."""
    # "stabilizes" the calculation by removing the trivial dependence of <a^2> on omega0 until the very end
    dAqp[:,0]*=omega0; dAqp[0,:]/=omega0
    dDqp[:,0]/=omega0; dDqp[:,0]/=omega0;
    
    if __has_scipy:
        # this seems to be more stable in borderline cases
        nC = sp.solve_continuous_are(-dAqp, np.zeros(dAqp.shape), dDqp, np.eye(dAqp.shape[-1]))            
    else:
        # solve "a' la MC thesis" using just numpy
        a, O = np.linalg.eig(dAqp) 
        O1 = np.linalg.inv(O)
        W = np.dot(np.dot(O1, dDqp),O1.T)
        for i in xrange(len(W)):
            for j in xrange(len(W)):
                W[i,j]/=a[i]+a[j]
        nC = np.dot(O,np.dot(W,O.T))
    nC[:,0]/=omega0;  nC[0,:]/=omega0
    return nC
    
def gleKernel(omega, Ap, Dp):
    """Given the Cp and Dp matrices for a harmonic oscillator of frequency omega_0, constructs the gle kernel for transformation of the velocity velocity autocorrelation function."""
    dw = abs(omega[1]-omega[0])
    ngrid = len(omega) 
    dKer = np.zeros((ngrid,ngrid), float)
    omlist = omega.copy()
    omlist[0] = max(omlist[0], dw*1e-2) # avoids a 0/0 instability
    om2list = omlist**2
    y = 0
    # outer loop over the physical frequency
    for omega_0 in omlist:
        # works in "scaled coordinates" to stabilize the machinery for small or large omegas
        dAqp = Aqp(omega_0, Ap)/omega_0
        dDqp = Dqp(omega_0, Dp)/omega_0
        dCqp = Cqp(omega_0, dAqp, dDqp)
        dAqp2 = np.dot(dAqp,dAqp)
        # diagonalizes dAqp2 to accelerate the evaluation further down in the inner loop
        w2, O = np.linalg.eig(dAqp2)
        w = np.sqrt(w2)
        O1 = np.linalg.inv(O)
        cqp1w = np.dot(dCqp[1,:],O) * w/omega_0 
        # re-scales by omega_0 to recover physical units
        cqpt1 = np.dot(O1,dCqp[:,1])
        x = 0
        om2om0 = om2list/omega_0**2 
        # keeps working in scaled coordinates at this point
        for oo0x in om2om0:        
            dKer[x,y] = np.real(np.dot(cqp1w, cqpt1/(w2+oo0x)))                         
            x+=1
        y += 1    
    return dKer*dw*2.0/np.pi

def ISRA(omega, ker, y, dparam):
    """Given the thermostatted vvac spectrum and the range of frequencies, constructs the vibration density of states"""
    delta_omega = abs(omega[1]-omega[0])
    steps = dparam[0]
    stride = dparam[1]
    ngrid = len(omega)
    f = y
    CT = ker.T
    CTC = np.dot(ker.T, ker)

    cnvg = np.zeros((steps,3))
    dvvac = np.zeros((int(steps/stride) + 1, len(f)))

    for i in range(steps):
        f = f * np.dot(CT, y) / np.dot(CTC, f)
        # Temporarty fix for NaNs
        #ii = np.argwhere(np.isnan(f))
        #f[ii] = f[ii+1]
        if(np.fmod(i,stride) == 0 and i != 0):
            dvvac[i/stride - 1] = f 
            cnvg[i/stride -1] = np.asarray((i, np.linalg.norm((np.dot(f,ker) - y))**2, np.linalg.norm(np.gradient(np.gradient(f)))**2))
        dvvac[i/stride - 1] = f 
        cnvg[i/stride -1] = np.asarray((i, np.linalg.norm((np.dot(f,ker) - y))**2, np.linalg.norm(np.gradient(np.gradient(f)))**2))
    return dvvac, cnvg

def unwind(path2iipi, path2ivvac, path2ker, oprefix, action, nrows, stride, dparam):
   
    # opens & parses the input file
    ifile = open(path2iipi,"r")
    xmlrestart = io_xml.xml_parse_file(ifile) # Parses the file.
    ifile.close()

    isimul = InputSimulation()
    isimul.parse(xmlrestart.fields[0][1])
    simul = isimul.fetch()

    ttype = str(type(simul.syslist[0].motion.thermostat).__name__)
    kbT = float(simul.syslist[0].ensemble.temp)

    # TODO: add i-pi units conversion
    if(ttype == "ThermoGLE"):
        Ap = simul.syslist[0].motion.thermostat.A  * 41.341373
        Cp = simul.syslist[0].motion.thermostat.C  / kbT
        print Cp
    elif(ttype == "ThermoLangevin"):
        Ap = np.asarray([1.0/simul.syslist[0].motion.thermostat.tau]).reshape((1,1)) * 41.341373
        Cp = np.asarray([1.0]).reshape((1,1))
    
    ivvac=input_vvac(path2ivvac, nrows, stride)
    ix=ivvac[:,0]
    iy=ivvac[:,1]

    dw = ix[1] - ix[0]
    if Ap[0,0] < 2.0 * dw:
        print "# WARNING: White-noise term is weaker than the spacing of the frequency grid. Will increase automatically to avoid instabilities in the numerical integration."
        Ap[0,0] = 2.0 * dw
    Dp = np.dot(Ap,Cp) + np.dot(Cp,Ap.T)

    # computes the vvac kernel
    if (path2ker == None):
        print "# computing the kernel."
        ker = gleKernel(ix, Ap, Dp)
        np.savetxt(oprefix + "-ker.data", ker)
    else:
        print "# importing the kernel."
        ker=np.loadtxt(path2ker)

    # (de-)convolutes the spectrum
    if(action == "conv"):
        print "# printing the output spectrum."
        output_vvac((ix, np.dot(iy,ker.T)), oprefix, input_vvac(path2ivvac, nrows, 1))
    elif(action == "deconv"):
        print "# deconvoluting the input spectrum."
        oy, ocnvg = ISRA(ix, ker, iy, dparam)
        output_vvac((ix, oy), oprefix, input_vvac(path2ivvac, nrows, 1))
        np.savetxt(oprefix + "-ISRA.data", ocnvg,  header="# step error Laplacian")

if __name__ == '__main__':
    # adds description of the program.
    parser=argparse.ArgumentParser(description="Given the parameters of a Generalized Langevin Equation and the vibrational density of states predicts the velocity-velocity autcorrelation obtained by the dynamics. Conversely, given the velocity-velocity autocorrelation function removes the disturbance affected by the thermostat and returns the underlying vibrational density of states. ")

    # adds arguments.
    parser.add_argument("-a","--action", nargs=1, choices=["conv","deconv"], default=None, help="choose conv if you want to obtain the response of the thermostat on the vibrational density of states; choose deconv if you want to obtain the micro-canonical density of states by removing the disturbance induced by the thermostat")
    parser.add_argument("-iipi", "--input_ipi", nargs=1, type=str, default=None, help="the relative path to the i-PI inputfile")
    parser.add_argument("-ivvac", "--input_vvac", nargs=1, type=str, default=None, help="the relative path to the input velocity-velocity autocorrelation function")
    parser.add_argument("-k", "--input_kernel", nargs=1, type=str, default=[None], help="the relative path to the kernel function")
    parser.add_argument("-mrows", "--maximum_rows", nargs=1, type=int, default=[-1], help="the index of the last row to be imported from INPUT_VVAC")
    parser.add_argument("-s", "--stride", nargs=1, type=int, default=[1], help="the stride for importing the IVVAC and computing the kernel")
    parser.add_argument("-dparam", "--deconv_parameters", nargs=1, type=int, default=[500,10], help="the parameters associated with the deconvolution. Since the operation is based on an iterative algorithm, it requires the total number of epochs NEPOCHS and the stride PSTRIDE at which the output spectrum is returned. Usage: [NEPOCHS,PSTRIDE]")
    parser.add_argument("-oprefix", "--output_prefix", nargs=1, type=str, default=["output.data"], help="the prefix of the (various) output files.")

    # parses arguments.
    if( len(sys.argv) > 1):
        args=parser.parse_args()
    else:
        parser.print_help()
        sys.exit()

    # stores the arguments
    path2iipi = str(args.input_ipi[-1])
    path2ivvac = str(args.input_vvac[-1])
    path2ker = args.input_kernel[-1]
    oprefix = str(args.output_prefix[-1])
    action = str(args.action[-1])
    nrows = int(args.maximum_rows[-1])
    stride = int(args.stride[-1])
    dparam = np.asarray(args.deconv_parameters, dtype=int)

    unwind(path2iipi, path2ivvac, path2ker, oprefix, action, nrows, stride, dparam)
