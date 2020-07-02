# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.


#changed init subroutine ...

import time
import numpy as np
from numpy import linalg as LA
import os.path,os,glob
import sys
import random
from netCDF4 import Dataset
#from netcdf import Dataset


#from ipi.utils.depend import dstrip
from ipi.engine.motion import Motion
from ipi.utils.softexit import softexit
from ipi.utils.depend import *
from ipi.engine.thermostats import ThermoLangevin
from ipi.engine.motion.dynamics import *

from pymd.ebath  import *
from pymd.phbath  import *
from pymd.spectrum  import *
from pymd.functions  import *
from pymd.iout import Write2NetCDFFile,ReadNetCDFVar

__all__ = ['Displace']

class Constants(object):
    """
    unit conversion between ipy and jt's md units
    The most tedious part of the code is to do the unit conversion between jt's
    md code and ipy code.
    
    1. JT's units used in pymd:
        length:  0.06466 Ang
        time:   0.6582e-15 second
        energy: eV
        hbar:  1
        mass:  atomic mass

    2. ipy's unit:
        length: Bohr
        time: 2.4188843e-17 second
        energy: Hartree
        mass: electron mass

    """
    #unit conversion
    bohr2ang = 0.529177208
    ang2bohr = 1.0/bohr2ang
    #JT md Length unit in terms of angstrom
    jtmdlen = 0.06466
    #from ipy to jt unit
    lconv = bohr2ang/jtmdlen

    #from atomic mass to electron mass
    amass2emass = 1822.88848149
    emass2amass = 1.0/amass2emass
    eV2hartree = 0.0367493
    hartree2eV = 1.0/eV2hartree
    #1 Hartree/Bohr = 51.422 eV/ang
    hartreebohr2eVang = 51.42208619083232
    eVang2hartreebohr = 1.0/hartreebohr2eVang

    #conversion from i-pi t to pymd t:
    tconv=2.4188843e-17/0.658211814201041e-15

class Displace(Motion):
    """Calculator object that just displaces all non-fixed atoms by a specific amount for each step

    Attributes:
        dv: the rigid displacement vector

    Depend objects:
        None really meaningful.

    Todo:

    """

    def __init__(self, timestep=1.0, nmd=1024, temp=4.2, vbias=0.0, eta=0.0, \
                 memlen=100,nc=1,berry=1,renorm=1,thermostat=None,\
                 fixcom=False, fixatoms=None, displacement=None):
        """
        Initialises Replay.
        Args:
           temp: The system temperature.
           fixcom: An optional boolean which decides whether the centre of mass
              motion will be constrained or not. Defaults to False.
           displacement: displacement vector of coordinates
        """
        super(Displace, self).__init__(fixcom=fixcom, fixatoms=fixatoms)
        #JT#time step for the dynamics
        dself = dd(self)
        dself.dt = depend_value(name='dt', value=timestep)
        dself.nmd = depend_value(name='nmd', value=nmd)
        dself.temp = depend_value(name='temp', value=temp)
        dself.vbias = depend_value(name='vbias', value=vbias)
        dself.eta= depend_value(name='eta', value=eta)
        dself.memlen= depend_value(name='memlen', value=memlen)
        dself.nc = depend_value(name='nc', value=nc)
        dself.berry = depend_value(name='berry', value=berry)
        dself.renorm = depend_value(name='renorm', value=renorm)


        print 'self.eta', self.eta
        print 'self.memlen', self.memlen


        self.debyefactor = 2.0

        self.phbath = False
        self.ebath = False



        print "time step (unit: 2.4188843e-17 s):"
        print self.dt

        
        if displacement is None:
            raise ValueError("Must provide a displacement vector")
        self.displacement = displacement
        if self.displacement.size != 3:
            raise ValueError("Displacement vector must be 3 values")

        #JT#our Langevin integrator 
        self.integrator = GLEIntegrator()

        #counter of md steps
        self.mdstep = 0
        #This should be readin from ipy input.xml
     
        #--------------------------------------------------------------
        #reading in from nc file
        #--------------------------------------------------------------
        phfn="PHSigma.nc"
        ephfn="EPH.nc"
        #AuBZ structure
        ephfn2="EPH_AuBZ.nc"
        ephfn2="EPH-Au.nc"
        if os.path.isfile(phfn):
                 print "Reading ", phfn
                 phs=ReadPHSCFile(phfn)
                 #initial xyz coordinates from netcdf
                 self.XYZEq = phs.XYZEq
                 #flattened XYZ
                 self.XYZEqf = phs.XYZEq.flatten()
                 self.lenall= len(self.XYZEqf)
                 self.hw = phs.hw
                 self.U = phs.U
                 self.DynMat = phs.DynMat
                 self.dynatoms = phs.DynamicAtoms        
        
                 self.SigL=phs.SigL
                 self.SigR=phs.SigR
                 self.gwl=phs.wl        
                 phcatsl = phs.lcats
                 phcatsr = phs.rcats
                 self.fixatoms = phs.fixatoms
                 #masses of dynamical atoms
                 self.DynMasses=phs.DynamicMasses
                 self.phbath = True
                 
        if os.path.isfile(ephfn2):
                 eph=ReadEPHNCFile(ephfn2)

                 if self.phbath :
                 #ToDo:
                 #CHECK information in the two files are consistent!
                     pass

                 #initial xyz coordinates from netcdf
                 self.XYZEq = eph.XYZEq
                 #flattened XYZ
                 self.XYZEqf = eph.XYZEq.flatten()
                 self.lenall= len(self.XYZEqf)
                 self.hw = eph.hw
                 self.U = eph.U
                 self.DynMat = eph.DynMat
                 self.dynatoms = eph.DynamicAtoms        
                 self.DynMasses=eph.DynamicMasses
        
                 self.SigL=eph.SigL
                 self.SigR=eph.SigR
                 self.gwl=eph.wl        

                 #indices of atoms that connect to the bath in python index
                 #(starting from 0) 
                 ecats=[i+18 for i in range(13)]
                 phcatsl=[i+18 for i in range(5)] 
                 phcatsr=[i+26 for i in range(5)] 
                 #atoms that are fixed
                 leftfixed = np.arange(18)
                 rightfixed = np.arange(31,58)
                 self.fixatoms=np.concatenate((leftfixed,rightfixed), axis=0)
                 self.ebath = True
                 self.phbath = True

                 self.constraint = eph.constraint

        #AuBZ
        if os.path.isfile(ephfn):
                 eph=ReadEPHNCFile(ephfn)
                 #-----------------------------------------------
                 #-----------------------------------------------
                 #-----------------------------------------------
                 #-----------------------------------------------
                 #-----------------------------------------------
                 #-----------------------------------------------
                 #add artifical damping
                 #print eph.efric.diagonal()
                 print "adding art. damping"
                 for i in range(eph.efric.shape[0]):
                     eph.efric[i,i] = eph.efric[i,i]+5.0e-4
                 #-----------------------------------------------
                 #-----------------------------------------------
                 #-----------------------------------------------
                 #-----------------------------------------------
                 #-----------------------------------------------
                 print eph.efric.diagonal()
                 #stoppp

                 if self.phbath :
                 #ToDo:
                 #CHECK information in the two files are consistent!
                     pass

                 #initial xyz coordinates from netcdf
                 self.XYZEq = eph.XYZEq
                 #flattened XYZ
                 self.XYZEqf = eph.XYZEq.flatten()
                 self.lenall= len(self.XYZEqf)
                 self.hw = eph.hw
                 self.U = eph.U
                 self.DynMat = eph.DynMat
                 self.dynatoms = eph.DynamicAtoms
                 self.DynMasses=eph.DynamicMasses
        
                 self.SigL=eph.SigL
                 self.SigR=eph.SigR
                 self.gwl=eph.wl

                 #manual settings:
                 #indices of atoms that connect to the bath in python index
                 #(starting from 0) 
                 #ecats=[i+32 for i in range(20)]
                 ecats=eph.ecats
                 print "ecats",ecats
                 #phcatsl=[i+32 for i in range(5)] 
                 phcatsl=eph.lcats
                 print "lcats:",phcatsl
                 #phcatsr=[i+32+15 for i in range(5)] 
                 phcatsr = eph.rcats
                 print "rcats:",phcatsr
                 #atoms that are fixed
                 #leftfixed = np.arange(2*16)
                 #rightfixed = np.arange(2*16+20,2*16+20+3*16)
                 #self.fixatoms=np.concatenate((leftfixed,rightfixed), axis=0)
                 self.fixatoms=eph.FixedAtoms
                 #a vector rotation along which is forbidden
                 self.constraint = eph.constraint

                 self.ebath = True
                 self.phbath = True




        self.dynmass3 = np.array([[a,a,a] for a in self.DynMasses]).flatten()
        self.dynindex = np.array([[a*3,a*3+1,a*3+2] for a in\
                                  self.dynatoms],dtype=int).flatten()


        self.nph = len(self.dynindex)     
        #self.ml = 200
        self.debye = max(self.gwl)

        v,vv=LA.eigh(self.DynMat)
        print "hw from DynMat", np.sqrt(v)
        print "hw from NetCDF", self.hw
        print 'displace:fixatoms', self.fixatoms
        #print self.dynatoms
        #print self.hw
        #print phcatsl
        #print phcatsr
        #print self.dynmass3
        #stoppp


   
        self.thermolist=[]

        if self.ebath:
            #Electron bath
            self.elbath = ThermoQLE(temp=self.temp,dt=self.dt,ethermo=0.0,\
                        cats=ecats,nmd=self.nmd,wmax=1.0,nw=500,bias=self.vbias,\
                        efric=eph.efric,exim=eph.xim*self.nc,exip=eph.xip,\
                        zeta1=eph.zeta1*self.renorm,zeta2=eph.zeta2*self.berry,\
                        mdstep=self.mdstep,xyzeqf=self.XYZEqf,ml=self.memlen)

            self.thermolist.append(self.elbath)
        if self.phbath:
            #Phonon bath
            #Left Phonon bath  
            #cats: python index of system atoms that are connecting to the left phonon bath
            #eta_ad: add artificial damping to the system, useful when
            #there are sharp peaks in the imaginary of the self-energy (bath
            #DOS). It is like the small imaginary part of the frequency in the
            #Green's functions.
            cats=phcatsl 
            print 'phcatsl', phcatsl
            nwph = 3000  #number of energy/frequency points within [0,self.debye*self.debyefactor]

            self.phononbath_l = ThermoQLE_ph(temp=self.temp, dt=self.dt,\
                        ethermo=0.0, cats=cats, nmd=self.nmd, nw_ph=nwph,\
                        debye=self.debye,ml=self.memlen,mcof=self.debyefactor,\
                        sig=self.SigL,gwl=self.gwl,eta_ad=self.eta, mdstep=self.mdstep)   
            cats=phcatsr
            print 'phcatsr', phcatsr
            self.phononbath_r = ThermoQLE_ph(temp=self.temp, dt=self.dt,\
                        ethermo=0.0, cats=cats, nmd=self.nmd, nw_ph=nwph,\
                        debye=self.debye,ml=self.memlen,mcof=self.debyefactor,\
                        sig=self.SigR,gwl=self.gwl,eta_ad=self.eta, mdstep=self.mdstep)
            self.thermolist.append(self.phononbath_l)
            self.thermolist.append(self.phononbath_r)


        #list of thermostats that connect to the system
        #self.thermolist=[self.elbath]#,self.phononbath_l,self.phononbath_r]
        #self.thermolist=[self.phononbath_l,self.phononbath_r]

        #self.thermolist=[]
        self.thermostat = MultiThermo(temp=self.temp,dt=self.dt,\
                                ethermo=0.0,thermolist=self.thermolist)

                                                 
        #self.cats=ecats
        self.trajectories=np.zeros((self.nmd,self.nph))
        self.energy = np.zeros(self.nmd)
        self.t = 0




    def bind(self, ens, beads, nm, cell, bforce, prng):
        super(Displace, self).bind(ens, beads, nm, cell, bforce, prng)
        # Binds integrators
        self.integrator.bind(self)
        # depending on the kind, the thermostat might work in the normal mode or the bead representation.
        fixdof = len(self.fixatoms) * 3 * self.beads.nbeads
        if self.fixcom:
            fixdof += 3
        dself = dd(self)
        # first makes sure that the thermostat has the correct temperature, then proceed with binding it.
        #dpipe(dself.dt, dd(self.thermostat).dt)
        self.thermostat.bind(beads=self.beads, nm=self.nm, prng=prng, fixdof=fixdof)

    def initialipyqp(self):
        """
        initialize the initial coordinates and momenta of the system
        according to given temperature, and eigen phonon energies.
        self.hw                 list of eigen energies
        self.U                  the corresponding eigen vectors
        self.TT                 temperature
        """
        ##########################################################################
        #print "initialise the velocity and displacment"
        dis,vel = initialise(self.hw,self.U,self.temp)

        #q0 is the initial coordinates (ipy unit)
        ipyddis0 = dis/self.dynmass3**0.5/Constants.lconv
        #print "dis(Bohr):",ipyddis0

        ipydis = np.zeros(len(self.XYZEqf))
        for ind,val in enumerate(self.dynindex):
            ipydis[val] = ipyddis0[ind]
        self.q0 = self.XYZEqf/Constants.bohr2ang + ipydis
    
        #p0 is the initial momentum (ipy unit)
        self.p0 =  np.zeros(len(self.XYZEqf))
        ipydvel0 = vel*self.dynmass3**0.5*Constants.amass2emass/Constants.lconv*Constants.tconv
        for ind,val in enumerate(self.dynindex):
            self.p0[val] = ipydvel0[ind]


    def dump(self,ipie,id,ip):
        """
        dump md information
        """
        outfile="MD"+str(id)+"-"+str(ip)+".nc"
        NCfile = Dataset(outfile,'w','Created '+time.ctime(time.time()))
        NCfile.title = 'Output from ipi run'
        NCfile.createDimension('nph',self.nph)
        NCfile.createDimension('nwl',len(self.gwl))
        NCfile.createDimension('nall',self.lenall)
        NCfile.createDimension('one',1)
        NCfile.createDimension('two',2)
        NCfile.createDimension('mem',self.memlen)
        NCfile.createDimension('nmd',self.nmd)
        NCfile.createDimension('nnmd',None)
        for i in range(len(self.thermolist)):
            NCfile.createDimension('n'+str(i),self.thermolist[i].nndof)     
            (a,b,c)=self.thermolist[i].sig.shape
            NCfile.createDimension('nw'+str(i),a)  
            NCfile.createDimension('ndd'+str(i),b)

        # #save all the histories of p,q or not
        # if self.savepq:
        #     Write2NetCDFFile(NCfile,self.ps,'ps',('nnmd','nph',),units='')
        #     Write2NetCDFFile(NCfile,self.qs,'qs',('nnmd','nph',),units='')
        #power spectrum
        Write2NetCDFFile(NCfile,self.power,'power',('nnmd','two',),units='')
        #energy
        Write2NetCDFFile(NCfile,self.etot,'energy',('nnmd',),units='')
        #velocity
        Write2NetCDFFile(NCfile, self.beads.p[0][:],'p',('nall',),units='ipi')
        #Write2NetCDFFile(NCfile,reducedp(self.dynindex,(self.beads.p[0][:])),'p',('nph',),units='ipi')
        #last trajectory
        Write2NetCDFFile(NCfile, self.beads.q[0][:],'q',('nall',),units='ipi')
        #equilibrium force
        for i in range(self.beads.nbeads):
          Write2NetCDFFile(NCfile,self.integrator.eqforce[i],'eqforce',('nall',),units='ipi')        
        #Write2NetCDFFile(NCfile,reducedp(self.dynindex,(self.beads.q[0][:])),'q',('nph',),units='ipi')
        #current time
        Write2NetCDFFile(NCfile,self.t,'t',('one',),units='ipi time')
        #current mdstep
        Write2NetCDFFile(NCfile,self.mdstep,'mdstep',('one',),units='')
        #current segment
        Write2NetCDFFile(NCfile,ipie,'ipie',('one',),units='')
        if len(self.thermolist) > 0:
            # #memory kernel for p
            Write2NetCDFFile(NCfile,reducedpm(self.dynindex,self.thermolist[-1].phis0),'phis',('mem','nph',),units='')
        # #memory kernel for q
        # Write2NetCDFFile(NCfile,self.qhis,'qhis',('mem','nph',),units='')
        #noise series
        for i in range(len(self.thermolist)):
            for j in range(self.beads.nbeads):
                 Write2NetCDFFile(NCfile,self.thermolist[i].noise[j],'noise'+str(i),\
                          ('nnmd','n'+str(i),),units='pymd units')
        #friction series (for ebath only dummy)
        for i in range(len(self.thermolist)):
                 Write2NetCDFFile(NCfile,self.thermolist[i].phfric,'fric'+str(i),\
                          ('mem','n'+str(i),'n'+str(i)),units='pymd units')                          
        #Write2NetCDFFile(NCfile,self.fhis[i],'fhis'+str(i),('nnmd','nph',),units='')
        #Trajectories in pymd units
        Write2NetCDFFile(NCfile,self.trajectories,'traj',('nnmd','nph',),units='pymd units')


        #md self-energies
        for i in range(len(self.thermolist)):
                 Write2NetCDFFile(NCfile,self.thermolist[i].sig.real,'ReSig'+str(i),\
                          ('nw'+str(i),'ndd'+str(i),'ndd'+str(i)),units='eV^2')                          
                 Write2NetCDFFile(NCfile,self.thermolist[i].sig.imag,'ImSig'+str(i),\
                          ('nw'+str(i),'ndd'+str(i),'ndd'+str(i)),units='eV^2')                          
        #Dynamical Matrix
        Write2NetCDFFile(NCfile,self.DynMat,'DynMat',('nph','nph'),units='eV^2')
        #energy grid
        Write2NetCDFFile(NCfile,self.gwl,'Wlist',('nwl',),units='eV')
        print 'Finished writing.'
        NCfile.close()
        
    def step(self, step=None):
        """Step the displacement."""
        """Now use the integrator"""
        if self.mdstep is None:
            softexit.trigger("Unknown step count. Exiting simulation")
            return
        else:
            print "current step:"+str(self.mdstep)

        #initialize system
        nrrun=0
        #initialize q and p
        if self.mdstep == 0:
          fn="MD"+str(nrrun)+".nc"
          #fnm="MD"+str(j-1)+".nc"
          if os.path.isfile(fn):
                print "found file: "+fn+"\n"
                #ipie = int(ReadNetCDFVar(fn,'ipie')[0])
                self.mdstep = int(ReadNetCDFVar(fn,'mdstep')[0])
                print 'restarting run at'
                print 'mdstep', self.mdstep, 'from file'
                #if(ipie+1 < self.npie):
                print "reading resume information"
                p0 = ReadNetCDFVar(fn,'p')
                q0 = ReadNetCDFVar(fn,'q')
                #self.beads.q[0][:] = enlargep(len(self.XYZEqf),self.dynindex,q0) 
                #self.beads.p[0][:] = enlargep(len(self.XYZEqf),self.dynindex,p0)
                self.beads.q[0][:] = q0 
                self.beads.p[0][:] = p0 
                #    self.t = ReadNetCDFVar(fn,'t')[0]
                #    self.qhis = ReadNetCDFVar(fn,'qhis')
                #equilibrium force          
                for i in range(self.beads.nbeads): 
                        self.integrator.eqforce[i]=ReadNetCDFVar(fn,'eqforce') 
                self.phis = ReadNetCDFVar(fn,'phis')
                #enlarge to full system size
                for i in range(len(self.thermolist)):
                   self.thermolist[i].phis0 =  enlargepm(self.memlen,len(self.XYZEqf),self.dynindex,self.phis)
                self.power =ReadNetCDFVar(fn,'power')
                self.etot = ReadNetCDFVar(fn,'energy')
                self.trajectories = ReadNetCDFVar(fn,'traj')
                #    if self.savepq:
                #        self.qs = ReadNetCDFVar(fn,'qs')
                #        self.ps = ReadNetCDFVar(fn,'ps')
                #noise series
                for i in range(len(self.thermolist)):
                    for j in range(self.beads.nbeads):
                        self.thermolist[i].noise[j]=ReadNetCDFVar(fn,'noise'+str(i)) 
                        self.thermolist[i].hasnoise = 1  
                #friction series (for ebath only dummy)
                for i in range(len(self.thermolist)):
                        self.thermolist[i].phfric=ReadNetCDFVar(fn,'fric'+str(i))
                                          
          else:
             #force at equiblirum positions 
             self.integrator.equforce()
             print "the force at equilibrium positions:"
             print self.integrator.eqforce.shape
             print "initializing p and q"
             for i in range(self.beads.nbeads):
                self.initialipyqp()
                self.beads.q[i] = self.q0
                self.beads.p[i] = self.p0

        print 'step',self.mdstep
        self.integrator.step(self.mdstep)
        #time.sleep(120)

        #saving trajectories, Jing-Tao version
        #only saved the first bead??
        self.trajectories[self.mdstep,:]=reducedp(self.dynindex,\
                        (self.beads.q[0][:]*Constants.bohr2ang-self.XYZEqf)\
                                /Constants.jtmdlen)*self.dynmass3**0.5


        print 'Storing trajectory'
        with open("traj.ANI","a") as Tr:
          Tr.write("%i  \n"%len(self.XYZEq))
          Tr.write("\n")
          for i in range(0,self.beads.q.shape[1],3):
            Tr.write("%s %f %f %f \n"%(self.beads.names[i/3],Constants.bohr2ang*self.beads.q[0][i],\
                                       Constants.bohr2ang*self.beads.q[0][i+1],\
                                       Constants.bohr2ang*self.beads.q[0][i+2])) 
        
        #Harmonic energy of the system
        p0=reducedp(self.dynindex,self.beads.p[0])
        energy0 = 0.5*np.dot(p0/self.dynmass3/Constants.amass2emass,p0)/Constants.eV2hartree
        print "kinetic energy: ",energy0
        q0=reducedp(self.dynindex,\
                (self.beads.q[0][:]*Constants.bohr2ang-self.XYZEqf)\
                    /Constants.jtmdlen)*self.dynmass3**0.5
        energy1 = 0.5*np.dot(q0,np.dot(self.DynMat,q0))
        print "potential energy: ",energy1
        print "total: ", energy0+energy1
        self.etot = energy0+energy1
        print 'Storing energy (eV)'
        with open("energy.dat","a") as Tr2:
            Tr2.write("%f     %f\n"%(energy0,energy1))

        #dumping into nc file
        self.nrep = 8 #how many total runs
        self.idd = 0  #current run
        if self.mdstep==self.nmd:
            self.idd += 1
        modint = self.nmd/64
        if self.mdstep%modint == 0 and self.mdstep !=0:
            #writing out the power spectrum
            f = open("power"+str(self.idd)+"-"+str(self.mdstep/modint)+".dat","w")
            self.power = powerspec(self.trajectories,self.dt*Constants.tconv,self.nmd) #self.dt*self.au2fs
            for i in range(len(self.power)):
                    #max output frequency set to 2*max(hw)
                    if(self.power[i,0] < np.max(self.hw)*2.0):
                       f.write("%f     %f \n"%(self.power[i,0],self.power[i,1]))             
	    f.close()
            #dump current state
            self.dump(self.nrep,self.idd,self.mdstep/modint)


        #counter
        self.t+=self.mdstep*self.dt
        self.mdstep += 1
        print "end of step", self.mdstep
        
class DummyIntegrator(dobject):
    """ No-op integrator for (PI)MD """
    def __init__(self):
        pass

    def bind(self, motion):
        """ Reference all the variables for simpler access."""

        self.beads = motion.beads
        self.bias = motion.ensemble.bias
        self.ensemble = motion.ensemble
        self.forces = motion.forces
        self.prng = motion.prng
        self.nm = motion.nm
        self.thermostat = motion.thermostat
        self.fixcom = motion.fixcom
        self.fixatoms = motion.fixatoms
        self.dynindex = motion.dynindex
        self.constraint = motion.constraint
        dself = dd(self)
        dself.dt = dd(motion).dt




    def pstep(self):
        """Dummy momenta propagator which does nothing."""
        pass

    def qcstep(self):
        """Dummy centroid position propagator which does nothing."""
        pass

    def step(self, step=None):
        """Dummy simulation time step which does nothing."""
        pass

    def pconstraints(self):
        """Dummy centroid momentum step which does nothing."""
        pass



class GLEIntegrator(DummyIntegrator):
    """ 
    Integrator object for Langevin molecular dynamics with electrical current.
    25Apr2018 - created by Jingtao Lu to implement the GLE with current within
    ipi.


    Attributes:
        ptime: The time taken in updating the velocities.
        qtime: The time taken in updating the positions.
        ttime: The time taken in applying the thermostat steps.

    Depend objects:
    """

    def bind(self,motion):
        self.beads = motion.beads
        self.bias = motion.ensemble.bias
        self.ensemble = motion.ensemble
        self.forces = motion.forces
        self.prng = motion.prng
        self.nm = motion.nm
        self.thermostat = motion.thermostat
        self.fixcom = motion.fixcom
        self.fixatoms = motion.fixatoms
        self.dynindex = motion.dynindex
        self.constraint= motion.constraint
        dself = dd(self)
        dself.dt = dd(motion).dt

        #flag to reset the DM file
        self.reset = 0


	
        #self.mdstep=step

        #the projector projects out forces acting on
        #fix atoms
        tlen = len(self.beads.p.flatten())
        plen = tlen/self.beads.nbeads
        self.eqforce = np.zeros((self.beads.nbeads,tlen))
        
        #print 'eq len',self.eqforce.shape 
        self.projector = np.ones((self.beads.nbeads,plen))
        #project out the fixed atoms firstly
        for i in self.fixatoms:
            for j in range(3):
                self.projector[:,i*3+j]=0.0
        print "force projector:"
        print self.projector

        #self.projector = np.zeros((self.beads.nbeads,plen))
        #for j in self.dynindex:
        #    self.projector[:,j] = 1.0
        #print self.projector

    def equforce(self):
        #the oldforce save at n-th step is used for the n+1-th step
        #self.eqforce= dstrip(self.forces.f)
        #print self.beads.q
        self.eqforce= dstrip(self.forces.f)*1.0
        print "force at equilibrium positions:"
        print self.eqforce

        
    def pstep(self):
        """Velocity Verlet momenta propagator."""
        #print self.forces.f
        #self.beads.p += dstrip(self.forces.f) * (self.dt * 0.5)
        #project out the fixed atoms
        print "project out fixed atoms\n"
        self.tmpf = ((dstrip(self.forces.f)-self.eqforce) * self.projector)

	#set a threshold for the force
	#if the force ia larger than the threshold, reset to zero
	#We need this because sometimes siesta gives a huge force that lead to instability,
	#This should be a bug of siesta
        maxf = np.amax(np.absolute(self.tmpf))
	print "maximum value of the force:", maxf


	if  maxf > 1.0: 
            print "####################################"
            print "####################################"
	    print "#WARNING: find huge force, reset the DM file"
            print "####################################"
            print "####################################"
	    #self.tmpf = self.tmpf*0.0
	    #use a random force near equilibrium
            self.tmpf = self.eqforce * self.projector * random.uniform(-1,1)
            #rm the DM file and start from scratch
	    fns = glob.glob("./*.DM")
            for file in fns:
                print "removing: ", file
                os.remove(file)


	#min dist
        #project out other constraints one by one
        #print "project out the constraint (momentum)\n"
        #tmpp = self.tmpf*0.0
        ##loop over constraint
        #for vec in self.constraint:
        #    #loop over beads
        #    for i in range(self.beads.nbeads):
        #        dott = np.dot(self.tmpf[i],vec)
        #        tmpp[i] +=dott*vec
        #self.tmpf -= tmpp
        self.beads.p +=  self.tmpf * (self.dt * 0.5)


    def qstep(self):
        """Velocity Verlet displacement propagator."""
        #print self.forces.f
        #self.beads.p has the updated (to half time step) momenta now
        #self.beads.q += dstrip(self.beads.p) * self.dt /dstrip(self.beads.m3).flatten()

	#project out the constraints given by self.constraint
        qtmp = dstrip(self.beads.p) * self.dt /dstrip(self.beads.m3).flatten()
        print "project out the constraint (disp)\n"
        qtmp2 = qtmp*0.0
        print "shape of qtmp:",qtmp.shape
        for vec in self.constraint:
            for i in range(self.beads.nbeads):
                #vecq=vec/dstrip(self.beads.m3[i])
                #vecq = vecq/np.sqrt(np.dot(vecq,vecq))
                vecq = vec
	        dott = np.dot(vecq,qtmp[i])
                print "dott:",dott
		#print "m3:",self.beads.m3[i]
		#print "dott:",dott
		#print "dot*vec:",dott*vec
		#stoppp
                qtmp2[i] +=dott*vecq
        #print "qtmp:",qtmp
        #print "qtmp2:",qtmp2
        #self.tmpf == 0 means we got a huge force (see psetp())
        #keep the displacement unchanged in this case
	#if np.any(self.tmpf):
        self.beads.q += qtmp-qtmp2

    def step(self, step=None):
        """
        Does one simulation time step.
        step counter of md current step
        """
        #does the order of pstep matters?
        #thermostat first or integrator first?

        #perform half step of the momenta propagation 
        self.ptime = -time.time()
        self.thermostat.pstep(step,id=0)
        self.pstep()
        self.pconstraints()
        self.ptime += time.time()


        #propagate q using self.beads.p 
        #at this point, it stores momenta 
        #that has been propagated for half time step
        self.qtime = -time.time()
        self.qstep()

        #perform the 2nd half step of the momenta propagation 
        self.ptime -= time.time()
        self.thermostat.pstep(step,id=1)
        self.pstep()
        self.pconstraints()
        self.ptime += time.time()
        print "done", step
        #self.mdstep=step




class Thermostat(dobject):
    """Base thermostat class.

    Gives the standard methods and attributes needed in all the thermostat
    classes.

    Attributes:
       prng: A pseudo random number generator object.
       ndof: The number of degrees of freedom that the thermostat will be
          attached to.

    Depend objects:
       dt: The time step used in the algorithms. Depends on the simulation dt.
       temp: The simulation temperature. Higher than the system temperature by
          a factor of the number of beads. Depends on the simulation temp.
       ethermo: The total energy exchanged with the bath due to the thermostat.
       p: The momentum vector that the thermostat is coupled to. Depends on the
          beads p object.
       m: The mass vector associated with p. Depends on the beads m object.
       sm: The square root of the mass vector.
    """

    def __init__(self, temp=1.0, dt=1.0, ethermo=0.0):
        """Initialises Thermostat.

        Args:
           temp: The simulation temperature. Defaults to 1.0.
           dt: The simulation time step. Defaults to 1.0.
           ethermo: The initial heat energy transferred to the bath.
              Defaults to 0.0. Will be non-zero if the thermostat is
              initialised from a checkpoint file.
        """

        dself = dd(self)
        dself.temp = depend_value(name='temp', value=temp)
        dself.dt = depend_value(name='dt', value=dt)
        dself.ethermo = depend_value(name='ethermo', value=ethermo)

    def bind(self, beads=None, atoms=None, pm=None, nm=None, prng=None, fixdof=None):
        """Binds the appropriate degrees of freedom to the thermostat.

        This takes an object with degrees of freedom, and makes their momentum
        and mass vectors members of the thermostat. It also then creates the
        objects that will hold the data needed in the thermostat algorithms
        and the dependency network.

        Args:
           beads: An optional beads object to take the mass and momentum vectors
              from.
           atoms: An optional atoms object to take the mass and momentum vectors
              from.
           pm: An optional tuple containing a single momentum value and its
              conjugate mass.
           prng: An optional pseudo random number generator object. Defaults to
              Random().
           fixdof: An optional integer which can specify the number of constraints
              applied to the system. Defaults to zero.

        Raises:
           TypeError: Raised if no appropriate degree of freedom or object
              containing a momentum vector is specified for
              the thermostat to couple to.
        """

        if prng is None:
            warning("Initializing thermostat from standard random PRNG", verbosity.medium)
            self.prng = Random()
        else:
            self.prng = prng

        dself = dd(self)
        if not beads is None:
            dself.p = beads.p.flatten()
            dself.m = beads.m3.flatten()
        elif not atoms is None:
            dself.p = dd(atoms).p
            dself.m = dd(atoms).m3
        elif not pm is None:
            dself.p = pm[0].flatten()  # MR this should allow to simply pass the cell momenta in the anisotropic barostat
            dself.m = pm[1].flatten()
        else:
            raise TypeError("Thermostat.bind expects either Beads, Atoms, NormalModes, or a (p,m) tuple to bind to")

        if fixdof is None:
            self.ndof = len(self.p)
        else:
            self.ndof = float(len(self.p) - fixdof)

        dself.sm = depend_array(name="sm", value=np.zeros(len(dself.m)),
                                func=self.get_sm, dependencies=[dself.m])

    def get_sm(self):
        """Retrieves the square root of the mass matrix.

        Returns:
           A vector of the square root of the mass matrix with one value for
           each degree of freedom.
        """

        return np.sqrt(self.m)

    def step(self):
        """Dummy thermostat step."""

        pass


        
        

class ThermoQLE(Thermostat):
    """
    Represents an electron thermostat.

    Attributes:
           cats: indices of atoms that connect to the thermal bath (python
           index)
           temp: The simulation temperature. Defaults to 1.0.
           dt: The simulation time step. Defaults to 1.0.
           timesteps: The number of time steps to do the simulation.
           gamma: The single friction coefficient.
           ethermo: The initial heat energy transferred to the bath.
              Defaults to 0.0. Will be non-zero if the thermostat is
              initialised from a checkpoint file.

           xyzeqf: flattened equilibrium coordinates of the atoms in angstrom

    Depend objects:


    TODO:
    1. how to visit number of beads in the __init__? self.beads.nbeads
    """

    def __init__(self, temp=1.0, dt=1.0, ethermo=0.0, cats=None, nmd=2**10, wmax=None, nw=None, bias=0.,\
                 efric=None,exim=None,exip=None,zeta1=None,zeta2=None,classical=False,zpmotion=True,\
                 mdstep=None,xyzeqf=None, ml=None):
        super(ThermoQLE, self).__init__(temp,dt,ethermo)
        dself = dd(self)

        #electron bath parameters
        self.cats = np.array(cats,dtype='int')
        #number of md steps
        self.nmd = nmd
        self.wmax=wmax
        self.nw=nw
        self.bias=bias
        self.efric=efric
        #print 'efric', len(efric), efric.shape
        self.exim=exim
        self.exip=exip
        self.zeta1=zeta1
        self.zeta2=zeta2
        self.classical=classical
        self.zpmotion=zpmotion
        #number of degrees of freedom attached to the bath in a single bead
        self.nndof = len(cats)*3
        #counter
        self.counter = 0
        self.hasnoise = 0
        self.hasfriction = 0        
        
        self.temp = temp
        self.dt = dt
        self.kernel = None
        self.local = False
        self.wl = [self.wmax*i/nw for i in range(nw)]
        self.cur = np.zeros(nmd)     
        self.mdstep=mdstep
        self.XYZEqf=xyzeqf
        self.ml = ml
     

         

        print "thermostat parameters:\n"
        print "number of md steps: "+str(self.nmd)




    def bind(self, beads=None, atoms=None, pm=None, nm=None, prng=None, fixdof=None):
        """Binds the appropriate degrees of freedom to the thermostat.

        This takes an object with degrees of freedom, and makes their momentum
        and mass vectors members of the thermostat. It also then creates the
        objects that will hold the data needed in the thermostat algorithms
        and the dependency network.

        Args:
           beads: An optional beads object to take the mass and momentum vectors
              from.
           atoms: An optional atoms object to take the mass and momentum vectors
              from.
           pm: An optional tuple containing a single momentum value and its
              conjugate mass.
           prng: An optional pseudo random number generator object. Defaults to
              Random().
           fixdof: An optional integer which can specify the number of constraints
              applied to the system. Defaults to zero.

        Raises:
           TypeError: Raised if no appropriate degree of freedom or object
              containing a momentum vector is specified for
              the thermostat to couple to.
        """
        super(ThermoQLE, self).bind(beads=beads, atoms=atoms, pm=pm, prng=prng, fixdof=fixdof)
        dself=dd(self)
        dself.beads=beads
        if not beads is None:
            #the displacement
            dself.q = beads.q.flatten()
        else:
            raise TypeError("ThermoQLE.bind expects Beads to bind to")
       
        self.noise=np.zeros((self.beads.nbeads,self.nmd,self.nndof))
        self.phfric=np.zeros((self.ml,self.nndof,self.nndof)) #this is not used in the electron bath
        self.phis0=np.zeros((self.ml,len(self.XYZEqf)))   #this is not used in the electron bath


        #constants
        ##unit conversion
        #Constants.bohr2ang = Constants.bohr2ang
        ##JT md Length unit in terms of angstrom
        #Constants.jtmdlen = Constants.jtmdlen
        ##from ipy to jt unit
        #Constants.lconv = Constants.bohr2ang/Constants.jtmdlen
        ##from atomic mass to electron mass
        #Constants.amass2emass = Constants.amass2emass
        #Constants.hartree2eV = Constants.hartree2eV
        #Constants.eV2hartree = 1./Constants.hartree2eV
        ##1 Hartree/Bohr = 51.422 eV/ang
        #Constants.hartreebohr2eVang = Constants.hartreebohr2eVang
        ##conversion from i-pi t to pymd t:
        #Constants.tconv=Constants.tconv

        ##we need to convert it to Hartree/Bohr
        self.fconv = (dstrip(self.sm)/Constants.amass2emass**0.5)\
                /Constants.jtmdlen/Constants.hartreebohr2eVang
        #conversion from i-pi p to pymd p:
        #in i-pi, the momentum p has unit emass*bohr/time, while in pymd, 
        #p has unit: atomic mass**0.5*0.06466 ang/0.6582118e-15 second
        self.pconv = Constants.lconv/dstrip(self.sm)/Constants.amass2emass**0.5/Constants.tconv        

    def SetFric(self):
        """
        make electron friction matrix for all beads
        buid a block diagonal matrix from efric
        """

        dself=dd(self)
        dself.efric=np.kron(np.eye(dself.beads.nbeads), np.array(dself.efric))
        dself.exim=np.kron(np.eye(dself.beads.nbeads), np.array(dself.exim))
        dself.zeta1=np.kron(np.eye(dself.beads.nbeads), np.array(dself.zeta1))
        dself.zeta2=np.kron(np.eye(dself.beads.nbeads), np.array(dself.zeta2))
        
    def GenNoi(self):
        """
        initialize the electron noise array
        """
        self.noise=np.zeros((self.beads.nbeads,self.nmd,self.nndof))

        ##intialize the ebath class
        iebath = ebath(self.cats, self.temp, Constants.tconv*self.dt, self.nmd, self.wmax,\
               self.nw, self.bias, self.efric,self.exim,\
               self.exip,self.zeta1,self.zeta2,self.classical,self.zpmotion)
        self.sig=iebath.sig
        ##loop over beads
        for i, bead in enumerate(self.beads):
            #generate a set of noise
            iebath.gnoi()
            #average of the noise
            print "average of noise:"
            print numpy.average(self.noise,axis=0)
            #store it
            self.noise[i]=iebath.noise



    def force(self,step=None,id=0):
        """
        force from the baths:
        det   --   the deterministric part, friction
        fluc  --   fluctuating force

        """

        if step is None:
            softexit.trigger("Unknown step count. Exiting simulation")
            return

        """first call, seting up the matrices, parameters...
        I do not know how to visit some variables in __init___,
        so had to put it here.
        This can be moved to __init__ if we know how,
        i.e., number of beads, the momentum vector p, ...
        """
        print 'calculate force from electron bath'
        self.mdstep = step
        print 'step',self.mdstep
        
        #list of indices that attached to bath for all the beads
        ldof = np.array([[3*n+i for i in range(3)] for n in\
                             self.cats], dtype='int').flatten()
        self.ldof = self.SetAttachedList(ldof)

        #unit conversion
        #Constants.bohr2ang = 0.529177208
        #from atomic mass to electron mass
        #Constants.amass2emass = 1822.88848149
        #1 Hartree/Bohr = 51.422 eV/ang
        #Constants.hartreebohr2eVang = 51.42208619083232
        #JT md Length unit in terms of angstrom
        #Constants.jtmdlen = 0.06466
        #Constants.lconv = Constants.bohr2ang/Constants.jtmdlen
        #noise generated from JT pymd code needs to be converted to Hartree/Bohr
        #in pymd, the noise has unit eV/0.06466 Ang/atomic mass**0.5

        ##we need to convert it to Hartree/Bohr
        #self.fconv = (dstrip(self.sm)/Constants.amass2emass**0.5)\
                #        /Constants.jtmdlen/Constants.hartreebohr2eVang
        #conversion from i-pi t to pymd t:
        #Constants.tconv=2.4188843e-17/0.658211814201041e-15
        #conversion from i-pi p to pymd p:
        #in i-pi, the momentum p has unit emass*bohr/time, while in pymd, 
        #p has unit: atomic mass**0.5*0.06466 ang/0.6582118e-15 second
        
        self.fconv = (dstrip(self.sm)/Constants.amass2emass**0.5)\
                /Constants.jtmdlen/Constants.hartreebohr2eVang
        self.pconv = Constants.lconv/dstrip(self.sm)/Constants.amass2emass**0.5/Constants.tconv        

        #self.hasnoise = step
        if self.hasnoise == 0:
            #self.SetFric()
            self.GenNoi()
            #run the above codes only once
            self.hasnoise = 1
        if self.hasfriction == 0:    #this has to run at every new start and restart, it increases fric to nr of beads
            self.SetFric()
            self.hasfriction == 1
        #print 'fric', self.efric.shape, self.zeta1.shape            
            
        #momentum transformed to jt unit
        p = dstrip(self.p).copy()
        rp = reducedp(self.ldof,p*self.pconv)
        #displacement in Angstrom
        q = dstrip(self.q)*Constants.bohr2ang - self.XYZEqf
        #transform to jt md length unit
        q = q/Constants.jtmdlen
        #including square root of mass in atomic mass unit
        q = q*dstrip(self.sm)/Constants.amass2emass**0.5
        rq = reducedp(self.ldof,q)
        #nonequilibrium forces here
        #friction, non-conservative, renormalization
        # and berry force, respectively
        det = -np.dot(self.efric,rp)\
                +np.dot(self.bias*self.exim,rq)\
                -np.dot(self.bias*self.zeta1,rq)\
                -np.dot(self.bias*self.zeta2,rp)
        fluc = self.noise[:,step+id,:].flatten()
        #friction and noise
        sfor = det + fluc

        lfor = enlargep(len(p),self.ldof,sfor)
        #converting force unit: JT md unit to ipy  Hartree/Bohr unit
        lfor *= self.fconv

        llfor = np.array(np.split(lfor,self.beads.nbeads))
        return llfor



    def SetAttachedList(self,ldof):
        ntdof = len(self.p)/self.beads.nbeads
        lldof = np.array([],dtype='int')
        for i in range(self.beads.nbeads):

            lldof =np.append(lldof,ldof+i*ntdof)
        return lldof




    def step(self):
        """Updates the bound momentum vector with a GLE thermostat"""

        pass




    def pstep(self,step=None,id=0):
        """momenta propagator."""
        """
        We should update self.beads.p and self.beads.q
        """

        if step is None:
            softexit.trigger("Unknown step count. Exiting simulation")
            return
        elif step+id >= self.nmd:
            return

        self.beads.p += (self.force(step,id))  * (self.dt * 0.5)

        

    def qstep(self,step=None,id=0):
        """displacement propagator."""
        pass

        #if step is None:
        #    softexit.trigger("Unknown step count. Exiting simulation")
        #    return
        #elif step+id >= self.nmd:
        #    return
        #self.beads.q += self.beads.p * self.dt/self.beads.m3
        #self.beads.q += (self.force(step,id))* self.dt**2/2.0/self.beads.m3
        #self.beads.q += self.beads.p * self.dt/self.beads.m3\
                #      +(self.force(step,id))* self.dt**2/2.0/self.beads.m3


       

class ThermoQLE_ph(Thermostat):
    """
    Represents an electron thermostat.
    Attributes:
           cats: indices of atoms that connect to the thermal bath (python
           index)
           temp: The simulation temperature. Defaults to 1.0.
           dt: The simulation time step. Defaults to 1.0.
           timesteps: The number of time steps to do the simulation.
           gamma: The single friction coefficient.
           ethermo: The initial heat energy transferred to the bath.
              Defaults to 0.0. Will be non-zero if the thermostat is
              initialised from a checkpoint file.

    Depend objects:


    """

    def __init__(self, temp=1.0, dt=1.0, ethermo=0.0, cats=None, nmd=2**10, nw_ph=None, wmax_ph=None,\
                 debye=None, ml=None, mcof=2.0, sig=None,gamma=None,gwl=None,classical=False,zpmotion=True,\
                 K00=None,K01=None,V01=None,eta_ad=0, mdstep=None):

        super(ThermoQLE_ph, self).__init__(temp,dt,ethermo)
        dself = dd(self)


        #electron bath parameters

        self.cats = np.array(cats,dtype='int')
        #number of md steps
        self.nmd = nmd
        #number of degrees of freedom attached to the bath in a single bead
        self.nndof = len(cats)*3
        self.classical=classical
        self.zpmotion=zpmotion

        #phonon bath parameters

        self.temp,self.debye = temp,debye
        self.cats=np.array(cats,dtype='int')        
        self.dt,self.ml = dt,ml
        self.kernel = None
        self.mcof=mcof
        self.nw_ph=nw_ph
        self.wmax_ph = mcof*debye
        self.local = False
        self.wl = [self.wmax_ph*i/nw_ph for i in range(nw_ph)]
        self.cur = np.zeros(nmd)
        self.gamma = gamma
        self.sig = sig     
        self.gwl = gwl
        self.K00,self.K01,self.V01 = K00,K01,V01
        self.eta_ad=eta_ad
        self.mdstep=mdstep
        #counter     
        self.hasnoise = 0
        self.hasfriction = 0        
        self.counter = 0
          
        self.phis0=None  
        #conversion from i-pi t to pymd t:
        #Constants.tconv=2.4188843e-17/0.658211814201041e-15







    def bind(self, beads=None, atoms=None, pm=None, nm=None, prng=None, fixdof=None):
        """Binds the appropriate degrees of freedom to the thermostat.

        This takes an object with degrees of freedom, and makes their momentum
        and mass vectors members of the thermostat. It also then creates the
        objects that will hold the data needed in the thermostat algorithms
        and the dependency network.

        Args:
           beads: An optional beads object to take the mass and momentum vectors
              from.
           atoms: An optional atoms object to take the mass and momentum vectors
              from.
           pm: An optional tuple containing a single momentum value and its
              conjugate mass.
           prng: An optional pseudo random number generator object. Defaults to
              Random().
           fixdof: An optional integer which can specify the number of constraints
              applied to the system. Defaults to zero.

        Raises:
           TypeError: Raised if no appropriate degree of freedom or object
              containing a momentum vector is specified for
              the thermostat to couple to.
        """

        super(ThermoQLE_ph, self).bind(beads=beads, atoms=atoms, pm=pm, prng=prng, fixdof=fixdof)
        dself=dd(self)
        dself.beads=beads
        if not beads is None:
            #the displacement
            dself.q = beads.q.flatten()
        else:
            raise TypeError("ThermoQLE.bind expects Beads to bind to")
       
        self.noise=np.zeros((self.beads.nbeads,self.nmd,self.nndof))
        self.phfric=np.zeros((self.ml,self.nndof,self.nndof))
        self.phis0=np.zeros((self.ml,len(self.beads.p)))

        #constants
        ##unit conversion
        #Constants.bohr2ang = Constants.bohr2ang
        ##JT md Length unit in terms of angstrom
        #Constants.jtmdlen = Constants.jtmdlen
        ##from ipy to jt unit
        #Constants.lconv = Constants.bohr2ang/Constants.jtmdlen
        ##from atomic mass to electron mass
        #Constants.amass2emass = Constants.amass2emass
        #Constants.hartree2eV = Constants.hartree2eV
        #Constants.eV2hartree = 1./Constants.hartree2eV
        ##1 Hartree/Bohr = 51.422 eV/ang
        #Constants.hartreebohr2eVang = Constants.hartreebohr2eVang
        ##conversion from i-pi t to pymd t:
        #Constants.tconv=Constants.tconv

        ##we need to convert it to Hartree/Bohr
        self.fconv = (dstrip(self.sm)/Constants.amass2emass**0.5)\
                /Constants.jtmdlen/Constants.hartreebohr2eVang
        #conversion from i-pi p to pymd p:
        #in i-pi, the momentum p has unit emass*bohr/time, while in pymd, 
        #p has unit: atomic mass**0.5*0.06466 ang/0.6582118e-15 second
        self.pconv = Constants.lconv/dstrip(self.sm)/Constants.amass2emass**0.5/Constants.tconv        


    def GenNoiFric(self):
        """
        initialize the phonon noise and friction array
        """
        self.noise=np.zeros((self.beads.nbeads,self.nmd,self.nndof))        

        ##intialize the phbath class   #T+dT/2 not implemented
        iphbath = phbath(self.temp,self.cats,self.debye,self.nw_ph,Constants.tconv*self.dt,self.nmd,\
                self.ml,self.mcof,self.sig, self.gamma,self.gwl,self.K00,\
                self.K01,self.V01,self.eta_ad,self.classical,self.zpmotion)
        #friction
        #generate a set of friction
        #iphbath.ggamma()
        iphbath.gmem()
        self.phfric= iphbath.kernel
        self.sig = iphbath.sig
        ##loop over beads
        for i, bead in enumerate(self.beads):
            #generate a set of noise
            iphbath.gnoi()
            #store it
            self.noise[i]=iphbath.noise
            #print iphbath.noise
 
                        
   
   
    def SetFric_ph(self):
        """
        make phonon friction matrix for all beads
        buid a block diagonal matrix from efric
        """

        dself=dd(self)
        #dself.phfric=self.phfric
        dself.phfric=np.kron(np.eye(dself.beads.nbeads), np.array(dself.phfric))  



    def force(self,step=None,id=0):
        """
        force from the baths:
        det   --   the deterministric part, friction
        fluc  --   fluctuating force

        """

        if step is None:
            softexit.trigger("Unknown step count. Exiting simulation")
            return

        """first call, seting up the matrices, parameters...
        I do not know how to visit some variables in __init___,
        so had to put it here.
        This can be moved to __init__ if we know how to visit,
        i.e., number of beads, the momentum vector p, ...
        """
        """
        counter
        """
        print 'counter', self.counter
        self.mdstep = step
        self.counter += 1
        print 'mdstep', self.mdstep


        """
        calculate force from phonon baths
        first left phonon bath
        """
        print 'calculate force from phonon bath'
        
        #list of indices that attached to bath for all the beads
        ldof = np.array([[3*n+i for i in range(3)] for n in\
                             self.cats], dtype='int').flatten()
        self.ldof = self.SetAttachedList(ldof)

        ##unit conversion
        #Constants.bohr2ang = 0.529177208
        ##from atomic mass to electron mass
        #Constants.amass2emass = 1822.88848149
        ##1 Hartree/Bohr = 51.422 eV/ang
        #Constants.hartreebohr2eVang = 51.42208619083232
        ##JT md Length unit in terms of angstrom
        #Constants.jtmdlen = 0.06466
        #Constants.lconv = Constants.bohr2ang/Constants.jtmdlen
        ##noise generated from JT pymd code needs to be converted to Hartree/Bohr
        ##in pymd, the noise has unit eV/0.06466 Ang/atomic mass**0.5
        ##we need to convert it to Hartree/Bohr
        #self.fconv = (dstrip(self.sm)/Constants.amass2emass**0.5)\
        #        /Constants.jtmdlen/Constants.hartreebohr2eVang
        ##conversion from i-pi t to pymd t:
        #Constants.tconv=2.4188843e-17/0.658211814201041e-15
        ##conversion from i-pi p to pymd p:
        ##in i-pi, the momentum p has unit emass*bohr/time, while in pymd, 
        ##p has unit: atomic mass**0.5*0.06466 ang/0.6582118e-15 second
        #self.pconv = Constants.lconv/dstrip(self.sm)/Constants.amass2emass**0.5/Constants.tconv        
        #
        

        #self.hasnoise = step
        if self.hasnoise == 0:
            self.GenNoiFric()
            #self.SetFric_ph()
            #run the above codes only once
            self.hasnoise = 1
        if self.hasfriction == 0:    #this has to run at every new start and restart, it increases fric size to nr of beads
            self.SetFric_ph()
            self.hasfriction == 1
    
        #print 'phfric', self.phfric

        #new momenta
        p = dstrip(self.p).copy()
        rp = reducedp(self.ldof,p*self.pconv)
        #print 'ldof', self.ldof
        #print 'rp', rp

        #initialize p history 
        if self.mdstep==0:
           self.phis0=np.zeros((self.ml,len(p)))
        #project p history on coupling atoms in cats
        self.phis = reducedpm(self.ldof,self.phis0)           
        #update phis for id=1 (the last call in each md step)
        if self.mdstep!=0 and id==1:        
           self.phis=np.concatenate((np.array([rp]), np.array(self.phis)[:-1]),axis=0)
        #integrate time kernel*velocity(=phis)
        #print 'phis', self.phis
        det=0.0
        for j in range(self.ml):
                det = det  - np.dot(self.phfric[j,:,:],self.phis[j,:])*self.dt*Constants.tconv
        fluc = self.noise[:,step+id,:].flatten()

        sfor = det + fluc

        lfor = enlargep(len(p),self.ldof,sfor)
        #converting force to Hartree/Bohr
        lfor *= self.fconv
        llfor = np.array(np.split(lfor,self.beads.nbeads))    
        return llfor



    def SetAttachedList(self,ldof):
        ntdof = len(self.p)/self.beads.nbeads
        lldof = np.array([],dtype='int')
        for i in range(self.beads.nbeads):

            lldof =np.append(lldof,ldof+i*ntdof)
        return lldof



    def updatep(self,p0,ldof,p):
        """
        replace in the matrix pp the degrees of freedom that are connected
        to the bath, given by ldof
        """
        pp=p0
        for i in ldof:
            pp[:,i] = p[:,ldof[0]-i]
        return pp

#    def updatep(self,p0,ldof,p):
#        """
#        replace in the matrix pp the degrees of freedom that are connected
#        to the bath, given by ldof
#        """
#        pp=p0
#        for i,id in enumerate(ldof):
#            pp[:,id] = p[:,i]
#        return pp



    def step(self):
        """Updates the bound momentum vector with a GLE thermostat"""
        f = self.force() 




    def pstep(self,step=None,id=0):
        """momenta propagator."""
        """
        We should update self.beads.p and self.beads.q
        """

        if step is None:
            softexit.trigger("Unknown step count. Exiting simulation")
            return
        elif step+id >= self.nmd:
            return


        self.beads.p += (self.force(step,id))  * (self.dt * 0.5) 

        #print "self.p:"
        #print self.p
        #print "self.beads.p:"
        #print self.beads.p
        #stop

        #update phis0 at the end of each md step
        if id==1:
            #new momenta
            p = dstrip(self.p).copy()
            self.phis0=np.concatenate((np.array([p])*self.pconv, np.array(self.phis0)[:-1]),axis=0)

    def qstep(self,step=None,id=0):
        """displacement propagator."""
        if step is None:
            softexit.trigger("Unknown step count. Exiting simulation")
            return
        elif step+id >= self.nmd:
            return
        #self.beads.q += self.beads.p * self.dt/self.beads.m3
        self.beads.q += (self.force(step,id))* self.dt**2/2.0/self.beads.m3



class MultiThermo(Thermostat):

    def __init__(self, temp=1.0, dt=1.0, ethermo=0.0, thermolist=None):
        """Initialises Thermostat.

        Args:
           temp: The simulation temperature. Defaults to 1.0.
           dt: The simulation time step. Defaults to 1.0.
           ethermo: The initial heat energy transferred to the bath.
              Defaults to 0.0. Will be non-zero if the thermostat is
              initialised from a checkpoint file.
        """
        dself = dd(self)

        self.tlist = thermolist
        dself.temp = depend_value(name='temp', value=temp)
        dself.dt = depend_value(name='dt', value=dt)
        dself.ethermo = depend_value(name='ethermo', value=ethermo)
        for t in self.tlist:
            dpipe(dself.dt, dd(t).dt)
            dpipe(dself.temp, dd(t).temp)

    def get_ethermo(self):
        et = 0.0
        for t in self.tlist:
            et += t.ethermo
        return et

    def bind(self, beads=None, atoms=None, pm=None, nm=None, prng=None, fixdof=None):
        """Binds the appropriate degrees of freedom to the thermostat."""

        # just binds all the sub-thermostats
        for t in self.tlist:
            t.bind(beads=beads, atoms=atoms, pm=pm, nm=nm, prng=prng, fixdof=fixdof)
            dd(self).ethermo.add_dependency(dd(t).ethermo)

        dd(self).ethermo._func = self.get_ethermo

    def step(self):
        """Steps through all sub-thermostats."""

        for t in self.tlist:
            t.step()
        pass

    def pstep(self,step=None,id=0):
        """Steps through all sub-thermostats."""

        for t in self.tlist:
            t.pstep(step,id)
        pass
        
    def qstep(self,step=None,id=0):
        """Steps through all sub-thermostats."""
        for t in self.tlist:
            t.qstep(step,id)
        pass



def ReadEPHNCFile(filename):
    """
    Reads a NetCDF file that describes dynamical matrix, self-energies
    """
    class eph:
        pass

    file = Dataset(filename,'r')
    print 'Reading from %s' % filename

    # General attributes
    eph.filename = filename
    eph.wl= np.array(file.variables['Wlist'])
    eph.hw= np.array(file.variables['hw'])
    eph.U= np.array(file.variables['U'])
    eph.DynMat= np.array(file.variables['DynMat'])
    eph.SigL= np.array(file.variables['ReSigL'])+1j*np.array(file.variables['ImSigL'])
    eph.SigR= np.array(file.variables['ReSigR'])+1j*np.array(file.variables['ImSigR'])
    eph.efric=np.array(file.variables['Friction'])
    eph.xim=np.array(file.variables['NC'])
    eph.xip=np.array(file.variables['NCP'])
    eph.zeta1=np.array(file.variables['zeta1'])
    eph.zeta2=np.array(file.variables['zeta2'])
    eph.XYZEq = np.array(file.variables['XYZEq'])
    eph.DynamicMasses= np.array(file.variables['DynamicMasses'])  #Dynamic atoms
    eph.DynamicAtoms= np.array(file.variables['DynamicAtoms'])

    try:
        eph.lcats= np.array(file.variables['lcats'])
    except:
        eph.lcats = None
    try:
        eph.rcats= np.array(file.variables['rcats'])
    except:
        eph.rcats = None
    try:
        eph.ecats= np.array(file.variables['ecats'])
    except:
        eph.ecats=None

    try:
        eph.FixedAtoms= np.array(file.variables['FixedAtoms'])
    except:
        eph.FixedAtoms= np.array([])

    try:
        eph.constraint = np.array(file.variables['constraint'])
    except:
        eph.constraint = np.array([])

    file.close()
    return eph

def ReadPHSCFile(filename):
    """
    Reads a NetCDF file that describes dynamical matrix, self-energies for phonons
    """
    class phs:
        pass

    file = Dataset(filename,'r')
    print 'Reading from %s' % filename

    # General attributes
    phs.filename = filename
    phs.wl= np.array(file.variables['Wlist'])
    phs.hw= np.array(file.variables['hw'])
    phs.U= np.array(file.variables['U'])
    phs.DynMat= np.array(file.variables['DynMat'])
    phs.SigL= np.array(file.variables['ReSigL'])+1j*np.array(file.variables['ImSigL'])
    phs.SigR= np.array(file.variables['ReSigR'])+1j*np.array(file.variables['ImSigR'])
    #eph.efric=np.array(file.variables['Friction'])
    #eph.xim=np.array(file.variables['NC'])
    #eph.xip=np.array(file.variables['NCP'])
    #eph.zeta1=np.array(file.variables['zeta1'])
    #eph.zeta2=np.array(file.variables['zeta2'])
    phs.XYZEq = np.array(file.variables['XYZEq'])
    phs.DynamicMasses= np.array(file.variables['DynamicMasses'])
    phs.DynamicAtoms= np.array(file.variables['DynamicAtoms'])
    phs.lcats= np.array(file.variables['lcats'])
    phs.rcats= np.array(file.variables['rcats'])
    #phs.leftfixed= np.array(file.variables['leftfixed'])
    phs.fixatoms= np.array(file.variables['FixedAtoms'])
    print 'Read %s done!' % filename
    file.close()
    return phs






def initialise(hw,U,T):
    """
    initial displacement and velocity from the dynamical matrix
    according to the temperature.
    """
    av=hw
    au=U

    dis=np.zeros(len(av))
    vel=np.zeros(len(av))
    for i in range(len(av)):
        #cutoff energy 0.005 eV
        #do not initialise motion due to slow modes
        #because it may gives large displacement
        if av[i] < 0.005:
            am=0.0
        else:
            am=((bose(av[i],T)+0.5)*2.0/av[i])**0.5
            #am=(0.5*2.0/av[i])**0.5
        r=np.random.rand()
        dis = dis + au[i]*am*np.cos(2.*np.pi*r)
        vel = vel - av[i]*au[i]*am*np.sin(2.*np.pi*r)

    return dis,vel


#def bose(w,T):
#    """
#    bose distribution
#    """
#    kb = 0.000086173423    #boltzmann constant
#    small = 10e-20
#    if T == 0.0:
#        if w == 0.0:
#            return 1/(np.exp(1.0/kb)-1)
#        elif w < 0.0:
#            return -1.0
#        else:
#            return 0.0
#    else:
#        if w == 0.0:
#            #return 1/small
#            #have problems for finite temperature for bias calculation
#            #return 0 seems solves it
#            return 0.0
#        else:
#            return 1.0/(np.exp(w/kb/T)-1.0)



#def Write2NetCDFFile(file,var,varLabel,dimensions,units=None,description=None):
#    print 'Write2NetCDFFile:', varLabel, dimensions
#    tmp = file.createVariable(varLabel,'d',dimensions)
#    tmp[:] = var
#    if units: tmp.units = units
#    if description: tmp.description = description

#def ReadNetCDFVar(file,var):
#    print "ReadNetCDFFile: reading "+ var
#    f = Dataset(file,'r')
#    vv=N.array(f.variables[var])
#    f.close()
#    return vv
#    


def enlargep(nn,ldof,p):
    """
    enlarge the vector with size ldof to the size nn
    """
    pp = np.zeros(nn)
    for i,id in enumerate(ldof):
        pp[id] = p[i]
    return pp

def enlargepm(n,nn,ldof,p):
    """
    replace in matrix with size [n, nn] the elements with index 
    given in list ldof by elements p
    """
    pp = np.zeros((n,nn))
    for i in ldof:
        pp[:,i] = p[:,i-ldof[0]]
    return pp

def reducedp(ldof,p):
    """
    pick the degrees of freedom in p that are connected
    to the bath, given by ldof
    """
    return np.array([p[i] for i in ldof])

def reducedpm(ldof,p):
    """
    pick the degrees of freedom in matrix p that are connected
    to the bath, given by ldof
    """
    return np.array([p[:,i] for i in ldof]).T



#    def reducedp(self,ldof,p):
#        """
#        pick the degrees of freedom in p that are connected
#        to the bath, given by ldof
#        """
#        return np.array([p[i] for i in ldof])

#    def reducedpm(self,ldof,p):
#        """
#        pick the degrees of freedom in matrix p that are connected
#        to the bath, given by ldof
#        """
#        return np.array([p[:,i] for i in ldof]).T
        
#    def enlargep(self,nn,ldof,p):
#        """
#        enlarge the vector with size ldof to the size nn
#        """
#        pp = np.zeros(nn)
#        for i,id in enumerate(ldof):
#            pp[id] = p[i]
#        return pp
#    def enlargepm(self,n,nn,ldof,p):
#        """
#        replace in matrix with size [n, nn] the elements with index 
#        given in list ldof by elements p
#        """
#        pp = np.zeros((n,nn))
#        for i in ldof:
#            pp[:,i] = p[:,i-ldof[0]]
#        return pp
"""
#--------------------------------------------------------------------------------------
#testing
#
#--------------------------------------------------------------------------------------

if __name__=="__main__":
    from netCDF4 import Dataset
    from pymd.ebath import *
    import matplotlib.pyplot as PP
    
    
    filename="EPH.nc"
    eph=ReadEPHNCFile(filename)
    
    #--------------------------------------------------------------------------------------
    #----------------------------------------------------
    # electron bath
    #    def __init__(self,cats,T=0.,wmax=None,nw=None,bias=0.,efric=None,exim=None,exip=None,dt=None,nmd=None):
    #----------------------------------------------------
    #ecats=range(13)
    #eb = ebath(ecats,T=300.0,dt=0.01,nmd=2**12,wmax=1.0,nw=500,bias=0.,efric=eph.efric,exim=eph.xim,exip=eph.xip)
    #eb.SetMDsteps(8,2**12)
    #eb.gnoi()
    #PP.plot(eb.noise[:,0])
    #PP.savefig("enoise.pdf")
    #PP.close()
    #sys.exit()

    ecats=range(13)
    qle = ThermoQLE(temp=1.0,dt=1.0,ethermo=0.0,cats=ecats,nmd=2**10,wmax=1.0,\
                    nw=500,bias=0.,efric=eph.efric,exim=eph.xim,exip=eph.xip)
    #def __init__(self, temp=1.0, dt=1.0, ethermo=0.0, cats=None, nmd=2**10, wmax=None, nw=None, bias=0.,\
            #             efric=None,exim=None,exip=None,zeta1=None,zeta2=None,classical=False,zpmotion=True):
#--------------------------------------------------------------------------------------
"""
