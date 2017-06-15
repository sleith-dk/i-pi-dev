"""Contains the classes that deal with the different dynamics required in
different types of ensembles.

Holds the algorithms required for normal mode propagators, and the objects to
do the constant temperature and pressure algorithms. Also calculates the
appropriate conserved energy quantity for the ensemble of choice.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.


import time

import numpy as np

from ipi.engine.motion import Motion
from ipi.utils.depend import *
from ipi.engine.thermostats import Thermostat
from ipi.engine.barostats import Barostat


#__all__ = ['Dynamics', 'NVEIntegrator', 'NVTIntegrator', 'NPTIntegrator', 'NSTIntegrator', 'SCIntegrator`']

class Dynamics(Motion):
    """self (path integral) molecular dynamics class.

    Gives the standard methods and attributes needed in all the
    dynamics classes.

    Attributes:
        beads: A beads object giving the atoms positions.
        cell: A cell object giving the system box.
        forces: A forces object giving the virial and the forces acting on
            each bead.
        prng: A random number generator object.
        nm: An object which does the normal modes transformation.

    Depend objects:
        econs: The conserved energy quantity appropriate to the given
            ensemble. Depends on the various energy terms which make it up,
            which are different depending on the ensemble.he
        temp: The system temperature.
        dt: The timestep for the algorithms.
        ntemp: The simulation temperature. Will be nbeads times higher than
            the system temperature as PIMD calculations are done at this
            effective classical temperature.
    """

    def __init__(self, timestep, mode="nve", splitting="obabo", thermostat=None, barostat=None, fixcom=False, fixatoms=None, nmts=None):
        """Initialises a "dynamics" motion object.

        Args:
            dt: The timestep of the simulation algorithms.
            fixcom: An optional boolean which decides whether the centre of mass
                motion will be constrained or not. Defaults to False.
        """

        super(Dynamics, self).__init__(fixcom=fixcom, fixatoms=fixatoms)

        # initialize time step. this is the master time step that covers a full time step
        dset(self, "dt", depend_value(name='dt', value=timestep))

        if thermostat is None:
            self.thermostat = Thermostat()
        else:
            self.thermostat = thermostat

        if barostat is None:
            self.barostat = Barostat()
        else:
            self.barostat = barostat

        if nmts is np.zeros(0,int):
           self.nmts = np.asarray([1],int)
        elif nmts is None or len(nmts) == 0:
           self.nmts = np.asarray([1],int)
        else:
           self.nmts=np.asarray(nmts)

        self.enstype = mode
        if self.enstype == "nve":
            self.integrator = NVEIntegrator()
        elif self.enstype == "nvt":
            self.integrator = NVTIntegrator()
        elif self.enstype == "npt":
            self.integrator = NPTIntegrator()
        elif self.enstype == "nst":
            self.integrator = NSTIntegrator()
        elif self.enstype == "sc":
            self.integrator = SCIntegrator()
        else:
            self.integrator = DummyIntegrator()

        # splitting mode for the integrators
        self.splitting = splitting

        # constraints
        self.fixcom = fixcom
        if fixatoms is None:
            self.fixatoms = np.zeros(0, int)
        else:
            self.fixatoms = fixatoms

    def bind(self, ens, beads, nm, cell, bforce, prng):
        """Binds ensemble beads, cell, bforce, and prng to the dynamics.

        This takes a beads object, a cell object, a forcefield object and a
        random number generator object and makes them members of the ensemble.
        It also then creates the objects that will hold the data needed in the
        ensemble algorithms and the dependency network. Note that the conserved
        quantity is defined in the init, but as each ensemble has a different
        conserved quantity the dependencies are defined in bind.

        Args:
            beads: The beads object from whcih the bead positions are taken.
            nm: A normal modes object used to do the normal modes transformation.
            cell: The cell object from which the system box is taken.
            bforce: The forcefield object from which the force and virial are
                taken.
            prng: The random number generator object which controls random number
                generation.
        """

        super(Dynamics, self).bind(ens, beads, nm, cell, bforce, prng)


        # Strips off depend machinery for easier referencing.
        dself = dd(self)
        dthrm = dd(self.thermostat)
        dbaro = dd(self.barostat)
        dnm = dd(self.nm)
        dens = dd(self.ensemble)

        # n times the temperature (for path integral partition function)
        dself.ntemp = depend_value(name='ntemp', func=self.get_ntemp, dependencies=[dens.temp])

        # fixed degrees of freedom count
        fixdof = len(self.fixatoms) * 3 * self.beads.nbeads
        if self.fixcom:
            fixdof += 3

        # first makes sure that the thermostat has the correct temperature and timestep, then proceeds with binding it.
        dpipe(dself.ntemp, dthrm.temp)
        dpipe(dself.dt, dthrm.dt)

        # depending on the kind, the thermostat might work in the normal mode or the bead representation.
        self.thermostat.bind(beads=self.beads, nm=self.nm, prng=prng, fixdof=fixdof)

        # first makes sure that the barostat has the correct stress andf timestep, then proceeds with binding it.
        dpipe(dself.ntemp, dbaro.temp)
        dpipe(dself.dt, dbaro.dt)
        dpipe(dens.pext, dbaro.pext)
        dpipe(dens.stressext, dbaro.stressext)
        self.barostat.bind(beads, nm, cell, bforce, prng=prng, fixdof=fixdof)

        # now we need to define timesteps for the different propagators. NOTE: O=stochastic B=momenta A=positions
        dset(self,"halfdt", depend_value(name="dt", func=(lambda : 0.5*self.dt) , dependencies=[dself.dt]))

        self.inmts = np.prod(self.nmts) # inner multiplier for MTS propagator        
        if self.splitting == "obabo":
            # sets the timstep of the thermostat and barostat to dt/2            
            dpipe(dself.halfdt, dthrm.dt)
            dpipe(dself.halfdt, dbaro.dt)

            # sets the timstep of the normalmode propagator to time step of the innermost MTS propagator            
            dself.deltat = depend_value(name="deltat", func=(lambda : self.dt/self.inmts) , dependencies=[dself.dt])
            dpipe(dself.deltat, dnm.dt)

        elif self.splitting == "baoab":
            # sets the timstep of the thermostat and barostat to dt/2
            dpipe(dself.halfdt, dbaro.dt)

            # sets the timstep of the normalmode propagator to HALF OF THE time step of the innermost MTS propagator
            dself.halfdeltat = depend_value(name="halfdeltat", func=(lambda : 0.5*self.dt/self.inmts) , dependencies=[dself.dt])
            dpipe(dself.halfdeltat, dnm.dt)

        # now that the timesteps are decided, we proceed to bind the integrator.
        self.integrator.bind(self)

        self.ensemble.add_econs(dthrm.ethermo)
        self.ensemble.add_econs(dbaro.ebaro)

        #!TODO THOROUGH CLEAN-UP AND CHECK
        if self.enstype == "nvt" or self.enstype == "npt" or self.enstype == "nst":
            if self.ensemble.temp < 0:
                raise ValueError("Negative or unspecified temperature for a constant-T integrator")
            if self.enstype == "npt":
                if type(self.barostat) is Barostat:
                    raise ValueError("The barostat and its mode have to be specified for constant-p integrators")
                if self.ensemble.pext < 0:
                    raise ValueError("Negative or unspecified pressure for a constant-p integrator")
            elif self.enstype == "nst":
                if np.trace(self.ensemble.stressext) < 0:
                    raise ValueError("Negative or unspecified stress for a constant-s integrator")

    def get_ntemp(self):
        """Returns the PI simulation temperature (P times the physical T)."""

        return self.ensemble.temp * self.beads.nbeads

    def step(self, step=None):
        self.integrator.step(step)


class DummyIntegrator(dobject):
    """ No-op integrator for (PI)MD """

    def __init__(self):
        pass

    def bind(self, motion):
        """ Reference all the variables for simpler access."""

        self.beads = motion.beads
        self.bias = motion.bias
        self.ensemble = motion.ensemble
        self.forces = motion.forces
        self.prng = motion.prng
        self.nm = motion.nm
        self.thermostat = motion.thermostat
        self.barostat = motion.barostat
        self.fixcom = motion.fixcom
        self.fixatoms = motion.fixatoms
        self.splitting = motion.splitting
        dset(self, "dt", dget(motion, "dt"))
        dset(self, "halfdt", dget(motion, "halfdt"))
        if motion.enstype == "mts" or motion.enstype == "nvt" or  motion.enstype == "nve": self.nmts=motion.nmts
        #mts on sc force in suzuki-chin
        if motion.enstype == "sc":
            if(motion.nmts.size > 1):
                raise ValueError("MTS for SC is not implemented yet....")
            else:
                # coefficients to get the (baseline) trotter to sc conversion
                self.coeffsc = np.ones((self.beads.nbeads,3*self.beads.natoms), float)
                self.coeffsc[::2] /= -3.
                self.coeffsc[1::2] /= 3.
                self.nmts=motion.nmts[-1]

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


class NVEIntegrator(DummyIntegrator):
    """ Integrator object for constant energy simulations.

    Has the relevant conserved quantity and normal mode propagator for the
    constant energy ensemble. Note that a temperature of some kind must be
    defined so that the spring potential can be calculated.

    Attributes:
        ptime: The time taken in updating the velocities.
        qtime: The time taken in updating the positions.
        ttime: The time taken in applying the thermostat steps.

    Depend objects:
        econs: Conserved energy quantity. Depends on the bead kinetic and
            potential energy, and the spring potential energy.
    """

    def pconstraints(self):
        """This removes the centre of mass contribution to the kinetic energy.

        Calculates the centre of mass momenta, then removes the mass weighted
        contribution from each atom. If the ensemble defines a thermostat, then
        the contribution to the conserved quantity due to this subtraction is
        added to the thermostat heat energy, as it is assumed that the centre of
        mass motion is due to the thermostat.

        If there is a choice of thermostats, the thermostat
        connected to the centroid is chosen.
        """

        if (self.fixcom):
            pcom = np.zeros(3, float)

            na3 = self.beads.natoms * 3
            nb = self.beads.nbeads
            p = depstrip(self.beads.p)
            m = depstrip(self.beads.m3)[:,0:na3:3]
            M = self.beads[0].M

            for i in range(3):
                pcom[i] = p[:,i:na3:3].sum()

            self.ensemble.eens += np.dot(pcom, pcom) / (2.0*M*nb)

            # subtracts COM velocity
            pcom *= 1.0 / (nb*M)
            for i in range(3):
                self.beads.p[:,i:na3:3] -= m*pcom[i]

        if len(self.fixatoms) > 0:
            for bp in self.beads.p:
                m = depstrip(self.beads.m)
                self.ensemble.eens += 0.5*np.dot(bp[self.fixatoms*3], bp[self.fixatoms*3]/m[self.fixatoms])
                self.ensemble.eens += 0.5*np.dot(bp[self.fixatoms*3+1], bp[self.fixatoms*3+1]/m[self.fixatoms])
                self.ensemble.eens += 0.5*np.dot(bp[self.fixatoms*3+2], bp[self.fixatoms*3+2]/m[self.fixatoms])
                bp[self.fixatoms*3] = 0.0
                bp[self.fixatoms*3+1] = 0.0
                bp[self.fixatoms*3+2] = 0.0

    def pstep(self, level=0, alpha=1.0):
        """Velocity Verlet monemtum propagator."""
        self.beads.p += self.forces.forces_mts(level)*self.halfdt/alpha
        if level == 0:
            self.beads.p += depstrip(self.bias.f)*(self.halfdt/alpha)        

    def qcstep(self, alpha=1.0):
        """Velocity Verlet centroid position propagator."""
        self.nm.qnm[0,:] += depstrip(self.nm.pnm)[0,:]/depstrip(self.beads.m3)[0]*self.halfdt/alpha        

    def mtsprop(self, index, alpha):
        """ Recursive MTS step """
        nmtslevels = len(self.nmts)
        mk = self.nmts[index]  # mtslevels starts at level zero, where nmts should be 1 in most cases
        alpha *= mk

        for i in range(mk):
            # propagate p for dt/2alpha with force at level index
            self.pstep(index, alpha)
            self.pconstraints()

            if index == nmtslevels-1:
            # call Q propagation for dt/alpha at the inner step
                self.qcstep(alpha)
                self.nm.free_qstep() 
                self.qcstep(alpha)
            else:
                self.mtsprop(index+1, alpha)

            # propagate p for dt/2alpha
            self.pstep(index, alpha)
            self.pconstraints()

    def step(self, step=None):
        """Does one simulation time step."""

        self.mtsprop(0,1.0)


class NVTIntegrator(NVEIntegrator):
    """Integrator object for constant temperature simulations.

    Has the relevant conserved quantity and normal mode propagator for the
    constant temperature ensemble. Contains a thermostat object containing the
    algorithms to keep the temperature constant.

    Attributes:
        thermostat: A thermostat object to keep the temperature constant.
    """

    def tstep(self):
        """Velocity Verlet thermostat step"""
        # the length of the thermostat step is controlled via depend objects
        self.thermostat.step()

    def mtsprop(self, index, alpha):
        """ Recursive MTS step """
        nmtslevels = len(self.nmts)        
        mk = self.nmts[index]  # mtslevels starts at level zero, where nmts should be 1 in most cases
        alpha *= mk

        if self.splitting == "obabo":
            for i in range(mk):
                # propagate p for dt/2alpha with force at level index
                self.pstep(index, alpha)
                self.pconstraints()

                if index == nmtslevels-1:
                # call Q propagation for dt/alpha at the inner step
                    self.qcstep(alpha)
                    self.nm.free_qstep() 
                    self.qcstep(alpha)
                else:
                    self.mtsprop(index+1, alpha)

                # propagate p for dt/2alpha
                self.pstep(index, alpha)
                self.pconstraints()

        elif self.splitting == "baoab":
            for i in range(mk):
                # propagate p for dt/2alpha with force at level index
                self.pstep(index, alpha)
                self.pconstraints()

                if index == nmtslevels-1:
                    # call Q propagation for dt/2alpha at the inner step
                    self.qcstep(alpha)
                    self.nm.free_qstep()  
                    self.tstep()
                    self.pconstraints()
                    self.qcstep(alpha)
                    self.nm.free_qstep() 
                else:
                    self.mtsprop(index+1, alpha)

                # propagate p for dt/2alpha
                self.pstep(index, alpha)
                self.pconstraints()

    def step(self, step=None):
        """Does one simulation time step."""

        if self.splitting == "obabo":
            # thermostat is applied at the outer loop
            self.tstep()
            self.pconstraints()

            self.mtsprop(0,1.0)

            self.tstep()
            self.pconstraints()
        elif self.splitting == "baoab":

            self.mtsprop(0,1.0)


class NPTIntegrator(NVTIntegrator):
    """Integrator object for constant pressure simulations.

    Has the relevant conserved quantity and normal mode propagator for the
    constant pressure ensemble. Contains a thermostat object containing the
    algorithms to keep the temperature constant, and a barostat to keep the
    pressure constant.
    """

    # should be enough to redefine these functions, and the step() from NVTIntegrator should do the trick
    def pstep(self, level=0, alpha=1.0):
        """Velocity Verlet monemtum propagator."""

        # since this  is thermostatted, should use half dt for every splitting
        self.barostat.pstep(level, 1.0/alpha)


    def qcstep(self, alpha=1.0):
        """Velocity Verlet centroid position propagator."""
        if self.splitting == "obabo": dt = 2.0
        elif self.splitting == "aboba" or self.splitting == "baoab": dt = 1.0

        self.barostat.qcstep(dtscale=dt/alpha)

    def tstep(self):
        """Velocity Verlet thermostat step"""
        # the length of the thermostat step is controlled via depend objects
        self.thermostat.step()
        self.barostat.thermostat.step()

class NSTIntegrator(NVTIntegrator):
    """Ensemble object for constant pressure simulations.

    Has the relevant conserved quantity and normal mode propagator for the
    constant pressure ensemble. Contains a thermostat object containing the
    algorithms to keep the temperature constant, and a barostat to keep the
    pressure constant.

    Attributes:
    barostat: A barostat object to keep the pressure constant.

    Depend objects:
    econs: Conserved energy quantity. Depends on the bead and cell kinetic
    and potential energy, the spring potential energy, the heat
    transferred to the beads and cell thermostat, the temperature and
    the cell volume.
    pext: External pressure.
    """

    def step(self, step=None):
        """NST time step.

        Note that the barostat only propagates the centroid coordinates. If this
        approximation is made a centroid virial pressure and stress estimator can
        be defined, so this gives the best statistical convergence. This is
        allowed as the normal mode propagation is approximately unaffected
        by volume fluctuations as long as the system box is much larger than
        the radius of gyration of the ring polymers.
        """

        self.ttime = -time.time()
        if self.splitting == "obabo":
            self.thermostat.step()
            self.barostat.thermostat.step()
            self.pconstraints()
            self.barostat.pstep()
            self.pconstraints()

            self.barostat.qcstep(dtscale=2.0)
            self.nm.free_qstep()

            self.barostat.pstep()
            self.pconstraints()
            self.barostat.thermostat.step()
            self.thermostat.step()
            self.pconstraints()
        elif self.splitting == "aboba":
            self.barostat.qcstep()
            self.nm.free_qstep()
            self.barostat.pstep()
            self.pconstraints()

            self.barostat.thermostat.step()
            self.thermostat.step()
            self.barostat.thermostat.step()
            self.pconstraints()

            self.barostat.pstep()
            self.pconstraints()

            self.barostat.qcstep()
            self.nm.free_qstep()
        elif self.splitting == "baoab":
            self.barostat.pstep()
            self.pconstraints()

            self.barostat.qcstep()
            self.nm.free_qstep()

            self.thermostat.step()
            self.pconstraints()

            self.barostat.qcstep()
            self.nm.free_qstep()

            self.barostat.pstep()
            self.pconstraints()

        self.ttime += time.time()

class SCIntegrator(NVTIntegrator):
    """Integrator object for constant temperature simulations.

    Has the relevant conserved quantity and normal mode propagator for the
    constant temperature ensemble. Contains a thermostat object containing the
    algorithms to keep the temperature constant.

    Attributes:
        thermostat: A thermostat object to keep the temperature constant.

    Depend objects:
        econs: Conserved energy quantity. Depends on the bead kinetic and
            potential energy, the spring potential energy and the heat
            transferred to the thermostat.
    """

    def bind(self, mover):
        """Binds ensemble beads, cell, bforce, bbias and prng to the dynamics.

        This takes a beads object, a cell object, a forcefield object and a
        random number generator object and makes them members of the ensemble.
        It also then creates the objects that will hold the data needed in the
        ensemble algorithms and the dependency network. Note that the conserved
        quantity is defined in the init, but as each ensemble has a different
        conserved quantity the dependencies are defined in bind.

        Args:
        beads: The beads object from whcih the bead positions are taken.
        nm: A normal modes object used to do the normal modes transformation.
        cell: The cell object from which the system box is taken.
        bforce: The forcefield object from which the force and virial are
            taken.
        prng: The random number generator object which controls random number
            generation.
        """

        super(SCIntegrator,self).bind(mover)
        self.ensemble.add_econs(dget(self.forces, "potsc"))

    def pstep(self):
        """Velocity Verlet momenta propagator."""


        # also include the baseline Tr2SC correction (the 2/3 & 4/3 V bit)
        self.beads.p += depstrip(self.forces.f + self.coeffsc*self.forces.f)*self.halfdt/self.nmts
        # also adds the bias force (TODO!!!)
        # self.beads.p += depstrip(self.bias.f)*(self.dt*0.5)

    def pscstep(self):
        """Velocity Verlet Suzuki-Chin momenta propagator."""

        # also adds the force assiciated with SuzukiChin correction (only the |f^2| term, so we remove the Tr2SC correction)
        self.beads.p += depstrip(self.forces.fsc - self.coeffsc*self.forces.f)*self.halfdt

    def qcstep(self):
        """Velocity Verlet centroid position propagator."""

        if self.splitting == "aboba" or self.splitting == "baoab": dt = self.halfdt
        elif self.splitting == "obabo": dt = self.dt

        self.nm.qnm[0,:] += depstrip(self.nm.pnm)[0,:]/depstrip(self.beads.m3)[0]*dt/self.nmts

    def step(self, step=None):
        """Does one simulation time step."""


        if self.splitting == "obabo":
            self.thermostat.step()
            self.pconstraints()

            self.pscstep()

            for i in range(self.nmts):
                self.pstep()
                self.pconstraints()

                self.qcstep()
                self.nm.free_qstep()

                self.pstep()

            self.pscstep()
            self.pconstraints()

            self.thermostat.step()
            self.pconstraints()
        elif self.splitting == "baoab":
            self.pscstep()
            for i in range(self.nmts):
                self.pstep()
                self.pconstraints()

                self.qcstep()
                self.nm.free_qstep()
                self.thermostat.step()
                self.pconstraints()
                self.qcstep()
                self.nm.free_qstep()

                self.pstep()

            self.pscstep()
            self.pconstraints()
