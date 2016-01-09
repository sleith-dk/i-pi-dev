"""TODO

"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.


import numpy as np

from ipi.utils.depend import depend_value, dset, dobject


__all__ = ['Motion']


class Motion(dobject):
    """Base motion calculation class.

    Gives the standard methods and attributes needed in all the
    motion calculation classes.

    Attributes:
        beads: A beads object giving the atoms positions.
        cell: A cell object giving the system box.
        forces: A forces object giving the virial and the forces acting on
            each bead.
        fixcom: A boolean which decides whether the centre of mass
            motion will be constrained or not.
        fixatoms: A list of atoms that should be held fixed to their
            initial positions.

    Depend objects:
        none
    """

    def __init__(self, fixcom=False, fixatoms=None):
        """Initialises Motion object.

        Args:
           fixcom: An optional boolean which decides whether the centre of mass
              motion will be constrained or not. Defaults to False.
           fixatoms: A list of atoms that should be held fixed to their
             initial positions.
        """

        dset(self, "dt", depend_value(name="dt", value=0.0))
        self.fixcom = fixcom
        if fixatoms is None:
            self.fixatoms = np.zeros(0, int)
        else:
            self.fixatoms = fixatoms
                        
        self.beads = self.cell = self.forces = self.prng = self.nm = None 

    def bind(self, ens, beads, nm, cell, bforce, prng):
        """Binds beads, cell, bforce, and prng to the calculator.

        This takes a beads object, a cell object, a forcefield object and a
        random number generator object and makes them members of the atom motion caclulator.
        It also then creates the objects that will hold the data needed in the
        ensemble algorithms and the dependency network. Note that the conserved
        quantity is defined in the init, but as each ensemble has a different
        conserved quantity the dependencies are defined in bind.

        Args:
            beads: The beads object from whcih the bead positions are taken.
            nm: A normal modes object used to do the normal modes transformation.
            cell: The cell object from which the system box is taken.
            bforce: The forcefield object from which the force and virial are taken.
            bbias: The forcefield object from which the bias forces are obtained
            prng: The random number generator object which controls random number
                generation.
        """

        # store local references to the different bits of the simulation
        self.beads = beads
        self.cell = cell
        self.forces = bforce
        self.prng = prng
        self.nm = nm
        self.ensemble = ens
        self.bias = self.ensemble.bias

    def step(self, step=None):
        """Dummy simulation time step which does nothing."""

        pass
