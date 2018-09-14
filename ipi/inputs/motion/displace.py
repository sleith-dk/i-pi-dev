"""Deals with creating the ensembles class.

Copyright (C) 2013, Joshua More and Michele Ceriotti

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http.//www.gnu.org/licenses/>.

Inputs created by Michele Ceriotti and Benjamin Helfrecht, 2015

Classes:
   InputGeop: Deals with creating the Geop object from a file, and
      writing the checkpoints.
"""

import numpy as np
import ipi.engine.initializer
from ipi.engine.motion import *
from ipi.utils.inputvalue import *
from ipi.inputs.initializer import *
from ipi.utils.units import *

__all__ = ["InputDisplace"]


class InputDisplace(InputDictionary):
    """Direct displacement calculation options."""

    attribs = {}
    fields = {
        "displacement": (InputArray, {"dtype": float,
                                      "default": np.zeros(3, float),
                                      "dimension": "length",
                                      "help": "The displacement for each step.",
        }),
        "timestep": (InputValue, {"dtype": float,
                                  "default": 1.0,
                                  "help": "The time step.",
                                  "dimension": "time"}),
        "nmd": (InputValue, {"dtype": int,
                                  "default": 1024,
                                  "help": "The number of md steps."}),
        "temp": (InputValue, {"dtype": float,
                                  "default": 4.2,
                                  "help": "The bath temperature."}),
        "vbias": (InputValue, {"dtype": float,
                                  "default": 0.0,
                                  "help": "The applied voltage in the electron\
                              bath."}),
        "eta": (InputValue, {"dtype": float,
                                  "default": 0.0,
                                  "help": "Artificial damping adding to the\
                             phonon baths."}),
        "memlen": (InputValue, {"dtype": int,
                                  "default": 100,
                                  "help": "Length of memory kernel for the\
                             phonon baths."}),
    }


    default_help = "Constant displacements of the non-fixed atoms."
    default_label = "DISPLACE"

    def store(self, displace):
        """ Store an `Displace` instance in a minimal representation

        Parameters
        ----------
        displace : Displace
           object
        """
        if displace == {}:
            return

        self.displacement.store(displace.displacement)
        self.nmd.store(displace.nmd)
        self.timestep.store(displace.dt)
        self.temp.store(displace.temp)
        self.vbias.store(displace.vbias)
        self.eta.store(displace.eta)
        self.memlen.store(displace.memlen)
