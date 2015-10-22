#!/usr/bin/env python

""" pos2centroid.py

Reads positions of individual beads from an i-PI run and
assemles them in a pdb describing the ring polymer connectivity.

Assumes the input files are in pdb format names prefix.pos_*.pdb.

Syntax:
   mergebeadspdb.py prefix
"""


import numpy as np
import sys, glob
from ipi.utils.io import *
from ipi.engine.atoms import Atoms
from ipi.engine.beads import Beads
from ipi.engine.cell import Cell
from ipi.utils.depend import *
from ipi.utils.units import *


def main(prefix):

   ipos=[]
   imode=[]
   for filename in sorted(glob.glob(prefix+".pos*")):
      imode.append(filename.split(".")[-1])
      ipos.append(open(filename,"r"))

   nbeads = len(ipos)
   natoms = 0
   ifr = 0
   while True:
      try:

         for i in range(nbeads):
            if (imode[i]=="xyz"):
               pos=read_file(imode[i],ipos[i])
               cell = Cell()
            else:
               pos, cell = read_file(imode[i],ipos[i])
            if natoms == 0:
               natoms = pos.natoms
               atoms = Atoms(natoms)

            atoms.q += pos.q
            atoms.names = pos.names
      except EOFError: # finished reading files
         sys.exit(0)
      atoms.q /= nbeads
      print_file(imode[0],atoms, cell)
      atoms.q[:]=0.0
      ifr+=1


if __name__ == '__main__':
   main(*sys.argv[1:])