"""Contains fundamental constants in atomic units.

Classes:
   Constants: Class whose members are fundamental contants.
   Elements: Class which contains the mass of different elements
   Units: Class which contains the methods needed to transform
      between different systems of units.
"""

import re

__all__ = ['Constants', 'Elements', 'UnitMap', 'unit_to_internal', 'unit_to_user' ]

class Constants:
   """Class whose members are fundamental contants.

   Attributes:
      kb: Boltzmann constant.
      hbar: Reduced Planck's constant.
      amu: Atomic mass unit.
   """

   kb = 3.1668152e-06
   hbar = 1.0
   amu = 1822.8885


class Elements(dict):
   """Class which contains the mass of different elements.

   Attributes:
      mass_list: A dictionary containing the masses of different elements.
         Has the form {"label": Mass in a.m.u.}. Note that the generic "X"
         label is assumed to be an electron.
   """

   mass_list={
      "X"   :    1.0000/Constants.amu, 
      "H"   :   1.00794,
      "D"   :    2.0141,
      "Z"   :  1.382943, #an interpolated H-D atom, based on y=1/sqrt(m) scaling
      "H2"  :    2.0160,
      "He"  :  4.002602,
      "Li"  :    6.9410,     
      "Be"  :  9.012182,
      "B"   :    10.811,
      "C"   :   12.0107,
      "N"   :  14.00674,
      "O"   :   15.9994,
      "F"   : 18.998403,
      "Ne"  :   20.1797,
      "Na"  : 22.989770,
      "Mg"  :   24.3050,
      "Al"  : 26.981538,
      "Si"  :   28.0855,
      "P"   : 30.973761,
      "S"   :    32.066,
      "Cl"  :   35.4527,
      "Ar"  :   39.9480,
      "K"   :   39.0983,
      "Ca"  :    40.078,
      "Sc"  : 44.955910,
      "Ti"  :    47.867,
      "V"   :   50.9415,
      "Cr"  :   51.9961,
      "Mn"  : 54.938049,
      "Fe"  :    55.845,
      "Co"  : 58.933200,
      "Ni"  :   58.6934,
      "Cu"  :    63.546,
      "Zn"  :     65.39,
      "Ga"  :    69.723,
      "Ge"  :     72.61,
      "As"  :  74.92160,
      "Se"  :     78.96,
      "Br"  :    79.904,
      "Kr"  :     83.80,
      "Rb"  :   85.4678,
      "Sr"  :     87.62,
      "Y"   :  88.90585,
      "Zr"  :    91.224,
      "Nb"  :  92.90638,
      "Mo"  :     95.94,
      "Tc"  :      98.0,
      "Ru"  :    101.07,
      "Rh"  : 102.90550,
      "Pd"  :    106.42,
      "Ag"  :  107.8682, 
      "Cd"  :   112.411,
      "In"  :   114.818,
      "Sn"  :   118.710,
      "Sb"  :   121.760,
      "Te"  :    127.60,
      "I"   : 126.90447,
      "Xe"  :    131.29
   }   
   
   @classmethod
   def mass(cls, label):
      """Function to access the mass_list attribute.
   
      Note that this does not require an instance of the Elements class to be 
      created, as this is a class method. Therefore using Elements.mass(label) 
      will give the mass of the element with the atomic symbol given by label.

      Args:
         label: The atomic symbol of the atom whose mass is required.

      Returns:
         A float giving the mass of the atom with atomic symbol label.
      """

      try:
         return cls.mass_list[label]*Constants.amu
      except KeyError:
         print "Unknown element given, you must specify the mass"
         return -1.0

# thes are the conversion FROM atomic units to the unit stated
UnitMap = { 
   "undefined": {
      ""             : 1.00 
      },      
   "energy":   {
      ""             : 1.00,
      "atomic_unit"  : 1.00, 
      "electronvolt" : 0.036749326,
      "j_mol"       : 0.00038087989,
      "cal_mol"     : 0.0015946679,
      "kelvin"       : 3.1668152e-06
      },
   "temperature":   {
      ""             : 1.00,
      "atomic_unit"  : 1.00, 
      "kelvin"       : 1.00
      },
   "time":     {
      ""             : 1.00,
      "atomic_unit"  : 1.00,
      "second"       : 4.1341373e+16
      },
   "frequency" :   {   # TODO fill up units here 
                       # also, we may or may not need some 2*pi factors here
      ""             : 1.00,
      "atomic_unit"  : 1.00,
      "invcm"        : 8.0685297e-28,
      "hertz"        : 2.4188843e-17
      },     
   "ms-momentum" :   {   # TODO fill up units here (mass-scaled momentum)
      ""             : 1.00,
      "atomic_unit"  : 1.00
      }, 
   "length" :     {
      ""             : 1.00,
      "atomic_unit"  : 1.00,
      "angstrom"     : 1.8897261,
      "meter"        : 1.8897261e+10
      },
   "velocity":    {
      ""            : 1.00,
      "atomic_unit" : 1.00,
      "SI_unit"     : 4.5710289e-7
      },           
   "momentum":    {
      ""             : 1.00,
      "atomic_unit"  : 1.00
      },           
   "mass":        {
      ""             : 1.00,
      "atomic_unit"  : 1.00,
      "dalton"       : 1.00*Constants.amu,
      "electronmass" : 1.00
      },
   "pressure" :     {
      ""             : 1.00,
      "atomic_unit"  : 1.00,
      "bar"          : 3.398827377e-9,
      "atmosphere"   : 3.44386184e-9,
      "pascal"       : 3.398827377e-14
      }
}

# a list of magnitude prefixes
UnitPrefix = {
   "" : 1.0,
   "yotta" : 1e24, "zetta" : 1e21, "exa" : 1e18, "peta" : 1e15,
   "tera" : 1e12, "giga" : 1e9, "mega" : 1e6, "kilo" : 1e3,
   "milli" : 1e-3, "micro" : 1e-6, "nano" : 1e-9, "pico" : 1e-12,
   "femto" : 1e-15, "atto" : 1e-18, "zepto" : 1e-21, "yocto" : 1e-24
}

# builds a RE to match prefix and split out the base unit
UnitPrefixRE=""
for key in UnitPrefix: UnitPrefixRE = UnitPrefixRE+key+"|"
UnitPrefixRE = " *("+UnitPrefixRE[1:-1] + ")(.*) *"
print "KEY DICTIONARY : ", UnitPrefixRE
UnitPrefixRE = re.compile(UnitPrefixRE)

########################################################################
#  Atomic units are used EVERYWHERE internally. In order to quickly    #
#  interface with any "outside" unit, we set up a simple conversion    #
#  library.                                                            # 
########################################################################

def unit_to_internal(family, unit, number):
   """ Converts a number of given dimensions and units into internal units. """
   
   if not (family == "number"  or family in UnitMap):
      raise IndexError(family + " is an undefined units kind.")
   if family == "number":
      return number


   if unit == "":
      prefix=""; base=""
   else:
      m = UnitPrefixRE.match(unit);   
      if m is None : raise ValueError("Unit "+unit+" is not structured with a prefix+base syntax.")   
      prefix = m.group(1); base = m.group(2); 
      
   if not prefix in UnitPrefix:
      raise TypeError(prefix + " is not a valid unit prefix.")      
   if not base in UnitMap[family]:
      raise TypeError(base + " is an undefined unit for kind " + family + ".")
      
   print "converted "+unit+" to "+  str(number*UnitMap[family][base]*UnitPrefix[prefix])
   return number*UnitMap[family][base]*UnitPrefix[prefix]

def unit_to_user(family, unit, number):
   """ Converts a number of given dimensions from internal to user units. """
   
   return number/unit_to_internal(family, unit, 1.0)
