import numpy, math

def compute_ih(h):
   """Inverts a 3*3 (upper-triangular) cell matrix"""
   ih = numpy.zeros((3,3), float)
   for i in range(3):
      ih[i,i] = 1.0/h[i,i]
   ih[0,1] = -ih[0,0]*h[0,1]*ih[1,1]
   ih[1,2] = -ih[1,1]*h[1,2]*ih[2,2]
   ih[0,2] = -ih[1,2]*h[0,1]*ih[0,0]-ih[0,0]*h[0,2]*ih[2,2]
   return ih

def compute_eigp(p):
   """Finds the eigenvector matrix of a 3*3 upper-triangular matrix"""
   eigp = numpy.zeros((3,3), float)
   eigvals = numpy.zeros(3, float)

   for i in range(3):
      eigp[i,i] = 1
   eigp[0,1] = -p[0,1]/(p[0,0] - p[1,1])
   eigp[1,2] = -p[1,2]/(p[1,1] - p[2,2])
   eigp[0,2] = -(p[0,1]*p[1,2] - p[0,2]*p[1,1] + p[0,2]*p[2,2])/((p[0,0] - p[2,2])*(p[2,2] - p[1,1]))

   for i in range(3):
      eigvals[i] = p[i,i]
   return eigp, eigvals

def volume(h):
   """Calculates the volume of the unit cell, assuming an upper-triangular
      unit vector matrix"""
   #return numpy.inner(self.h[0:3,0], numpy.cross(self.h[0:3,1], self.h[0:3,2])) # general matrix
   return h[0,0]*h[1,1]*h[2,2]   # upper-triangular matrix