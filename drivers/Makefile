# Makefile for the driver tests
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# 
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

.PHONY: all modules clean

NETCDFPATH=/zdata/groups/common/nicpa/2018-feb/XeonE5-2665/netcdf-serial/4.6.0/gnu-7.3.0
NETCDFFPATH=$(HOME)/.local/lib

NETCDFPATH=/usr/local/intelsoft/
NETCDFFPATH=/usr/local/intelsoft/

FLAGS=-g -O3 -Wall
CFLAGS=$(FLAGS)
FFLAGS=$(FLAGS) -ffree-line-length-none -ffixed-line-length-none -Wno-maybe-uninitialized -L$(NETCDFPATH)/lib -I$(NETCDFPATH)/include/ 

MODULES=distance.f90 LJ.f90 NETCDF.f90 SG.f90 pes/pswater.f90 pes/LEPS.f90
PES=pes/zundel.f pes/morse.f pes/qtip4pf.f pes/efield.f90 pes/eckart.f90 pes/utility.f pes/ch52008.f
OBJECTS=$(MODULES:%.f90=%.o) $(PES:%.f=%.o)
FC=ifort
CC=icc
all: driver.x #driver_pure.x

sockets.o: sockets.c
	$(CC) $(CFLAGS) -c -o sockets.o sockets.c

driver.x: $(OBJECTS) sockets.o fsockets.o driver.o | $(OBJECTS)
	$(FC) $(FFLAGS) -o driver.x $^ -lnetcdf  -lnetcdff
	ln -fs ../drivers/driver.x ../bin/i-pi-driver

driver_pure.x: $(OBJECTS) fsockets_pure.o driver.o | $(OBJECTS)
	$(FC) $(FFLAGS) -o driver_pure.x $^  -lnetcdf  -lnetcdff
	#$(FC) $(FFLAGS) -o driver_pure.x $^  -lnetcdf $(NETCDFFPATH)/libnetcdff.a 

%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

%.o: %.f
	$(FC) $(FFLAGS) -c $< -o $@

clean:
	rm -f *.o *.mod *.x */*.mod */*.o ../bin/i-pi-driver
