      MODULE NETCDFIO
      USE NETCDF
      IMPLICIT NONE

      CONTAINS
         SUBROUTINE READ_NETCDF_VAR2D(FILE_NAME,VAR_NAME,VAR_DATA)
            IMPLICIT NONE
            integer ncid,varid,numAtts,numDims,i
            integer, allocatable :: dimlen(:)
            integer, dimension(2) :: dimIDss
            !    NATOMS: THE NUMBER OF ATOMS IN THE SYSTEM.
            CHARACTER(LEN=*), INTENT(IN) :: FILE_NAME,VAR_NAME
            double precision, ALLOCATABLE, INTENT(OUT) :: VAR_DATA(:,:)

            ! Open the file. NF90_NOWRITE tells netCDF we want read-only access to 
            ! the file.
            call check( nf90_open(FILE_NAME, NF90_NOWRITE, ncid) )
            ! Get the varid of the variable DynMat, based on its name.
            call check( nf90_inq_varid(ncid, VAR_NAME, varid) )
            ! Get the dimensions of the variable
            call check(nf90_inquire_variable(ncid,varid,ndims = numDims, natts = numAtts))
            if (numDims /= 2) then
                    print *, "this works only for reading 2-dimensional array!"
                    stop
            else
                    allocate(dimlen(numDims))
            end if

            call check(nf90_inquire_variable(ncid, varid, dimids = dimIDss))

            do i=1,2
                call check(nf90_inquire_dimension(ncid, dimIDss(i), len = dimlen(i)))
            enddo
            !print *, "the first dimension of the array:", dimlen(1)
            !print *, "the second dimension of the array:", dimlen(2)
            allocate(VAR_DATA(dimlen(1),dimlen(2)))

            ! Read the data.
            call check( nf90_get_var(ncid, varid, VAR_DATA) )
            ! Check the data.
            !print *, VAR_DATA
            deallocate(dimlen)
            ! Close the file, freeing all resources.
            call check( nf90_close(ncid) )
            !print *,"*** SUCCESS reading file ", FILE_NAME, "! "
         END SUBROUTINE READ_NETCDF_VAR2D


         SUBROUTINE READ_NETCDF_VAR1D(FILE_NAME,VAR_NAME,VAR_DATA)
            IMPLICIT NONE
            integer ncid,varid,numAtts,numDims,i
            integer :: dimlen
            integer :: dimIDss(1)
            !    NATOMS: THE NUMBER OF ATOMS IN THE SYSTEM.
            CHARACTER(LEN=*), INTENT(IN) :: FILE_NAME,VAR_NAME
            !double precision, ALLOCATABLE, INTENT(OUT) :: VAR_DATA(:)
            integer, ALLOCATABLE, INTENT(OUT) :: VAR_DATA(:)

            ! Open the file. NF90_NOWRITE tells netCDF we want read-only access to 
            ! the file.
            call check( nf90_open(FILE_NAME, NF90_NOWRITE, ncid) )
            ! Get the varid of the variable DynMat, based on its name.
            call check( nf90_inq_varid(ncid, VAR_NAME, varid) )
            ! Get the dimensions of the variable
            call check(nf90_inquire_variable(ncid,varid,ndims = numDims, natts = numAtts))
            if (numDims /= 1) then
                    print *, "this works only for reading 1-dimensional array!"
                    stop
            end if

            call check(nf90_inquire_variable(ncid, varid, dimids = dimIDss))
            call check(nf90_inquire_dimension(ncid, dimIDss(1), len = dimlen))
            !print *, "the first dimension of the array:", dimlen
            allocate(VAR_DATA(dimlen))

            ! Read the data.
            call check( nf90_get_var(ncid, varid, VAR_DATA) )
            ! Check the data.
            !print *, VAR_DATA
            ! Close the file, freeing all resources.
            call check( nf90_close(ncid) )
            !print *,"*** SUCCESS reading file ", FILE_NAME, "! "
         END SUBROUTINE READ_NETCDF_VAR1D


         SUBROUTINE READ_NETCDF_VAR1DR(FILE_NAME,VAR_NAME,VAR_DATA)
            IMPLICIT NONE
            integer ncid,varid,numAtts,numDims,i
            integer :: dimlen
            integer :: dimIDss(1)
            !    NATOMS: THE NUMBER OF ATOMS IN THE SYSTEM.
            CHARACTER(LEN=*), INTENT(IN) :: FILE_NAME,VAR_NAME
            !double precision, ALLOCATABLE, INTENT(OUT) :: VAR_DATA(:)
            DOUBLE PRECISION, ALLOCATABLE, INTENT(OUT) :: VAR_DATA(:)

            ! Open the file. NF90_NOWRITE tells netCDF we want read-only access to 
            ! the file.
            call check( nf90_open(FILE_NAME, NF90_NOWRITE, ncid) )
            ! Get the varid of the variable DynMat, based on its name.
            call check( nf90_inq_varid(ncid, VAR_NAME, varid) )
            ! Get the dimensions of the variable
            call check(nf90_inquire_variable(ncid,varid,ndims = numDims, natts = numAtts))
            if (numDims /= 1) then
                    print *, "this works only for reading 1-dimensional array!"
                    stop
            end if

            call check(nf90_inquire_variable(ncid, varid, dimids = dimIDss))
            call check(nf90_inquire_dimension(ncid, dimIDss(1), len = dimlen))
            !print *, "the first dimension of the array:", dimlen
            allocate(VAR_DATA(dimlen))

            ! Read the data.
            call check( nf90_get_var(ncid, varid, VAR_DATA) )
            ! Check the data.
            !print *, VAR_DATA
            ! Close the file, freeing all resources.
            call check( nf90_close(ncid) )
            !print *,"*** SUCCESS reading file ", FILE_NAME, "! "
         END SUBROUTINE READ_NETCDF_VAR1DR


         SUBROUTINE CHECK(STATUS)
             IMPLICIT NONE
             INTEGER, INTENT ( IN) :: STATUS
             IF(STATUS /= NF90_NOERR) THEN 
               PRINT *, TRIM(NF90_STRERROR(STATUS))
               STOP "STOPPED"
             END IF
         END SUBROUTINE CHECK  


         subroutine write_matrix(a)
            double precision, dimension(:,:) :: a
            integer i,j
            write(*,*)
            
            do i = lbound(a,1), ubound(a,1)
               write(*,*) (a(i,j), j = lbound(a,2), ubound(a,2))
            end do
         end subroutine write_matrix
      END MODULE NETCDFIO



!      PROGRAM main 
!      use netcdfio
!      implicit none
!
!      DOUBLE PRECISION, allocatable :: dynmat(:,:)
!      double precision, allocatable :: atoms0(:,:)
!      integer, allocatable :: DynAtoms(:),dynind(:)
!      
!      character(len=*),parameter :: ncfn="EPH.nc"
!      integer :: i,j,n,ndat,dims(2)
!
!
!      double precision, allocatable :: atomslist(:)
!
!      !readin atoms0
!      CALL READ_NETCDF_VAR2D(ncfn,"XYZEq",atoms0)
!      !call write_matrix(transpose(atoms0))
!      dims = shape(atoms0)
!      !print*, dims
!      !atomslist=reshape(atoms0,[dims(1)*dims(2)])
!      !call write_matrix(reshape(atomslist,(/dims(2),dims(1)/),order=(/ 2, 1 /)))
!      !print*, reshape(reshape(atomslist,(/dims(2),dims(1)/),order=(/ 1, 2 /)),[dims(1)*dims(2)])
!      !readin the dynamical matrix
!      CALL READ_NETCDF_VAR2D(ncfn,"DynMat",DynMat)
!      !readin the dynamical atoms
!      CALL READ_NETCDF_VAR1D(ncfn,"DynamicAtoms",DynAtoms)
!      ndat = size(DynAtoms)
!      !note that the dynamical atom index are python index,
!      !counting from 0, we change to fortran index
!      do i=1,ndat
!        DynAtoms(i)= DynAtoms(i)+1
!      enddo
!
!      print *, dynatoms
!
!      allocate(dynind(3*ndat))
!
!      do i=1,ndat
!        n = dynatoms(i)
!        do j=1,3
!           dynind((i-1)*3+j) = (n-1)*3+j
!        enddo
!      enddo
!      print *, dynind
!
!      deallocate(dynind)
!
!
!      contains
!
!      subroutine write_matrix(a)
!         double precision, dimension(:,:) :: a
!         write(*,*)
!         
!         do i = lbound(a,1), ubound(a,1)
!            write(*,*) (a(i,j), j = lbound(a,2), ubound(a,2))
!         end do
!      end subroutine write_matrix
!
!
!      END PROGRAM main
