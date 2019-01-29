!=====================================================================
!
!               S p e c f e m 3 D  V e r s i o n  3 . 0
!               ---------------------------------------
!
!     Main historical authors: Dimitri Komatitsch and Jeroen Tromp
!                              CNRS, France
!                       and Princeton University, USA
!                 (there are currently many more authors!)
!                           (c) October 2017
!
! This program is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 3 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License along
! with this program; if not, write to the Free Software Foundation, Inc.,
! 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
!
!=====================================================================

!--------------------------------------------------------------------------------------------------
!
! GLL
!
! based on modified GLL mesh output from mesher
!
! used for iterative inversion procedures
!
!--------------------------------------------------------------------------------------------------

!! 2018-04-28 ktao : add this subroutine to read in anistropic model

  subroutine model_gll_aniso(myrank,nspec)

  use generate_databases_par, only: NGLLX,NGLLY,NGLLZ,FOUR_THIRDS,IMAIN,MAX_STRING_LEN,ATTENUATION

  use create_regions_mesh_ext_par, only: rhostore,kappastore,mustore,rho_vp,rho_vs,qkappa_attenuation_store,qmu_attenuation_store, &
            c11store,c12store,c13store,c14store,c15store,c16store, &
            c22store,c23store,c24store,c25store,c26store,c33store, &
            c34store,c35store,c36store,c44store,c45store,c46store, &
            c55store,c56store,c66store

  implicit none

  integer, intent(in) :: myrank,nspec
  character(len=MAX_STRING_LEN), parameter :: gll_path = 'DATA/GLL'

  ! local parameters
  real, dimension(:,:,:,:),allocatable :: rho_read
  real, dimension(:,:,:,:,:),allocatable :: cijkl_read
  integer :: ier
  character(len=MAX_STRING_LEN) :: prname_lp,filename

  ! user output
  if (myrank == 0) then
    write(IMAIN,*) '     using GLL model from: ',trim(gll_path)
  endif

  ! processors name
  write(prname_lp,'(a,i6.6,a)') trim(gll_path)// '/' //'proc',myrank,'_'

  !------ density
  allocate(rho_read(NGLLX,NGLLY,NGLLZ,nspec),stat=ier)
  if (ier /= 0) stop 'error allocating array rho_read'

  ! user output
  if (myrank == 0) write(IMAIN,*) '     reading in: rho.bin'

  filename = prname_lp(1:len_trim(prname_lp))//'rho.bin'
  open(unit=28,file=trim(filename),status='old',action='read',form='unformatted',iostat=ier)
  if (ier /= 0) then
    print *,'error opening file: ',trim(filename)
    stop 'error reading rho.bin file'
  endif

  read(28) rho_read
  close(28)

  !------  cijkl
  allocate(cijkl_read(21,NGLLX,NGLLY,NGLLZ,nspec),stat=ier)
  if (ier /= 0) stop 'error allocating array cijkl_read'

  ! user output
  if (myrank == 0) write(IMAIN,*) '     reading in: cijkl.bin'
  filename = prname_lp(1:len_trim(prname_lp))//'cijkl.bin'
  open(unit=28,file=trim(filename),status='old',action='read',form='unformatted',iostat=ier)
  if (ier /= 0) then
    print *,'error opening file: ',trim(filename)
    stop 'error reading cijkl.bin file'
  endif

  read(28) cijkl_read
  close(28)
 
  c11store(:,:,:,:) = cijkl_read(1,:,:,:,:)
  c12store(:,:,:,:) = cijkl_read(2,:,:,:,:)
  c13store(:,:,:,:) = cijkl_read(3,:,:,:,:)
  c14store(:,:,:,:) = cijkl_read(4,:,:,:,:)
  c15store(:,:,:,:) = cijkl_read(5,:,:,:,:)
  c16store(:,:,:,:) = cijkl_read(6,:,:,:,:)
  c22store(:,:,:,:) = cijkl_read(7,:,:,:,:)
  c23store(:,:,:,:) = cijkl_read(8,:,:,:,:)
  c24store(:,:,:,:) = cijkl_read(9,:,:,:,:)
  c25store(:,:,:,:) = cijkl_read(10,:,:,:,:)
  c26store(:,:,:,:) = cijkl_read(11,:,:,:,:)
  c33store(:,:,:,:) = cijkl_read(12,:,:,:,:)
  c34store(:,:,:,:) = cijkl_read(13,:,:,:,:)
  c35store(:,:,:,:) = cijkl_read(14,:,:,:,:)
  c36store(:,:,:,:) = cijkl_read(15,:,:,:,:)
  c44store(:,:,:,:) = cijkl_read(16,:,:,:,:)
  c45store(:,:,:,:) = cijkl_read(17,:,:,:,:)
  c46store(:,:,:,:) = cijkl_read(18,:,:,:,:)
  c55store(:,:,:,:) = cijkl_read(19,:,:,:,:)
  c56store(:,:,:,:) = cijkl_read(20,:,:,:,:)
  c66store(:,:,:,:) = cijkl_read(21,:,:,:,:)

  !!! update arrays that will be saved and used in the solver xspecfem3D
  !!! the following part is neccessary if you uncommented something above
  rhostore(:,:,:,:) = rho_read(:,:,:,:)
  !! voigt average (iso-strain)
  !! 9*kappa_voigt = C11+C22+C33 + 2*(C12+C13+C23)
  !! 15*mu_voigt = C11+C22+C33 - (C12+C13+C23) + 3*(C44+C55+C66)
  !! rho_vp = sqrt((kappa+4/3*mu)*rho)
  !! rho_vs = sqrt(mu*rho)
  kappastore = (c11store + c22store + c33store + 2.0*(c12store + c13store + c23store)) / 9.0
  mustore = (c11store + c22store + c33store - (c12store + c13store + c23store) + 3.0*(c44store + c55store + c66store)) / 15.0
  ! used for stacey absorbing boundary
  rho_vp = sqrt(rhostore * (kappastore + 4.0/3.0*mustore))
  rho_vs = sqrt(rhostore * mustore)

  !------ gets attenuation arrays from files
  if (ATTENUATION) then
    ! shear attenuation
    ! user output
    if (myrank == 0) write(IMAIN,*) '     reading in: qmu.bin'

    filename = prname_lp(1:len_trim(prname_lp))//'qmu.bin'
    open(unit=28,file=trim(filename),status='old',action='read',form='unformatted',iostat=ier)
    if (ier /= 0) then
      print *,'Error opening file: ',trim(filename)
      stop 'Error reading qmu.bin file'
    endif

    read(28) qmu_attenuation_store
    close(28)

    ! bulk attenuation
    ! user output
    if (myrank == 0) write(IMAIN,*) '     reading in: qkappa.bin'

    filename = prname_lp(1:len_trim(prname_lp))//'qkappa.bin'
    open(unit=28,file=trim(filename),status='old',action='read',form='unformatted',iostat=ier)
    if (ier /= 0) then
      print *,'error opening file: ',trim(filename)
      stop 'error reading qkappa.bin file'
    endif

    read(28) qkappa_attenuation_store
    close(28)
  endif

  !------ free memory
  deallocate(rho_read,cijkl_read)

  end subroutine

