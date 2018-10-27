!///////////////////////////////////////////////////////////////////////////////

subroutine xyz2cube_bounded_hex8(xyz_anchor, xyz, uvw, misloc, flag_inside)
!-mapping a given point in physical space (xyz) to the 
! reference cube (uvw) for a 8-node element,
! and also flag whether the point is inside the cube
! if the point lies outside the element, calculate the bounded (xi,eta,gamma)
! inside or on the surface of the reference unit cube.
!
!-inputs:
! (real) xyz_anchor(3,8): anchor points of the element
! (real) xyz(3): coordinates of the target point
!
!-outputs:
! (real) uvw(3): local coordinates in reference cube
! (real) misloc: location misfit abs(xyz - XYZ(uvw))
! (logical) flag_inside: flag whether the target point locates inside the element

  implicit none

  ! input/output
  real(kind=8), intent(in) :: xyz_anchor(3,8)
  real(kind=8), intent(in) :: xyz(3)

  real(kind=8), intent(out) :: uvw(3)
  real(kind=8), intent(out) :: misloc
  logical, intent(out) :: flag_inside

  ! local variables

  ! number of iterations used to locate point inside one element 
  integer, parameter :: niter = 5

  integer :: iter
  real(kind=8), dimension(3) :: xyzi ! iteratively improved xyz
  real(kind=8), dimension(3,3) :: DuvwDxyz
  real(kind=8), dimension(3) :: dxyz, duvw
  real(kind=8), parameter ::  ZERO=0.d0, ONE=1.d0, MINUS_ONE=-1.d0

  ! initialize 
  uvw = ZERO
  flag_inside = .true.

  ! iteratively update local coordinate uvw to approach the target xyz
  do iter = 1, niter

    ! predicted xyzi and Jacobian for the current uvw
    call jacobian_hex8(xyz_anchor, uvw, xyzi, DuvwDxyz)

    ! compute difference
    dxyz = xyz - xyzi

    ! compute increments
    duvw = matmul(DuvwDxyz, dxyz)

    ! update values
    uvw = uvw + duvw

    ! limit inside the cube
    if (any(uvw < MINUS_ONE .or. uvw > ONE)) then 
      where (uvw < MINUS_ONE) uvw = MINUS_ONE
      where (uvw > ONE) uvw = ONE
      ! set is_inside to false based on the last iteration
      if (iter == niter) then
        flag_inside = .false.
      endif
    endif

  enddo ! do iter_loop = 1,NUM_ITER
  
  ! calculate the predicted position 
  call jacobian_hex8(xyz_anchor, uvw, xyzi, DuvwDxyz)

  ! residual distance from the target point
  misloc = sqrt(sum((xyz-xyzi)**2))

end subroutine xyz2cube_bounded_hex8

!///////////////////////////////////////////////////////////////////////////////

subroutine jacobian_hex8(xyz_anchor, uvw, xyz, DuvwDxyz)
! compute 3D jacobian at a given point for a 8-node element
! map from local coordinate (uvw) to physical position (xyz)
! the shape the element is defined by the anchor points (xyz_anchor)
!
!-input
! xyz_anchor(3,8): xyz of anchor points, the order of the 8 nodes must be from
! subroutine anchor_index_hex8()
!
! uvw(3): local coordinate 
!
!-output
! xyz(3): map uvw to physical space
! DuvwDxyz(3,3): jacobian matrix

  implicit none
  
  ! input/output
  real(kind=8), intent(in) :: xyz_anchor(3,8), uvw(3)
  real(kind=8), intent(out) :: xyz(3), DuvwDxyz(3,3)
  
  ! local variables
  real(kind=8) :: ra1,ra2,rb1,rb2,rc1,rc2
  real(kind=8) :: DxyzDuvw(3,3), jacobian
  ! 3D shape functions and their derivatives at receiver
  real(kind=8) :: shape3D(8)
  real(kind=8) :: dershape3D(8,3)
  
  real(kind=8), parameter :: ONE_EIGHTH = 0.125d0
  real(kind=8), parameter :: ZERO = 0.d0
  real(kind=8), parameter :: ONE = 1.d0
  
  ! recompute jacobian for any (xi,eta,gamma) point, not necessarily a GLL point
  
  ! ***
  ! *** create the 3D shape functions and the Jacobian for an 8-node element
  ! ***
  
  !--- case of an 8-node 3D element (Dhatt-Touzot p. 115)
  
  ! trilinear interpolant
  ra1 = ONE + uvw(1)
  ra2 = ONE - uvw(1)
  rb1 = ONE + uvw(2)
  rb2 = ONE - uvw(2)
  rc1 = ONE + uvw(3)
  rc2 = ONE - uvw(3)
  
  ! interpolation weights at 8 anchor points
  shape3D(1) = ONE_EIGHTH*ra2*rb2*rc2
  shape3D(2) = ONE_EIGHTH*ra1*rb2*rc2
  shape3D(3) = ONE_EIGHTH*ra1*rb1*rc2
  shape3D(4) = ONE_EIGHTH*ra2*rb1*rc2
  shape3D(5) = ONE_EIGHTH*ra2*rb2*rc1
  shape3D(6) = ONE_EIGHTH*ra1*rb2*rc1
  shape3D(7) = ONE_EIGHTH*ra1*rb1*rc1
  shape3D(8) = ONE_EIGHTH*ra2*rb1*rc1
  
  ! derivative of interpolation weights w.r.t uvw
  dershape3D(1,1) = - ONE_EIGHTH*rb2*rc2
  dershape3D(2,1) = ONE_EIGHTH*rb2*rc2
  dershape3D(3,1) = ONE_EIGHTH*rb1*rc2
  dershape3D(4,1) = - ONE_EIGHTH*rb1*rc2
  dershape3D(5,1) = - ONE_EIGHTH*rb2*rc1
  dershape3D(6,1) = ONE_EIGHTH*rb2*rc1
  dershape3D(7,1) = ONE_EIGHTH*rb1*rc1
  dershape3D(8,1) = - ONE_EIGHTH*rb1*rc1
  
  dershape3D(1,2) = - ONE_EIGHTH*ra2*rc2
  dershape3D(2,2) = - ONE_EIGHTH*ra1*rc2
  dershape3D(3,2) = ONE_EIGHTH*ra1*rc2
  dershape3D(4,2) = ONE_EIGHTH*ra2*rc2
  dershape3D(5,2) = - ONE_EIGHTH*ra2*rc1
  dershape3D(6,2) = - ONE_EIGHTH*ra1*rc1
  dershape3D(7,2) = ONE_EIGHTH*ra1*rc1
  dershape3D(8,2) = ONE_EIGHTH*ra2*rc1
  
  dershape3D(1,3) = - ONE_EIGHTH*ra2*rb2
  dershape3D(2,3) = - ONE_EIGHTH*ra1*rb2
  dershape3D(3,3) = - ONE_EIGHTH*ra1*rb1
  dershape3D(4,3) = - ONE_EIGHTH*ra2*rb1
  dershape3D(5,3) = ONE_EIGHTH*ra2*rb2
  dershape3D(6,3) = ONE_EIGHTH*ra1*rb2
  dershape3D(7,3) = ONE_EIGHTH*ra1*rb1
  dershape3D(8,3) = ONE_EIGHTH*ra2*rb1
  
  ! compute coordinates
  xyz = matmul(xyz_anchor, shape3D)
  
  ! compute jacobian: Duvw/Dxyz
  ! Duvw/Dxyz = inv(Dxyz/Duvw) = adj(DxyzDuvw)/det(DxyzDuvw)  
  
  DxyzDuvw = matmul(xyz_anchor, dershape3D)
  
  ! adjoint matrix: adj(Dxyz/Duvw)
  DuvwDxyz(1,1) =   DxyzDuvw(2,2)*DxyzDuvw(3,3)-DxyzDuvw(3,2)*DxyzDuvw(2,3)
  DuvwDxyz(2,1) = -(DxyzDuvw(2,1)*DxyzDuvw(3,3)-DxyzDuvw(3,1)*DxyzDuvw(2,3))
  DuvwDxyz(3,1) =   DxyzDuvw(2,1)*DxyzDuvw(3,2)-DxyzDuvw(3,1)*DxyzDuvw(2,2)
  
  DuvwDxyz(1,2) = -(DxyzDuvw(1,2)*DxyzDuvw(3,3)-DxyzDuvw(3,2)*DxyzDuvw(1,3))
  DuvwDxyz(2,2) =   DxyzDuvw(1,1)*DxyzDuvw(3,3)-DxyzDuvw(3,1)*DxyzDuvw(1,3)
  DuvwDxyz(3,2) = -(DxyzDuvw(1,1)*DxyzDuvw(3,2)-DxyzDuvw(3,1)*DxyzDuvw(1,2))
  
  DuvwDxyz(1,3) =   DxyzDuvw(1,2)*DxyzDuvw(2,3)-DxyzDuvw(2,2)*DxyzDuvw(1,3)
  DuvwDxyz(2,3) = -(DxyzDuvw(1,1)*DxyzDuvw(2,3)-DxyzDuvw(2,1)*DxyzDuvw(1,3))
  DuvwDxyz(3,3) =   DxyzDuvw(1,1)*DxyzDuvw(2,2)-DxyzDuvw(2,1)*DxyzDuvw(1,2)
  
  ! jacobian = det(Dxyz/Duvw)
  jacobian =  DxyzDuvw(1,1)*DuvwDxyz(1,1) &
            + DxyzDuvw(1,2)*DuvwDxyz(2,1) &
            + DxyzDuvw(1,3)*DuvwDxyz(3,1)
  
  if (jacobian <= ZERO) then
    print *, "[ERROR] cube2xyz: 3D Jacobian undefined jacobian=", jacobian
    print *, "xyz_anchor(3,8)=", xyz_anchor
    print *, "uvw(3)=", uvw
    stop
  endif
  
  ! inverse matrix: adj(DxyzDuvw)/det(DxyzDuvw)  
  DuvwDxyz = DuvwDxyz/jacobian

end subroutine jacobian_hex8

!///////////////////////////////////////////////////////////////////////////////

subroutine anchor_index_hex8(NGLLX,NGLLY,NGLLZ,iax,iay,iaz)
! index of the anchor nodes as a 8-node element in a SEM element of NGLLX,NGLLY,NGLLZ
! nodes   
!
!-input
! NGLLX,NGLLY,NGLLZ
!
!-output
! iax,iay,iaz(3): index of the anchor nodes on x,y,z axes of the SEM element

  implicit none

  ! input/output
  integer, intent(in) :: NGLLX,NGLLY,NGLLZ
  integer, intent(out) :: iax(8), iay(8), iaz(8)

  integer :: nx, ny, nz

  nx = NGLLX -1
  ny = NGLLY -1
  nz = NGLLZ -1

  iax = (/0, nx, nx,  0,  0, nx, nx,  0/)
  iay = (/0,  0, ny, ny,  0,  0, ny, ny/)
  iaz = (/0,  0,  0,  0, nz, nz, nz, nz/)
 
end subroutine anchor_index_hex8
