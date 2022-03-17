! Copyright (c) 2021-2022 Miguel A. Caro and Patricia Hernández-León

module vertices_module

contains

subroutine max_vol_vertices(n_points, map_ind, dist_points, N, ind_anchor)

  implicit none

  real*8, intent(in) :: dist_points(:, :)
  integer, intent(in) :: n_points, map_ind(:), N
  integer, intent(out) :: ind_anchor(1:4)

  real*8 :: dist_vert(1:4, 1:4), A(1:5, 1:5), CM_det, V, tol = 1.d-10, V_opt
  integer :: i, vert(1:4), j, k, l, i2, j2, k2, l2

  if( n_points /= size(map_ind) )then
    write(*,*)"ERROR: The size of map_ind must be equal to n_points."
    return
  end if

  V_opt = 0.d0

  do i = 1, n_points
    i2 = map_ind(i) + 1
    do j = i+1, n_points
      j2 = map_ind(j) + 1
      do k = j+1, n_points
        k2 = map_ind(k) + 1
        do l = k+1, n_points
          l2 = map_ind(l) + 1
          vert(1:4) = [i2,j2,k2,l2]
          call get_dist_vertices(dist_points, vert, N, dist_vert)

          A(1:4, 1:4) = dist_vert(1:4, 1:4)
          A(5, :) = 1.d0
          A(:, 5) = 1.d0
          A(5, 5) = 0.d0

          CM_det = FindDet(A, 5)
          if( abs(CM_det) > tol )then
            V = sqrt(abs(CM_det)/2.d0**(N-1)) / factorial(N-1)
            if( V >= V_opt )then
              V_opt = V
              ind_anchor(1:4) = vert(1:4) - 1
            end if
          end if
        end do
      end do
    end do
  end do

end subroutine


! This implementation has a limited precision (4bytes int)
! so the result is wrong when n > 12
function factorial(n)

  implicit none

  integer :: n
  integer :: factorial
  integer :: i

  factorial = 1
  do i = 1, n
    factorial = factorial * i
  end do

end function



subroutine get_dist_vertices(dist_points, vert, N, dist_vertices)

  implicit none

  integer, intent(in) :: N
  real*8, intent(in) :: dist_points(:, :)
  integer, intent(in) :: vert(1:N)
  real*8, intent(out) :: dist_vertices(1:N, 1:N)

  integer :: i, j, i2, j2

  do i = 1, N
    i2 = vert(i)
    do j = 1, N
      j2 = vert(j)
      dist_vertices(i, j) = dist_points(i2, j2)
    end do
  end do

end subroutine



! This I copied from the Internet. We can use it for now. When merging
! with the
! repo, we should write our own det() function based on LAPACK routines
REAL*8 FUNCTION FindDet(matrix, n)
    IMPLICIT NONE
    REAL*8, DIMENSION(n,n) :: matrix
    INTEGER, INTENT(IN) :: n
    REAL*8 :: m, temp
    INTEGER :: i, j, k, l
    LOGICAL :: DetExists = .TRUE.
    l = 1
    !Convert to upper triangular form
    DO k = 1, n-1
        IF (matrix(k,k) == 0) THEN
            DetExists = .FALSE.
            DO i = k+1, n
                IF (matrix(i,k) /= 0) THEN
                    DO j = 1, n
                        temp = matrix(i,j)
                        matrix(i,j)= matrix(k,j)
                        matrix(k,j) = temp
                    END DO
                    DetExists = .TRUE.
                    l=-l
                    EXIT
                ENDIF
            END DO
            IF (DetExists .EQV. .FALSE.) THEN
                FindDet = 0
                return
            END IF
        ENDIF
        DO j = k+1, n
            m = matrix(j,k)/matrix(k,k)
            DO i = k+1, n
                matrix(j,i) = matrix(j,i) - m*matrix(k,i)
            END DO
        END DO
    END DO
   
    !Calculate determinant by finding product of diagonal elements
    FindDet = l
    DO i = 1, n
        FindDet = FindDet * matrix(i,i)
    END DO
   
END FUNCTION FindDet

end module

