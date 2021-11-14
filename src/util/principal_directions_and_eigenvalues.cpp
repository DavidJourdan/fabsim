#include "fsim/util/principal_directions_and_eigenvalues.h"

#include <Eigen/Dense>

void principal_directions_and_eigenvalues(
    const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix2d> &eigensolver,
    const Eigen::Matrix<double, 3, 2> &X,
    Eigen::Vector3d &max_dir,
    Eigen::Vector3d &min_dir,
    Eigen::Vector2d &eigs)
{
  Eigen::Matrix<double, 3, 2> dirs = X * eigensolver.eigenvectors();
  Eigen::Vector2d lambda = eigensolver.eigenvalues();

  // order eigenvalues and eigenvectors
  if(lambda(0) < lambda(1))
  {
    min_dir = dirs.col(0);
    max_dir = dirs.col(1);
    eigs = lambda;
  }
  else
  {
    min_dir = dirs.col(1);
    max_dir = dirs.col(0);
    eigs(0) = lambda(1);
    eigs(1) = lambda(0);
  }
}