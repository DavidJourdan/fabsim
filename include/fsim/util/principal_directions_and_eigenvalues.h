#pragma once

#include <Eigen/Dense>

/*
 * Find the solutions of a given 2d eigenproblem and project these solutions in the triangle basis X
 */
void principal_directions_and_eigenvalues(
    const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix2d> &eigensolver,
    const Eigen::Matrix<double, 3, 2> &X,
    Eigen::Vector3d &max_dir,
    Eigen::Vector3d &min_dir,
    Eigen::Vector2d &eigs);