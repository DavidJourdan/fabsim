// filter_var.h
// Filters out variables that need to stay fixed during the simulation/optimization
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 05/21/19

#pragma once

#include <Eigen/Dense>
#include <Eigen/SparseCore>

namespace fsim
{

/**
 * removes the entries of the specified indices from X (sets them to 0)
 * useful to remove variables (or fix degrees of freedom) from a linear system
 * @param X  list of variables
 * @param indices  list of indices such that if i is in indices, then X(i) = 0
 */
template <typename DerivedX>
void filter_var(Eigen::PlainObjectBase<DerivedX> &X, const std::vector<int> &indices)
{
  assert(X.cols() == 1);
  for(auto index: indices)
  {
    X(index) = 0;
  }
}

/**
 * removes the entries of the specified indices from M (sets lines and columns to 0, diagonal to 1)
 * useful to remove variables (or fix degrees of freedom) from a linear system
 * @param M  sparse matrix
 * @param indices  list of indices such that if i is in indices, then X(i) = 0
 */
template <typename scalar>
void filter_var(Eigen::SparseMatrix<scalar> &M, const std::vector<int> &indices);

} // namespace fsim