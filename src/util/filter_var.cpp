// filter_var.cpp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 05/21/19

#include "fsim/util/filter_var.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace fsim
{

void filter_var(Eigen::Ref<Eigen::VectorXd> X, const std::vector<int> &indices)
{
  for(auto index: indices)
  {
    X(index) = 0;
  }
}

void filter_var(Eigen::SparseMatrix<double> &M, const std::vector<int> &indices)
{
  if(!std::is_sorted(indices.begin(), indices.end()))
    throw(std::invalid_argument("indices are not sorted"));

  // clang-format off
  M.prune([&indices](const int row, const int col, const double val)
  {
    // clang-format on
    return !std::binary_search(indices.begin(), indices.end(), row) &&
           !std::binary_search(indices.begin(), indices.end(), col);
  });
  M.reserve(indices.size());
  for(int i: indices)
  {
    M.coeffRef(i, i) = 1;
  }
  M.makeCompressed();
}

} // namespace fsim
