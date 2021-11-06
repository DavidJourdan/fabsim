// MassSpring.cpp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 03/27/18

#include "fsim/MassSpring.h"

#include "fsim/util/vector_utils.h"

#include <algorithm>

namespace fsim
{

template <bool allow_compression>
MassSpring<allow_compression>::MassSpring(const Eigen::Ref<const Mat3<double>> V,
                                          const Eigen::Ref<const Mat3<int>> F,
                                          double young_modulus,
                                          double stress,
                                          double density)
    : _F(F), _stress(stress), _young_modulus(young_modulus)
{
  using namespace Eigen;
  nV = V.rows();
  assert(F.maxCoeff() == nV - 1);

  // create a list of sorted edge indices
  std::vector<std::pair<int, int>> edges;
  edges.reserve(3 * F.rows());
  for(int i = 0; i < F.rows(); ++i)
  {
    for(int j = 0; j < 3; ++j)
    {
      int k = (j + 1) % 3;
      if(F(i, j) < F(i, k))
        edges.emplace_back(F(i, j), F(i, k));
      else
        edges.emplace_back(F(i, k), F(i, j));
    }
  }
  std::sort(edges.begin(), edges.end());                 // std::unique only works if entries are sorted beforehand
  auto end_it = std::unique(edges.begin(), edges.end()); // remove duplicate edges
  _springs.reserve(end_it - edges.begin());
  for(auto it = edges.begin(); it != end_it; ++it)
  {
    int i = (*it).first;
    int j = (*it).second;
    _springs.emplace_back(i, j, (V.row(i) - V.row(j)).norm());
  }
}

template <bool allow_compression>
double MassSpring<allow_compression>::energy(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  double w = 0.0;
  // Spring
  for(const auto &s: _springs)
    w += _young_modulus * s.energy(X, _stress);

  return w;
}

template <bool allow_compression>
void MassSpring<allow_compression>::gradient(const Eigen::Ref<const Eigen::VectorXd> X,
                                             Eigen::Ref<Eigen::VectorXd> Y) const
{
  using namespace Eigen;
  assert(X.size() == 3 * nV);

  for(const auto &s: _springs)
  {
    Vector3d force = _young_modulus * s.force(X, _stress);
    Y.segment<3>(3 * s.i) -= force;
    Y.segment<3>(3 * s.j) += force;
  }
}

template <bool allow_compression>
std::vector<Eigen::Triplet<double>>
MassSpring<allow_compression>::hessian_triplets_upper(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;
  std::vector<Triplet<double>> triplets;
  triplets.reserve(3 * 9 * _springs.size());
  for(const auto &s: _springs)
  {
    Matrix3d h_ij = _young_modulus * s.hessian(X, _stress);
    // clang-format off
    auto fill = [&triplets](int a, int b, Matrix3d m)
    {
      for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j) triplets.emplace_back(3 * a + i, 3 * b + j, m(i, j));
    };
    // clang-format on
    if(s.i < s.j)
      fill(s.i, s.j, h_ij);
    else
      fill(s.j, s.i, h_ij);
    fill(s.i, s.i, -h_ij);
    fill(s.j, s.j, -h_ij);
  }
  return triplets;
}

template <bool allow_compression>
std::vector<Eigen::Triplet<double>>
MassSpring<allow_compression>::hessian_triplets(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  std::vector<std::vector<Triplet<double>>> local_triplets(_springs.size());

#pragma omp parallel for if(_springs.size() > 100)
  for(int k = 0; k < _springs.size(); ++k)
  {
    auto s = _springs[k];
    Matrix3d h_ij = _young_modulus * s.hessian(X, _stress);

    auto fill = [&](int a, int b, Matrix3d m) {
      for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
          local_triplets[k].emplace_back(3 * a + i, 3 * b + j, m(i, j));
    };

    local_triplets[k].reserve(4 * 9);
    fill(s.i, s.j, h_ij);
    fill(s.j, s.i, h_ij);

    fill(s.i, s.i, -h_ij);
    fill(s.j, s.j, -h_ij);
  }
  return unroll(local_triplets);
}

template <bool allow_compression>
Eigen::SparseMatrix<double> MassSpring<allow_compression>::hessian(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  assert(X.size() == 3 * nV);
  Eigen::SparseMatrix<double> hess(3 * nV, 3 * nV);
  std::vector<Eigen::Triplet<double>> triplets = hessian_triplets(X);
  hess.setFromTriplets(triplets.begin(), triplets.end());
  hess.makeCompressed();
  return hess;
}

// instantiation
template class MassSpring<true>;
template class MassSpring<false>;

} // namespace fsim
