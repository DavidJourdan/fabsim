// MassSpring.cpp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 03/27/18

#include "fsim/MassSpring.h"

#include <algorithm>

namespace fsim
{

MassSpring::MassSpring(const Eigen::Ref<const Mat3<double>> V,
                                           const Eigen::Ref<const Mat3<int>> F,
                                           double young_modulus)
    : _young_modulus(young_modulus)
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
  for(auto &pair: edges)
  {
    int i = pair.first;
    int j = pair.second;
    _springs.emplace_back(i, j, (V.row(i) - V.row(j)).norm());
  }
}

double MassSpring::energy(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  double w = 0.0;
  // Spring
  for(const auto &s: _springs)
    w += _young_modulus * s.energy(X);

  return w;
}

void MassSpring::gradient(const Eigen::Ref<const Eigen::VectorXd> X,
                                              Eigen::Ref<Eigen::VectorXd> Y) const
{
  using namespace Eigen;
  assert(X.size() == 3 * nV);

  for(const auto &s: _springs)
  {
    Vector3d force = _young_modulus * s.force(X);
    Y.segment<3>(3 * s.i) -= force;
    Y.segment<3>(3 * s.j) += force;
  }
}

Eigen::VectorXd MassSpring::gradient(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  VectorXd Y = VectorXd::Zero(X.size());
  gradient(X, Y);
  return Y;
}

std::vector<Eigen::Triplet<double>>
MassSpring::hessianTriplets(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  std::vector<Triplet<double>> triplets(_springs.size() * 9 * 3);

#pragma omp parallel for if(_springs.size() > 1000)
  for(int k = 0; k < _springs.size(); ++k)
  {
    auto s = _springs[k];
    Matrix3d h = _young_modulus * s.hessian(X);

    int id = 0;
    if(s.i < s.j)
      for(int a = 0; a < 3; ++a)
        for(int b = 0; b < 3; ++b)
          triplets[27 * k + id++] = Triplet<double>(3 * s.i + a, 3 * s.j + b, h(a, b));
    else
      for(int a = 0; a < 3; ++a)
        for(int b = 0; b < 3; ++b)
          triplets[27 * k + id++] = Triplet<double>(3 * s.j + a, 3 * s.i + b, h(a, b));

    for(int a = 0; a < 3; ++a)
      for(int b = 0; b < 3; ++b)
      {
        triplets[27 * k + id++] = Triplet<double>(3 * s.i + a, 3 * s.i + b, -h(a, b));
        triplets[27 * k + id++] = Triplet<double>(3 * s.j + a, 3 * s.j + b, -h(a, b));
      }
  }
  return triplets;
}

Eigen::SparseMatrix<double> MassSpring::hessian(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  assert(X.size() == 3 * nV);
  Eigen::SparseMatrix<double> hess(3 * nV, 3 * nV);
  std::vector<Eigen::Triplet<double>> triplets = hessianTriplets(X);
  hess.setFromTriplets(triplets.begin(), triplets.end());
  hess.makeCompressed();
  return hess;
}

} // namespace fsim
