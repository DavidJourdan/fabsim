// RodCollection.cpp
//
// Implementation of the Discrete Elastic Rods model of Bergou et al. as presented in
// "Discrete Viscous Threads" (https://doi.org/10.1145/1778765.1778853),
// see also  "A discrete, geometrically exact method for simulating nonlinear, elastic or
// non-elastic beams"  (https://hal.archives-ouvertes.fr/hal-02352879v1)
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 01/09/20

#include "fsim/RodCollection.h"
#include "fsim/ElasticRod.h"

namespace fsim
{

template <bool fullHess>
RodCollection<fullHess>::RodCollection(const Eigen::Ref<const Mat3<double>> V,
                                       const std::vector<std::vector<int>> &indices,
                                       const Eigen::Ref<const Mat2<int>> C,
                                       const Eigen::Ref<const Mat3<double>> N,
                                       const std::vector<double> &thicknesses,
                                       const std::vector<double> &widths,
                                       double young_modulus,
                                       double mass,
                                       CrossSection c)
{
  using namespace Eigen;

  this->_mass = mass;

  this->nV = V.rows();
  int nR = indices.size();
  assert(widths.size() == nR && thicknesses.size() == nR);
  MatrixX2i extremal_edges(nR, 2); // rod extremal edge indices
  this->nE = 0;
  for(int i = 0; i < nR; ++i)
  {
    Mat3<double> D1, D2;
    Map<VectorXi> E(const_cast<int *>(indices[i].data()), indices[i].size());
    ElasticRod<>::bishopFrame(V, E, N.row(i), D1, D2);
    for(int j = 0; j < E.size() - 1; ++j)
    {
      this->_frames.emplace_back((V.row(E(j + 1)) - V.row(E(j))).normalized(), D1.row(j), D2.row(j));
      this->_springs.emplace_back(E(j), E(j + 1), (V.row(E(j)) - V.row(E(j + 1))).norm());
    }

    if(c == CrossSection::Circle)
      rodData.emplace_back(pow(thicknesses[i], 3) * widths[i] * 3.1415 * young_modulus / 64,
                           pow(widths[i], 3) * thicknesses[i] * 3.1415 * young_modulus / 64, 
                           3.14159 / 4 * young_modulus * thicknesses[i] * widths[i], E.size() - 2);
    else if(c == CrossSection::Square)
      rodData.emplace_back(pow(thicknesses[i], 3) * widths[i] * young_modulus / 12,
                           pow(widths[i], 3) * thicknesses[i] * young_modulus / 12, 
                           young_modulus * thicknesses[0] * widths[0], E.size() - 2);

    extremal_edges(i, 0) = this->nE;
    for(int j = 1; j < E.size() - 1; ++j)
    {
      Matrix<int, 5, 1> dofs;
      dofs << E(j - 1), E(j), E(j + 1), 3 * V.rows() + this->nE, 3 * V.rows() + this->nE + 1;
      this->_stencils.emplace_back(V, this->_frames[this->nE], this->_frames[this->nE + 1], dofs);
      this->nE += 1;
    }
    extremal_edges(i, 1) = this->nE;
    this->nE += 1;
  }

  for(int k = 0; k < C.rows(); ++k)
  {
    Matrix<int, 5, 1> dofs;
    LocalFrame f1, f2;
    int i = C(k, 0), j = C(k, 1);
    int nI = indices[i].size(), nJ = indices[j].size();
    if(indices[i][0] == indices[j][0])
    {
      dofs << indices[i][1], indices[i][0], indices[j][1], 3 * V.rows() + extremal_edges(i, 0),
          3 * V.rows() + extremal_edges(j, 0);
      f1 = this->_frames[extremal_edges(i, 0)];
      f2 = this->_frames[extremal_edges(j, 0)];
    }
    else if(indices[i][0] == indices[j].back())
    {
      dofs << indices[i][1], indices[i][0], indices[j][nJ - 2], 3 * V.rows() + extremal_edges(i, 0),
          3 * V.rows() + extremal_edges(j, 1);
      f1 = this->_frames[extremal_edges(i, 0)];
      f2 = this->_frames[extremal_edges(j, 1)];
    }
    else if(indices[i].back() == indices[j][0])
    {
      dofs << indices[i][nI - 2], indices[j][0], indices[j][1], 3 * V.rows() + extremal_edges(i, 1),
          3 * V.rows() + extremal_edges(j, 0);
      f1 = this->_frames[extremal_edges(i, 1)];
      f2 = this->_frames[extremal_edges(j, 0)];
    }
    else if(indices[i].back() == indices[j].back())
    {
      dofs << indices[i][nI - 2], indices[i][nI - 1], indices[j][nJ - 2], 3 * V.rows() + extremal_edges(i, 1),
          3 * V.rows() + extremal_edges(j, 1);
      f1 = this->_frames[extremal_edges(i, 1)];
      f2 = this->_frames[extremal_edges(j, 1)];
    }
    else
      throw std::runtime_error("Non connected rods\n");

    if(f1.t.dot(V.row(dofs(1)) - V.row(dofs(0))) < 0)
    {
      f1.t *= -1;
      f1.d2 *= -1;
    }
    if(f2.t.dot(V.row(dofs(2)) - V.row(dofs(1))) < 0)
    {
      f2.t *= -1;
      f2.d2 *= -1;
    }
    this->_stencils.emplace_back(V, f1, f2, dofs);

    double wn = (thicknesses[i] + thicknesses[j]) / 2;
    double wb = (widths[i] + widths[j]) / 2;
    if(c == CrossSection::Circle)
      connectionData.emplace_back(pow(wn, 3) * wb * 3.1415 * young_modulus / 128,
                                  pow(wb, 3) * wn * 3.1415 * young_modulus / 128,
                                  3.1415 / 4 * wb * wn * young_modulus);
    else if(c == CrossSection::Square)
      connectionData.emplace_back(pow(wn, 3) * wb * young_modulus / 24, 
                                  pow(wb, 3) * wn * young_modulus / 24,
                                  wb * wn * young_modulus);
  }

  assert(this->_springs.size() == this->_frames.size());
}

template <bool fullHess>
RodCollection<fullHess>::RodCollection(const Eigen::Ref<const Mat3<double>> V,
                                       const std::vector<std::vector<int>> &indices,
                                       const Eigen::Ref<const Mat2<int>> C,
                                       const Eigen::Ref<const Mat3<double>> N,
                                       const RodParams &p)
  : RodCollection<fullHess>(V, indices, C, N, std::vector<double>(indices.size(), p.thickness), 
    std::vector<double>(indices.size(), p.width), p.E, p.mass, p.crossSection)
{
  rodData.clear();
  if(p.crossSection == CrossSection::Circle)
    rodData.emplace_back(pow(p.thickness, 3) * p.width * 3.1415 * p.E / 64,
                          pow(p.width, 3) * p.thickness * 3.1415 * p.E / 64, 
                          3.1415 / 4 * p.width * p.thickness * p.E,
                          this->_stencils.size() - connectionData.size());
  else if(p.crossSection == CrossSection::Square)
    rodData.emplace_back(pow(p.thickness, 3) * p.width * p.E / 12, 
                          pow(p.width, 3) * p.thickness * p.E / 12,
                          p.width * p.thickness * p.E,
                          this->_stencils.size()- connectionData.size());
}

template <bool fullHess>
double RodCollection<fullHess>::energy(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  // static_cast<const ElasticRod<fullHess>*>(this)->energy(X);

  double result = 0;
  int k = 0;
  for(auto &data: rodData)
  {
    for(int i = k; i < k + std::get<3>(data); ++i)
    {
      auto e = this->_stencils[i]; // make a copy

      // update properties
      LocalFrame f1 = this->getFrame(X, e.idx(0), e.idx(1), e.idx(3));
      f1.update(X.segment<3>(3 * e.idx(0)), X.segment<3>(3 * e.idx(1)));
      LocalFrame f2 = this->getFrame(X, e.idx(1), e.idx(2), e.idx(4));
      f2.update(X.segment<3>(3 * e.idx(1)), X.segment<3>(3 * e.idx(2)));

      e.updateReferenceTwist(f1, f2);

      result += e.energy(X, f1, f2, Vector2d(std::get<0>(data), std::get<1>(data)), std::get<2>(data), this->_mass);
    }
    k += std::get<3>(data);
  }
  for(auto &data: connectionData)
  {
    auto e = this->_stencils[k++]; // make a copy

    // update properties
    LocalFrame f1 = this->getFrame(X, e.idx(0), e.idx(1), e.idx(3));
    f1.update(X.segment<3>(3 * e.idx(0)), X.segment<3>(3 * e.idx(1)));
    LocalFrame f2 = this->getFrame(X, e.idx(1), e.idx(2), e.idx(4));
    f2.update(X.segment<3>(3 * e.idx(1)), X.segment<3>(3 * e.idx(2)));

    e.updateReferenceTwist(f1, f2);

    result += e.energy(X, f1, f2, Vector2d(std::get<0>(data), std::get<1>(data)), std::get<2>(data), this->_mass / 2);
  }
  return result;
}

template <bool fullHess>
void RodCollection<fullHess>::gradient(const Eigen::Ref<const Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> Y) const
{
  using namespace Eigen;

  int k = 0;
  for(auto &data: rodData)
  {
    for(int i = k; i < k + std::get<3>(data); ++i)
    {
      auto &e = this->_stencils[i];
      LocalFrame f1 = this->getFrame(X, e.idx(0), e.idx(1), e.idx(3));
      LocalFrame f2 = this->getFrame(X, e.idx(1), e.idx(2), e.idx(4));
      auto grad = e.gradient(X, f1, f2, Vector2d(std::get<0>(data), std::get<1>(data)), std::get<2>(data), this->_mass);

      for(int j = 0; j < 3; ++j)
        Y.segment<3>(3 * e.idx(j)) += grad.template segment<3>(3 * j);

      Y(e.idx(3)) += grad(9);
      Y(e.idx(4)) += grad(10);
    }
    k += std::get<3>(data);
  }
  for(auto &data: connectionData)
  {
    auto &e = this->_stencils[k++];
    LocalFrame f1 = this->getFrame(X, e.idx(0), e.idx(1), e.idx(3));
    LocalFrame f2 = this->getFrame(X, e.idx(1), e.idx(2), e.idx(4));
    auto grad = e.gradient(X, f1, f2, Vector2d(std::get<0>(data), std::get<1>(data)), std::get<2>(data), this->_mass / 2);

    for(int j = 0; j < 3; ++j)
      Y.segment<3>(3 * e.idx(j)) += grad.template segment<3>(3 * j);

    Y(e.idx(3)) += grad(9);
    Y(e.idx(4)) += grad(10);
  }
}

template <bool fullHess>
Eigen::VectorXd RodCollection<fullHess>::gradient(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  VectorXd Y = VectorXd::Zero(X.size());
  gradient(X, Y);
  return Y;
}

template <bool fullHess>
std::vector<Eigen::Triplet<double>>
RodCollection<fullHess>::hessianTriplets(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  std::vector<Triplet<double>> triplets(77 * this->_stencils.size());
  int k = 0;
  for(auto &data: rodData)
  {
    for(int i = k; i < k + std::get<3>(data); ++i)
    {
      auto &e = this->_stencils[i];
      LocalFrame f1 = this->getFrame(X, e.idx(0), e.idx(1), e.idx(3));
      LocalFrame f2 = this->getFrame(X, e.idx(1), e.idx(2), e.idx(4));
      auto hess = e.hessian(X, f1, f2, Vector2d(std::get<0>(data), std::get<1>(data)), std::get<2>(data), this->_mass);

      int id = 0;
      for(int j = 0; j < 3; ++j)
        for(int k = 0; k < 3; ++k)
          if(e.idx(j) <= e.idx(k))
            for(int l = 0; l < 3; ++l)
              for(int m = 0; m < 3; ++m)
                triplets[77 * i + id++] = Triplet<double>(3 * e.idx(j) + l, 3 * e.idx(k) + m, hess(3 * j + l, 3 * k + m));

      for(int j = 0; j < 2; ++j)
        for(int k = 0; k < 3; ++k)
        {
          if(3 * e.idx(k) < e.idx(3 + j))
            for(int l = 0; l < 3; ++l)
              triplets[77 * i + id++] = Triplet<double>(3 * e.idx(k) + l, e.idx(3 + j), hess(3 * k + l, 9 + j));
          else
            for(int l = 0; l < 3; ++l)
              triplets[77 * i + id++] = Triplet<double>(e.idx(3 + j), 3 * e.idx(k) + l, hess(9 + j, 3 * k + l));
        }

      for(int j = 0; j < 2; ++j)
        for(int k = 0; k < 2; ++k)
          if(e.idx(3 + j) <= e.idx(3 + k))
            triplets[77 * i + id++] = Triplet<double>(e.idx(3 + j), e.idx(3 + k), hess(9 + j, 9 + k));
    }
    k += std::get<3>(data);
  }
  for(auto &data: connectionData)
  {
    auto &e = this->_stencils[k++];
    LocalFrame f1 = this->getFrame(X, e.idx(0), e.idx(1), e.idx(3));
    LocalFrame f2 = this->getFrame(X, e.idx(1), e.idx(2), e.idx(4));
    auto hess = e.hessian(X, f1, f2, Vector2d(std::get<0>(data), std::get<1>(data)), std::get<2>(data),  this->_mass / 2);

    int id = 0;
    for(int j = 0; j < 3; ++j)
      for(int k = 0; k < 3; ++k)
        if(e.idx(j) <= e.idx(k))
          for(int l = 0; l < 3; ++l)
            for(int m = 0; m < 3; ++m)
              triplets[77 * k + id++] = Triplet<double>(3 * e.idx(j) + l, 3 * e.idx(k) + m, hess(3 * j + l, 3 * k + m));

    for(int j = 0; j < 2; ++j)
      for(int k = 0; k < 3; ++k)
      {
        if(3 * e.idx(k) < e.idx(3 + j))
          for(int l = 0; l < 3; ++l)
            triplets[77 * k + id++] = Triplet<double>(3 * e.idx(k) + l, e.idx(3 + j), hess(3 * k + l, 9 + j));
        else
          for(int l = 0; l < 3; ++l)
            triplets[77 * k + id++] = Triplet<double>(e.idx(3 + j), 3 * e.idx(k) + l, hess(9 + j, 3 * k + l));
      }

    for(int j = 0; j < 2; ++j)
      for(int k = 0; k < 2; ++k)
        if(e.idx(3 + j) <= e.idx(3 + k))
          triplets[77 * k + id++] = Triplet<double>(e.idx(3 + j), e.idx(3 + k), hess(9 + j, 9 + k));
  }
  return triplets;
}

template class RodCollection<true>;
template class RodCollection<false>;

} // namespace fsim
