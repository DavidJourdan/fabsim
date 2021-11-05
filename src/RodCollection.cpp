// RodCollection.cpp
//
// Implementation of the Discrete Elastic Rods model of Bergou et al. as presented in
// "Discrete Viscous Threads" (https://doi.org/10.1145/1778765.1778853),
// see also  "A discrete, geometrically exact method for simulating nonlinear, elastic or
// non-elastic beams"  (https://hal.archives-ouvertes.fr/hal-02352879v1)
//
// Please note that, at the moment this implementation assumes a rectangular cross-section whose
// dimensions are given by the normal and binormal widths variables
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 01/09/20

#include "fsim/RodCollection.h"

#include "fsim/util/vector_utils.h"

RodCollection::RodCollection(const Eigen::Ref<const fsim::Mat3<double>> V,
                             const std::vector<std::vector<int>> &indices,
                             const Eigen::MatrixX2i &C,
                             const Eigen::Ref<const fsim::Mat3<double>> N,
                             const std::vector<std::vector<double>> &normal_widths,
                             const std::vector<std::vector<double>> &binormal_widths,
                             double young_modulus,
                             double incompressibility)
{
  using namespace Eigen;

  _stretch_modulus = incompressibility;
  nV = V.rows();
  int nR = indices.size();
  assert(normal_widths.size() == nR && binormal_widths.size() == nR);
  MatrixX2i extremal_edges(nR, 2); // rod extremal edge indices
  int nE = 0;
  for(int i = 0; i < nR; ++i)
  {
    fsim::Mat3<double> D1, D2;
    Map<VectorXi> E(const_cast<int *>(indices[i].data()), indices[i].size());
    ElasticRod::bishop_frame(V, E, N.row(i), D1, D2);
    for(int j = 0; j < E.size() - 1; ++j)
    {
      _frames.emplace_back((V.row(E(j + 1)) - V.row(E(j))).normalized(), D1.row(j), D2.row(j));
      _springs.emplace_back(E(j), E(j + 1), (V.row(E(j)) - V.row(E(j + 1))).norm());
    }
    extremal_edges(i, 0) = nE;
    for(int j = 1; j < E.size() - 1; ++j)
    {
      Matrix<int, 5, 1> dofs;
      dofs << E(j - 1), E(j), E(j + 1), 3 * nV + nE, 3 * nV + nE + 1;
      double wn = (normal_widths[i][j - 1] + normal_widths[i][j]) / 2;
      double wb = (binormal_widths[i][j - 1] + binormal_widths[i][j]) / 2;
      _stencils.emplace_back(V, _frames[nE], _frames[nE + 1], dofs, Vector2d(wn, wb), young_modulus);
      nE += 1;
    }
    extremal_edges(i, 1) = nE;
    nE += 1;
  }

  for(int k = 0; k < C.rows(); ++k)
  {
    Matrix<int, 5, 1> dofs;
    Vector2d widths;
    LocalFrame f1, f2;
    int i = C(k, 0), j = C(k, 1);
    int nI = indices[i].size(), nJ = indices[j].size();
    if(indices[i][0] == indices[j][0])
    {
      widths << (normal_widths[i][0] + normal_widths[j][0]) / 2, (binormal_widths[i][0] + binormal_widths[j][0]) / 2;
      dofs << indices[i][1], indices[i][0], indices[j][1], 3 * nV + extremal_edges(i, 0), 3 * nV + extremal_edges(j, 0);
      f1 = _frames[extremal_edges(i, 0)];
      f2 = _frames[extremal_edges(j, 0)];
    }
    else if(indices[i][0] == indices[j].back())
    {
      widths << (normal_widths[i][0] + normal_widths[j].back()) / 2,
          (binormal_widths[i][0] + binormal_widths[j].back()) / 2;
      dofs << indices[i][1], indices[i][0], indices[j][nJ - 2], 3 * nV + extremal_edges(i, 0),
          3 * nV + extremal_edges(j, 1);
      f1 = _frames[extremal_edges(i, 0)];
      f2 = _frames[extremal_edges(j, 1)];
    }
    else if(indices[i].back() == indices[j][0])
    {
      widths << (normal_widths[i].back() + normal_widths[j][0]) / 2,
          (binormal_widths[i].back() + binormal_widths[j][0]) / 2;
      dofs << indices[i][nI - 2], indices[j][0], indices[j][1], 3 * nV + extremal_edges(i, 1),
          3 * nV + extremal_edges(j, 0);
      f1 = _frames[extremal_edges(i, 1)];
      f2 = _frames[extremal_edges(j, 0)];
    }
    else if(indices[i].back() == indices[j].back())
    {
      widths << (normal_widths[i].back() + normal_widths[j].back()) / 2,
          (binormal_widths[i].back() + binormal_widths[j].back()) / 2;
      dofs << indices[i][nI - 2], indices[i][nI - 1], indices[j][nJ - 2], 3 * nV + extremal_edges(i, 1),
          3 * nV + extremal_edges(j, 1);
      f1 = _frames[extremal_edges(i, 1)];
      f2 = _frames[extremal_edges(j, 1)];
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
    _stencils.emplace_back(V, f1, f2, dofs, widths, young_modulus / 2);
  }

  assert(_springs.size() == _frames.size());
}

RodCollection::RodCollection(const Eigen::Ref<const fsim::Mat3<double>> V,
                             const std::vector<std::vector<int>> &indices,
                             const Eigen::MatrixX2i &C,
                             const Eigen::Ref<const fsim::Mat3<double>> N,
                             const std::vector<double> &W_n,
                             const std::vector<double> &W_b,
                             double young_modulus,
                             double incompressibility)
    : RodCollection(V, indices, C, N, constant(W_n, indices), constant(W_b, indices), young_modulus, incompressibility)
{}

RodCollection::RodCollection(const Eigen::Ref<const fsim::Mat3<double>> V,
                             const std::vector<std::vector<int>> &indices,
                             const Eigen::MatrixX2i &C,
                             const Eigen::Ref<const fsim::Mat3<double>> N,
                             double w_n,
                             double w_b,
                             double young_modulus,
                             double incompressibility)
    : RodCollection(V,
                    indices,
                    C,
                    N,
                    std::vector<double>(indices.size(), w_n),
                    std::vector<double>(indices.size(), w_b),
                    young_modulus,
                    incompressibility)
{}
