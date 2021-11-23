// ElasticRod.cpp
//
// Implementation of the Discrete Elastic Rods model of Bergou et al. as presented in
// "Discrete Viscous Threads" (https://doi.org/10.1145/1778765.1778853),
// see also  "A discrete, geometrically exact method for simulating nonlinear, elastic or
// non-elastic beams"  (https://hal.archives-ouvertes.fr/hal-02352879v1)
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 01/09/20

#include "fsim/ElasticRod.h"

namespace fsim
{

template <bool fullHess>
ElasticRod<fullHess>::ElasticRod(const Eigen::Ref<const Mat3<double>> V,
                                 const Eigen::Ref<const Eigen::VectorXi> indices,
                                 const Eigen::Vector3d &n,
                                 const RodParams &p)
    : _mass{p.mass}
{
  using namespace Eigen;

  _stretch = p.E * p.thickness * p.width;
  _stiffness << pow(p.thickness, 3) * p.width, pow(p.width, 3) * p.thickness;
  if(p.crossSection == CrossSection::Circle)
    _stiffness *= 3.1415 * p.E / 64;
  else if(p.crossSection == CrossSection::Square)
    _stiffness *= p.E / 12;

  Map<VectorXi> E(const_cast<int *>(indices.data()), indices.size());
  
  nV = V.rows();
  nDOFs = 3 * nV + E.size() - 1;

  Mat3<double> D1, D2;
  ElasticRod<fullHess>::bishopFrame(V, E, n, D1, D2);
  for(int j = 0; j < E.size() - 1; ++j)
  {
    _frames.emplace_back((V.row(E(j + 1)) - V.row(E(j))).normalized(), D1.row(j), D2.row(j));
    _springs.emplace_back(E(j), E(j + 1), (V.row(E(j)) - V.row(E(j + 1))).norm());
  }

  for(int j = 1; j < E.size() - 1; ++j)
  {
    Matrix<int, 5, 1> dofs;
    dofs << E(j - 1), E(j), E(j + 1), 3 * nV + j - 1, 3 * nV + j;
    _stencils.emplace_back(V, _frames[j - 1], _frames[j], dofs);
  }

  assert(_springs.size() == _frames.size());
}

template <bool fullHess>
ElasticRod<fullHess>::ElasticRod(const Eigen::Ref<const Mat3<double>> V,
                                 const Eigen::Vector3d &n,
                                 const RodParams &params)
    : ElasticRod(V, Eigen::VectorXi::LinSpaced(V.rows(), 0, V.rows() - 1), n, params)
{}

template <bool fullHess>
double ElasticRod<fullHess>::energy(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  double result = 0;
  for(int i = 0; i < _stencils.size(); ++i)
  {
    auto e = _stencils[i]; // make a copy

    // update properties
    LocalFrame f1 = getFrame(X, e.idx(0), e.idx(1), e.idx(3));
    f1.update(X.segment<3>(3 * e.idx(0)), X.segment<3>(3 * e.idx(1)));
    LocalFrame f2 = getFrame(X, e.idx(1), e.idx(2), e.idx(4));
    f2.update(X.segment<3>(3 * e.idx(1)), X.segment<3>(3 * e.idx(2)));

    e.updateReferenceTwist(f1, f2);

    result += e.energy(X, f1, f2, _stiffness, _stretch, _mass);
  }

  return result;
}

template <bool fullHess>
void ElasticRod<fullHess>::gradient(const Eigen::Ref<const Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> Y) const
{
  using namespace Eigen;

  for(auto &e: _stencils)
  {
    LocalFrame f1 = getFrame(X, e.idx(0), e.idx(1), e.idx(3));
    LocalFrame f2 = getFrame(X, e.idx(1), e.idx(2), e.idx(4));
    auto grad = e.gradient(X, f1, f2, _stiffness, _stretch, _mass);

    for(int j = 0; j < 3; ++j)
      Y.segment<3>(3 * e.idx(j)) += grad.template segment<3>(3 * j);

    Y(e.idx(3)) += grad(9);
    Y(e.idx(4)) += grad(10);
  }

}

template <bool fullHess>
Eigen::VectorXd ElasticRod<fullHess>::gradient(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  VectorXd Y = VectorXd::Zero(X.size());
  gradient(X, Y);
  return Y;
}

template <bool fullHess>
Eigen::SparseMatrix<double> ElasticRod<fullHess>::hessian(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  std::vector<Triplet<double>> triplets = hessianTriplets(X);

  SparseMatrix<double> hess(X.size(), X.size());
  hess.setFromTriplets(triplets.begin(), triplets.end());
  return hess;
}

template <bool fullHess>
std::vector<Eigen::Triplet<double>>
ElasticRod<fullHess>::hessianTriplets(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  std::vector<Triplet<double>> triplets(11 * 7 * _stencils.size());

#pragma omp parallel for if(_stencils.size() > 1000)
  for(int i = 0; i < _stencils.size(); ++i)
  {
    auto &e = _stencils[i];
    LocalFrame f1 = getFrame(X, e.idx(0), e.idx(1), e.idx(3));
    LocalFrame f2 = getFrame(X, e.idx(1), e.idx(2), e.idx(4));
    auto hess = e.hessian(X, f1, f2, _stiffness, _stretch, _mass);

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

  return triplets;
}

template <bool fullHess>
void ElasticRod<fullHess>::updateProperties(const Eigen::Ref<const Eigen::VectorXd> X)
{
  int k = 0;
  for(auto &spring: _springs)
  {
    _frames[k++].update(X.segment<3>(3 * spring.i), X.segment<3>(3 * spring.j));
  }

  for(auto &e: _stencils)
  {
    LocalFrame f1 = getFrame(X, e.idx(0), e.idx(1), e.idx(3));
    LocalFrame f2 = getFrame(X, e.idx(1), e.idx(2), e.idx(4));
    e.updateReferenceTwist(f1, f2);
  }
}

template <bool fullHess>
void ElasticRod<fullHess>::setParams(const RodParams &p)
{
  _stretch = 1e3 * p.E * p.thickness * p.width;
  _stiffness << pow(p.thickness, 3) * p.width, pow(p.width, 3) * p.thickness;

  if(p.crossSection == CrossSection::Circle)
    _stiffness *= 3.1415 * p.E / 64;
  else if(p.crossSection == CrossSection::Square)
    _stiffness *= p.E / 12;

  _mass = p.mass;
}

template <bool fullHess>
void ElasticRod<fullHess>::getReferenceDirectors(Mat3<double> &D1, Mat3<double> &D2) const
{
  using namespace Eigen;
  D1.resize(_frames.size(), 3);
  D2.resize(_frames.size(), 3);
  for(int i = 0; i < _frames.size(); ++i)
  {
    D1.row(i) = _frames[i].d1;
    D2.row(i) = _frames[i].d2;
  }
}

template <bool fullHess>
void ElasticRod<fullHess>::getRotatedDirectors(const Eigen::Ref<const Eigen::VectorXd> theta,
                                               Mat3<double> &P1,
                                               Mat3<double> &P2) const
{
  using namespace Eigen;
  P1.resize(_frames.size(), 3);
  P2.resize(_frames.size(), 3);
  for(int i = 0; i < _frames.size(); ++i)
  {
    P1.row(i) = cos(theta(i)) * _frames[i].d1 + sin(theta(i)) * _frames[i].d2;
    P2.row(i) = -sin(theta(i)) * _frames[i].d1 + cos(theta(i)) * _frames[i].d2;
  }
}

template <bool fullHess>
LocalFrame ElasticRod<fullHess>::getFrame(const Eigen::Ref<const Eigen::VectorXd> X, int x0, int x1, int k) const
{
  LocalFrame f = _frames[k - 3 * nV];
  if(f.t.dot(X.segment<3>(3 * x1) - X.segment<3>(3 * x0)) < 0)
  {
    f.t *= -1;
    f.d2 *= -1;
  }
  return f;
}

template <bool fullHess>
void ElasticRod<fullHess>::bishopFrame(const Eigen::Ref<const Mat3<double>> V,
                                       const Eigen::Ref<const Eigen::VectorXi> E,
                                       const Eigen::Vector3d &n,
                                       Mat3<double> &P1,
                                       Mat3<double> &P2)
{
  using namespace Eigen;
  int nE = E.rows() - 1; // number of edges

  P1.resize(nE, 3);
  P2.resize(nE, 3);

  // define first frame
  Vector3d t = (V.row(E(1)) - V.row(E(0))).normalized();
  Vector3d u;
  // if n isn't orthogonal to the first edge, the direction chosen to be the normal will be as close as possible to n
  if(abs(t.dot(n)) < 1e-6)
    u = n.normalized();
  else
    u = (n - n.dot(t) * t).normalized();
  Vector3d v = t.cross(u);
  P1.row(0) = u;
  P2.row(0) = v;

  for(int j = 1; j < nE; ++j)
  {
    u = parallel_transport(u, t, (V.row(E(j + 1)) - V.row(E(j))).normalized());
    t = (V.row(E(j + 1)) - V.row(E(j))).normalized();

    u = (u - u.dot(t) * t).normalized();
    P1.row(j) = u;
    P2.row(j) = t.cross(u);
  }
}

template <bool fullHess>
Mat3<double> ElasticRod<fullHess>::curvatureBinormals(const Eigen::Ref<const Mat3<double>> P,
                                                      const Eigen::Ref<const Eigen::VectorXi> E)
{
  using namespace Eigen;
  Mat3<double> KB(E.size() - 2, 3);

  for(int i = 1; i < E.size() - 1; ++i)
  {
    Vector3d t0 = (P.row(E(i)) - P.row(E(i - 1))).normalized();
    Vector3d t1 = (P.row(E(i + 1)) - P.row(E(i))).normalized();
    KB.row(i - 1) = 2 * t0.cross(t1) / (1 + t0.dot(t1));
  }
  return KB;
}

template class ElasticRod<true>;
template class ElasticRod<false>;

} // namespace fsim
