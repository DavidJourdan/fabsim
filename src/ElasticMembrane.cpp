// ElasticMembrane.ipp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 12/06/21

#include "fsim/ElasticMembrane.h"

#include <exception>

namespace fsim
{

template <class Element>
ElasticMembrane<Element>::ElasticMembrane(const Eigen::Ref<const Mat3<double>> V,
                                          const Eigen::Ref<const Mat3<int>> F,
                                          double thickness,
                                          double young_modulus,
                                          double poisson_ratio,
                                          double mass)
    : ElasticMembrane(V, F, std::vector<double>(F.rows(), thickness), young_modulus, poisson_ratio, mass)
{
  _thickness = thickness;
}

template <class Element>
ElasticMembrane<Element>::ElasticMembrane(const Eigen::Ref<const Mat3<double>> V,
                                          const Eigen::Ref<const Mat3<int>> F,
                                          const std::vector<double> &thicknesses,
                                          double young_modulus,
                                          double poisson_ratio,
                                          double mass)
    : _E(young_modulus), _nu(poisson_ratio), _mass(mass)
{
  using namespace Eigen;

  nV = V.rows();

  _lambda = _E * _nu / (1 - std::pow(_nu, 2));
  _mu = 0.5 * _E / (1 + _nu);

  int nF = F.rows();
  this->_elements.reserve(nF);
  for(int i = 0; i < nF; ++i)
    this->_elements.emplace_back(V, F.row(i), thicknesses[i]);
}

template <class Element>
double ElasticMembrane<Element>::energy(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  return ModelBase<Element>::energy(X, _lambda, _mu, _mass);
}

template <>
double ElasticMembrane<IncompressibleNeoHookeanElement>::energy(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  return ModelBase<IncompressibleNeoHookeanElement>::energy(X, _mu, _mass);
}

template <class Element>
void ElasticMembrane<Element>::gradient(const Eigen::Ref<const Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> Y) const
{
  ModelBase<Element>::gradient(X, Y, _lambda, _mu, _mass);
}

template <>
void ElasticMembrane<IncompressibleNeoHookeanElement>::gradient(const Eigen::Ref<const Eigen::VectorXd> X,
                                                                Eigen::Ref<Eigen::VectorXd> Y) const
{
  ModelBase<IncompressibleNeoHookeanElement>::gradient(X, Y, _mu, _mass);
}

template <class Element>
Eigen::VectorXd ElasticMembrane<Element>::gradient(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  VectorXd Y = VectorXd::Zero(X.size());
  gradient(X, Y);
  return Y;
}

template <class Element>
Eigen::SparseMatrix<double> ElasticMembrane<Element>::hessian(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  return ModelBase<Element>::hessian(X, _lambda, _mu, _mass);
}

template <>
Eigen::SparseMatrix<double>
ElasticMembrane<IncompressibleNeoHookeanElement>::hessian(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  return ModelBase<IncompressibleNeoHookeanElement>::hessian(X, _mu, _mass);
}

template <class Element>
std::vector<Eigen::Triplet<double>>
ElasticMembrane<Element>::hessianTriplets(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  return ModelBase<Element>::hessianTriplets(X, _lambda, _mu, _mass);
}

template <>
std::vector<Eigen::Triplet<double>>
ElasticMembrane<IncompressibleNeoHookeanElement>::hessianTriplets(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  return ModelBase<IncompressibleNeoHookeanElement>::hessianTriplets(X, _mu, _mass);
}

template <class Element>
void ElasticMembrane<Element>::setPoissonRatio(double poisson_ratio)
{
  _nu = poisson_ratio;
  _lambda = _E * _nu / (1 - std::pow(_nu, 2));
  _mu = 0.5 * _E / (1 + _nu);
}

template <class Element>
void ElasticMembrane<Element>::setYoungModulus(double young_modulus)
{
  _E = young_modulus;
  _lambda = _E * _nu / (1 - std::pow(_nu, 2));
  _mu = 0.5 * _E / (1 + _nu);
}

template <class Element>
void ElasticMembrane<Element>::setMass(double mass)
{
  _mass = mass;
}

template <class Element>
void ElasticMembrane<Element>::setThickness(double t)
{
  if(_thickness <= 0)
    throw std::runtime_error(
        "Warning: membrane may have a locally varying thickness\nCan't set it to a constant value\n");
  for(auto &elem: this->_elements)
    elem.coeff *= t / _thickness;

  _thickness = t;
}

template class ElasticMembrane<StVKElement>;
template class ElasticMembrane<IncompressibleNeoHookeanElement>;
template class ElasticMembrane<NeoHookeanElement>;
} // namespace fsim
