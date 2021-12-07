// OrthotropicStVKMembrane.cpp
//
// StVK version of "Stable Orthotropic Materials" by Li and Barbič (https://doi.org/10.1109/tvcg.2015.2448105)
// Parameterizes the elasticity tensor with 2 Young's moduli and 1 Poisson's ratio
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 10/30/21

#include "fsim/OrthotropicStVKMembrane.h"

#include <array>
#include <iostream>

namespace fsim
{

OrthotropicStVKMembrane::OrthotropicStVKMembrane(const Eigen::Ref<const Mat2<double>> V,
                                                     const Eigen::Ref<const Mat3<int>> F,
                                                     double thickness,
                                                     double E1,
                                                     double E2,
                                                     double poisson_ratio,
                                                     double mass)
    : OrthotropicStVKMembrane(V, F, std::vector<double>(F.rows(), thickness), E1, E2, poisson_ratio, mass)
{
  _thickness = thickness;
}

OrthotropicStVKMembrane::OrthotropicStVKMembrane(const Eigen::Ref<const Mat2<double>> V,
                                                     const Eigen::Ref<const Mat3<int>> F,
                                                     const std::vector<double> &thicknesses,
                                                     double E1,
                                                     double E2,
                                                     double poisson_ratio,
                                                     double mass)
    : _poisson_ratio{poisson_ratio}, _E1{E1}, _E2{E2}, _mass{mass}
{
  using namespace Eigen;

  nV = V.rows();
  int nF = F.rows();
  this->_elements.reserve(nF);
  for(int i = 0; i < nF; ++i)
    this->_elements.emplace_back(V, F.row(i), thicknesses[i]);
}

double OrthotropicStVKMembrane::energy(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  Eigen::Matrix3d C;
  C << _E1, _poisson_ratio * sqrt(_E1 * _E2), 0, 
       _poisson_ratio * sqrt(_E1 * _E2), _E2, 0, 
       0, 0, 0.5 * sqrt(_E1 * _E2) * (1 - _poisson_ratio);
  C /= (1 - std::pow(_poisson_ratio, 2));
  return ModelBase<OrthotropicStVKElement>::energy(X, C, _mass);
}

void OrthotropicStVKMembrane::gradient(const Eigen::Ref<const Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> Y) const
{
  Eigen::Matrix3d C;
  C << _E1, _poisson_ratio * sqrt(_E1 * _E2), 0, 
       _poisson_ratio * sqrt(_E1 * _E2), _E2, 0, 
       0, 0, 0.5 * sqrt(_E1 * _E2) * (1 - _poisson_ratio);
  C /= (1 - std::pow(_poisson_ratio, 2));
  ModelBase<OrthotropicStVKElement>::gradient(X, Y, C, _mass);
}

Eigen::VectorXd OrthotropicStVKMembrane::gradient(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  VectorXd Y = VectorXd::Zero(X.size());
  gradient(X, Y);
  return Y;
}

Eigen::SparseMatrix<double> OrthotropicStVKMembrane::hessian(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  Eigen::Matrix3d C;
  C << _E1, _poisson_ratio * sqrt(_E1 * _E2), 0, 
       _poisson_ratio * sqrt(_E1 * _E2), _E2, 0, 
       0, 0, 0.5 * sqrt(_E1 * _E2) * (1 - _poisson_ratio);
  C /= (1 - std::pow(_poisson_ratio, 2));
  return ModelBase<OrthotropicStVKElement>::hessian(X, C, _mass);
}

std::vector<Eigen::Triplet<double>>
OrthotropicStVKMembrane::hessianTriplets(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  Eigen::Matrix3d C;
  C << _E1, _poisson_ratio * sqrt(_E1 * _E2), 0, 
       _poisson_ratio * sqrt(_E1 * _E2), _E2, 0, 
       0, 0, 0.5 * sqrt(_E1 * _E2) * (1 - _poisson_ratio);
  C /= (1 - std::pow(_poisson_ratio, 2));
  return ModelBase<OrthotropicStVKElement>::hessianTriplets(X, C, _mass);
}


void OrthotropicStVKMembrane::setPoissonRatio(double poisson_ratio)
{
  _poisson_ratio = poisson_ratio;
}

void OrthotropicStVKMembrane::setYoungModuli(double E1, double E2)
{
  _E1 = E1;
  _E2 = E2;
}

void OrthotropicStVKMembrane::setThickness(double t)
{
  if(_thickness <= 0)
    throw std::runtime_error(
        "Warning: membrane may have a locally varying thickness\nCan't set it to a constant value\n");
  for(auto &elem: this->_elements)
  {
    elem.coeff *= t / _thickness;
  }
  _thickness = t;
}

void OrthotropicStVKMembrane::setMass(double mass)
{
  _mass = mass;
}

} // namespace fsim
