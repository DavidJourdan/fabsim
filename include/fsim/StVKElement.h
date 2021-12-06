// StVKElement.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 30/10/21

#pragma once

#include "ElementBase.h"
#include "util/typedefs.h"

namespace fsim
{

template <int id = 0>
class StVKElement : public ElementBase<3>
{
public:
  /**
   * Constructor for the StVKElement class
   * @param V  n by 2 list of vertex positions (each row is a vertex)
   * @param face  list of 3 indices, one per vertex of the triangle
   */
  StVKElement(const Eigen::Ref<const Mat3<double>> V, const Eigen::Vector3i &face, double thickness);

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  energy of the triangle element for a given material model
   */
  double energy(const Eigen::Ref<const Eigen::VectorXd> X) const;

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  gradient of the energy (9 by 1 vector), derivatives are stacked in the order of the triangle indices
   */
  LocalVector gradient(const Eigen::Ref<const Eigen::VectorXd> X) const;

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  hessian matrix of the energy w.r.t. all 9 degrees of freedom of the triangle
   */
  LocalMatrix hessian(const Eigen::Ref<const Eigen::VectorXd> X) const;

  /**
   * Computes the Green strain tensor E = \frac 1 2 (F^T F - I)  where F is the deformation gradient
   * Uses Voigt's notation to express it as a vector
   * @param V  n by 3 list of vertex positions (each row is a vertex)
   * @return Green strain
   */
  Eigen::Matrix2d strain(const Eigen::Ref<const Eigen::VectorXd> X) const;

  /**
   * Computes the Second Piola-Kirchhoff stress tensor S = \frac{\partial f}{\partial E} where E is the Green strain
   * Uses Voigt's notation to express it as a vector
   * @param V  n by 3 list of vertex positions (each row is a vertex)
   * @return second Piola-Kirchhoff stress
   */
  Eigen::Matrix2d stress(const Eigen::Ref<const Eigen::VectorXd> X) const;

  static void setParameters(double young_modulus, double poisson_ratio);

  double coeff;
  Eigen::Matrix<double, 2, 2> _R;
  static double mass;
  static double lambda;
  static double mu;
};

// https://stackoverflow.com/questions/3229883/static-member-initialization-in-a-class-template
template <int id>
double StVKElement<id>::lambda = 0;

template <int id>
double StVKElement<id>::mu = 0;

template <int id>
double StVKElement<id>::mass = 0;

template <int id>
void StVKElement<id>::setParameters(double E, double nu)
{
  lambda = E * nu / pow(1 - nu, 2);
  mu = 0.5 * E / (1 + nu);
}

template <int id>
StVKElement<id>::StVKElement(const Eigen::Ref<const Mat3<double>> V, const Eigen::Vector3i &E, double thickness)
{
  using namespace Eigen;

  idx = E;

  Vector3d e1 = V.row(E(0)) - V.row(E(2));
  Vector3d e2 = V.row(E(1)) - V.row(E(2));

  _R.col(0) << e1.squaredNorm(), 0;
  _R.col(1) << e2.dot(e1), e2.cross(e1).norm();
  _R /= e1.norm();
  _R = _R.inverse().eval();

  coeff = thickness / 2 * e1.cross(e2).norm();
}

template <int id>
Eigen::Matrix2d StVKElement<id>::strain(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  Matrix<double, 3, 2> Ds;
  Ds.col(0) = X.segment<3>(3 * idx(0)) - X.segment<3>(3 * idx(2));
  Ds.col(1) = X.segment<3>(3 * idx(1)) - X.segment<3>(3 * idx(2));
  Matrix<double, 3, 2> F = Ds * _R;

  return 0.5 * (F.transpose() * F - Matrix2d::Identity());
}

template <int id>
Eigen::Matrix2d StVKElement<id>::stress(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;
  Matrix2d E = strain(X);
  return 2 * mu * E + lambda * E.trace() * Matrix2d::Identity();
}

template <int id>
double StVKElement<id>::energy(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;
  Matrix2d E = strain(X);
  return coeff * (mu * (E * E).trace() + lambda / 2 * pow(E.trace(), 2) +
                  9.8 * mass * (X(3 * idx(0) + 2) + X(3 * idx(1) + 2) + X(3 * idx(2) + 2)) / 3);
}

template <int id>
Vec<double, 9> StVKElement<id>::gradient(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;
  Matrix2d S = stress(X);

  Matrix<double, 3, 2> Ds;
  Ds.col(0) = X.segment<3>(3 * idx(0)) - X.segment<3>(3 * idx(2));
  Ds.col(1) = X.segment<3>(3 * idx(1)) - X.segment<3>(3 * idx(2));
  Matrix<double, 3, 2> F = Ds * _R;

  Matrix<double, 3, 2> H = coeff * F * (S * _R.transpose());

  Vec<double, 9> grad;
  grad.segment<3>(0) = H.col(0);
  grad.segment<3>(3) = H.col(1);
  grad.segment<3>(6) = -H.col(0) - H.col(1);

  grad(2) += 9.8 * coeff / 3 * mass;
  grad(5) += 9.8 * coeff / 3 * mass;
  grad(8) += 9.8 * coeff / 3 * mass;

  return grad;
}

template <int id>
Mat<double, 9, 9> StVKElement<id>::hessian(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  Matrix2d S = stress(X);
  Matrix<double, 3, 2> Ds;
  Ds.col(0) = X.segment<3>(3 * idx(0)) - X.segment<3>(3 * idx(2));
  Ds.col(1) = X.segment<3>(3 * idx(1)) - X.segment<3>(3 * idx(2));
  Matrix<double, 3, 2> F = Ds * _R;

  Matrix3d A = (lambda + 2 * mu) * F.col(0) * F.col(0).transpose() + S(0, 0) * Matrix3d::Identity() +
               mu * F.col(1) * F.col(1).transpose();
  Matrix3d B = (lambda + 2 * mu) * F.col(1) * F.col(1).transpose() + S(1, 1) * Matrix3d::Identity() +
               mu * F.col(0) * F.col(0).transpose();
  Matrix3d C = lambda * F.col(0) * F.col(1).transpose() + S(0, 1) * Matrix3d::Identity() +
               mu * F.col(1) * F.col(0).transpose();

  Matrix<double, 9, 9> hess;

  for(int i = 0; i < 2; ++i)
    for(int j = i; j < 2; ++j)
      hess.block<3, 3>(3 * i, 3 * j) =
          _R(i, 0) * _R(j, 0) * A + _R(i, 1) * _R(j, 1) * B + _R(i, 0) * _R(j, 1) * C + _R(i, 1) * _R(j, 0) * C.transpose();

  hess.block<3, 3>(3, 0) = hess.block<3, 3>(0, 3).transpose();
  hess.block<6, 3>(0, 6) = -hess.block<6, 3>(0, 0) - hess.block<6, 3>(0, 3);
  hess.block<3, 6>(6, 0) = hess.block<6, 3>(0, 6).transpose();
  hess.block<3, 3>(6, 6) = -hess.block<3, 3>(0, 6) - hess.block<3, 3>(3, 6);
  return coeff * hess;
}
} // namespace fsim
