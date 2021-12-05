// OrthotropicStVKElement.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 30/10/21

#pragma once

#include "ElementBase.h"
#include "util/typedefs.h"

namespace fsim
{

template <int id = 0>
class OrthotropicStVKElement : public ElementBase<3>
{
public:
  /**
   * Constructor for the OrthotropicStVKElement class
   * @param V  n by 2 list of vertex positions (each row is a vertex)
   * @param face  list of 3 indices, one per vertex of the triangle
   */
  OrthotropicStVKElement(const Eigen::Ref<const Mat2<double>> V, const Eigen::Vector3i &face, double thickness);

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
  Eigen::Vector3d strain(const Eigen::Ref<const Eigen::VectorXd> X) const;

  /**
   * Computes the Second Piola-Kirchhoff stress tensor S = \frac{\partial f}{\partial E} where E is the Green strain
   * Uses Voigt's notation to express it as a vector
   * @param V  n by 3 list of vertex positions (each row is a vertex)
   * @return second Piola-Kirchhoff stress
   */
  Eigen::Vector3d stress(const Eigen::Ref<const Eigen::VectorXd> X) const;

  double coeff;
  Eigen::Matrix<double, 3, 2> _R;
  static Eigen::Matrix3d _C;
  static double mass;
};

// https://stackoverflow.com/questions/3229883/static-member-initialization-in-a-class-template
template <int id>
Eigen::Matrix3d OrthotropicStVKElement<id>::_C = Eigen::Matrix3d::Zero();

template <int id>
double OrthotropicStVKElement<id>::mass = 0;

template <int id>
OrthotropicStVKElement<id>::OrthotropicStVKElement(const Eigen::Ref<const Mat2<double>> V,
                                                   const Eigen::Vector3i &E,
                                                   double thickness)
{
  using namespace Eigen;

  idx = E;

  _R << V(E(1), 1) - V(E(2), 1), V(E(1), 0) - V(E(2), 0), 
        V(E(2), 1) - V(E(0), 1), V(E(2), 0) - V(E(0), 0),
        V(E(0), 1) - V(E(1), 1), V(E(0), 0) - V(E(1), 0);

  double d = Vector3d(V(E(0), 0), V(E(1), 0), V(E(2), 0)).dot(_R.col(0));

  _R /= d;
  coeff = thickness / 2 * std::abs(d);
}

template <int id>
Eigen::Vector3d OrthotropicStVKElement<id>::strain(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;
  Matrix3d P;
  P.col(0) = X.segment<3>(3 * idx(0));
  P.col(1) = X.segment<3>(3 * idx(1));
  P.col(2) = X.segment<3>(3 * idx(2));
  Matrix<double, 3, 2> F = P * _R;

  Vector3d res;
  res(0) = 0.5 * (F.col(0).dot(F.col(0)) - 1);
  res(1) = 0.5 * (F.col(1).dot(F.col(1)) - 1);
  res(2) = F.col(1).dot(F.col(0));

  return res;
}

template <int id>
Eigen::Vector3d OrthotropicStVKElement<id>::stress(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;
  Vector3d E = strain(X);
  return _C * E;
}

template <int id>
double OrthotropicStVKElement<id>::energy(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;
  Vector3d E = strain(X);
  return coeff * (0.5 * E.dot(_C * E) + 9.8 * mass * (X(3 * idx(0) + 2) + X(3 * idx(1) + 2) + X(3 * idx(2) + 2)) / 3);
}

template <int id>
Vec<double, 9> OrthotropicStVKElement<id>::gradient(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;
  Vector3d S = stress(X);
  Matrix2d SMat = (Matrix2d(2, 2) << S(0), S(2), S(2), S(1)).finished();

  Matrix3d P;
  P.col(0) = X.segment<3>(3 * idx(0));
  P.col(1) = X.segment<3>(3 * idx(1));
  P.col(2) = X.segment<3>(3 * idx(2));
  Matrix<double, 3, 2> F = P * _R;

  Matrix3d grad = coeff * F * (SMat * _R.transpose());
  grad.row(2) += Vector3d::Constant(9.8 * coeff / 3 * mass);
  return Map<Vec<double, 9>>(grad.data(), 9);
}

template <int id>
Mat<double, 9, 9> OrthotropicStVKElement<id>::hessian(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;
  Vector3d S = stress(X);
  Matrix3d P;
  P.col(0) = X.segment<3>(3 * idx(0));
  P.col(1) = X.segment<3>(3 * idx(1));
  P.col(2) = X.segment<3>(3 * idx(2));
  Matrix<double, 3, 2> F = P * _R;

  Matrix3d A = _C(0, 0) * F.col(0) * F.col(0).transpose() + S(0) * Matrix3d::Identity() +
               _C(2, 2) * F.col(1) * F.col(1).transpose();
  Matrix3d B = _C(1, 1) * F.col(1) * F.col(1).transpose() + S(1) * Matrix3d::Identity() +
               _C(2, 2) * F.col(0) * F.col(0).transpose();
  Matrix3d C = _C(0, 1) * F.col(0) * F.col(1).transpose() + S(2) * Matrix3d::Identity() +
               _C(2, 2) * F.col(1) * F.col(0).transpose();

  Matrix<double, 9, 9> hess;

  for(int i = 0; i < 3; ++i)
    for(int j = i; j < 3; ++j)
      hess.block<3, 3>(3 * i, 3 * j) =
          _R(i, 0) * _R(j, 0) * A + _R(i, 1) * _R(j, 1) * B + _R(i, 0) * _R(j, 1) * C + _R(i, 1) * _R(j, 0) * C.transpose();

  hess.block<3, 3>(3, 0) = hess.block<3, 3>(0, 3).transpose();
  hess.block<3, 6>(6, 0) = hess.block<6, 3>(0, 6).transpose();
  return coeff * hess;
}
} // namespace fsim
