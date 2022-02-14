// OrthotropicStVKElement.cpp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 10/30/21

#include "fsim/OrthotropicStVKElement.h"

namespace fsim
{

OrthotropicStVKElement::OrthotropicStVKElement(const Eigen::Ref<const Mat2<double>> V,
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

Eigen::Vector3d OrthotropicStVKElement::strain(const Eigen::Ref<const Eigen::VectorXd> X) const
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

Eigen::Vector3d OrthotropicStVKElement::stress(const Eigen::Ref<const Eigen::VectorXd> X,
                                               const Eigen::Matrix3d &_C) const
{
  using namespace Eigen;
  Vector3d E = strain(X);
  return _C * E;
}

double
OrthotropicStVKElement::energy(const Eigen::Ref<const Eigen::VectorXd> X, const Eigen::Matrix3d &_C, double mass) const
{
  using namespace Eigen;
  Vector3d E = strain(X);
  return coeff * (0.5 * E.dot(_C * E) + 9.8 * mass * (X(3 * idx(0) + 2) + X(3 * idx(1) + 2) + X(3 * idx(2) + 2)) / 3);
}

Vec<double, 9> OrthotropicStVKElement::gradient(const Eigen::Ref<const Eigen::VectorXd> X,
                                                const Eigen::Matrix3d &_C,
                                                double mass) const
{
  using namespace Eigen;
  Vector3d S = stress(X, _C);
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

Mat<double, 9, 9>
OrthotropicStVKElement::hessian(const Eigen::Ref<const Eigen::VectorXd> X, const Eigen::Matrix3d &_C, double mass) const
{
  using namespace Eigen;
  Vector3d S = stress(X, _C);
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
