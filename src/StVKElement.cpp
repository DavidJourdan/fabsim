// StVKElement.cpp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 12/06/21

#include "fsim/StVKElement.h"

namespace fsim
{

StVKElement::StVKElement(const Eigen::Ref<const Mat3<double>> V, const Eigen::Vector3i &E, double thickness)
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

Eigen::Matrix2d StVKElement::strain(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  Matrix<double, 3, 2> Ds;
  Ds.col(0) = X.segment<3>(3 * idx(0)) - X.segment<3>(3 * idx(2));
  Ds.col(1) = X.segment<3>(3 * idx(1)) - X.segment<3>(3 * idx(2));
  Matrix<double, 3, 2> F = Ds * _R;

  return 0.5 * (F.transpose() * F - Matrix2d::Identity());
}

Eigen::Matrix2d StVKElement::stress(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu) const
{
  using namespace Eigen;
  Matrix2d E = strain(X);
  return 2 * mu * E + lambda * E.trace() * Matrix2d::Identity();
}

double StVKElement::energy(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double mass) const
{
  using namespace Eigen;
  Matrix2d E = strain(X);
  return coeff * (mu * (E * E).trace() + lambda / 2 * pow(E.trace(), 2) +
                  9.8 * mass * (X(3 * idx(0) + 2) + X(3 * idx(1) + 2) + X(3 * idx(2) + 2)) / 3);
}

Vec<double, 9>
StVKElement::gradient(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double mass) const
{
  using namespace Eigen;
  Matrix2d S = stress(X, lambda, mu);

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

Mat<double, 9, 9>
StVKElement::hessian(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double mass) const
{
  using namespace Eigen;

  Matrix2d S = stress(X, lambda, mu);
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
