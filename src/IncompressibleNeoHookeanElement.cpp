// IncompressibleNeoHookeanElement.cpp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 10/30/21

#include "fsim/IncompressibleNeoHookeanElement.h"
#include "fsim/util/geometry.h"

#include <cmath>

namespace fsim
{

double IncompressibleNeoHookeanElement::k = 1e4;

IncompressibleNeoHookeanElement::IncompressibleNeoHookeanElement(const Eigen::Ref<const Mat3<double>> V,
                                                                 const Eigen::Vector3i &E,
                                                                 double thickness)
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

Eigen::Matrix<double, 3, 2>
IncompressibleNeoHookeanElement::deformationGradient(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  Matrix<double, 3, 2> Ds;
  Ds.col(0) = X.segment<3>(3 * idx(0)) - X.segment<3>(3 * idx(2));
  Ds.col(1) = X.segment<3>(3 * idx(1)) - X.segment<3>(3 * idx(2));
  Matrix<double, 3, 2> F = Ds * _R;

  return F;
}

Eigen::Matrix2d IncompressibleNeoHookeanElement::stress(const Eigen::Ref<const Eigen::VectorXd> X, double mu) const
{
  using namespace Eigen;

  Matrix<double, 3, 2> F = deformationGradient(X);
  Matrix2d C = F.transpose() * F;
  Matrix2d Cinv = C.inverse();
  double J = sqrt(C.determinant());

  return mu * ((Matrix2d::Identity() - 0.5 * C.trace() * Cinv) / J + k * J * (J - 1) * Cinv);
}

Eigen::Matrix2d IncompressibleNeoHookeanElement::stress(const Eigen::Matrix2d &Cinv, double mu) const
{
  using namespace Eigen;

  double detC = 1 / Cinv.determinant();
  double J = sqrt(detC);
  double trC = Cinv.trace() * detC;

  return mu * ((Matrix2d::Identity() - 0.5 * trC * Cinv) / J + k * J * (J - 1) * Cinv);
}

/**
 * All the formula for the elasticity tensors in this library (among other things) are taken from
 * "Nonlinear continuum mechanics for finite element analysis" by J. Bonet and R. D. Wood
 * However, the formulas for the case of _nearly_ incompressible neohookean materials the equations should be modified
 * in eq. 5.53a, III should be replaced by \mathcal{I}, and eq. 5.53b shoud read:
 * \mathcal{C}_p = k J(2 J - 1) C^{-1} \bigotimes C^{-1} - 2 p J \mathcal{I}
 */
Eigen::Matrix3d IncompressibleNeoHookeanElement::elasticityTensor(const Eigen::Matrix2d &Cinv, double mu) const
{
  using namespace Eigen;

  double detC = 1 / Cinv.determinant();
  double J = sqrt(detC);
  double trC = Cinv.trace() * detC;

  Matrix3d _C;
  _C << Cinv(0, 0) * Cinv(0, 0), Cinv(0, 1) * Cinv(0, 1), Cinv(0, 0) * Cinv(0, 1), 
        Cinv(1, 0) * Cinv(1, 0), Cinv(1, 1) * Cinv(1, 1), Cinv(0, 1) * Cinv(1, 1), 
        Cinv(0, 1) * Cinv(0, 0), Cinv(0, 1) * Cinv(1, 1), (Cinv(0, 0) * Cinv(1, 1) + Cinv(0, 1) * Cinv(0, 1)) / 2;
  _C *= mu / J * trC - 2 * mu * k * J * (J - 1);

  Vector3d CinvVec(Cinv(0, 0), Cinv(1, 1), Cinv(0, 1));
  _C += (mu / J * trC / 2 + mu * k * J * (2 * J - 1)) * outer_prod(CinvVec, CinvVec);
  _C += -2 * mu / J * sym(outer_prod(Vector3d(1, 1, 0), CinvVec));

  return _C;
}

double IncompressibleNeoHookeanElement::energy(const Eigen::Ref<const Eigen::VectorXd> X, double mu, double mass) const
{
  using namespace Eigen;

  Matrix<double, 3, 2> F = deformationGradient(X);
  Matrix2d C = F.transpose() * F;
  double J = sqrt(C.determinant());

  return coeff * (mu / 2 * (C.trace() / J - 2 + k * pow(J - 1, 2)) +
                  9.8 * mass * (X(3 * idx(0) + 2) + X(3 * idx(1) + 2) + X(3 * idx(2) + 2)) / 3);
}

Vec<double, 9>
IncompressibleNeoHookeanElement::gradient(const Eigen::Ref<const Eigen::VectorXd> X, double mu, double mass) const
{
  using namespace Eigen;

  Matrix<double, 3, 2> F = deformationGradient(X);
  Matrix2d Cinv = (F.transpose() * F).inverse();
  Matrix2d S = stress(Cinv, mu);

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
IncompressibleNeoHookeanElement::hessian(const Eigen::Ref<const Eigen::VectorXd> X, double mu, double mass) const
{
  using namespace Eigen;

  Matrix<double, 3, 2> F = deformationGradient(X);

  Matrix2d Cinv = (F.transpose() * F).inverse();
  Matrix2d S = stress(Cinv, mu);
  Matrix3d _C = elasticityTensor(Cinv, mu);

  Matrix3d A = _C(0, 0) * F.col(0) * F.col(0).transpose() + S(0, 0) * Matrix3d::Identity() +
               _C(2, 2) * F.col(1) * F.col(1).transpose() + 2 * _C(0, 2) * sym(F.col(0) * F.col(1).transpose());
  Matrix3d B = _C(1, 1) * F.col(1) * F.col(1).transpose() + S(1, 1) * Matrix3d::Identity() +
               _C(2, 2) * F.col(0) * F.col(0).transpose() + 2 * _C(1, 2) * sym(F.col(0) * F.col(1).transpose());
  Matrix3d C = _C(0, 1) * F.col(0) * F.col(1).transpose() + S(0, 1) * Matrix3d::Identity() +
               _C(2, 2) * F.col(1) * F.col(0).transpose() + _C(0, 2) * F.col(0) * F.col(0).transpose() +
               _C(1, 2) * F.col(1) * F.col(1).transpose();

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
