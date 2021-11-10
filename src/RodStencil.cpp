// RodStencil.cpp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 02/21/20

#include "fsim/RodStencil.h"

#include "fsim/ElasticRod.h"

#include <fsim/util/geometry.h>

#include <iostream>

namespace fsim
{

// matrix of material directors
Eigen::Matrix<double, 2, 3> materialMatrix(const LocalFrame &f, double theta)
{
  using namespace Eigen;
  Matrix<double, 2, 3> res;
  res.row(0) = -sin(theta) * f.d1 + cos(theta) * f.d2;
  res.row(1) = -cos(theta) * f.d1 - sin(theta) * f.d2;
  return res;
}

RodStencil::RodStencil(const Eigen::Ref<const Mat3<double>> V,
                       const LocalFrame &f1,
                       const LocalFrame &f2,
                       const Eigen::Matrix<int, 5, 1> &dofs,
                       const Eigen::Vector2d &widths,
                       double young_modulus)
{
  using namespace Eigen;

  idx = dofs;
  Vector3d x0 = V.row(idx(0));
  Vector3d x1 = V.row(idx(1));
  Vector3d x2 = V.row(idx(2));

  _vertex_length = (x1 - x0).norm() + (x2 - x1).norm();
  _stiffnesses << pow(widths(0), 3) * widths(1) / 12., pow(widths(1), 3) * widths(0) / 12.;
  _stiffnesses *= young_modulus;

  Vector3d kb = 2 * f1.t.cross(f2.t) / (1 + f1.t.dot(f2.t));
  _restK = (materialMatrix(f1, 0) + materialMatrix(f2, 0)) / 2 * kb;

  _ref_twist = 0;
}

double RodStencil::mass = 0;

double RodStencil::energy(const Eigen::Ref<const Eigen::VectorXd> X, const LocalFrame &f1, const LocalFrame &f2) const
{
  using namespace Eigen;

  Vector2d k = materialCurvature(X, f1, f2);

  return ((k - _restK).dot(bendMatrix() * (k - _restK)) + twistCoeff() * pow(twistAngle(X), 2)) / (2 * _vertex_length)
    + 9.81 * mass * X(3 * idx(1) + 2) * _vertex_length / 2;
}

RodStencil::LocalVector
RodStencil::gradient(const Eigen::Ref<const Eigen::VectorXd> X, const LocalFrame &f1, const LocalFrame &f2) const
{
  using namespace Eigen;

  Vector2d k = materialCurvature(X, f1, f2);
  auto dK = materialCurvatureDerivative(X, f1, f2);
  auto dT = twistAngleDerivative(X, f1, f2);

  LocalVector res = (dK * bendMatrix() * (k - _restK) + dT * twistCoeff() * twistAngle(X)) / _vertex_length;
  res(5) += 9.81 * mass * _vertex_length / 2;

  return res;
}

RodStencil::LocalMatrix
RodStencil::hessian(const Eigen::Ref<const Eigen::VectorXd> X, const LocalFrame &f1, const LocalFrame &f2) const
{
  using namespace Eigen;

  Matrix<double, 11, 11> ddK, ddT;
  Matrix<double, 11, 2> dK = materialCurvatureDerivative(X, f1, f2, &ddK);
  auto dT = twistAngleDerivative(X, f1, f2, &ddT);

  return (dK * bendMatrix() * dK.transpose() + ddK + twistCoeff() * (dT * dT.transpose() + twistAngle(X) * ddT)) /
         _vertex_length;
}

Eigen::Vector2d RodStencil::materialCurvature(const Eigen::Ref<const Eigen::VectorXd> X,
                                              const LocalFrame &f1,
                                              const LocalFrame &f2) const
{
  using namespace Eigen;

  Vector3d kb = 2 * f1.t.cross(f2.t) / (1 + f1.t.dot(f2.t));
  double theta0, theta1;
  if(idx(0) < idx(1))
    theta0 = X(idx(3));
  else
    theta0 = -X(idx(3));
  if(idx(1) < idx(2))
    theta1 = X(idx(4));
  else
    theta1 = -X(idx(4));
  return (materialMatrix(f1, theta0) + materialMatrix(f2, theta1)) / 2 * kb;
}

Eigen::Matrix<double, 11, 2> RodStencil::materialCurvatureDerivative(const Eigen::Ref<const Eigen::VectorXd> X,
                                                                     const LocalFrame &f1,
                                                                     const LocalFrame &f2,
                                                                     Eigen::Matrix<double, 11, 11> *ddK) const
{
  using namespace Eigen;
  double theta0, theta1;
  if(idx(0) < idx(1))
    theta0 = X(idx(3));
  else
    theta0 = -X(idx(3));
  if(idx(1) < idx(2))
    theta1 = X(idx(4));
  else
    theta1 = -X(idx(4));

  Matrix<double, 2, 3> M1 = materialMatrix(f1, theta0);
  Matrix<double, 2, 3> M2 = materialMatrix(f2, theta1);
  double d = 1 + f1.t.dot(f2.t);
  Matrix<double, 2, 3> M = (M1 + M2) / d;

  Vector3d kb = 2 * f1.t.cross(f2.t) / d;
  Vector2d k = (materialMatrix(f1, theta0) + materialMatrix(f2, theta1)) / 2 * kb;

  Vector3d mean_t = (f1.t + f2.t) / d;

  double l0 = (X.segment<3>(3 * idx(1)) - X.segment<3>(3 * idx(0))).norm();
  double l1 = (X.segment<3>(3 * idx(2)) - X.segment<3>(3 * idx(1))).norm();

  Matrix<double, 11, 2> dK;
  dK.block<3, 2>(0, 0) = (-cross_matrix(f2.t) * M.transpose() + outer_prod(mean_t, k)) / l0;
  dK.block<3, 2>(6, 0) = (-cross_matrix(f1.t) * M.transpose() - outer_prod(mean_t, k)) / l1;

  dK.block<3, 2>(3, 0) = -dK.block<3, 2>(0, 0) - dK.block<3, 2>(6, 0);

  dK.row(9) << kb.dot(M1.row(1)) / 2, -kb.dot(M1.row(0)) / 2;
  dK.row(10) << kb.dot(M2.row(1)) / 2, -kb.dot(M2.row(0)) / 2;

  if(idx(0) > idx(1))
    dK.row(9) *= -1;
  if(idx(1) > idx(2))
    dK.row(10) *= -1;

  if(ddK) // compute second derivative wrt u = bendMatrix * (k - _restK)
  {
    Vector2d u = bendMatrix() * (k - _restK);

    // \frac{ \partial^2 \kappa_i \cdot u }{ \partial x_j \partial x_k }
    Vector3d v = M.transpose() * u;
    double c = u.dot(k);
    Matrix3d A = (2 * c * outer_prod(mean_t, mean_t) - 2 * sym(outer_prod(f2.t.cross(v), mean_t)) -
                  c / d * (Matrix3d::Identity() - outer_prod(f1.t, f1.t)) +
                  0.5 * sym(outer_prod<3, 3>(kb, M1.transpose() * u))) /
                 l0 / l0;
    Matrix3d B = (2 * c * outer_prod(mean_t, mean_t) + 2 * sym(outer_prod(f1.t.cross(v), mean_t)) -
                  c / d * (Matrix3d::Identity() - outer_prod(f2.t, f2.t)) +
                  0.5 * sym(outer_prod<3, 3>(kb, M2.transpose() * u))) /
                 l1 / l1;
    Matrix3d C = (2 * c * outer_prod(mean_t, mean_t) + outer_prod(mean_t, f1.t.cross(v)) -
                  c / d * (Matrix3d::Identity() + outer_prod(f1.t, f2.t)) - outer_prod(f2.t.cross(v), mean_t) -
                  cross_matrix(v)) /
                 l0 / l1;

    // clang-format off
    ddK->block<3,3>(0,0) =  A;
    ddK->block<3,3>(0,3) = -A             + C;
    ddK->block<3,3>(0,6) =                - C;
    ddK->block<3,3>(3,3) =  A + B - 2 * sym(C);
    ddK->block<3,3>(3,6) =     -B         + C;
    ddK->block<3,3>(6,6) =      B;
    // clang-format on
    // \frac{ \partial^2 \kappa_i \cdot u }{ \partial\theta_j \partial\theta_j }
    (*ddK)(9, 9) = -u.dot(M1 * kb) / 2;
    (*ddK)(10, 10) = -u.dot(M2 * kb) / 2;
    (*ddK)(9, 10) = 0;

    // \frac{ \partial^2 \kappa_i \cdot u }{ \partial\theta_{i-1} \partial x_k }
    v = -u(1) * M1.row(0) + u(0) * M1.row(1);
    ddK->block<3, 1>(0, 9) = (-f2.t.cross(v) / d + mean_t * v.dot(kb) / 2) / l0;
    ddK->block<3, 1>(6, 9) = (-f1.t.cross(v) / d - mean_t * v.dot(kb) / 2) / l1;
    ddK->block<3, 1>(3, 9) = -ddK->block<3, 1>(0, 9) - ddK->block<3, 1>(6, 9);

    // \frac{ \partial^2 \kappa_i \cdot u }{ \partial\theta_i \partial\x_k }
    v = -u(1) * M2.row(0) + u(0) * M2.row(1);
    ddK->block<3, 1>(0, 10) = (f2.t.cross(v) / d + v.dot(kb) / 2 * mean_t) / l0;
    ddK->block<3, 1>(6, 10) = (f1.t.cross(v) / d - v.dot(kb) / 2 * mean_t) / l1;
    ddK->block<3, 1>(3, 10) = -ddK->block<3, 1>(0, 10) - ddK->block<3, 1>(6, 10);

    if(idx(0) > idx(1))
      ddK->block<9, 1>(0, 9) *= -1;
    if(idx(1) > idx(2))
      ddK->block<9, 1>(0, 10) *= -1;

    *ddK = ddK->selfadjointView<Upper>();
  }
  return dK;
}

double RodStencil::twistAngle(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  // to measure angles in a consistent way across shared degrees of freedom, we check for the order of vertex indices
  double theta0, theta1;
  if(idx(0) < idx(1))
    theta0 = X(idx(3));
  else
    theta0 = -X(idx(3));
  if(idx(1) < idx(2))
    theta1 = X(idx(4));
  else
    theta1 = -X(idx(4));
  return theta1 - theta0 + _ref_twist;
}

Eigen::Matrix<double, 11, 1> RodStencil::twistAngleDerivative(const Eigen::Ref<const Eigen::VectorXd> X,
                                                              const LocalFrame &f1,
                                                              const LocalFrame &f2,
                                                              Eigen::Matrix<double, 11, 11> *dderiv) const
{
  using namespace Eigen;

  const double d = 1 + f1.t.dot(f2.t);
  Vector3d kb = 2 * f1.t.cross(f2.t) / d;

  const double l0 = (X.segment<3>(3 * idx(1)) - X.segment<3>(3 * idx(0))).norm();
  const double l1 = (X.segment<3>(3 * idx(2)) - X.segment<3>(3 * idx(1))).norm();

  Matrix<double, 11, 1> V;
  V.segment<3>(0) = -0.5 * kb / l0;
  V.segment<3>(6) = 0.5 * kb / l1;
  V.segment<3>(3) = -V.segment<3>(0) - V.segment<3>(6);
  V(9) = idx(0) < idx(1) ? -1 : 1;
  V(10) = idx(1) < idx(2) ? 1 : -1;

  if(dderiv)
  {
    *dderiv = Matrix<double, 11, 11>::Zero();

    Vector3d mean_t = (f1.t + f2.t) / d;

    // $\frac{\partial^2 m}{\partial(e^{i-1})^2}$
    Matrix3d A = -sym(outer_prod<3, 3>(kb, f1.t + mean_t)) / (2 * l0 * l0);
    // $\frac{\partial^2 m}{\partial(e^{i})^2}$
    Matrix3d B = -sym(outer_prod<3, 3>(kb, f2.t + mean_t)) / (2 * l1 * l1);
    // $\frac{\partial^2 m}{\partial e^{i-1}\partial e^i$
    Matrix3d C = (2 * cross_matrix(f1.t) / d - outer_prod<3, 3>(kb, mean_t)) / (2 * l0 * l1);

    // clang-format off
    dderiv->block<3, 3>(0, 0) =  A;
    dderiv->block<3, 3>(0, 3) = -A     + C;
    dderiv->block<3, 3>(0, 6) =         -C;
    dderiv->block<3, 3>(3, 3) =  A + B - C - C.transpose();
    dderiv->block<3, 3>(3, 6) =     -B + C;
    dderiv->block<3, 3>(6, 6) =      B;
    // clang-format on

    *dderiv = dderiv->selfadjointView<Upper>();
  }

  return V;
}

void RodStencil::updateReferenceTwist(const LocalFrame &f1, const LocalFrame &f2)
{
  using namespace Eigen;

  // transport reference frame to next edge
  Vector3d u = parallel_transport(f1.d1, f1.t, f2.t);
  // rotate by current value of reference twist
  u = AngleAxis<double>(_ref_twist, f2.t) * u;
  // compute increment to reference twist to align reference frames
  _ref_twist += signed_angle(u, f2.d1, f2.t);
}

} // namespace fsim
