// HingeElement.cpp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 01/13/20

#include "fsim/HingeElement.h"

#include "fsim/util/geometry.h"

namespace fsim
{

template <class Formulation, bool fullHess>
HingeElement<Formulation, fullHess>::HingeElement(const Eigen::Ref<const Mat3<double>> V,
                                                  const Eigen::Vector4i &E,
                                                  double coeff)
    : _coeff{coeff}, _hinge(V, E)
{
  idx = E;
}

template <class Formulation, bool fullHess>
double HingeElement<Formulation, fullHess>::energy(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  Map<Mat3<double>> V(const_cast<double *>(X.data()), X.size() / 3, 3);

  Vector3d n0 = (V.row(idx(0)) - V.row(idx(2))).cross(V.row(idx(1)) - V.row(idx(2))).normalized();
  Vector3d n1 = (V.row(idx(1)) - V.row(idx(3))).cross(V.row(idx(0)) - V.row(idx(3))).normalized();
  Vector3d axis = V.row(idx(1)) - V.row(idx(0));
  return _coeff * _hinge.func(n0, n1, axis);
}

template <class Formulation, bool fullHess>
typename HingeElement<Formulation, fullHess>::LocalVector
HingeElement<Formulation, fullHess>::gradient(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;
  Map<Mat3<double>> V(const_cast<double *>(X.data()), X.size() / 3, 3);

  Vector3d n0 = (V.row(idx(0)) - V.row(idx(2))).cross(V.row(idx(1)) - V.row(idx(2))).normalized();
  Vector3d n1 = (V.row(idx(1)) - V.row(idx(3))).cross(V.row(idx(0)) - V.row(idx(3))).normalized();
  Vector3d axis = V.row(idx(1)) - V.row(idx(0));
  return _coeff * _hinge.deriv(n0, n1, axis) * bendAngleGradient(V);
}

template <class Formulation, bool fullHess>
typename HingeElement<Formulation, fullHess>::LocalMatrix
HingeElement<Formulation, fullHess>::hessian(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;
  Map<Mat3<double>> V(const_cast<double *>(X.data()), X.size() / 3, 3);

  LocalMatrix bend_hess;
  LocalVector bend_grad;
  if(fullHess)
    bend_grad = bendAngleGradient(V, &bend_hess);
  else
    bend_grad = bendAngleGradient(V);

  Vector3d n0 = (V.row(idx(0)) - V.row(idx(2))).cross(V.row(idx(1)) - V.row(idx(2))).normalized();
  Vector3d n1 = (V.row(idx(1)) - V.row(idx(3))).cross(V.row(idx(0)) - V.row(idx(3))).normalized();
  Vector3d axis = V.row(idx(1)) - V.row(idx(0));

  double dderiv, deriv;
  deriv = _hinge.deriv(n0, n1, axis, &dderiv);

  LocalMatrix hess = dderiv * bend_grad * bend_grad.transpose();
  if(fullHess)
  {
    hess += deriv * bend_hess;
  }
  return _coeff * hess;
}

template <class Formulation, bool fullHess>
typename HingeElement<Formulation, fullHess>::LocalVector
HingeElement<Formulation, fullHess>::bendAngleGradient(const Eigen::Ref<const Mat3<double>> V,
                                                       LocalMatrix *hessian) const
{
  using namespace Eigen;

  Vector3d e0 = V.row(idx(1)) - V.row(idx(0));
  Vector3d e1 = V.row(idx(2)) - V.row(idx(0));
  Vector3d e2 = V.row(idx(3)) - V.row(idx(0));
  Vector3d e3 = V.row(idx(2)) - V.row(idx(1));
  Vector3d e4 = V.row(idx(3)) - V.row(idx(1));

  Vector3d n0 = e1.cross(e3);
  Vector3d n1 = e4.cross(e2);

  double norm_e0 = normalize(e0);

  LocalVector gradient;
  gradient.segment<3>(0) = -(e0.dot(e3) / n0.squaredNorm() * n0 + e0.dot(e4) / n1.squaredNorm() * n1);
  gradient.segment<3>(3) = (e0.dot(e1) / n0.squaredNorm() * n0 + e0.dot(e2) / n1.squaredNorm() * n1);
  gradient.segment<3>(6) = -norm_e0 / n0.squaredNorm() * n0;
  gradient.segment<3>(9) = -norm_e0 / n1.squaredNorm() * n1;

  if(hessian)
  {
    hessian->setZero();

    double norm_e1 = normalize(e1);
    double norm_e2 = normalize(e2);
    double norm_e3 = normalize(e3);
    double norm_e4 = normalize(e4);
    double norm_n0 = normalize(n0);
    double norm_n1 = normalize(n1);

    double h00 = norm_n0 / norm_e0;
    double h01 = norm_n1 / norm_e0;
    double h1 = norm_n0 / norm_e1;
    double h2 = norm_n1 / norm_e2;
    double h3 = norm_n0 / norm_e3;
    double h4 = norm_n1 / norm_e4;

    double cos1 = e0.dot(e1);
    double cos2 = e0.dot(e2);
    double cos3 = -e0.dot(e3);
    double cos4 = -e0.dot(e4);

    Vector3d m00 = e0.cross(n0);
    Vector3d m01 = -e0.cross(n1);
    Vector3d m1 = -e1.cross(n0);
    Vector3d m3 = e3.cross(n0);
    Vector3d m2 = e2.cross(n1);
    Vector3d m4 = -e4.cross(n1);

    Matrix3d B0 = outer_prod(n0, m00) / pow(norm_e0, 2);

    hessian->block<3, 3>(0, 0) += 2 * sym(cos3 / pow(h3, 2) * outer_prod(m3, n0)) - B0;
    hessian->block<3, 3>(0, 3) += (cos3 * outer_prod(m1, n0) + cos1 * outer_prod(n0, m3)) / h3 / h1 + B0;
    hessian->block<3, 3>(0, 6) += (cos3 * outer_prod(m00, n0) - outer_prod(n0, m3)) / h3 / h00;

    hessian->block<3, 3>(3, 3) += 2 * sym(cos1 / pow(h1, 2) * outer_prod(m1, n0)) - B0;
    hessian->block<3, 3>(3, 6) += (cos1 * outer_prod(m00, n0) - outer_prod(n0, m1)) / h1 / h00;

    hessian->block<3, 3>(6, 6) += -2 * sym(outer_prod(n0, m00) / h00 / h00);

    Matrix3d B1 = outer_prod(n1, m01) / pow(norm_e0, 2);

    hessian->block<3, 3>(0, 0) += 2 * sym(cos4 / pow(h4, 2) * outer_prod(m4, n1)) - B1;
    hessian->block<3, 3>(0, 3) += (cos4 * outer_prod(m2, n1) + cos2 * outer_prod(n1, m4)) / h4 / h2 + B1;
    hessian->block<3, 3>(0, 9) += (cos4 * outer_prod(m01, n1) - outer_prod(n1, m4)) / h4 / h01;

    hessian->block<3, 3>(3, 3) += 2 * sym(cos2 / pow(h2, 2) * outer_prod(m2, n1)) - B1;
    hessian->block<3, 3>(3, 9) += (cos2 * outer_prod(m01, n1) - outer_prod(n1, m2)) / h2 / h01;

    hessian->block<3, 3>(9, 9) += -2 * sym(outer_prod(n1, m01) / h01 / h01);

    *hessian = hessian->selfadjointView<Upper>();
  }
  return gradient;
}

SquaredAngleFormulation::SquaredAngleFormulation(const Eigen::Ref<const Mat3<double>> V, const Eigen::Vector4i &E)
{
  using namespace Eigen;

  Vector3d n0 = (V.row(E(0)) - V.row(E(2))).cross(V.row(E(1)) - V.row(E(2))).normalized();
  Vector3d n1 = (V.row(E(1)) - V.row(E(3))).cross(V.row(E(0)) - V.row(E(3))).normalized();
  Vector3d axis = V.row(E(1)) - V.row(E(0));
  _rest_angle = signed_angle(n0, n1, axis);
}

double
SquaredAngleFormulation::func(const Eigen::Vector3d &n0, const Eigen::Vector3d &n1, const Eigen::Vector3d &axis) const
{
  using namespace Eigen;

  double angle = signed_angle(n0, n1, axis);

  return pow(angle - _rest_angle, 2);
}

double SquaredAngleFormulation::deriv(const Eigen::Vector3d &n0,
                                      const Eigen::Vector3d &n1,
                                      const Eigen::Vector3d &axis,
                                      double *second_deriv) const
{
  using namespace Eigen;

  double angle = signed_angle(n0, n1, axis);
  double res = 2 * (angle - _rest_angle);

  if(second_deriv)
    *second_deriv = 2;

  return res;
}

TanAngleFormulation::TanAngleFormulation(const Eigen::Ref<const Mat3<double>> V, const Eigen::Vector4i &E)
{
  using namespace Eigen;

  Vector3d n0 = (V.row(E(0)) - V.row(E(2))).cross(V.row(E(1)) - V.row(E(2))).normalized();
  Vector3d n1 = (V.row(E(1)) - V.row(E(3))).cross(V.row(E(0)) - V.row(E(3))).normalized();
  assert(n0.norm() > 1e-10 && "Vertices shouldn't be colinear");
  assert(n1.norm() > 1e-10 && "Vertices shouldn't be colinear");
  Vector3d axis = V.row(E(1)) - V.row(E(0));
  _rest_tangent = tan_angle_2(n0, n1, axis);
}

double
TanAngleFormulation::func(const Eigen::Vector3d &n0, const Eigen::Vector3d &n1, const Eigen::Vector3d &axis) const
{
  using namespace Eigen;

  assert(n0.norm() > 1e-10 && "Vertices shouldn't be colinear");
  assert(n1.norm() > 1e-10 && "Vertices shouldn't be colinear");
  double tangent = tan_angle_2(n0, n1, axis);

  return pow(2 * tangent - 2 * _rest_tangent, 2);
}

double TanAngleFormulation::deriv(const Eigen::Vector3d &n0,
                                  const Eigen::Vector3d &n1,
                                  const Eigen::Vector3d &axis,
                                  double *second_deriv) const
{
  using namespace Eigen;

  double tangent = tan_angle_2(n0, n1, axis);
  double sec2 = 4 / (n0 + n1).squaredNorm(); // 1/cos^2(theta/2)
  double res = 2 * sec2 * (2 * tangent - 2 * _rest_tangent);

  if(second_deriv)
    *second_deriv = 2 * sec2 * (sec2 + 2 * tangent * (tangent - _rest_tangent));

  return res;
}

template class HingeElement<SquaredAngleFormulation, false>;
template class HingeElement<SquaredAngleFormulation, true>;
template class HingeElement<TanAngleFormulation, false>;
template class HingeElement<TanAngleFormulation, true>;

} // namespace fsim
