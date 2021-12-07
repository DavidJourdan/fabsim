// Spring.cpp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 04/10/18

#include <fsim/Spring.h>

namespace fsim
{

Spring::Spring(int _i, int _j, double length) : i(_i), j(_j), rest_length(length)
{}

double Spring::energy(const Eigen::Ref<const Eigen::VectorXd> pos) const
{
  double diff = (pos.segment<3>(3 * i) - pos.segment<3>(3 * j)).norm() - rest_length;
  return pow(diff, 2) / 2.;
}

Eigen::Vector3d Spring::force(const Eigen::Ref<const Eigen::VectorXd> pos) const
{
  using namespace Eigen;

  Vector3d u = pos.segment<3>(3 * i);
  Vector3d v = pos.segment<3>(3 * j);
  double r = rest_length / (u - v).norm();
  return -(u - v) * (1 - r);
}

Eigen::Matrix3d Spring::hessian(const Eigen::Ref<const Eigen::VectorXd> pos) const
{
  using namespace Eigen;

  Vector3d u = pos.segment<3>(3 * i);
  Vector3d v = pos.segment<3>(3 * j);
  double ratio = rest_length / (u - v).norm();
  return -((1 - ratio) * Matrix3d::Identity() + ratio / (u - v).squaredNorm() * (u - v) * (u - v).transpose());
}

} // namespace fsim
