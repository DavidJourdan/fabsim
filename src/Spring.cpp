// Spring.cpp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 04/10/18

#include <fsim/Spring.h>

namespace fsim
{

template <bool allowCompression>
Spring<allowCompression>::Spring(int _i, int _j, double length) : i(_i), j(_j), rest_length(length) {}

template <bool allowCompression>
double Spring<allowCompression>::energy(const Eigen::Ref<const Eigen::VectorXd> pos) const
{
  double diff = (pos.segment<3>(3 * i) - pos.segment<3>(3 * j)).norm() - rest_length;
  if(allowCompression)
    return pow(diff, 2) / 2.;
  else
    return diff > 0 ? pow(diff, 2) / 2. : 0.;
}

template <bool allowCompression>
Eigen::Vector3d Spring<allowCompression>::force(const Eigen::Ref<const Eigen::VectorXd> pos) const
{
  using namespace Eigen;

  Vector3d u = pos.segment<3>(3 * i);
  Vector3d v = pos.segment<3>(3 * j);
  double r = rest_length / (u - v).norm();
  VectorXd f = -(u - v) * (1 - r);
  if(allowCompression)
    return f;
  else
    return r < 1 ? f : Vector3d::Zero();
}

template <bool allowCompression>
Eigen::Matrix3d Spring<allowCompression>::hessian(const Eigen::Ref<const Eigen::VectorXd> pos) const
{
  using namespace Eigen;

  Vector3d u = pos.segment<3>(3 * i);
  Vector3d v = pos.segment<3>(3 * j);
  double ratio = rest_length / (u - v).norm();
  Matrix3d h = -((1 - ratio) * Matrix3d::Identity() + ratio / (u - v).squaredNorm() * (u - v) * (u - v).transpose());
  if(allowCompression)
    return h;
  else
    return ratio < 1 ? h : Matrix3d::Zero();
}

template class Spring<true>;
template class Spring<false>;

} // namespace fsim
