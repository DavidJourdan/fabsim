// Spring.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 04/10/18

#pragma once

#include <Eigen/Dense>

template <bool allow_compression>
struct Spring
{
  int i, j;
  double rest_length;

  Spring(int _i, int _j, double length) : i(_i), j(_j), rest_length(length) {}

  double energy(const Eigen::Ref<const Eigen::VectorXd> pos, double stretch = 1.0) const
  {
    double diff = (pos.segment<3>(3 * i) - pos.segment<3>(3 * j)).norm() - rest_length / stretch;
    if(allow_compression)
      return pow(diff, 2) / 2.;
    else
      return diff > 0 ? pow(diff, 2) / 2. : 0.;
  }

  Eigen::Vector3d force(const Eigen::Ref<const Eigen::VectorXd> pos, double stretch = 1.0) const
  {
    using namespace Eigen;

    Vector3d u = pos.segment<3>(3 * i);
    Vector3d v = pos.segment<3>(3 * j);
    double r = rest_length / (u - v).norm() / stretch;
    VectorXd f = -(u - v) * (1 - r);
    if(allow_compression)
      return f;
    else
      return r < 1 ? f : Vector3d::Zero();
  }

  Eigen::Matrix3d hessian(const Eigen::Ref<const Eigen::VectorXd> pos, double stretch = 1.0) const
  {
    using namespace Eigen;

    Vector3d u = pos.segment<3>(3 * i);
    Vector3d v = pos.segment<3>(3 * j);
    double ratio = rest_length / (u - v).norm() / stretch;
    Matrix3d h = -((1 - ratio) * Matrix3d::Identity() + ratio / (u - v).squaredNorm() * (u - v) * (u - v).transpose());
    if(allow_compression)
      return h;
    else
      return ratio < 1 ? h : Matrix3d::Zero();
  }
};
