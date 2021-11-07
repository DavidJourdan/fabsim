// Spring.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 04/10/18

#pragma once

#include <Eigen/Dense>

namespace fsim
{

template <bool allowCompression = true>
struct Spring
{
  int i, j;
  double rest_length;

  Spring(int _i, int _j, double length);

  double energy(const Eigen::Ref<const Eigen::VectorXd> pos) const;
  Eigen::Vector3d force(const Eigen::Ref<const Eigen::VectorXd> pos) const;
  Eigen::Matrix3d hessian(const Eigen::Ref<const Eigen::VectorXd> pos) const;
};

} // namespace fsim
