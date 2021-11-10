// LocalFrame.h
//
// local frame attached to a rod's edge
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 02/09/20

#pragma once

#include <Eigen/Dense>

namespace fsim
{

struct LocalFrame
{
  LocalFrame(const Eigen::Vector3d &_t, const Eigen::Vector3d &_d1, const Eigen::Vector3d &_d2);
  LocalFrame() = default;

  void update(const Eigen::Vector3d &x0, const Eigen::Vector3d &x1);

  Eigen::Vector3d t;
  Eigen::Vector3d d1;
  Eigen::Vector3d d2;
};

LocalFrame updateFrame(const LocalFrame &f, const Eigen::Vector3d &x0, const Eigen::Vector3d &x1);

} // namespace fsim