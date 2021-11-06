// geometry.cpp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 8/20/19

#include "fsim/util/geometry.h"

namespace fsim
{

double signed_angle(const Eigen::Vector3d &u, const Eigen::Vector3d &v, const Eigen::Vector3d &n)
{
  Eigen::Vector3d w = u.cross(v);
  double a = w.norm(), b = u.dot(v);
  double angle = atan2(a, b);
  if(n.dot(w) < 0)
    return -angle;
  else
    return angle;
}

double sin_angle(const Eigen::Vector3d &u, const Eigen::Vector3d &v, const Eigen::Vector3d &n)
{
  Eigen::Vector3d w = u.cross(v);
  if(w.dot(n) < 0)
    return -w.norm();
  else
    return w.norm();
}

double tan_angle_2(const Eigen::Vector3d &u, const Eigen::Vector3d &v, const Eigen::Vector3d &n)
{
  assert((u + v).norm() > 1e-10);
  if(u.cross(v).dot(n) < 0)
    return -(u - v).norm() / (u + v).norm();
  else
    return +(u - v).norm() / (u + v).norm();
}

Eigen::Matrix3d cross_matrix(const Eigen::Vector3d &a)
{
  Eigen::Matrix3d A;
  // clang-format off
  A <<  0,   -a(2),  a(1),
        a(2),   0,  -a(0),
       -a(1), a(0),    0;
  // clang-format on
  return A;
}

// Equations (3.2) and (3.3) from "A discrete, geometrically exact method for simulating nonlinear,
// elastic or non-elastic beams" by Lestringant et al. (2019)
Eigen::Vector3d parallel_transport(const Eigen::Vector3d &u, const Eigen::Vector3d &t1, const Eigen::Vector3d &t2)
{
  using namespace Eigen;

  double c = t1.dot(t2);
  Vector3d b = t1.cross(t2);
  if(c == -1)
    return -u;
  else if(c > 0.99)
    return c * u + b.cross(u) + (0.5 + (1 - c) / 4 + pow(1 - c, 2) / 8 * (1 + b.squaredNorm() / 4)) * b * b.dot(u);
  else
    return (t2 * (t1 - c * t2).dot(u) + (2 * c * t2 - t1) * u.dot(-c * t1 + t2) + b * b.dot(u)) / (1 - c * c);
}

bool point_in_segment(const Eigen::Vector3d &x, const Eigen::Vector3d &a, const Eigen::Vector3d &b)
{
  if((x - a).cross(b - a).norm() > 1e-6)
    return false;
  double r = (x - a).dot(b - a) / (b - a).squaredNorm();
  return (r >= 0 && r <= 1);
}

} // namespace fsim
