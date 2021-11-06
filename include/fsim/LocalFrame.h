// LocalFrame.h
//
// local frame attached to a rod's edge
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 02/09/20

#pragma once

#include "util/geometry.h"
#include "util/typedefs.h"

#include <Eigen/Dense>

namespace fsim
{

template <typename T = double>
struct LocalFrame
{
  LocalFrame(const Vec3<T> &_t, const Vec3<T> &_d1, const Vec3<T> &_d2) : t(_t), d1(_d1), d2(_d2) {}
  LocalFrame() = default;

  void update(const Vec3<T> &x0, const Vec3<T> &x1)
  {
    using namespace fsim;
    Vec3<T> t_new = (x1 - x0).normalized();
    d1 = parallel_transport(d1, t, t_new);
    d2 = t_new.cross(d1);
    t = t_new;
  }

  Vec3<T> t;
  Vec3<T> d1;
  Vec3<T> d2;
};

template <typename T = double>
LocalFrame<T> update_frame(const LocalFrame<T> &f, const Vec3<T> &x0, const Vec3<T> &x1)
{
  LocalFrame<T> f_updated;
  f_updated.t = (x1 - x0).normalized();
  f_updated.d1 = parallel_transport(f.d1, f.t, f_updated.t);
  f_updated.d2 = f_updated.t.cross(f_updated.d1);
  return f_updated;
}
} // namespace fsim
