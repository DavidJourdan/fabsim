#include "fsim/LocalFrame.h"

#include "fsim/util/geometry.h"

namespace fsim
{

LocalFrame::LocalFrame(const Eigen::Vector3d &_t, const Eigen::Vector3d &_d1, const Eigen::Vector3d &_d2)
    : t(_t), d1(_d1), d2(_d2)
{}

void LocalFrame::update(const Eigen::Vector3d &x0, const Eigen::Vector3d &x1)
{
  using namespace fsim;
  Eigen::Vector3d t_new = (x1 - x0).normalized();
  d1 = parallel_transport(d1, t, t_new);
  d2 = t_new.cross(d1);
  t = t_new;
}

LocalFrame updateFrame(const LocalFrame &f, const Eigen::Vector3d &x0, const Eigen::Vector3d &x1)
{
  LocalFrame f_updated;
  f_updated.t = (x1 - x0).normalized();
  f_updated.d1 = parallel_transport(f.d1, f.t, f_updated.t);
  f_updated.d2 = f_updated.t.cross(f_updated.d1);
  return f_updated;
}
} // namespace fsim