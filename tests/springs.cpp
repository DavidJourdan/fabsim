#include "catch.hpp"
#include "helpers.h"

#include <fsim/Spring.h>

using namespace fsim;

TEST_CASE("Spring")
{
  using namespace Eigen;

  double rest_length = GENERATE(take(5, random(0., 1.)));
  Spring<> e(0, 1, rest_length);

  SECTION("Translate invariance")
  {
    VectorXd var = GENERATE(take(5, vector_random(6)));

    Vector3d randomDir = GENERATE(take(5, vector_random(3)));
    VectorXd var2 = var;
    for(int i = 0; i < 2; ++i)
      var2.segment<3>(3 * i) = var.segment<3>(3 * i) + randomDir;

    REQUIRE(e.energy(var) == Approx(e.energy(var2)).epsilon(1e-10));
  }

  SECTION("Rotation invariance")
  {
    VectorXd var = GENERATE(take(5, vector_random(6)));

    Vector3d axis = GENERATE(take(5, vector_random(3))).normalized();
    double angle = GENERATE(take(5, random(0., M_PI)));
    AngleAxisd rotation(angle, axis);

    VectorXd var2 = var;
    for(int i = 0; i < 2; ++i)
      var2.segment<3>(3 * i) = rotation * var.segment<3>(3 * i);

    REQUIRE(e.energy(var) == Approx(e.energy(var2)).epsilon(1e-10));
  }
}
