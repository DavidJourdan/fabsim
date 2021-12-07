#include "catch.hpp"
#include "helpers.h"

#include <fsim/MassSpring.h>
#include <fsim/Spring.h>
#include <fsim/util/typedefs.h>

using namespace fsim;

TEST_CASE("Spring")
{
  using namespace Eigen;

  double rest_length = GENERATE(take(5, random(0., 1.)));
  Spring s(0, 1, rest_length);

  SECTION("Translate invariance")
  {
    VectorXd var = GENERATE(take(5, vector_random(6)));

    Vector3d randomDir = GENERATE(take(5, vector_random(3)));
    VectorXd var2 = var;
    for(int i = 0; i < 2; ++i)
      var2.segment<3>(3 * i) = var.segment<3>(3 * i) + randomDir;

    REQUIRE(s.energy(var) == Approx(s.energy(var2)).epsilon(1e-10));
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

    REQUIRE(s.energy(var) == Approx(s.energy(var2)).epsilon(1e-10));
  }
}

TEST_CASE("MassSpring")
{
  using namespace Eigen;

  Mat3<double> V = GENERATE(take(5, matrix_random(3, 3)));
  Mat3<int> F = (Mat3<int>(1, 3) << 0, 1, 2).finished();
  double E = GENERATE(take(5, random(0., 0.5)));
  MassSpring membrane(V, F, E);

  SECTION("Young's modulus")
  {
    double E = GENERATE(take(5, random(0., 1.)));
    membrane.setYoungModulus(E);
    REQUIRE(membrane.getYoungModulus() == E);

    VectorXd var = GENERATE(take(5, vector_random(9)));
    double prev_energy = membrane.energy(var);

    membrane.setYoungModulus(2 * E);

    REQUIRE(membrane.energy(var) == Approx(2 * prev_energy).epsilon(1e-6));
  }
}