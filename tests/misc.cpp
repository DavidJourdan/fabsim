#include "catch.hpp"
#include "helpers.h"

#include "fsim/util/filter_var.h"
#include "fsim/util/geometry.h"

using namespace fsim;

TEST_CASE("filter_var")
{
  using namespace Eigen;

  SECTION("Vector")
  {
    VectorXd X = GENERATE(take(5, vector_random(100)));
    std::vector<int> idx = {1, 6, 18, 23, 67};
    filter_var(X, idx);

    for(int i : idx)
      REQUIRE(X(i) == 0);
  }

  SECTION("Matrix")
  {
    MatrixXd M = GENERATE(take(5, matrix_random(100, 100)));
    std::vector<int> idx = {1, 6, 18, 23, 67};
    filter_var(M, idx);

    for(int i : idx)
    {
      REQUIRE(M(i, i) == 1);
      for(int j = 0; j < 100; ++j)
        if(j != i)
        {
          REQUIRE(M(j, i) == 0);
          REQUIRE(M(i, j) == 0);
        }
    }
  }
}

TEST_CASE("geometry")
{
  using namespace Eigen;

  Vector3d u = GENERATE(take(5, vector_random(3)));
  Vector3d v = GENERATE(take(5, vector_random(3)));

  SECTION("cross_matrix")
  {
    REQUIRE_THAT(cross_matrix(u) * v, ApproxEquals(u.cross(v)));
  }

  SECTION("tan_angle_2")
  {
    u.normalize();
    v.normalize();
    Vector3d w = u.cross(v);
    REQUIRE(tan_angle_2(u, v, w) == Approx(tan(acos(u.dot(v)) / 2)));
  }

  SECTION("sin_angle")
  {
    u.normalize();
    v.normalize();
    Vector3d w = u.cross(v);
    REQUIRE(sin_angle(u, v, w) == Approx(sin(acos(u.dot(v)))));
  }

  SECTION("point_in_segment")
  {
    Vector3d w = (u + v) / 2;
    REQUIRE(point_in_segment(w, u, v));
  }
}

