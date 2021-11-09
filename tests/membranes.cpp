#include "catch.hpp"
#include "helpers.h"

#include <fsim/ElasticMembraneModel.h>
#include <fsim/TriangleElement.h>
#include <fsim/OrthotropicStVKModel.h>

using namespace fsim;

TEMPLATE_TEST_CASE("TriangleElement", "", StVKElement<>, NeoHookeanElement<>, NHIncompressibleElement<>)
{
  using namespace Eigen;

  Mat3<double> V = GENERATE(take(2, matrix_random(3, 3)));
  double thickness = GENERATE(take(2, random(0., 1.)));
  double young_modulus = GENERATE(take(2, random(0., 1.)));
  double mass = GENERATE(take(2, random(0., 1.)));
  double poisson_ratio = GENERATE(take(2, random(0., 0.5)));
  TestType::nu = poisson_ratio;
  TestType::E = young_modulus;
  TestType::mass = mass;
  TestType e(V, Vector3i(0, 1, 2), thickness);

  SECTION("Translate invariance") { translate_invariance(e); }
  SECTION("Rotation invariance") { rotational_invariance(e); }
}

TEST_CASE("StVKMembrane class")
{
  using namespace Eigen;

  Mat3<double> V = GENERATE(take(2, matrix_random(3, 3)));
  Mat3<int> F = (Mat3<int>(1, 3) << 0, 1, 2).finished();

  StVKMembraneModel<0> instance0(V, F, 0.1, 10, 0.3);
  StVKMembraneModel<1> instance1(V, F, 0.9, 100, 0.4);

  SECTION("getPoissonRatio") { REQUIRE(instance0.getPoissonRatio() != instance1.getPoissonRatio()); }
  SECTION("getYoungModulus") { REQUIRE(instance0.getYoungModulus() != instance1.getYoungModulus()); }
  SECTION("getThickness") { REQUIRE(instance1.getThickness() == 0.9); }
  SECTION("setThickness")
  {
    instance0.setThickness(0.3);
    REQUIRE(instance0.getThickness() == 0.3);
  }
}

TEST_CASE("NeoHookeanMembrane class")
{
  using namespace Eigen;

  Mat3<double> V = GENERATE(take(2, matrix_random(3, 3)));
  Mat3<int> F = (Mat3<int>(1, 3) << 0, 1, 2).finished();

  NeoHookeanMembraneModel<0> instance0(V, F, 0.1, 10, 0.3);
  NeoHookeanMembraneModel<1> instance1(V, F, 0.9, 100, 0.4);

  SECTION("getThickness") { REQUIRE(instance1.getThickness() == 0.9); }
  SECTION("getPoissonRatio") { REQUIRE(instance0.getPoissonRatio() != instance1.getPoissonRatio()); }
  SECTION("getYoungModulus") { REQUIRE(instance0.getYoungModulus() != instance1.getYoungModulus()); }
  SECTION("setThickness")
  {
    instance0.setThickness(0.3);
    REQUIRE(instance0.getThickness() == 0.3);
  }
}

TEST_CASE("NHIncompressibleMembrane class")
{
  using namespace Eigen;

  Mat3<double> V = GENERATE(take(2, matrix_random(3, 3)));
  Mat3<int> F = (Mat3<int>(1, 3) << 0, 1, 2).finished();

  NHIncompressibleMembraneModel<0> instance0(V, F, 0.1, 10, 0.3);
  NHIncompressibleMembraneModel<1> instance1(V, F, 0.9, 100, 0.4);

  SECTION("getThickness") { REQUIRE(instance1.getThickness() == 0.9); }
  SECTION("getPoissonRatio") { REQUIRE(instance0.getPoissonRatio() != instance1.getPoissonRatio()); }
  SECTION("getYoungModulus") { REQUIRE(instance0.getYoungModulus() != instance1.getYoungModulus()); }
  SECTION("setThickness")
  {
    instance0.setThickness(0.3);
    REQUIRE(instance0.getThickness() == 0.3);
  }
}

TEST_CASE("OrthotropicStVKMembrane class")
{
  using namespace Eigen;

  Mat2<double> V = GENERATE(take(2, matrix_random(3, 2)));
  Mat3<int> F = (Mat3<int>(1, 3) << 0, 1, 2).finished();

  OrthotropicStVKModel<0> instance0(V, F, 0.1, 10, 5, 0.3);
  OrthotropicStVKModel<1> instance1(V, F, 0.9, 100, 50, 0.4);

  SECTION("getThickness") { REQUIRE(instance1.getThickness() == 0.9); }
  SECTION("getPoissonRatio") { REQUIRE(instance0.getPoissonRatio() != instance1.getPoissonRatio()); }
  SECTION("setThickness")
  {
    instance0.setThickness(0.3);
    REQUIRE(instance0.getThickness() == 0.3);
  }
}

TEST_CASE("Small strain equivalence")
{
  using namespace Eigen;

  Mat3<double> V = GENERATE(take(2, matrix_random(3, 3)));

  double young_modulus = GENERATE(take(2, random(1e10, 1e11)));
  double poisson_ratio = GENERATE(take(2, random(0., 0.5)));
  double thickness = GENERATE(take(2, random(0., 1.)));

  StVKElement<> e1(V, Vector3i(0, 1, 2), thickness);
  NeoHookeanElement<> e2(V, Vector3i(0, 1, 2), thickness);

  StVKElement<>::nu = poisson_ratio;
  StVKElement<>::E = young_modulus;
  NeoHookeanElement<>::nu = poisson_ratio;
  NeoHookeanElement<>::E = young_modulus;

  VectorXd var = 1e-5 * GENERATE(take(2, vector_random(9)));
  var.segment<3>(0) += V.row(0);
  var.segment<3>(3) += V.row(1);
  var.segment<3>(6) += V.row(2);

  // doesn't need to be super precise, just check that they are approximately the same
  REQUIRE(e1.energy(var) == Approx(e2.energy(var)).epsilon(1e-2));
}

TEST_CASE("OrthotropicStVKElement")
{
  using namespace Eigen;

  double poisson_ratio = GENERATE(take(2, random(0., 0.5)));
  double thickness = GENERATE(take(2, random(0., 1.)));
  MatrixX2d V = GENERATE(take(2, matrix_random(3, 2)));

  double E1 = GENERATE(take(2, random(0., 1.)));
  double E2 = GENERATE(take(2, random(0., 1.)));
  OrthotropicStVKElement<>::_C << E1, poisson_ratio * sqrt(E1 * E2), 0,
                                  poisson_ratio * sqrt(E1 * E2), E2, 0,
                                  0, 0, 0.5 * sqrt(E1 * E2) * (1 - poisson_ratio);
  OrthotropicStVKElement<>::_C /= (1 - std::pow(poisson_ratio, 2));
  OrthotropicStVKElement<> e(V, Vector3i(0, 1, 2), thickness);

  SECTION("Energy")
  {
    VectorXd var = VectorXd::Zero(9);
    var.segment<2>(0) = V.row(0);
    var.segment<2>(3) = V.row(1);
    var.segment<2>(6) = V.row(2);

    REQUIRE(e.energy(var) == Approx(0.).margin(1e-10));
  }

  SECTION("Isotropic Equivalence")
  {
    Matrix3d V1(3, 3);
    V1 << V, VectorXd::Zero(9);
    StVKElement<> e1(V1, Vector3i(0, 1, 2), thickness);
    StVKElement<>::nu = poisson_ratio;
    StVKElement<>::E = 1;

    double E1 = 1;
    double E2 = 1;
    OrthotropicStVKElement<>::_C << E1, poisson_ratio * sqrt(E1 * E2), 0,
                                   poisson_ratio * sqrt(E1 * E2), E2, 0,
                                   0, 0, 0.5 * sqrt(E1 * E2) * (1 - poisson_ratio);
    OrthotropicStVKElement<>::_C /= (1 - std::pow(poisson_ratio, 2));

    VectorXd var = GENERATE(take(2, vector_random(9)));
    REQUIRE(e1.energy(var) == Approx(e.energy(var)).epsilon(1e-6));
  }
}