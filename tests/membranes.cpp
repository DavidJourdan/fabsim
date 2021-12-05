#include "catch.hpp"
#include "helpers.h"

#include <fsim/IncompressibleNeoHookeanElement.h>
#include <fsim/IncompressibleNeoHookeanMembrane.h>
#include <fsim/NeoHookeanElement.h>
#include <fsim/NeoHookeanMembrane.h>
#include <fsim/OrthotropicStVKMembrane.h>
#include <fsim/StVKMembrane.h>
#include <fsim/StVKElement.h>
#include <fsim/util/typedefs.h>

using namespace fsim;

TEMPLATE_TEST_CASE("TriangleElement", "", StVKElement<>, NeoHookeanElement<>, IncompressibleNeoHookeanElement<>)
{
  using namespace Eigen;

  Mat3<double> V = GENERATE(take(5, matrix_random(3, 3)));
  Vector3d params = GENERATE(take(5, vector_random(3, 0., 0.5)));
  TestType::setParameters(params(0), params(1));
  TestType::mass = 0;
  TestType e(V, Vector3i(0, 1, 2), params(2));

  SECTION("Translate invariance") { translate_invariance(e); }
  SECTION("Rotation invariance") { rotational_invariance(e); }

  SECTION("Energy")
  {
    VectorXd var = VectorXd::Zero(9);
    var.segment<3>(0) = V.row(0);
    var.segment<3>(3) = V.row(1);
    var.segment<3>(6) = V.row(2);

    REQUIRE(e.energy(var) == Approx(0.).margin(1e-10));
  }
}

TEST_CASE("OrthotropicStVKElement")
{
  using namespace Eigen;

  Mat2<double> V = GENERATE(take(5, matrix_random(3, 2)));
  VectorXd params = GENERATE(take(5, vector_random(4, 0., 0.5)));

  double E1 = params(0);
  double E2 = params(1);
  double thickness = params(2);
  double poisson_ratio = params(3);

  OrthotropicStVKElement<>::_C << E1, poisson_ratio * sqrt(E1 * E2), 0,
                                  poisson_ratio * sqrt(E1 * E2), E2, 0,
                                  0, 0, 0.5 * sqrt(E1 * E2) * (1 - poisson_ratio);
  OrthotropicStVKElement<>::_C /= (1 - std::pow(poisson_ratio, 2));
  OrthotropicStVKElement e(V, Vector3i(0, 1, 2), thickness);

  SECTION("Energy")
  {
    VectorXd var = VectorXd::Zero(9);
    var.segment<2>(0) = V.row(0);
    var.segment<2>(3) = V.row(1);
    var.segment<2>(6) = V.row(2);

    REQUIRE(e.energy(var) == Approx(0.).margin(1e-10));
  }
  SECTION("Strain/stress")
  {
    VectorXd var = GENERATE(take(5, vector_random(9)));

    REQUIRE_THAT(e.stress(var), ApproxEquals(OrthotropicStVKElement<>::_C * e.strain(var)));
  }
}

TEST_CASE("StVKMembrane class")
{
  using namespace Eigen;

  Mat3<double> V = GENERATE(take(5, matrix_random(3, 3)));
  Mat3<int> F = (Mat3<int>(1, 3) << 0, 1, 2).finished();

  StVKMembrane<0> instance0(V, F, 0.1, 10, 0.3);
  StVKMembrane<1> instance1(V, F, 0.9, 100, 0.4);

  SECTION("getPoissonRatio") { REQUIRE(instance0.getPoissonRatio() != instance1.getPoissonRatio()); }
  SECTION("getYoungModulus") { REQUIRE(instance0.getYoungModulus() != instance1.getYoungModulus()); }
  SECTION("getThickness") { REQUIRE(instance1.getThickness() == 0.9); }
  SECTION("setThickness")
  {
    instance0.setThickness(0.3);
    REQUIRE(instance0.getThickness() == 0.3);
  }
  SECTION("setPoissonRatio")
  {
    instance0.setPoissonRatio(0.5);
    REQUIRE(instance0.getPoissonRatio() == 0.5);
  }
  SECTION("Young's modulus")
  {
    double E = GENERATE(take(5, random(0., 1.)));
    instance0.setYoungModulus(E);
    REQUIRE(instance0.getYoungModulus() == E);

    VectorXd var = GENERATE(take(5, vector_random(9)));
    double prev_energy = instance0.energy(var);

    instance0.setYoungModulus(2 * E);

    REQUIRE(instance0.energy(var) == Approx(2 * prev_energy).epsilon(1e-6));
  }
  SECTION("setMass")
  {
    instance0.setMass(2);
    REQUIRE(instance0.getMass() == 2);
  }
  SECTION("nbDOFS") { REQUIRE(instance0.nbDOFs() == 9); }
}

TEST_CASE("NeoHookeanMembrane class")
{
  using namespace Eigen;

  Mat3<double> V = GENERATE(take(5, matrix_random(3, 3)));
  Mat3<int> F = (Mat3<int>(1, 3) << 0, 1, 2).finished();

  NeoHookeanMembrane<0> instance0(V, F, 0.1, 10, 0.3);
  NeoHookeanMembrane<1> instance1(V, F, 0.9, 100, 0.4);

  SECTION("getThickness") { REQUIRE(instance1.getThickness() == 0.9); }
  SECTION("getPoissonRatio") { REQUIRE(instance0.getPoissonRatio() != instance1.getPoissonRatio()); }
  SECTION("getYoungModulus") { REQUIRE(instance0.getYoungModulus() != instance1.getYoungModulus()); }
  SECTION("setThickness")
  {
    instance0.setThickness(0.3);
    REQUIRE(instance0.getThickness() == 0.3);
  }
  SECTION("setPoissonRatio")
  {
    instance0.setPoissonRatio(0.5);
    REQUIRE(instance0.getPoissonRatio() == 0.5);
  }
  SECTION("Young's modulus")
  {
    double E = GENERATE(take(5, random(0., 1.)));
    instance0.setYoungModulus(E);
    REQUIRE(instance0.getYoungModulus() == E);

    VectorXd var = GENERATE(take(5, vector_random(9)));
    double prev_energy = instance0.energy(var);

    instance0.setYoungModulus(2 * E);

    REQUIRE(instance0.energy(var) == Approx(2 * prev_energy).epsilon(1e-6));
  }
  SECTION("setMass")
  {
    instance0.setMass(2);
    REQUIRE(instance0.getMass() == 2);
  }
  SECTION("nbDOFS") { REQUIRE(instance0.nbDOFs() == 9); }
}

TEST_CASE("IncompressibleNeoHookeanMembrane class")
{
  using namespace Eigen;

  Mat3<double> V = GENERATE(take(5, matrix_random(3, 3)));
  Mat3<int> F = (Mat3<int>(1, 3) << 0, 1, 2).finished();

  IncompressibleNeoHookeanMembrane<0> instance0(V, F, 0.1, 10, 0.3);
  IncompressibleNeoHookeanMembrane<1> instance1(V, F, 0.9, 100, 0.4);

  SECTION("getThickness") { REQUIRE(instance1.getThickness() == 0.9); }
  SECTION("getPoissonRatio") { REQUIRE(instance0.getPoissonRatio() != instance1.getPoissonRatio()); }
  SECTION("getYoungModulus") { REQUIRE(instance0.getYoungModulus() != instance1.getYoungModulus()); }
  SECTION("setThickness")
  {
    instance0.setThickness(0.3);
    REQUIRE(instance0.getThickness() == 0.3);
  }
  SECTION("setPoissonRatio")
  {
    instance0.setPoissonRatio(0.5);
    REQUIRE(instance0.getPoissonRatio() == 0.5);
  }
  SECTION("Young's modulus")
  {
    double E = GENERATE(take(5, random(0., 1.)));
    instance0.setYoungModulus(E);
    REQUIRE(instance0.getYoungModulus() == E);

    VectorXd var = GENERATE(take(5, vector_random(9)));
    double prev_energy = instance0.energy(var);

    instance0.setYoungModulus(2 * E);

    REQUIRE(instance0.energy(var) == Approx(2 * prev_energy).epsilon(1e-6));
  }
  SECTION("setMass")
  {
    instance0.setMass(2);
    REQUIRE(instance0.getMass() == 2);
  }
  SECTION("nbDOFS") { REQUIRE(instance0.nbDOFs() == 9); }
}

TEST_CASE("OrthotropicStVKMembrane class")
{
  using namespace Eigen;

  Mat2<double> V = GENERATE(take(5, matrix_random(3, 2)));
  Mat3<int> F = (Mat3<int>(1, 3) << 0, 1, 2).finished();

  OrthotropicStVKMembrane<0> instance0(V, F, 0.1, 10, 5, 0.3);
  OrthotropicStVKMembrane<1> instance1(V, F, 0.9, 100, 50, 0.4);

  SECTION("getThickness") { REQUIRE(instance1.getThickness() == 0.9); }
  SECTION("getPoissonRatio") { REQUIRE(instance0.getPoissonRatio() != instance1.getPoissonRatio()); }
  SECTION("setThickness")
  {
    instance0.setThickness(0.3);
    REQUIRE(instance0.getThickness() == 0.3);
  }
  SECTION("setPoissonRatio")
  {
    instance0.setPoissonRatio(0.5);
    REQUIRE(instance0.getPoissonRatio() == 0.5);
  }
  SECTION("setYoungModuli")
  {
    instance0.setYoungModuli(5, 6);
    std::array<double, 2> youngModuli = instance0.getYoungModuli();
    REQUIRE(youngModuli[0] == 5);
    REQUIRE(youngModuli[1] == 6);
  }
  SECTION("setMass")
  {
    instance0.setMass(2);
    REQUIRE(instance0.getMass() == 2);
  }
  SECTION("nbDOFS") { REQUIRE(instance0.nbDOFs() == 9); }
}

TEST_CASE("Small strain equivalence")
{
  using namespace Eigen;

  Mat3<double> V = GENERATE(take(5, matrix_random(3, 3)));

  double E = GENERATE(take(5, random(1e10, 1e11)));
  double nu = GENERATE(take(5, random(0., 0.5)));
  double thickness = GENERATE(take(5, random(0., 1.)));

  StVKElement<> e1(V, Vector3i(0, 1, 2), thickness);
  NeoHookeanElement<> e2(V, Vector3i(0, 1, 2), thickness);

  StVKElement<>::lambda = E * nu / pow(1 - nu, 2);
  StVKElement<>::mu = E / (1 + nu) / 2;
  NeoHookeanElement<>::lambda = E * nu / pow(1 - nu, 2);
  NeoHookeanElement<>::mu = E / (1 + nu) / 2;
  
  VectorXd var = 1e-5 * GENERATE(take(5, vector_random(9)));
  var.segment<3>(0) += V.row(0);
  var.segment<3>(3) += V.row(1);
  var.segment<3>(6) += V.row(2);

  // doesn't need to be super precise, just check that they are approximately the same
  REQUIRE(e1.energy(var) == Approx(e2.energy(var)).epsilon(1e-2));
}
