#include "catch.hpp"
#include "helpers.h"

#include <fsim/ElasticMembraneModel.h>
#include <fsim/TriangleElement.h>
#include <fsim/OrthotropicStVKElement.h>

using namespace fsim;

TEMPLATE_TEST_CASE("TriangleElement", "[TriElem]", StVKElement<>, NeoHookeanElement<>, NHIncompressibleElement<>)
{
  using namespace Eigen;

  MatrixX3d V = GENERATE(take(2, matrix_random(3, 3)));
  double thickness = GENERATE(take(1, random(0., 1.)));
  double young_modulus = GENERATE(take(2, random(0., 1.)));
  double poisson_ratio = GENERATE(range(0.01, 0.5, 0.05));
  TestType::alpha = poisson_ratio / (1.0 - pow(poisson_ratio, 2));
  TestType::beta = 0.5 / (1 + poisson_ratio);
  TestType e(V, Vector3i(0, 1, 2), thickness, young_modulus);

  SECTION("Gradient") { test_gradient(e); }
  SECTION("Hessian") { test_hessian(e); }
  SECTION("Translate invariance") { translate_invariance(e); }
  SECTION("Rotation invariance") { rotational_invariance(e); }
  // SECTION("Scale invariance")
  // {
  //   VectorXd var = GENERATE(take(10, vector_random(9)));
  //   double stretch = GENERATE(take(2, random(0., 2.)));

  //   double scaled_energy = e.energy(var) / pow(stretch, 3);

  //   e.set_stretch_factor(stretch, 1);
  //   TestType::thickness /= stretch;

  //   INFO(scaled_energy / e.energy(var / stretch))
  //   REQUIRE(e.energy(var / stretch) == Approx(scaled_energy).margin(1e-10));
  // }
}

TEST_CASE("StVKMembrane class", "[StVK]")
{
  using namespace Eigen;

  MatrixX3d V = GENERATE(take(2, matrix_random(3, 3)));
  MatrixX3i F(1, 3);
  F.row(0) = Vector3i(0, 1, 2);

  StVKMembraneModel<0> instance0(V, F, 10, 0.3, 0.1);
  StVKMembraneModel<1> instance1(V, F, 100, 0.4, 0.9);

  SECTION("get_thickness") { REQUIRE(instance1.get_thickness() == 0.9); }
  SECTION("get_lame_alpha") { REQUIRE(instance0.get_lame_alpha() != instance1.get_lame_alpha()); }
  SECTION("get_lame_beta") { REQUIRE(instance0.get_lame_beta() != instance1.get_lame_beta()); }
  SECTION("set_thickness")
  {
    instance0.set_thickness(0.3);
    REQUIRE(instance0.get_thickness() == 0.3);
  }
  SECTION("Gradient") { test_gradient(instance0); }
  SECTION("Hessian") { test_hessian(instance0); }
}

TEST_CASE("NeoHookeanMembrane class", "[NeoHookean]")
{
  using namespace Eigen;

  MatrixX3d V = GENERATE(take(2, matrix_random(3, 3)));
  MatrixX3i F(1, 3);
  F.row(0) = Vector3i(0, 1, 2);

  NeoHookeanMembraneModel<0> instance0(V, F, 10, 0.3, 0.1);
  NeoHookeanMembraneModel<1> instance1(V, F, 100, 0.4, 0.9);

  SECTION("get_thickness") { REQUIRE(instance1.get_thickness() == 0.9); }
  SECTION("get_lame_alpha") { REQUIRE(instance0.get_lame_alpha() != instance1.get_lame_alpha()); }
  SECTION("get_lame_beta") { REQUIRE(instance0.get_lame_beta() != instance1.get_lame_beta()); }
  SECTION("set_thickness")
  {
    instance0.set_thickness(0.3);
    REQUIRE(instance0.get_thickness() == 0.3);
  }
  SECTION("Gradient") { test_gradient(instance0); }
  SECTION("Hessian") { test_hessian(instance0); }
}

TEST_CASE("NHIncompressibleMembrane class", "[IncompressibleNeoHookean]")
{
  using namespace Eigen;

  MatrixX3d V = GENERATE(take(2, matrix_random(3, 3)));
  MatrixX3i F(1, 3);
  F.row(0) = Vector3i(0, 1, 2);

  NHIncompressibleMembraneModel<0> instance0(V, F, 10, 0.3, 0.1);
  NHIncompressibleMembraneModel<1> instance1(V, F, 100, 0.4, 0.9);

  SECTION("get_thickness") { REQUIRE(instance1.get_thickness() == 0.9); }
  SECTION("get_lame_alpha") { REQUIRE(instance0.get_lame_alpha() != instance1.get_lame_alpha()); }
  SECTION("get_lame_beta") { REQUIRE(instance0.get_lame_beta() != instance1.get_lame_beta()); }
  SECTION("set_thickness")
  {
    instance0.set_thickness(0.3);
    REQUIRE(instance0.get_thickness() == 0.3);
  }
  SECTION("Gradient") { test_gradient(instance0); }
  SECTION("Hessian") { test_hessian(instance0); }
}

TEST_CASE("Small strain equivalence", "[Membrane]")
{
  using namespace Eigen;

  MatrixX3d V = GENERATE(take(2, matrix_random(3, 3)));

  double young_modulus = GENERATE(take(2, random(1e10, 1e11)));
  double poisson_ratio = GENERATE(range(0.01, 0.5, 0.05));
  double thickness = GENERATE(take(1, random(0., 1.)));

  StVKElement<> e1(V, Vector3i(0, 1, 2), thickness, young_modulus);
  NeoHookeanElement<> e2(V, Vector3i(0, 1, 2), thickness, young_modulus);

  StVKElement<>::alpha = poisson_ratio / (1.0 - pow(poisson_ratio, 2));
  StVKElement<>::beta = 0.5 / (1 + poisson_ratio);

  NeoHookeanElement<>::alpha = poisson_ratio / (1.0 - pow(poisson_ratio, 2));
  NeoHookeanElement<>::beta = 0.5 / (1 + poisson_ratio);

  VectorXd var = 1e-5 * GENERATE(take(10, vector_random(9)));
  var.segment<3>(0) += V.row(0);
  var.segment<3>(3) += V.row(1);
  var.segment<3>(6) += V.row(2);

  // doesn't need to be super precise, just check that they are approximately the same
  REQUIRE(e1.energy(var) == Approx(e2.energy(var)).epsilon(1e-2));
}

TEST_CASE("OrthotropicStVKElement", "[StVK]")
{
  using namespace Eigen;

  double poisson_ratio = GENERATE(range(0.01, 0.5, 0.05));
  double thickness = GENERATE(take(1, random(0., 1.)));
  MatrixX2d V = GENERATE(take(2, matrix_random(3, 2)));

  double E1 = GENERATE(take(1, random(0., 1.)));
  double E2 = GENERATE(take(1, random(0., 1.)));
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
    StVKElement<> e1(V1, Vector3i(0, 1, 2), thickness, 1);
    StVKElement<>::alpha = poisson_ratio / (1.0 - pow(poisson_ratio, 2));
    StVKElement<>::beta = 0.5 / (1 + poisson_ratio);

    double E1 = 1;
    double E2 = 1;
    OrthotropicStVKElement<>::_C << E1, poisson_ratio * sqrt(E1 * E2), 0,
                                   poisson_ratio * sqrt(E1 * E2), E2, 0,
                                   0, 0, 0.5 * sqrt(E1 * E2) * (1 - poisson_ratio);
    OrthotropicStVKElement<>::_C /= (1 - std::pow(poisson_ratio, 2));

    VectorXd var = GENERATE(take(2, vector_random(9)));
    REQUIRE(e1.energy(var) == Approx(e.energy(var)).epsilon(1e-6));
  }

  SECTION("Gradient") { test_gradient(e); }
  SECTION("Hessian") { test_hessian(e); }
}