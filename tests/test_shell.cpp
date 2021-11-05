#include "catch.hpp"
#include "helpers.h"

#include <fsim/DiscreteShell.h>
#include <fsim/util/geometry.h>

bool check_normals(const Eigen::VectorXd &X)
{
  using namespace Eigen;
  Vector3d n0 = (X.segment<3>(0) - X.segment<3>(6)).cross(X.segment<3>(3) - X.segment<3>(6));
  Vector3d n1 = (X.segment<3>(3) - X.segment<3>(9)).cross(X.segment<3>(0) - X.segment<3>(9));

  return (n0.norm() > 1e-5 && n1.norm() > 1e-5 && (n0.normalized() + n1.normalized()).norm() > 1e-2);
};

TEMPLATE_TEST_CASE("HingeElement", "[HingeElem]", SquaredAngleFormulation, TanAngleFormulation)
{
  using namespace Eigen;

  double coeff = GENERATE(take(2, random(0., 1.)));
  VectorXd X = GENERATE(take(10, filter(check_normals, vector_random(12))));
  Map<fsim::Mat3<double>> V(X.data(), 4, 3);
  HingeElement<TestType, true> e1(V, Vector4i(0, 1, 2, 3), coeff);

  RandomRange random_generator(-1.0, 1.0);

  SECTION("Gradient") { test_gradient(e1, 5e-5, check_normals); }
  SECTION("Hessian") { test_hessian(e1, 5e-5, check_normals); }
  SECTION("Translate invariance") { translate_invariance(e1); }
  SECTION("Rotation invariance") { rotational_invariance(e1); }

  SECTION("Equivalence")
  {
    HingeElement<TestType, false> e2(V, Vector4i(0, 1, 2, 3), coeff);
    VectorXd X = VectorXd::NullaryExpr(12, random_generator);
    REQUIRE(e1.energy(X) == Approx(e2.energy(X)).epsilon(1e-10));
    REQUIRE((e1.gradient(X) - e2.gradient(X)).norm() == Approx(0).margin(1e-10));
  }
}

TEST_CASE("signed angle")
{
  using namespace Eigen;

  Vector3d u = GENERATE(take(5, vector_random(3)));
  Vector3d axis = GENERATE(take(5, vector_random(3)));
  axis = u.cross(axis).normalized();
  double angle = GENERATE(take(5, random(-M_PI, M_PI)));

  Vector3d v = AngleAxis<double>(angle, axis) * u;

  REQUIRE(signed_angle(u, v, axis) == Approx(angle).epsilon(1e-10));
}

TEST_CASE("DiscreteShell class", "[Shell]")
{
  using namespace Eigen;

  VectorXd X = GENERATE(take(10, filter(check_normals, vector_random(12))));
  Map<fsim::Mat3<double>> V(X.data(), 4, 3);
  MatrixX3i F(2, 3);
  F << 0, 1, 2, 0, 2, 3;

  double young_modulus = GENERATE(take(2, random(0., 1.)));
  double poisson_ratio = GENERATE(take(2, random(0.01, 0.5)));
  double thickness = GENERATE(take(2, random(0., 1.)));
  ElasticShell shell(V, F, young_modulus, poisson_ratio, thickness);

  SECTION("Gradient") { test_gradient(shell, 5e-5, check_normals); }
  SECTION("Hessian") { test_hessian(shell, 5e-5, check_normals); }

  SECTION("Thickness")
  {
    double t = GENERATE(take(5, random(0., 1.)));
    shell.set_thickness(t);
    REQUIRE(shell.get_thickness() == t);

    VectorXd var = GENERATE(take(2, filter(check_normals, vector_random(12))));
    double prev_energy = shell.energy(var);

    shell.set_thickness(2 * t);

    REQUIRE(shell.energy(var) == Approx(8 * prev_energy).margin(1e-10));
  }
  SECTION("Young's Modulus")
  {
    double E = GENERATE(take(5, random(0., 1.)));
    shell.set_young_modulus(E);
    REQUIRE(shell.get_young_modulus() == E);

    VectorXd var = GENERATE(take(2, filter(check_normals, vector_random(12))));
    double prev_energy = shell.energy(var);

    shell.set_young_modulus(2 * E);

    REQUIRE(shell.energy(var) == Approx(2 * prev_energy).margin(1e-10));
  }
  SECTION("Poisson's ratio")
  {
    double v = GENERATE(range(0.01, 0.5, 0.05));
    shell.set_poisson_ratio(v);
    REQUIRE(shell.get_poisson_ratio() == v);
  }
}
