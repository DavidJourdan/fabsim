#include "catch.hpp"
#include "fsim/HingeElement.h"
#include "helpers.h"

#include <fsim/ElasticShell.h>
#include <fsim/util/geometry.h>

using namespace fsim;

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

  double coeff = GENERATE(take(5, random(0., 1.)));
  VectorXd X = GENERATE(take(5, filter(check_normals, vector_random(12))));
  Map<Mat3<double>> V(X.data(), 4, 3);
  HingeElement<TestType> e(V, Vector4i(0, 1, 2, 3), coeff);

  SECTION("Translate invariance") { translate_invariance(e); }
  SECTION("Rotation invariance") { rotational_invariance(e); }
}

TEST_CASE("Approx hessian")
{
  using namespace Eigen;

  double coeff = GENERATE(take(5, random(0., 1.)));
  VectorXd X = GENERATE(take(5, filter(check_normals, vector_random(12))));
  Map<Mat3<double>> V(X.data(), 4, 3);
  HingeElement<SquaredAngleFormulation> e(V, Vector4i(0, 1, 2, 3), coeff);

  VectorXd var = GENERATE(take(5, filter(check_normals, vector_random(12))));
  MatrixXd h = e.hessianApprox(var);

  VectorXd z = GENERATE(take(5, filter(check_normals, vector_random(12))));
  REQUIRE(z.dot(h * z) >= 0);
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

TEST_CASE("ElasticShell")
{
  using namespace Eigen;

  VectorXd X = GENERATE(take(5, filter(check_normals, vector_random(12))));
  Map<Mat3<double>> V(X.data(), 4, 3);
  MatrixX3i F(2, 3);
  F << 0, 1, 2, 0, 2, 3;

  double young_modulus = GENERATE(take(5, random(0., 1.)));
  double poisson_ratio = GENERATE(take(5, random(0., 0.5)));
  double thickness = GENERATE(take(5, random(0., 1.)));
  ElasticShell<> shell(V, F, thickness, young_modulus, poisson_ratio);

  SECTION("Thickness")
  {
    double t = GENERATE(take(5, random(0., 1.)));
    
    // compare constant thickness constructor vs list of thicknesses constructor
    ElasticShell<> shell1(V, F, t, young_modulus, poisson_ratio);
    ElasticShell<> shell2(V, F, {t, t}, young_modulus, poisson_ratio);

    VectorXd var = GENERATE(take(5, filter(check_normals, vector_random(12))));

    REQUIRE(shell1.energy(var) == Approx(shell2.energy(var)).epsilon(1e-10));
  }
  SECTION("Young's Modulus")
  {
    double E = GENERATE(take(5, random(0., 1.)));
    shell.setYoungModulus(E);
    REQUIRE(shell.getYoungModulus() == E);

    VectorXd var = GENERATE(take(5, filter(check_normals, vector_random(12))));
    double prev_energy = shell.energy(var);

    shell.setYoungModulus(2 * E);

    REQUIRE(shell.energy(var) == Approx(2 * prev_energy).epsilon(1e-6));
  }
  SECTION("Poisson's ratio")
  {
    double poisson_ratio = GENERATE(take(5, random(0., 0.5)));
    shell.setPoissonRatio(poisson_ratio);
    REQUIRE(shell.getPoissonRatio() == poisson_ratio);
  }
  SECTION("Nb vertices/edges/faces/dofs")
  {
    REQUIRE(shell.nbDOFs() == 12);
    REQUIRE(shell.nbVertices() == 4);
    REQUIRE(shell.nbFaces() == 2);
    REQUIRE(shell.nbEdges() == 1);
  }
}

TEST_CASE("DiscreteShell")
{
  using namespace Eigen;

  VectorXd X = GENERATE(take(5, filter(check_normals, vector_random(12))));
  Map<Mat3<double>> V(X.data(), 4, 3);
  MatrixX3i F(2, 3);
  F << 0, 1, 2, 0, 2, 3;

  double young_modulus = GENERATE(take(5, random(0., 1.)));
  double poisson_ratio = GENERATE(take(5, random(0., 0.5)));
  double thickness = GENERATE(take(5, random(0., 1.)));
  DiscreteShell<> shell(V, F, thickness, young_modulus, poisson_ratio);

  SECTION("Thickness")
  {
    double t1 = GENERATE(take(5, random(0., 1.)));
    double t2 = GENERATE(take(5, random(0., 1.)));
    double t = (t1 + t2) / 2;
    VectorXd var = GENERATE(take(5, filter(check_normals, vector_random(12))));

    DiscreteShell<> shell1(V, F, t, young_modulus, poisson_ratio);
    DiscreteShell<> shell2(V, F, 2 * t, young_modulus, poisson_ratio);

    // check that energy scales accordingly
    REQUIRE(shell2.energy(var) == Approx(8 * shell1.energy(var)).margin(1e-10));

    // compare vertex-based vs face-based vs edge-based list of thicknesses constructor
    DiscreteShell<> shell3(V, F, std::vector<double>(1, t), young_modulus, poisson_ratio);
    DiscreteShell<> shell4(V, F, {t1, t2}, young_modulus, poisson_ratio);
    DiscreteShell<> shell5(V, F, {t1, 0, t2, 0}, young_modulus, poisson_ratio);
    REQUIRE(shell1.energy(var) == Approx(shell3.energy(var)).margin(1e-10));
    REQUIRE(shell1.energy(var) == Approx(shell4.energy(var)).margin(1e-10));
    REQUIRE(shell1.energy(var) == Approx(shell5.energy(var)).margin(1e-10));
  }
}