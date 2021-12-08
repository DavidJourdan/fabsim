#include "catch.hpp"
#include "fsim/HingeElement.h"
#include "fsim/util/first_fundamental_form.h"
#include "helpers.h"

#include <fsim/ElasticMembrane.h>
#include <fsim/ElasticRod.h>
#include <fsim/ElasticShell.h>
#include <fsim/IncompressibleNeoHookeanElement.h>
#include <fsim/MassSpring.h>
#include <fsim/NeoHookeanElement.h>
#include <fsim/OrthotropicStVKElement.h>
#include <fsim/OrthotropicStVKMembrane.h>
#include <fsim/RodCollection.h>
#include <fsim/RodStencil.h>
#include <fsim/Spring.h>
#include <fsim/StVKElement.h>
#include <fsim/util/typedefs.h>

using namespace fsim;

// MEMBRANES

TEMPLATE_TEST_CASE("TriangleElement", "", StVKElement, NeoHookeanElement)
{
  using namespace Eigen;

  Mat3<double> V = GENERATE(take(2, matrix_random(3, 3)));
  VectorXd params = GENERATE(take(4, vector_random(4, 0., 1.)));

  TestType e(V, Vector3i(0, 1, 2), params(3));

  SECTION("Gradient")
  {
    test_gradient([&](auto &X) { return e.energy(X, params(0), params(1), params(2)); },
                  [&](auto &X) { return e.gradient(X, params(0), params(1), params(2)); }, 9, 1e-5);
  }
  SECTION("Hessian")
  {
    test_hessian([&](auto &X) { return e.gradient(X, params(0), params(1), params(2)); },
                 [&](auto &X) { return e.hessian(X, params(0), params(1), params(2)); }, 9, 1e-5);
  }
}

TEST_CASE("IncompressibleNeoHookeanElement")
{
  using namespace Eigen;

  Mat3<double> V = GENERATE(take(5, matrix_random(3, 3)));
  Vector3d params = GENERATE(take(5, vector_random(3, 0., 1.)));
  IncompressibleNeoHookeanElement e(V, Vector3i(0, 1, 2), params(2));

  SECTION("Gradient")
  {
    test_gradient([&](auto &X) { return e.energy(X, params(0), params(1)); },
                  [&](auto &X) { return e.gradient(X, params(0), params(1)); }, 9, 1e-5);
  }
  SECTION("Hessian")
  {
    test_hessian([&](auto &X) { return e.gradient(X, params(0), params(1)); },
                 [&](auto &X) { return e.hessian(X, params(0), params(1)); }, 9, 1e-5);
  }
}

TEST_CASE("OrthotropicStVKElement")
{
  using namespace Eigen;

  Mat2<double> V = GENERATE(take(2, matrix_random(3, 2)));
  double thickness = GENERATE(take(2, random(0., 1.)));
  double poisson_ratio = GENERATE(take(2, random(0., 0.5)));

  double E1 = GENERATE(take(2, random(0., 1.)));
  double E2 = GENERATE(take(2, random(0., 1.)));
  double mass = GENERATE(take(2, random(0., 1.)));
  Matrix3d C;
  C << E1, poisson_ratio * sqrt(E1 * E2), 0, 
       poisson_ratio * sqrt(E1 * E2), E2, 0, 0, 
       0, 0.5 * sqrt(E1 * E2) * (1 - poisson_ratio);
  C /= (1 - std::pow(poisson_ratio, 2));
  OrthotropicStVKElement e(V, Vector3i(0, 1, 2), thickness);

  SECTION("Gradient")
  {
    test_gradient([&](auto &X) { return e.energy(X, C, mass); },
                  [&](auto &X) { return e.gradient(X, C, mass); }, 9, 1e-5);
  }
  SECTION("Hessian")
  {
    test_hessian([&](auto &X) { return e.gradient(X, C, mass); },
                 [&](auto &X) { return e.hessian(X, C, mass); }, 9, 1e-5);
  }
}

TEST_CASE("OrthotropicStVKMembrane")
{
  using namespace Eigen;

  Mat2<double> V = GENERATE(take(5, matrix_random(3, 2)));
  Mat3<int> F = (Mat3<int>(1, 3) << 0, 1, 2).finished();
  VectorXd params = GENERATE(take(5, vector_random(5, 0., 0.5)));
  OrthotropicStVKMembrane membrane(V, F, params(0), params(1), params(2), params(3), params(4));

  SECTION("Gradient") { test_gradient(membrane); }
  SECTION("Hessian") 
  {
    test_hessian([&](auto &X) { return membrane.gradient(X); },
                 [&](auto &X) { return MatrixXd(MatrixXd(membrane.hessian(X)).selfadjointView<Upper>()); }, 9, 1e-5);
  }
}

TEMPLATE_TEST_CASE("ElasticMembrane", "", StVKMembrane, NeoHookeanMembrane, IncompressibleNeoHookeanMembrane)
{
  using namespace Eigen;

  Mat3<double> V = GENERATE(take(5, matrix_random(3, 3)));
  Mat3<int> F = (Mat3<int>(1, 3) << 0, 1, 2).finished();
  VectorXd params = GENERATE(take(4, vector_random(5, 0., 0.5)));
  TestType membrane(V, F, params(0), params(1), params(2), params(3));

  SECTION("Gradient") { test_gradient(membrane); }
  SECTION("Hessian") 
  {
    test_hessian([&](auto &X) { return membrane.gradient(X); },
                 [&](auto &X) { return MatrixXd(MatrixXd(membrane.hessian(X)).selfadjointView<Upper>()); }, 9, 1e-5);
  }
}

TEST_CASE("MassSpring")
{
  using namespace Eigen;

  Mat3<double> V = GENERATE(take(5, matrix_random(3, 3)));
  Mat3<int> F = (Mat3<int>(1, 3) << 0, 1, 2).finished();
  double E = GENERATE(take(5, random(0., 0.5)));
  MassSpring membrane(V, F, E);

  SECTION("Gradient") { test_gradient(membrane); }
  SECTION("Hessian") 
  {
    test_hessian([&](auto &X) { return membrane.gradient(X); },
                 [&](auto &X) { return MatrixXd(MatrixXd(membrane.hessian(X)).selfadjointView<Upper>()); }, 9, 1e-5);
  }
}

// RODS

TEST_CASE("RodStencil")
{
  using namespace Eigen;

  Mat3<double> V = GENERATE(take(2, matrix_random(3, 3)));
  Vector2d widths = GENERATE(take(2, vector_random(2, 0, 1)));
  double young_modulus = GENERATE(take(2, random(0., 1.)));
  double stretch = GENERATE(take(2, random(0., 1.)));
  double mass = GENERATE(take(2, random(0., 1.)));

  Vector3d n1 = GENERATE(take(2, vector_random(3))).normalized();
  Vector3d t1 = (V.row(1) - V.row(0)).normalized();
  n1 = (n1 - n1.dot(t1) * t1).normalized(); // make sure the frame is orthogonal

  VectorXi dofs(5);
  dofs << 0, 1, 2, 9, 10;

  Vector2d stiffnesses(pow(widths(0), 3) * widths(1), pow(widths(1), 3) * widths(0));
  stiffnesses *= young_modulus / 12;

  LocalFrame f1(t1, n1, t1.cross(n1));
  LocalFrame f2(f1);
  f2.update(V.row(1), V.row(2));
  RodStencil rod(V, f1, f2, dofs);

  SECTION("Gradient")
  {
    test_gradient(
        [&](const Eigen::VectorXd &X) {
          LocalFrame new_f1 = updateFrame(f1, X.segment<3>(3 * 0), X.segment<3>(3 * 1));
          LocalFrame new_f2 = updateFrame(f2, X.segment<3>(3 * 1), X.segment<3>(3 * 2));

          double ref_twist = rod.getReferenceTwist();
          rod.updateReferenceTwist(new_f1, new_f2);
          double res = rod.energy(X, new_f1, new_f2, stiffnesses, stretch, mass);
          rod.setReferenceTwist(ref_twist);
          return res;
        },
        [&](const Eigen::VectorXd &X) {
          LocalFrame new_f1 = updateFrame(f1, X.segment<3>(3 * 0), X.segment<3>(3 * 1));
          LocalFrame new_f2 = updateFrame(f2, X.segment<3>(3 * 1), X.segment<3>(3 * 2));

          double ref_twist = rod.getReferenceTwist();
          rod.updateReferenceTwist(new_f1, new_f2);
          auto res = rod.gradient(X, new_f1, new_f2, stiffnesses, stretch, mass);
          rod.setReferenceTwist(ref_twist);
          return res;
        },
        rod.nbDOFs(), 1e-5, [](auto &X) { return true; },
        [&](const Eigen::VectorXd &X) {
          f1.update(X.segment<3>(3 * 0), X.segment<3>(3 * 1));
          f2.update(X.segment<3>(3 * 1), X.segment<3>(3 * 2));
          rod.updateReferenceTwist(f1, f2);
        });
  }
  SECTION("Hessian")
  {
    VectorXd var(rod.nbDOFs());
    MatrixXd hessian_computed = MatrixXd::Zero(rod.nbDOFs(), rod.nbDOFs());
    MatrixXd hessian_numerical = MatrixXd::Zero(rod.nbDOFs(), rod.nbDOFs());

    for(int i = 0; i < 10; ++i)
    {
      var = VectorXd::NullaryExpr(rod.nbDOFs(), RandomRange(-1.0, 1.0));

      f1.update(var.segment<3>(3 * 0), var.segment<3>(3 * 1));
      f2.update(var.segment<3>(3 * 1), var.segment<3>(3 * 2));
      rod.updateReferenceTwist(f1, f2);

      hessian_computed += rod.hessian(var, f1, f2, stiffnesses, stretch, mass);
      hessian_numerical += sym(finite_differences(
          [&](const Eigen::VectorXd &X) {
            LocalFrame new_f1 = updateFrame(f1, X.segment<3>(3 * 0), X.segment<3>(3 * 1));
            LocalFrame new_f2 = updateFrame(f2, X.segment<3>(3 * 1), X.segment<3>(3 * 2));

            double ref_twist = rod.getReferenceTwist();
            rod.updateReferenceTwist(new_f1, new_f2);
            auto res = rod.gradient(X, new_f1, new_f2, stiffnesses, stretch, mass);
            rod.setReferenceTwist(ref_twist);
            return res;
          },
          var));
    }

    hessian_computed /= 10;
    hessian_numerical /= 10;
    MatrixXd diff = hessian_computed - hessian_numerical;

    INFO("Numerical hessian\n" << hessian_numerical);
    INFO("Computed hessian\n" << hessian_computed);
    INFO("Difference\n" << diff);
    REQUIRE(diff.norm() / hessian_numerical.norm() == Approx(0.0).margin(1e-4));
  }
}

TEST_CASE("ElasticRod")
{
  using namespace Eigen;

  Mat3<double> V = GENERATE(take(5, matrix_random(3, 3)));
  VectorXd params = GENERATE(take(5, vector_random(4, 0, 1)));
  // bool closed = GENERATE(true, false);
  CrossSection crossSection = GENERATE(CrossSection::Circle, CrossSection::Square);
  Vector3d n = Vector3d::UnitZ();
  ElasticRod rod(V, n, {params(0), params(1), params(2), params(3), crossSection});

  SECTION("Gradient")
  {
    test_gradient(
        rod, 5e-4, [](auto &X) { return true; }, [&](const Eigen::VectorXd &X) { rod.updateProperties(X); });
  }
  SECTION("Hessian")
  {
    VectorXd var(rod.nbDOFs());
    MatrixXd hessian_computed = MatrixXd::Zero(rod.nbDOFs(), rod.nbDOFs());
    MatrixXd hessian_numerical = MatrixXd::Zero(rod.nbDOFs(), rod.nbDOFs());

    for(int i = 0; i < 10; ++i)
    {
      var = VectorXd::NullaryExpr(rod.nbDOFs(), RandomRange(-1.0, 1.0));
      rod.updateProperties(var);

      hessian_computed += MatrixXd(rod.hessian(var)).selfadjointView<Upper>();
      hessian_numerical += sym(finite_differences(
          [&](const Eigen::VectorXd &X) {
            ElasticRod r = rod;
            r.updateProperties(X);
            return r.gradient(X);
          },
          var));
    }

    hessian_computed /= 10;
    hessian_numerical /= 10;
    MatrixXd diff = hessian_computed - hessian_numerical;

    INFO("Numerical hessian\n" << hessian_numerical);
    INFO("Computed hessian\n" << hessian_computed);
    INFO("Difference\n" << diff);
    REQUIRE(diff.norm() / hessian_numerical.norm() == Approx(0.0).margin(1e-4));
  }
}

TEST_CASE("RodCollection")
{
  using namespace Eigen;

  MatrixXd V = GENERATE(take(5, matrix_random(3, 3)));
  VectorXd params = GENERATE(take(5, vector_random(4, 0, 1)));
  Mat3<double> N(2, 3);
  N << 0, 0, 1, 0, 0, 1;

  int Case = GENERATE(0, 1);
  std::vector<std::vector<int>> indices;
  Mat2<int> C; //(2, 2);

  if(Case == 0)
  {
    indices = {{0, 1, 2}};
  }
  else if(Case == 1)
  {
    indices = {{0, 1}, {1, 2}};
    C.resize(2, 2);
    C << 0, 1, 1, 0;
  }

  CrossSection crossSection = GENERATE(CrossSection::Circle, CrossSection::Square);
  RodCollection rod(V, indices, C, N, {params(0), params(1), params(2), params(3), crossSection});

  SECTION("Gradient")
  {
    test_gradient(
        rod, 1e-5, [](auto &X) { return true; }, [&](const Eigen::VectorXd &X) { rod.updateProperties(X); });
  }
  SECTION("Hessian")
  {
    VectorXd var(rod.nbDOFs());
    MatrixXd hessian_computed = MatrixXd::Zero(rod.nbDOFs(), rod.nbDOFs());
    MatrixXd hessian_numerical = MatrixXd::Zero(rod.nbDOFs(), rod.nbDOFs());

    for(int i = 0; i < 10; ++i)
    {
      var = VectorXd::NullaryExpr(rod.nbDOFs(), RandomRange(-1.0, 1.0));
      rod.updateProperties(var);

      hessian_computed += MatrixXd(rod.hessian(var)).selfadjointView<Upper>();
      hessian_numerical += sym(finite_differences(
          [&](const Eigen::VectorXd &X) {
            RodCollection r = rod;
            r.updateProperties(X);
            return r.gradient(X);
          },
          var));
    }

    hessian_computed /= 10;
    hessian_numerical /= 10;
    MatrixXd diff = hessian_computed - hessian_numerical;

    INFO("Numerical hessian\n" << hessian_numerical);
    INFO("Computed hessian\n" << hessian_computed);
    INFO("Difference\n" << diff);
    REQUIRE(diff.norm() / hessian_numerical.norm() == Approx(0.0).margin(1e-4));
  }
}

// SHELLS

bool check_normals(const Eigen::VectorXd &X)
{
  using namespace Eigen;
  Vector3d n0 = (X.segment<3>(0) - X.segment<3>(6)).cross(X.segment<3>(3) - X.segment<3>(6));
  Vector3d n1 = (X.segment<3>(3) - X.segment<3>(9)).cross(X.segment<3>(0) - X.segment<3>(9));

  return (n0.norm() > 1e-5 && n1.norm() > 1e-5 && (n0.normalized() + n1.normalized()).norm() > 1e-2);
};

TEMPLATE_TEST_CASE("HingeElement", "", SquaredAngleFormulation, TanAngleFormulation)
{
  using namespace Eigen;

  double coeff = GENERATE(take(2, random(0., 1.)));
  VectorXd X = GENERATE(take(2, filter(check_normals, vector_random(12))));
  Map<Mat3<double>> V(X.data(), 4, 3);
  HingeElement<TestType> e1(V, Vector4i(0, 1, 2, 3), coeff);

  SECTION("Gradient") { test_gradient(e1, 5e-5, check_normals); }
  SECTION("Hessian") { test_hessian(e1, 5e-5, check_normals); }
}

// SPRINGS

TEST_CASE("Spring")
{
  using namespace Eigen;

  double rest_length = GENERATE(take(2, random(0., 1.)));
  Spring s(0, 1, rest_length);

  SECTION("Gradient")
  {
    test_gradient([&s](const auto &X) { return s.energy(X); },
                  [&s](const auto &X) {
                    Vec<double, 6> res;
                    res.segment<3>(0) = -s.force(X);
                    res.segment<3>(3) = s.force(X);
                    return res;
                  },
                  6);
  }
  SECTION("Hessian")
  {
    test_hessian(
        [&s](const auto &X) {
          Vec<double, 6> res;
          res.segment<3>(0) = -s.force(X);
          res.segment<3>(3) = s.force(X);
          return res;
        },
        [&s](const auto &X) {
          MatrixXd res(6, 6);
          Matrix3d h = s.hessian(X);
          res.block<3, 3>(0, 0) = -h;
          res.block<3, 3>(3, 3) = -h;
          res.block<3, 3>(0, 3) = h;
          res.block<3, 3>(3, 0) = h;
          return res;
        },
        6);
  }
}

TEST_CASE("First Fundamental form")
{
  using namespace Eigen;

  Vector3i face(0, 1, 2);
  auto i = GENERATE(0, 1, 2, 3);

  SECTION("Gradient")
  {
    test_gradient(
        [&](const auto &X) {
          return first_fundamental_form(Map<Mat3<double>>(const_cast<double *>(X.data()), 3, 3), face)(i);
        },
        [&](const auto &X) {
          Matrix<double, 4, 9> deriv;
          first_fundamental_form(Map<Mat3<double>>(const_cast<double *>(X.data()), 3, 3), face, &deriv);
          return deriv.row(i);
        },
        9);
  }
  SECTION("Hessian")
  {
    test_hessian(
        [&](const auto &X) {
          Matrix<double, 4, 9> deriv;
          first_fundamental_form(Map<Mat3<double>>(const_cast<double *>(X.data()), 3, 3), face, &deriv);
          return deriv.row(i);
        },
        [&](const auto &X) {
          Matrix<double, 36, 9> dderiv;
          first_fundamental_form(Map<Mat3<double>>(const_cast<double *>(X.data()), 3, 3), face, nullptr, &dderiv);
          return dderiv.block<9, 9>(9 * i, 0);
        },
        9);
  }
}

TEST_CASE("bendAngleGradient")
{
  using namespace Eigen;

  Vec<int, 4> indices;
  indices << 0, 1, 2, 3;

  SECTION("Gradient")
  {
    test_gradient(
        [&](const Eigen::VectorXd &X) {
          Vector3d n0 = (X.segment<3>(6) - X.segment<3>(0)).cross(X.segment<3>(6) - X.segment<3>(3));
          Vector3d n1 = (X.segment<3>(9) - X.segment<3>(3)).cross(X.segment<3>(9) - X.segment<3>(0));
          return signed_angle(n0, n1, X.segment<3>(3) - X.segment<3>(0));
        },
        [&](const auto &X) {
          return HingeElement<>::bendAngleGradient(Map<Mat3<double>>(const_cast<double *>(X.data()), 4, 3), indices);
        },
        12);
  }
  SECTION("Hessian")
  {
    test_hessian(
        [&](const auto &X) {
          return HingeElement<>::bendAngleGradient(Map<Mat3<double>>(const_cast<double *>(X.data()), 4, 3), indices);
        },
        [&](const auto &X) {
          Mat<double, 12, 12> dderiv;
          HingeElement<>::bendAngleGradient(Map<Mat3<double>>(const_cast<double *>(X.data()), 4, 3), indices, &dderiv);
          return dderiv;
        },
        12);
  }
}