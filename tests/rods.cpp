#include "catch.hpp"
#include "helpers.h"

#include <fsim/ElasticRod.h>
#include <fsim/RodCollection.h>
#include <fsim/RodStencil.h>
#include <fsim/util/geometry.h>

using namespace fsim;

TEST_CASE("ElasticRod")
{
  using namespace Eigen;

  Mat3<double> V = GENERATE(take(5, matrix_random(3, 3)));
  VectorXd params = GENERATE(take(5, vector_random(4)));
  // bool closed = GENERATE(true, false);
  CrossSection crossSection = GENERATE(CrossSection::Circle, CrossSection::Square);
  Vector3d n = Vector3d::UnitZ();
  ElasticRod rod(V, n, {params(0), params(1), params(2), params(3), crossSection});

  SECTION("getReferenceDirectors")
  {
    Mat3<double> D1, D2, D3, D4;
    // if(closed)
    //   ElasticRod<>::bishopFrame(V, (VectorXi(4) << 0, 1, 2, 0).finished(), n, D1, D2);
    // else
      ElasticRod<>::bishopFrame(V, Vector3i(0, 1, 2), n, D1, D2);

    rod.getReferenceDirectors(D3, D4);
    REQUIRE_THAT(D1, ApproxEquals(D3));
    REQUIRE_THAT(D2, ApproxEquals(D4));

    REQUIRE_THAT(D1.row(1).transpose(), ApproxEquals(
      parallel_transport(D1.row(0), (V.row(1) - V.row(0)).normalized(), (V.row(2) - V.row(1)).normalized())));

    REQUIRE_THAT(D2.row(1).transpose(), ApproxEquals(
      parallel_transport(D2.row(0), (V.row(1) - V.row(0)).normalized(), (V.row(2) - V.row(1)).normalized())));

    Vector2d theta = GENERATE(take(5, vector_random(2)));
    rod.getRotatedDirectors(theta, D3, D4);

    for(int i = 0; i < 2; ++i)
    {
      REQUIRE(D1.row(i).dot(D2.row(i)) == Approx(0).margin(1e-10));
      REQUIRE(D1.row(i).norm() == Approx(1).margin(1e-10));
      REQUIRE(D2.row(i).norm() == Approx(1).margin(1e-10));
      REQUIRE_THAT(D3.row(i).transpose(), 
        ApproxEquals(AngleAxis<double>(theta(i), 
          (V.row(i + 1) - V.row(i)).normalized()) * D1.row(i).transpose()).epsilon(1e-6));
      REQUIRE_THAT(D4.row(i).transpose(), 
        ApproxEquals(AngleAxis<double>(theta(i), 
          (V.row(i + 1) - V.row(i)).normalized()) * D2.row(i).transpose()).epsilon(1e-6));
    }
  }

  SECTION("setMass")
  {
    rod.setMass(2);
    REQUIRE(rod.getMass() == 2);
  }

  SECTION("stiffness")
  {
    VectorXd params = GENERATE(take(5, vector_random(3)));
    rod.setParams({params(0), params(1), params(2)});

    Vector2d s = rod.stiffness();
    REQUIRE(s(0) == Approx(pow(params(0), 3) * params(1) * 3.1415 * params(2) / 64));
    REQUIRE(s(1) == Approx(pow(params(1), 3) * params(0) * 3.1415 * params(2) / 64));
  }

  SECTION("Orthonormal frames")
  {
    VectorXd X = GENERATE(take(5, vector_random(9)));
    Vector2d theta = GENERATE(take(5, vector_random(2)));

    Vector3d t0 = (X.segment<3>(3) - X.segment<3>(0)).normalized();
    Vector3d t1 = (X.segment<3>(6) - X.segment<3>(3)).normalized();

    rod.updateProperties(X);
    Mat3<double> D1, D2;
    rod.getReferenceDirectors(D1, D2);

    REQUIRE(t0.dot(D1.row(0)) == Approx(0).margin(1e-10));
    REQUIRE(t0.dot(D2.row(0)) == Approx(0).margin(1e-10));
    REQUIRE(t1.dot(D1.row(1)) == Approx(0).margin(1e-10));
    REQUIRE(t1.dot(D2.row(1)) == Approx(0).margin(1e-10));

    rod.getRotatedDirectors(theta, D1, D2);

    REQUIRE(t0.dot(D1.row(0)) == Approx(0).margin(1e-10));
    REQUIRE(t0.dot(D2.row(0)) == Approx(0).margin(1e-10));
    REQUIRE(t1.dot(D1.row(1)) == Approx(0).margin(1e-10));
    REQUIRE(t1.dot(D2.row(1)) == Approx(0).margin(1e-10));

    REQUIRE_THAT(D1.block(0, 0, 2, 3).rowwise().norm(), ApproxEquals(Vector2d(1, 1)));
    REQUIRE_THAT(D2.block(0, 0, 2, 3).rowwise().norm(), ApproxEquals(Vector2d(1, 1)));
  }

  SECTION("Translate invariance")
  {
    VectorXd var = GENERATE(take(5, vector_random(4 * 3 - 1)));

    Vector3d randomDir = GENERATE(take(5, vector_random(3)));
    VectorXd var2 = var;
    for(int i = 0; i < 3; ++i)
      var2.segment<3>(3 * i) += randomDir;

    rod.setMass(0);
    REQUIRE(rod.energy(var) == Approx(rod.energy(var2)).epsilon(1e-10));
  }

  // This test fails systematically, probably due to the incorrect averaging of material directors for the bending part
  // (the above section proves, however, that the curvature binormal itself is rotationally invariant)
  // Since the twist part also fails this test, there might be a problem with the parallel transport and/or the
  // reference twist update
  // SECTION("Rotation invariance")
  // {
  //   VectorXd var = GENERATE(take(5, vector_random(11)));

  //   Vector3d axis = GENERATE(take(5, vector_random(3))).normalized();
  //   double angle = GENERATE(take(5, random(0., M_PI)));
  //   AngleAxisd rotation(angle, axis);

  //   VectorXd var2 = var;
  //   for(int i = 0; i < 3; ++i)
  //     var2.segment<3>(3 * i) = rotation * var.segment<3>(3 * i);

  //   REQUIRE(rod.energy(var) == Approx(rod.energy(var2)).margin(1e-10));
  // }

  // SECTION("Rods are independent of index ordering")
  // {
  //   ElasticRod e2(V, Vector3i(2, 1, 0), n, W_n, W_n, young_modulus);
  //   VectorXd X = GENERATE(take(10, vector_random(11)));
  //   VectorXd Y = X;
  //   Y(9) = -X(10);
  //   Y(10) = -X(9);

  //   REQUIRE(rod.energy(X) == Approx(e2.energy(Y)));

  //   rod.updateProperties(X);
  //   e2.updateProperties(Y);

  //   VectorXd grad_e = rod.gradient(X);
  //   VectorXd grad_e2 = e2.gradient(Y);

  //   MatrixXd hess_e = rod.hessian(X);
  //   MatrixXd hess_e2 = e2.hessian(Y);

  //   REQUIRE_THAT(grad_e.head(9), ApproxEquals(grad_e2.head(9)));
  //   REQUIRE(grad_e(9) == Approx(grad_e2(10)));
  //   REQUIRE(grad_e(10) == Approx(grad_e2(9)));
  //   REQUIRE_THAT(hess_e.block(0,0,9,9), ApproxEquals(hess_e2.block(0,0,9,9)));
  // }
}

TEST_CASE("RodCollection")
{
  using namespace Eigen;

  MatrixXd V = GENERATE(take(5, matrix_random(3, 3)));
  V.col(2).setZero();
  VectorXd params = GENERATE(take(5, vector_random(3)));
  Mat3<double> N(2, 3);
  N << 0, 0, 1, 0, 0, 1;
  std::vector<std::vector<int>> indices = {{0, 1}, {1, 2}};
  Mat2<int> C(2, 2);
  C << 0, 1, 1, 0;
  RodCollection rod(V, indices, C, N, {params(0), params(1), params(2)});

  SECTION("Orthonormal frames")
  {
    VectorXd X = GENERATE(take(10, vector_random(9)));
    Vector2d theta = GENERATE(take(1, vector_random(2)));

    Vector3d t0 = (X.segment<3>(3) - X.segment<3>(0)).normalized();
    Vector3d t1 = (X.segment<3>(6) - X.segment<3>(3)).normalized();

    rod.updateProperties(X);
    Mat3<double> D1, D2;
    rod.getReferenceDirectors(D1, D2);

    REQUIRE(t0.dot(D1.row(0)) == Approx(0).margin(1e-10));
    REQUIRE(t0.dot(D2.row(0)) == Approx(0).margin(1e-10));
    REQUIRE(t1.dot(D1.row(1)) == Approx(0).margin(1e-10));
    REQUIRE(t1.dot(D2.row(1)) == Approx(0).margin(1e-10));

    rod.getRotatedDirectors(theta, D1, D2);

    REQUIRE(t0.dot(D1.row(0)) == Approx(0).margin(1e-10));
    REQUIRE(t0.dot(D2.row(0)) == Approx(0).margin(1e-10));
    REQUIRE(t1.dot(D1.row(1)) == Approx(0).margin(1e-10));
    REQUIRE(t1.dot(D2.row(1)) == Approx(0).margin(1e-10));

    REQUIRE_THAT(D1.rowwise().norm(), ApproxEquals(Vector2d(1, 1)));
    REQUIRE_THAT(D2.rowwise().norm(), ApproxEquals(Vector2d(1, 1)));
  }

  SECTION("Translate invariance")
  {
    VectorXd var = GENERATE(take(5, vector_random(11)));

    Vector3d randomDir = GENERATE(take(5, vector_random(3)));
    VectorXd var2 = var;
    for(int i = 0; i < 3; ++i)
      var2.segment<3>(3 * i) += randomDir;

    REQUIRE(rod.energy(var) == Approx(rod.energy(var2)).epsilon(1e-10));
  }
}

// TEST_CASE("Class equivalences")
// {
//   using namespace Eigen;

//   double young_modulus = 1;
//   MatrixXd V = GENERATE(take(5, matrix_random(3, 3)));
//   Vector2d W_n = GENERATE(take(5, vector_random(2, 0, 1)));
//   Vector2d W_b = GENERATE(take(5, vector_random(2, 0, 1)));
//   MatrixXd n = GENERATE(take(5, vector_random(3))).normalized();

//   Mat3<double> D1, D2;
//   ElasticRod<>::bishopFrame(V, Vector3i(0, 1, 2), n, D1, D2);
//   LocalFrame f1{(V.row(1) - V.row(0)).normalized(), D1.row(0), D2.row(0)};
//   LocalFrame f2{(V.row(2) - V.row(1)).normalized(), D1.row(1), D2.row(1)};

//   VectorXi dofs(5);
//   dofs << 0, 1, 2, 9, 10;
//   RodStencil stencil(V, f1, f2, dofs, Vector2d(W_n.sum() / 2, W_b.sum() / 2), young_modulus);

//   VectorXd X = GENERATE(take(5, vector_random(11)));
//   LocalFrame new_f1 = updateFrame(f1, X.segment<3>(0), X.segment<3>(3));
//   LocalFrame new_f2 = updateFrame(f2, X.segment<3>(3), X.segment<3>(6));

//   stencil.updateReferenceTwist(new_f1, new_f2);

//   SECTION("RodStencil ~ RodCollection")
//   {
//     RodCollection rodCol(V, {{0, 1}, {1, 2}}, 
//       (MatrixX2i(2, 2) << 0, 1, 1, 0).finished(), 
//       (Mat3<double>(2, 3) << n.transpose(), n.transpose()).finished(), 
//       {W_n(0), W_n(1)}, {W_b(0), W_b(1)}, young_modulus);
//     rodCol.updateProperties(X);

//     SECTION("Energy") { REQUIRE(rodCol.energy(X) == Approx(stencil.energy(X, new_f1, new_f2))); }
//     SECTION("Gradient") { REQUIRE_THAT(rodCol.gradient(X), ApproxEquals(stencil.gradient(X, new_f1, new_f2))); }
//     SECTION("Hessian") { REQUIRE_THAT(rodCol.hessian(X), ApproxEquals(stencil.hessian(X, new_f1, new_f2))); }
//   }
// }

TEST_CASE("Parallel transport")
{
  using namespace Eigen;

  Vector3d t1 = GENERATE(take(50, vector_random(3)));
  Vector3d t2 = GENERATE(take(50, vector_random(3)));
  Vector3d u = GENERATE(take(50, vector_random(3)));

  t1.normalize();
  t2.normalize();

  Vector3d res1 = parallel_transport(u, t1, t2);

  REQUIRE(res1.dot(t2) == Approx(u.dot(t1)).margin(1e-10));

  SECTION("expansion for t1 ~ t2")
  {
    double c = t1.dot(t2);
    Vector3d b = t1.cross(t2);

    if(c > 0.99)
    {
      Vector3d res2 =
          (t2 * (t1 - c * t2).dot(u) + (2 * c * t2 - t1) * u.dot(-c * t1 + t2) + b * b.dot(u)) / b.squaredNorm();

      INFO("cos = " << c)
      REQUIRE((res1 - res2).norm() == Approx(0).margin(1e-6));
    }
  }
}

TEST_CASE("Local Frames")
{
  using namespace Eigen;
  Vector3d t = GENERATE(take(5, vector_random(3))).normalized();
  Vector3d n = GENERATE(take(5, vector_random(3))).normalized();
  n = (n - n.dot(t) * t).normalized(); // make sure the frame is orthogonal
  LocalFrame f{t, n, t.cross(n)};

  Vector3d x0 = GENERATE(take(5, vector_random(3)));
  Vector3d x1 = GENERATE(take(5, vector_random(3)));
  f.update(x0, x1);

  // vectors stay normal
  REQUIRE(f.t.norm() == Approx(1));
  REQUIRE(f.d1.norm() == Approx(1));
  REQUIRE(f.d2.norm() == Approx(1));
  // vectors stay orthogonal
  REQUIRE(f.t.dot(f.d1) == Approx(0).margin(1e-10));
  REQUIRE(f.d1.dot(f.d2) == Approx(0).margin(1e-10));
  REQUIRE(f.d2.dot(f.t) == Approx(0).margin(1e-10));
}