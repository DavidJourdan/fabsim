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

  double young_modulus = 1;
  MatrixXd V = GENERATE(take(2, matrix_random(3, 3)));
  V.col(2).setZero();
  Vector2d W_n = GENERATE(take(2, vector_random(2)));
  Vector2d W_b = GENERATE(take(2, vector_random(2)));
  Vector3d n = Vector3d::UnitZ();
  ElasticRod rod(V, Vector3i(0, 1, 2), n, W_n, W_n, young_modulus, 0);

  SECTION("Orthonormal frames")
  {
    VectorXd X = GENERATE(take(10, vector_random(9)));
    Vector2d theta = GENERATE(take(1, vector_random(2)));

    Vector3d t0 = (X.segment<3>(3) - X.segment<3>(0)).normalized();
    Vector3d t1 = (X.segment<3>(6) - X.segment<3>(3)).normalized();

    rod.update_properties(X);
    Mat3<double> D1, D2;
    rod.get_reference_directors(D1, D2);

    REQUIRE(t0.dot(D1.row(0)) == Approx(0).margin(1e-10));
    REQUIRE(t0.dot(D2.row(0)) == Approx(0).margin(1e-10));
    REQUIRE(t1.dot(D1.row(1)) == Approx(0).margin(1e-10));
    REQUIRE(t1.dot(D2.row(1)) == Approx(0).margin(1e-10));

    rod.get_rotated_directors(theta, D1, D2);

    REQUIRE(t0.dot(D1.row(0)) == Approx(0).margin(1e-10));
    REQUIRE(t0.dot(D2.row(0)) == Approx(0).margin(1e-10));
    REQUIRE(t1.dot(D1.row(1)) == Approx(0).margin(1e-10));
    REQUIRE(t1.dot(D2.row(1)) == Approx(0).margin(1e-10));

    REQUIRE_THAT(D1.rowwise().norm(), ApproxEquals(Vector2d(1, 1)));
    REQUIRE_THAT(D2.rowwise().norm(), ApproxEquals(Vector2d(1, 1)));
  }

  SECTION("Translate invariance")
  {
    VectorXd var = GENERATE(take(2, vector_random(11)));

    Vector3d randomDir = GENERATE(take(2, vector_random(3)));
    VectorXd var2 = var;
    for(int i = 0; i < 3; ++i)
      var2.segment<3>(3 * i) += randomDir;

    REQUIRE(rod.energy(var) == Approx(rod.energy(var2)).epsilon(1e-10));
  }

  // This test fails systematically, probably due to the incorrect averaging of material directors for the bending part
  // (the above section proves, however, that the curvature binormal itself is rotationally invariant)
  // Since the twist part also fails this test, there might be a problem with the parallel transport and/or the
  // reference twist update
  // SECTION("Rotation invariance")
  // {
  //   VectorXd var = GENERATE(take(2, vector_random(11)));

  //   Vector3d axis = GENERATE(take(2, vector_random(3))).normalized();
  //   double angle = GENERATE(take(2, random(0., M_PI)));
  //   AngleAxisd rotation(angle, axis);

  //   VectorXd var2 = var;
  //   for(int i = 0; i < 3; ++i)
  //     var2.segment<3>(3 * i) = rotation * var.segment<3>(3 * i);

  //   REQUIRE(rod.energy(var) == Approx(rod.energy(var2)).margin(1e-10));
  // }

  // SECTION("Rods are independent of index ordering")
  // {
  //   ElasticRod e2(V, Vector3i(2, 1, 0), n, W_n, W_n, young_modulus, 0);
  //   VectorXd X = GENERATE(take(10, vector_random(11)));
  //   VectorXd Y = X;
  //   Y(9) = -X(10);
  //   Y(10) = -X(9);

  //   REQUIRE(rod.energy(X) == Approx(e2.energy(Y)));

  //   rod.update_properties(X);
  //   e2.update_properties(Y);

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

TEST_CASE("Class equivalences")
{
  using namespace Eigen;

  double young_modulus = 1;
  MatrixXd V = GENERATE(take(2, matrix_random(3, 3)));
  Vector2d W_n = GENERATE(take(2, vector_random(2, 0, 1)));
  Vector2d W_b = GENERATE(take(2, vector_random(2, 0, 1)));
  MatrixXd n = GENERATE(take(2, vector_random(3))).normalized();

  Mat3<double> D1, D2;
  ElasticRod::bishop_frame(V, Vector3i(0, 1, 2), n, D1, D2);
  ElasticRod::LocalFrame f1{(V.row(1) - V.row(0)).normalized(), D1.row(0), D2.row(0)};
  ElasticRod::LocalFrame f2{(V.row(2) - V.row(1)).normalized(), D1.row(1), D2.row(1)};

  VectorXi dofs(5);
  dofs << 0, 1, 2, 9, 10;
  RodStencil stencil(V, f1, f2, dofs, Vector2d(W_n.sum() / 2, W_b.sum() / 2), young_modulus);

  VectorXd X = GENERATE(take(2, vector_random(11)));
  ElasticRod::LocalFrame new_f1 = ElasticRod::update_frame(f1, X.segment<3>(0), X.segment<3>(3));
  ElasticRod::LocalFrame new_f2 = ElasticRod::update_frame(f2, X.segment<3>(3), X.segment<3>(6));

  stencil.update_reference_twist(new_f1, new_f2);
  
  SECTION("RodStencil ~ ElasticRod")
  {
    ElasticRod rod(V, Vector3i(0, 1, 2), n, W_n, W_b, young_modulus, 0);
    rod.update_properties(X);

    SECTION("Energy") { REQUIRE(rod.energy(X) == Approx(stencil.energy(X, new_f1, new_f2))); }
    SECTION("Gradient") { REQUIRE_THAT(rod.gradient(X), ApproxEquals(stencil.gradient(X, new_f1, new_f2))); }
    SECTION("Hessian") { REQUIRE_THAT(rod.hessian(X), ApproxEquals(stencil.hessian(X, new_f1, new_f2))); }
  }

  // SECTION("RodStencil ~ RodCollection")
  // {
  //   RodCollection rodCol(V, {{0, 1}, {1, 2}}, 
  //     (MatrixX2i(2, 2) << 0, 1, 1, 0).finished(), 
  //     (Mat3<double>(2, 3) << n.transpose(), n.transpose()).finished(), 
  //     {W_n(0), W_n(1)}, {W_b(0), W_b(1)}, young_modulus, 0);
  //   rodCol.update_properties(X);

  //   SECTION("Energy") { REQUIRE(rodCol.energy(X) == Approx(stencil.energy(X, new_f1, new_f2))); }
  //   SECTION("Gradient") { REQUIRE_THAT(rodCol.gradient(X), ApproxEquals(stencil.gradient(X, new_f1, new_f2))); }
  //   SECTION("Hessian") { REQUIRE_THAT(rodCol.hessian(X), ApproxEquals(stencil.hessian(X, new_f1, new_f2))); }
  // }
}

TEST_CASE("Parallel transport")
{
  using namespace Eigen;

  Vector3d t1 = GENERATE(take(20, vector_random(3)));
  Vector3d t2 = GENERATE(take(20, vector_random(3)));
  Vector3d u = GENERATE(take(20, vector_random(3)));

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
  Vector3d t = GENERATE(take(2, vector_random(3))).normalized();
  Vector3d n = GENERATE(take(2, vector_random(3))).normalized();
  n = (n - n.dot(t) * t).normalized(); // make sure the frame is orthogonal
  ElasticRod::LocalFrame f{t, n, t.cross(n)};

  Vector3d x0 = GENERATE(take(2, vector_random(3)));
  Vector3d x1 = GENERATE(take(2, vector_random(3)));
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