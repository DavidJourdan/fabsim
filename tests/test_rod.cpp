#include "catch.hpp"
#include "helpers.h"

#include <fsim/ElasticRod.h>
#include <fsim/RodCollection.h>
#include <fsim/RodStencil.h>
#include <fsim/util/geometry.h>

using namespace fsim;

TEST_CASE("Local Frames")
{
  using namespace Eigen;
  Vector3d t = GENERATE(take(5, vector_random(3))).normalized();
  Vector3d n = GENERATE(take(5, vector_random(3))).normalized();
  n = (n - n.dot(t) * t).normalized(); // make sure the frame is orthogonal
  LocalFrame<double> f{t, n, t.cross(n)};

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

struct BundledRodStencil
{
  BundledRodStencil(const Eigen::Ref<const Mat3<double>> V,
                    const Eigen::Vector3d &t,
                    const Eigen::Vector3d &n,
                    const Eigen::Matrix<int, 5, 1> &dofs,
                    const Eigen::Vector2d &widths,
                    double young_modulus)
      : f1(t, n, t.cross(n)), f2(f1), stencil(V, f1, f2, dofs, widths, young_modulus)
  {
    f2.update(V.row(dofs(1)), V.row(dofs(2)));
  }

  constexpr int nb_dofs() const { return stencil.nb_dofs(); }

  void prepare_data(const Eigen::Ref<const Eigen::VectorXd> X) const
  {
    f1.update(X.segment<3>(3 * stencil.idx(0)), X.segment<3>(3 * stencil.idx(1)));
    f2.update(X.segment<3>(3 * stencil.idx(1)), X.segment<3>(3 * stencil.idx(2)));

    stencil.update_reference_twist(f1, f2);
  }

  double energy(const Eigen::Ref<const Eigen::VectorXd> X) const
  {
    LocalFrame<double> new_f1 =
        update_frame<double>(f1, X.segment<3>(3 * stencil.idx(0)), X.segment<3>(3 * stencil.idx(1)));
    LocalFrame<double> new_f2 =
        update_frame<double>(f2, X.segment<3>(3 * stencil.idx(1)), X.segment<3>(3 * stencil.idx(2)));

    double ref_twist = stencil.get_reference_twist();
    stencil.update_reference_twist(new_f1, new_f2);
    double res = stencil.energy(X, new_f1, new_f2);
    stencil.set_reference_twist(ref_twist);
    return res;
  }
  RodStencil::LocalVector gradient(const Eigen::Ref<const Eigen::VectorXd> X) const
  {
    LocalFrame<double> new_f1 =
        update_frame<double>(f1, X.segment<3>(3 * stencil.idx(0)), X.segment<3>(3 * stencil.idx(1)));
    LocalFrame<double> new_f2 =
        update_frame<double>(f2, X.segment<3>(3 * stencil.idx(1)), X.segment<3>(3 * stencil.idx(2)));

    double ref_twist = stencil.get_reference_twist();
    stencil.update_reference_twist(new_f1, new_f2);
    auto res = stencil.gradient(X, new_f1, new_f2);
    stencil.set_reference_twist(ref_twist);
    return res;
  }
  RodStencil::LocalMatrix hessian(const Eigen::Ref<const Eigen::VectorXd> X) const
  {
    return stencil.hessian(X, f1, f2);
  }

  static const int NB_VERTICES = 3;
  static const int NB_DOFS = 11;

  mutable RodStencil stencil;
  mutable LocalFrame<double> f1;
  mutable LocalFrame<double> f2;
};

TEST_CASE("RodStencil")
{
  using namespace Eigen;

  double young_modulus = 1;
  MatrixXd V = GENERATE(take(5, matrix_random(3, 3)));
  Vector2d widths = GENERATE(take(1, vector_random(2, 0, 1)));

  Vector3d n1 = GENERATE(take(5, vector_random(3))).normalized();
  Vector3d t1 = (V.row(1) - V.row(0)).normalized();
  n1 = (n1 - n1.dot(t1) * t1).normalized(); // make sure the frame is orthogonal

  VectorXi dofs(5);
  dofs << 0, 1, 2, 9, 10;

  BundledRodStencil rod(V, t1, n1, dofs, widths, young_modulus);

  SECTION("Gradient") { test_gradient(rod, 1e-5); }
  // SECTION("Hessian") 
  // { 
  //   VectorXd var(rod.nb_dofs());
  //   MatrixXd hessian_computed = MatrixXd::Zero(rod.nb_dofs(), rod.nb_dofs());
  //   MatrixXd hessian_numerical = MatrixXd::Zero(rod.nb_dofs(), rod.nb_dofs());

  //   std::uniform_real_distribution<double> dis(-1, 1);
  //   std::mt19937 gen(std::random_device{}()); // Standard mersenne_twister_engine seeded with rd()
  //   for(int i = 0; i < 10; ++i)
  //   {
  //     var = VectorXd::NullaryExpr(rod.nb_dofs(), [&]() { return dis(gen); });
  //     rod.prepare_data(var);

  //     hessian_computed += MatrixXd(rod.hessian(var));
  //     hessian_numerical += sym(MatrixXd(finite_differences([&rod](const Eigen::VectorXd &X) { return rod.gradient(X); }, var)));
  //   }

  //   hessian_computed /= 10;
  //   hessian_numerical /= 10;
  //   MatrixXd diff = hessian_computed - hessian_numerical;

  //   INFO("Numerical hessian\n" << hessian_numerical);
  //   INFO("Computed hessian\n" << hessian_computed);
  //   INFO("Difference\n" << diff);
  //   REQUIRE(diff.norm() / hessian_numerical.norm() == Approx(0.0).margin(1e-5));
  //  }
  SECTION("Translate invariance") { translate_invariance(rod); }

  // This test fails systematically, probably due to the incorrect averaging of material directors for the bending part
  // (the above section proves, however, that the curvature binormal itself is rotationally invariant)
  // Since the twist part also fails this test, there might be a problem with the parallel transport and/or the
  // reference twist update
  // SECTION("Rotation invariance") { rotational_invariance(rod); }
}

TEST_CASE("Class equivalences")
{
  using namespace Eigen;

  double young_modulus = 1;
  MatrixXd V = GENERATE(take(5, matrix_random(3, 3)));
  Vector2d W_n = GENERATE(take(1, vector_random(2, 0, 1)));
  Vector2d W_b = GENERATE(take(1, vector_random(2, 0, 1)));
  MatrixXd n = GENERATE(take(5, vector_random(3))).normalized();

  Matrix<double, -1, 3, RowMajor> D1, D2;
  ElasticRod::bishop_frame(V, Vector3i(0, 1, 2), n, D1, D2);
  LocalFrame<double> f1{(V.row(1) - V.row(0)).normalized(), D1.row(0), D2.row(0)};
  LocalFrame<double> f2{(V.row(2) - V.row(1)).normalized(), D1.row(1), D2.row(1)};

  VectorXi dofs(5);
  dofs << 0, 1, 2, 9, 10;
  RodStencil stencil(V, f1, f2, dofs, Vector2d(W_n.sum() / 2, W_b.sum() / 2), young_modulus);

  VectorXd X = GENERATE(take(10, vector_random(11)));
  LocalFrame<double> new_f1 = update_frame<double>(f1, X.segment<3>(0), X.segment<3>(3));
  LocalFrame<double> new_f2 = update_frame<double>(f2, X.segment<3>(3), X.segment<3>(6));

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

TEST_CASE("ElasticRod class", "[Rod]")
{
  using namespace Eigen;

  double young_modulus = 1;
  MatrixXd V = GENERATE(take(5, matrix_random(3, 3)));
  V.col(2).setZero();
  Vector2d W_n = GENERATE(take(5, vector_random(2)));
  Vector2d W_b = GENERATE(take(5, vector_random(2)));
  Vector3d n = Vector3d::UnitZ();
  ElasticRod e(V, Vector3i(0, 1, 2), n, W_n, W_n, young_modulus, 0);

  SECTION("Orthonormal frames")
  {
    VectorXd X = GENERATE(take(10, vector_random(9)));
    Vector2d theta = GENERATE(take(1, vector_random(2)));

    Vector3d t0 = (X.segment<3>(3) - X.segment<3>(0)).normalized();
    Vector3d t1 = (X.segment<3>(6) - X.segment<3>(3)).normalized();

    e.update_properties(X);
    Mat3<double> D1, D2;
    e.get_reference_directors(D1, D2);

    REQUIRE(t0.dot(D1.row(0)) == Approx(0).margin(1e-10));
    REQUIRE(t0.dot(D2.row(0)) == Approx(0).margin(1e-10));
    REQUIRE(t1.dot(D1.row(1)) == Approx(0).margin(1e-10));
    REQUIRE(t1.dot(D2.row(1)) == Approx(0).margin(1e-10));

    e.get_rotated_directors(theta, D1, D2);

    REQUIRE(t0.dot(D1.row(0)) == Approx(0).margin(1e-10));
    REQUIRE(t0.dot(D2.row(0)) == Approx(0).margin(1e-10));
    REQUIRE(t1.dot(D1.row(1)) == Approx(0).margin(1e-10));
    REQUIRE(t1.dot(D2.row(1)) == Approx(0).margin(1e-10));

    REQUIRE_THAT(D1.rowwise().norm(), ApproxEquals(Vector2d(1, 1)));
    REQUIRE_THAT(D2.rowwise().norm(), ApproxEquals(Vector2d(1, 1)));
  }

  // SECTION("Rods are independent of index ordering")
  // {
  //   ElasticRod e2(V, Vector3i(2, 1, 0), n, W_n, W_n, young_modulus, 0);
  //   VectorXd X = GENERATE(take(10, vector_random(11)));
  //   VectorXd Y = X;
  //   Y(9) = -X(10);
  //   Y(10) = -X(9);

  //   REQUIRE(e.energy(X) == Approx(e2.energy(Y)));

  //   e.update_properties(X);
  //   e2.update_properties(Y);

  //   VectorXd grad_e = e.gradient(X);
  //   VectorXd grad_e2 = e2.gradient(Y);

  //   MatrixXd hess_e = e.hessian(X);
  //   MatrixXd hess_e2 = e2.hessian(Y);

  //   REQUIRE_THAT(grad_e.head(9), ApproxEquals(grad_e2.head(9)));
  //   REQUIRE(grad_e(9) == Approx(grad_e2(10)));
  //   REQUIRE(grad_e(10) == Approx(grad_e2(9)));
  //   REQUIRE_THAT(hess_e.block(0,0,9,9), ApproxEquals(hess_e2.block(0,0,9,9)));
  // }
}

TEST_CASE("Parallel transport", "[Rod]")
{
  using namespace Eigen;

  Vector3d t1 = GENERATE(take(100, vector_random(3)));
  Vector3d t2 = GENERATE(take(100, vector_random(3)));
  Vector3d u = GENERATE(take(5, vector_random(3)));

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
