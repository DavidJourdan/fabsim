#include "catch.hpp"
#include "helpers.h"

#include <fsim/Spring.h>

TEST_CASE("Spring", "[Spr]")
{
  using namespace Eigen;

  double rest_length = GENERATE(take(10, random(0., 1.)));
  Spring<true> e(0, 1, rest_length);

  std::uniform_real_distribution<double> dis(-1, 1);
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()

  SECTION("Gradient")
  {
    VectorXd var(6);
    Vector3d gradient_computed = Vector3d::Zero();
    Vector3d gradient_numerical = Vector3d::Zero();

    for(int i = 0; i < 20; ++i)
    {
      var.setRandom();
      gradient_computed += -e.force(var);
      gradient_numerical += finite_differences([&e](const Eigen::VectorXd &X) { return e.energy(X); }, var).head(3);
    }
    gradient_computed /= 20;
    gradient_numerical /= 20;
    VectorXd diff = gradient_computed - gradient_numerical;

    INFO("Numerical gradient\n" << gradient_numerical);
    INFO("Computed gradient\n" << gradient_computed);
    INFO("Difference\n" << diff);
    INFO("Average error: " << diff.cwiseAbs().sum() / var.size());

    int j;
    double max_error = diff.cwiseAbs().maxCoeff(&j);
    INFO("Max error at " << j << ": " << max_error);

    REQUIRE(diff.norm() / gradient_numerical.norm() == Approx(0.0).margin(1e-6));
  }

  SECTION("Hessian")
  {
    VectorXd var(6);
    Matrix3d hessian_computed = Matrix3d::Zero();
    Matrix3d hessian_numerical = Matrix3d::Zero();

    for(int i = 0; i < 20; ++i)
    {
      var.setRandom();
      hessian_computed += e.hessian(var);
      hessian_numerical +=
          MatrixXd(finite_differences([&e](const Eigen::VectorXd &X) { return e.force(X); }, var)).block<3, 3>(0, 0);
    }

    hessian_computed /= 20;
    hessian_numerical /= 20;
    MatrixXd diff = hessian_computed - hessian_numerical;

    INFO("Numerical hessian\n" << hessian_numerical);
    INFO("Computed hessian\n" << hessian_computed);
    INFO("Difference\n" << diff);
    INFO("Average error: " << std::sqrt(diff.squaredNorm() / diff.nonZeros()));
    int row, col;
    double max_error = diff.cwiseAbs().maxCoeff(&row, &col);
    INFO("Max error at at (" << row << ", " << col << "): " << max_error);

    REQUIRE(diff.norm() / hessian_numerical.norm() == Approx(0.0).margin(1e-6));
  }

  SECTION("Translate invariance")
  {
    VectorXd var = GENERATE(take(10, vector_random(6)));

    Vector3d randomDir = Vector3d::NullaryExpr(3, [&]() { return dis(gen); });
    VectorXd var2 = var;
    for(int i = 0; i < 2; ++i)
      var2.segment<3>(3 * i) = var.segment<3>(3 * i) + randomDir;

    REQUIRE(e.energy(var) == Approx(e.energy(var2)).epsilon(1e-10));
  }

  SECTION("Rotation invariance")
  {
    VectorXd var = GENERATE(take(10, vector_random(6)));

    Vector3d axis = Vector3d::NullaryExpr(3, [&]() { return dis(gen); }).normalized();
    double angle = GENERATE(take(10, random(0., M_PI)));
    AngleAxisd rotation(angle, axis);

    VectorXd var2 = var;
    for(int i = 0; i < 2; ++i)
      var2.segment<3>(3 * i) = rotation * var.segment<3>(3 * i);

    REQUIRE(e.energy(var) == Approx(e.energy(var2)).epsilon(1e-10));
  }
}
