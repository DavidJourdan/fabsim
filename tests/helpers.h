//
// Created by djourdan on 2/17/20.
//

#pragma once

#include "catch.hpp"

#include <fsim/util/finite_differences.h>

#include <Eigen/Dense>

#include <random>
#include <type_traits>

namespace detail
{
template <class>
struct sfinae_true : std::true_type
{};

template <class T, class A0>
static auto test_prepare_data(int) -> sfinae_true<decltype(std::declval<T>().prepare_data(std::declval<A0>()))>;
template <class, class A0>
static auto test_prepare_data(long) -> std::false_type;
} // namespace detail

// test if a class has a method called 'prepare_data'
template <class T, class Arg>
struct has_prepare_data : decltype(detail::test_prepare_data<T, Arg>(0))
{};

template <typename Scalar>
struct RandomRange
{
  RandomRange(Scalar low, Scalar high) : dis(low, high), gen()
  {
    gen = std::mt19937(std::random_device{}()); // Standard mersenne_twister_engine
  }
  const Scalar operator()() const { return dis(gen); }
  mutable std::uniform_real_distribution<Scalar> dis;
  mutable std::mt19937 gen;
};

class RandomVectorGenerator : public Catch::Generators::IGenerator<Eigen::VectorXd>
{
public:
  RandomVectorGenerator(int size, double _min, double _max) : dis(_min, _max), gen()
  {
    gen = std::mt19937(std::random_device{}()); // Standard mersenne_twister_engine
    current_value = Eigen::VectorXd::NullaryExpr(size, [this]() { return dis(gen); });
    static_cast<void>(next());
  }
  const Eigen::VectorXd &get() const override;
  bool next() override
  {
    current_value = Eigen::VectorXd::NullaryExpr(current_value.size(), [this]() { return dis(gen); });
    return true;
  }

private:
  std::uniform_real_distribution<double> dis;
  std::mt19937 gen;
  Eigen::VectorXd current_value;
};

// Avoids -Wweak-vtables
const Eigen::VectorXd &RandomVectorGenerator::get() const
{
  return current_value;
}

// This helper function provides a nicer UX when instantiating the generator
// Notice that it returns an instance of GeneratorWrapper<int>, which
// is a value-wrapper around std::unique_ptr<IGenerator<int>>.
Catch::Generators::GeneratorWrapper<Eigen::VectorXd> vector_random(int size, double min = -1, double max = 1)
{
  return Catch::Generators::GeneratorWrapper<Eigen::VectorXd>(
      std::unique_ptr<Catch::Generators::IGenerator<Eigen::VectorXd>>(new RandomVectorGenerator(size, min, max)));
}

class RandomMatrixGenerator : public Catch::Generators::IGenerator<Eigen::MatrixXd>
{
public:
  RandomMatrixGenerator(int nRows, int nCols, double _min, double _max) : dis(_min, _max), gen()
  {
    gen = std::mt19937(std::random_device{}()); // Standard mersenne_twister_engine
    current_value = Eigen::MatrixXd::NullaryExpr(nRows, nCols, [this]() { return dis(gen); });
    static_cast<void>(next());
  }
  const Eigen::MatrixXd &get() const override;
  bool next() override
  {
    current_value =
        Eigen::MatrixXd::NullaryExpr(current_value.rows(), current_value.cols(), [this]() { return dis(gen); });
    return true;
  }

private:
  std::uniform_real_distribution<double> dis;
  std::mt19937 gen;
  Eigen::MatrixXd current_value;
};

// Avoids -Wweak-vtables
const Eigen::MatrixXd &RandomMatrixGenerator::get() const
{
  return current_value;
}

// This helper function provides a nicer UX when instantiating the generator
// Notice that it returns an instance of GeneratorWrapper<int>, which
// is a value-wrapper around std::unique_ptr<IGenerator<int>>.
Catch::Generators::GeneratorWrapper<Eigen::MatrixXd>
matrix_random(int nRows, int nCols, double min = -1, double max = 1)
{
  return Catch::Generators::GeneratorWrapper<Eigen::MatrixXd>(
      std::unique_ptr<Catch::Generators::IGenerator<Eigen::MatrixXd>>(
          new RandomMatrixGenerator(nRows, nCols, min, max)));
}

struct EigenApproxMatcher : Catch::MatcherBase<Eigen::MatrixXd>
{
  EigenApproxMatcher(Eigen::MatrixXd const &comparator) : _comparator(comparator) {}

  bool match(Eigen::MatrixXd const &mat) const override
  {
    if(_comparator.rows() != mat.rows())
      return false;
    if(_comparator.cols() != mat.cols())
      return false;
    for(int i = 0; i < mat.rows(); ++i)
      for(int j = 0; j < mat.cols(); ++j)
        if(_comparator(i, j) != approx(mat(i, j)))
          return false;
    return true;
  }
  // Produces a string describing what this matcher does. It should
  // include any provided data (the begin/ end in this case) and
  // be written as if it were stating a fact (in the output it will be
  // preceded by the value under test).
  virtual std::string describe() const override
  {
    std::ostringstream ss;
    ss << "\napproximately equals:\n" << _comparator;
    return ss.str();
  }

  EigenApproxMatcher &epsilon(double newEpsilon)
  {
    approx.epsilon(newEpsilon);
    return *this;
  }

  EigenApproxMatcher &margin(double newMargin)
  {
    approx.margin(newMargin);
    return *this;
  }

  EigenApproxMatcher &scale(double newScale)
  {
    approx.scale(newScale);
    return *this;
  }

  Eigen::MatrixXd const &_comparator;
  mutable Catch::Detail::Approx approx = Catch::Detail::Approx::custom();
};

// The builder function
inline EigenApproxMatcher ApproxEquals(const Eigen::MatrixXd &M)
{
  return EigenApproxMatcher(M);
}

std::function<bool(const Eigen::VectorXd &)> no_op = [](auto X) { return true; };

template <class Element>
void test_gradient(const Element &e,
                   double eps = 1e-6,
                   const std::function<bool(const Eigen::VectorXd &)> &filter = no_op)
{
  using namespace Eigen;

  VectorXd var(e.nbDOFs());
  VectorXd gradient_computed = VectorXd::Zero(e.nbDOFs());
  VectorXd gradient_numerical = VectorXd::Zero(e.nbDOFs());

  for(int i = 0; i < 10; ++i)
  {
    var = VectorXd::NullaryExpr(e.nbDOFs(), RandomRange(-1.0, 1.0));
    if(filter(var))
    {
      if constexpr(has_prepare_data<Element, Eigen::VectorXd>{})
        e.prepare_data(var);

      REQUIRE_FALSE(isnan(e.energy(var)));

      gradient_computed += e.gradient(var);
      gradient_numerical += fsim::finite_differences([&e](const Eigen::VectorXd &X) { return e.energy(X); }, var);
    }
  }
  gradient_computed /= 10;
  gradient_numerical /= 10;
  VectorXd diff = gradient_computed - gradient_numerical;

  INFO("Numerical gradient\n" << gradient_numerical);
  INFO("Computed gradient\n" << gradient_computed);
  INFO("Difference\n" << diff);
  INFO("Average error: " << diff.cwiseAbs().sum() / var.size());

  int j;
  double max_error = diff.cwiseAbs().maxCoeff(&j);
  INFO("Max error at " << j << ": " << max_error);

  REQUIRE(diff.norm() / gradient_numerical.norm() == Approx(0.0).margin(eps));
}

template <class Element>
void test_hessian(const Element &e,
                  double eps = 1e-5,
                  const std::function<bool(const Eigen::VectorXd &)> &filter = no_op)
{
  using namespace Eigen;
  VectorXd var(e.nbDOFs());
  MatrixXd hessian_computed = MatrixXd::Zero(e.nbDOFs(), e.nbDOFs());
  MatrixXd hessian_numerical = MatrixXd::Zero(e.nbDOFs(), e.nbDOFs());

  for(int i = 0; i < 10; ++i)
  {
    var = VectorXd::NullaryExpr(e.nbDOFs(), RandomRange(-1.0, 1.0));
    if(filter(var))
    {
      if constexpr(has_prepare_data<Element, Eigen::VectorXd>{})
        e.prepare_data(var);

      hessian_computed += MatrixXd(e.hessian(var));
      hessian_numerical += MatrixXd(fsim::finite_differences([&e](const Eigen::VectorXd &X) { return e.gradient(X); }, var));
    }
  }

  hessian_computed /= 10;
  hessian_numerical /= 10;
  MatrixXd diff = hessian_computed - hessian_numerical;

  INFO("Numerical hessian\n" << hessian_numerical);
  INFO("Computed hessian\n" << hessian_computed);
  INFO("Difference\n" << diff);
  INFO("Average error: " << std::sqrt(diff.squaredNorm() / diff.nonZeros()));
  int row, col;
  double max_error = diff.cwiseAbs().maxCoeff(&row, &col);
  INFO("Max error at at (" << row << ", " << col << "): " << max_error);

  REQUIRE(diff.norm() / hessian_numerical.norm() == Approx(0.0).margin(eps));
}

template <class Element>
void rotational_invariance(const Element &e, double eps = 1e-10)
{
  using namespace Eigen;

  VectorXd var = GENERATE(take(10, vector_random(Element::NB_DOFS)));

  Vector3d axis = Vector3d::NullaryExpr(3, RandomRange(-1.0, 1.0)).normalized();
  double angle = GENERATE(take(10, random(0., M_PI)));
  AngleAxisd rotation(angle, axis);

  VectorXd var2 = var;
  for(int i = 0; i < Element::NB_VERTICES; ++i)
    var2.segment<3>(3 * i) = rotation * var.segment<3>(3 * i);

  REQUIRE(e.energy(var) == Approx(e.energy(var2)).epsilon(eps));
}

template <class Element>
void translate_invariance(const Element &e, double eps = 1e-10)
{
  using namespace Eigen;

  VectorXd var = GENERATE(take(10, vector_random(Element::NB_DOFS)));

  Vector3d randomDir = Vector3d::NullaryExpr(3, RandomRange(-1.0, 1.0));
  VectorXd var2 = var;
  for(int i = 0; i < Element::NB_VERTICES; ++i)
    var2.segment<3>(3 * i) = var.segment<3>(3 * i) + randomDir;

  REQUIRE(e.energy(var) == Approx(e.energy(var2)).epsilon(eps));
}
