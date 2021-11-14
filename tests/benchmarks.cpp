//
// Created by djourdan on 2/25/20.
//

#include "catch.hpp"
#include "helpers.h"

#include <fsim/ElasticShell.h>
#include <fsim/IncompressibleNeoHookeanMembrane.h>
#include <fsim/MassSpring.h>
#include <fsim/NeoHookeanMembrane.h>
#include <fsim/OrthotropicStVKMembrane.h>
#include <fsim/StVKMembrane.h>
#include <fsim/util/io.h>

using namespace fsim;

TEMPLATE_TEST_CASE("ElasticMembrane", "", NeoHookeanMembrane<>, IncompressibleNeoHookeanMembrane<>)
{
  using namespace Eigen;

  Mat3<double> V;
  Mat3<int> F;
  readOFF("../tests/mesh.off", V, F);

  TestType membrane(V, F, 10, 0.3, 0.1);

  static int nX = 3 * V.rows();
  BENCHMARK_ADVANCED("hessian triplets")(Catch::Benchmark::Chronometer meter)
  {
    VectorXd X = GENERATE(take(1, vector_random(nX)));
    meter.measure([&membrane, &X] { return membrane.hessianTriplets(X); });
  };
}

TEST_CASE("StVKMembrane")
{
  using namespace Eigen;

  Mat3<double> V;
  Mat3<int> F;
  readOFF("../tests/mesh.off", V, F);

  StVKMembrane membrane(V.leftCols(2), F, 10, 0.3, 0.1);

  static int nX = 3 * V.rows();
  BENCHMARK_ADVANCED("hessian triplets")(Catch::Benchmark::Chronometer meter)
  {
    VectorXd X = GENERATE(take(1, vector_random(nX)));
    meter.measure([&membrane, &X] { return membrane.hessianTriplets(X); });
  };
}

TEST_CASE("OrthotropicStVKMembrane")
{
  using namespace Eigen;

  Mat3<double> V;
  Mat3<int> F;
  readOFF("../tests/mesh.off", V, F);

  OrthotropicStVKMembrane membrane(V.leftCols(2), F, 10, 5, 0.3, 0.1);

  static int nX = 3 * V.rows();
  BENCHMARK_ADVANCED("membrane triplets")(Catch::Benchmark::Chronometer meter)
  {
    VectorXd X = GENERATE(take(1, vector_random(nX)));
    meter.measure([&membrane, &X] { return membrane.hessianTriplets(X); });
  };
}

TEMPLATE_TEST_CASE("ElasticShell", "", SquaredAngleFormulation, TanAngleFormulation)
{
  using namespace Eigen;

  Mat3<double> V;
  Mat3<int> F;
  readOFF("../tests/mesh.off", V, F);

  ElasticShell<TestType> shell(V, F, 10, 0.3, 0.1);

  static int nX = 3 * V.rows();
  BENCHMARK_ADVANCED("shell triplets")(Catch::Benchmark::Chronometer meter)
  {
    VectorXd X = GENERATE(take(1, vector_random(nX)));
    meter.measure([&shell, &X] { return shell.hessianTriplets(X); });
  };
}

TEST_CASE("MassSpring")
{
  using namespace Eigen;

  Mat3<double> V;
  Mat3<int> F;
  readOFF("../tests/mesh.off", V, F);

  MassSpring membrane(V, F, 10);

  static int nX = 3 * V.rows();
  BENCHMARK_ADVANCED("membrane triplets")(Catch::Benchmark::Chronometer meter)
  {
    VectorXd X = GENERATE(take(1, vector_random(nX)));
    meter.measure([&membrane, &X] { return membrane.hessianTriplets(X); });
  };
}