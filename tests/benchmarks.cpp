//
// Created by djourdan on 2/25/20.
//

#include "catch.hpp"
#include "helpers.h"

#include <fsim/util/io.h>
#include <fsim/ElasticMembraneModel.h>
#include <fsim/DiscreteShell.h>
#include <fsim/MassSpring.h>
#include <fsim/OrthotropicStVKModel.h>

using namespace fsim;

TEMPLATE_TEST_CASE("ElasticMembraneModel", "", StVKMembrane, NeoHookeanMembrane, NHIncompressibleMembrane)
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
    meter.measure([&membrane, &X] { return membrane.hessian_triplets(X); });
  };
}

TEST_CASE("OrthotropicStVKModel")
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
    meter.measure([&membrane, &X] { return membrane.hessian_triplets(X); });
  };
}

TEMPLATE_TEST_CASE("DiscreteShell", "", SquaredAngleFormulation, TanAngleFormulation)
{
  using namespace Eigen;

  Mat3<double> V;
  Mat3<int> F;
  readOFF("../tests/mesh.off", V, F);

  DiscreteShell<TestType, true> shell(V, F, 10, 0.3, 0.1);

  static int nX = 3 * V.rows();
  BENCHMARK_ADVANCED("shell triplets")(Catch::Benchmark::Chronometer meter)
  {
    VectorXd X = GENERATE(take(1, vector_random(nX)));
    meter.measure([&shell, &X] { return shell.hessian_triplets(X); });
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
    meter.measure([&membrane, &X] { return membrane.hessian_triplets(X); });
  };
}