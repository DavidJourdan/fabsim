//
// Created by djourdan on 2/25/20.
//

#include "catch.hpp"
#include "helpers.h"

#include <fsim/util/io.h>
#include <fsim/ElasticMembraneModel.h>
#include <fsim/DiscreteShell.h>

TEST_CASE("StVKMembrane class", "[StVK]")
{
  using namespace Eigen;

  MatrixX3d V;
  MatrixX3i F;
  read_from_file("../../../tests/mesh.off", V, F);

  StVKMembrane membrane(V, F, 10, 0.3, 0.1, 1.5);

  static int nX = 3 * V.rows();
  VectorXd X = GENERATE(take(10, vector_random(nX)));
  BENCHMARK("hessian triplets")
  {
    return membrane.hessian_triplets(X);
  };
}

TEST_CASE("DiscreteShell class", "[Shell]")
{
  using namespace Eigen;

  MatrixX3d V;
  MatrixX3i F;
  read_from_file("../../../tests/mesh.off", V, F);

  ElasticShell shell(V, F, 10, 0.3, 0.1);

  static int nX = 3 * V.rows();
  VectorXd X = GENERATE(take(10, vector_random(nX)));
  BENCHMARK("shell triplets")
  {
    return shell.hessian_triplets(X);
  };
}