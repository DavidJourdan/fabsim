// finite_differences.cpp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 04/10/18

#include "fsim/util/finite_differences.h"

#include "fsim/util/filter_var.h"

#include <cmath> // std::abs
#include <iostream>

namespace fsim
{

Eigen::VectorXd finite_differences(const std::function<double(const Eigen::Ref<const Eigen::VectorXd>)> &func,
                                   const Eigen::Ref<const Eigen::VectorXd> var,
                                   bool parallelism_enabled)
{
  using namespace Eigen;
  const double epsilon = 1.1e-8;
  int n = var.size();

  VectorXd derivative = VectorXd::Constant(n, -func(var));

#pragma omp parallel for if(n > 200 && parallelism_enabled)
  for(int i = 0; i < n; ++i)
  {
    VectorXd e = VectorXd::Zero(var.size());
    e(i) = epsilon;
    derivative(i) += func(var + e); // derivative(i) = func(var + e) - func(var)
  }
  derivative /= epsilon;

  return derivative;
}

Eigen::MatrixXd finite_differences(const std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd>)> &func,
                                   const Eigen::Ref<const Eigen::VectorXd> var,
                                   bool parallelism_enabled)
{
  using namespace Eigen;
  const double epsilon = 1.1e-8;
  int n = var.size();

  VectorXd func_var = func(var);
  MatrixXd derivative(func_var.size(), n);

#pragma omp parallel for if(n > 50 && parallelism_enabled)
  for(int i = 0; i < n; ++i)
  {
    VectorXd e = VectorXd::Zero(n);
    e(i) = epsilon;
    VectorXd diff = (func(var + e) - func_var) / epsilon;

    derivative.col(i) = diff;
  }
  return derivative;
}

Eigen::SparseMatrix<double>
finite_differences_sparse(const std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd>)> &func,
                          const Eigen::Ref<const Eigen::VectorXd> var,
                          bool parallelism_enabled)
{
  using namespace Eigen;
  const double epsilon = 1.1e-8;
  int n = var.size();

  VectorXd func_var = func(var);
  SparseMatrix<double, ColMajor> derivative(func_var.size(), n);

#pragma omp parallel for if(n > 50 && parallelism_enabled)
  for(int i = 0; i < n; ++i)
  {
    VectorXd e = VectorXd::Zero(n);
    e(i) = epsilon;
    VectorXd diff = (func(var + e) - func_var) / epsilon;

#pragma omp critical
    derivative.col(i) = diff.sparseView();
  }
  derivative.makeCompressed();

  return derivative;
}

// void derivative_check(const std::function<double(const Eigen::Ref<const Eigen::VectorXd>)> &func,
//                       const std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd>)> &derivative,
//                       const Eigen::Ref<const Eigen::VectorXd> var,
//                       bool dump_matrices)
// {
//   std::vector<int> empty_container{};
//   derivative_check(func, derivative, var, empty_container, dump_matrices);
// }

// void derivative_check(const std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd>)> &func,
//                       const std::function<Eigen::MatrixXd(const Eigen::Ref<const Eigen::VectorXd>)> &derivative,
//                       const Eigen::Ref<const Eigen::VectorXd> var,
//                       bool dump_matrices)
// {
//   std::vector<int> empty_container{};
//   derivative_check(func, derivative, var, empty_container, dump_matrices);
// }

// void derivative_check(const std::function<double(const Eigen::Ref<const Eigen::VectorXd>)> &func,
//                       const std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd>)> &derivative,
//                       const Eigen::Ref<const Eigen::VectorXd> var,
//                       const std::vector<int> &indices_to_be_filtered,
//                       bool dump_matrices)
// {
//   using namespace Eigen;
//   VectorXd diff = finite_differences(func, var);
//   filter_var(diff, indices_to_be_filtered);

//   if(dump_matrices)
//   {
//     std::cout << "FINITE DIFFERENCES\n" << std::defaultfloat;
//     std::cout << diff.transpose() << "\n";
//     std::cout << "GRADIENT\n";
//     std::cout << derivative(var).transpose() << "\n";
//     diff -= derivative(var);
//     std::cout << "FINITE DIFFERENCES - GRADIENT\n";
//     std::cout << diff.transpose() << "\n";
//   }
//   else
//     diff -= derivative(var);
//   std::cout << "Average error: " << diff.cwiseAbs().sum() / var.size() << std::endl;
//   VectorXd::Index i;
//   double max_error = diff.cwiseAbs().maxCoeff(&i);
//   std::cout << "Max error at " << i << ": " << max_error << "\n";
// }

// void derivative_check(const std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd>)> &func,
//                       const std::function<Eigen::MatrixXd(const Eigen::Ref<const Eigen::VectorXd>)> &derivative,
//                       const Eigen::Ref<const Eigen::VectorXd> var,
//                       std::vector<int> &indices_to_be_filtered,
//                       bool dump_matrices)
// {
//   using namespace Eigen;
//   MatrixXd diff = finite_differences(func, var);
//   filter_var(diff, indices_to_be_filtered);

//   if(dump_matrices)
//   {
//     std::cout << "FINITE DIFFERENCES\n" << std::defaultfloat;
//     std::cout << diff << "\n\n";
//     std::cout << "HESSIAN\n";
//     std::cout << derivative(var) << "\n\n";
//     diff -= derivative(var);
//     std::cout << "FINITE DIFFERENCES - HESSIAN\n";
//     std::cout << diff << "\n\n";
//   }
//   else
//     diff -= derivative(var);

//   std::cout << "Average error: " << diff.cwiseAbs().sum() / var.size() << std::endl;

//   double max = 0;
//   int row_max = 0;
//   int col_max = 0;

//   double max_error = diff.cwiseAbs().maxCoeff(&row_max, &col_max);
//   std::cout << "Max error at (" << row_max << ", " << col_max << "): " << max << std::endl;
// }

} // namespace fsim
