// finite_differences.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 04/10/18

#pragma once

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <functional>

namespace fsim
{

/**
 * computes the approximate gradient of an energy function using finite differences
 * @param func  energy function
 * @param var  list of variables, i.e. point in R^n to compute the gradient at
 * @param parallelism_enabled  whether or not to use multiple threads to compute it
 * @return  gradient of func evaluated at var
 */
Eigen::VectorXd finite_differences(const std::function<double(const Eigen::Ref<const Eigen::VectorXd>)> &func,
                                   const Eigen::Ref<const Eigen::VectorXd> var,
                                   bool parallelism_enabled = false);

/**
 * computes the approximate hessian of an energy function using finite differences
 * @param func  gradient function
 * @param var  list of variables, i.e. point in R^n to compute the hessian at
 * @param parallelism_enabled  whether or not to use multiple threads to compute it
 * @return  hessian of func evaluated at var,
 */
Eigen::MatrixXd
finite_differences(const std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd>)> &func,
                   const Eigen::Ref<const Eigen::VectorXd> var,
                   bool parallelism_enabled = false);

Eigen::SparseMatrix<double>
finite_differences_sparse(const std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd>)> &func,
                          const Eigen::Ref<const Eigen::VectorXd> var,
                          bool parallelism_enabled = false);

/**
 * compares the provided derivative function with the finite difference approximation
 * @param func  energy function
 * @param var  list of variables, i.e. point in R^n to compute the gradient at
 * @param dump_matrices  whether or not to display the full content of the gradients     (otherwise just prints max error)
 */
void derivative_check(const std::function<double(const Eigen::Ref<const Eigen::VectorXd>)> &func,
                      const std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd>)> &derivative,
                      const Eigen::Ref<const Eigen::VectorXd> var,
                      bool dump_matrices = false);

/**
 * compares the provided derivative function with the finite difference approximation
 * @param func  gradient function
 * @param var  list of variables, i.e. point in R^n to compute the hessian at
 * @param dump_matrices  whether or not to display the full content of the hessians (otherwise just prints max error)
 */
void derivative_check(
    const std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd>)> &func,
    const std::function<Eigen::MatrixXd(const Eigen::Ref<const Eigen::VectorXd>)> &derivative,
    const Eigen::Ref<const Eigen::VectorXd> var,
    bool dump_matrices = false);

void derivative_check(const std::function<double(const Eigen::Ref<const Eigen::VectorXd>)> &func,
                      const std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd>)> &derivative,
                      const Eigen::Ref<const Eigen::VectorXd> var,
                      const std::vector<int> &indices_to_be_filtered,
                      bool dump_matrices = false);

void derivative_check(
    const std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd>)> &func,
    const std::function<Eigen::MatrixXd(const Eigen::Ref<const Eigen::VectorXd>)> &derivative,
    const Eigen::Ref<const Eigen::VectorXd> var,
    std::vector<int> &indices_to_be_filtered,
    bool dump_matrices = false);

} // namespace fsim