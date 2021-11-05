// finite_differences.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 04/10/18

#pragma once

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <functional>

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
Eigen::SparseMatrix<double, Eigen::ColMajor>
finite_differences(const std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd>)> &func,
                   const Eigen::Ref<const Eigen::VectorXd> var,
                   bool parallelism_enabled = false);

void derivative_check(const std::function<double(const Eigen::Ref<const Eigen::VectorXd>)> &func,
                      const std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd>)> &derivative,
                      const Eigen::Ref<const Eigen::VectorXd> var,
                      bool dump_matrices = false);

void derivative_check(
    const std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd>)> &func,
    const std::function<Eigen::SparseMatrix<double>(const Eigen::Ref<const Eigen::VectorXd>)> &derivative,
    const Eigen::Ref<const Eigen::VectorXd> var,
    bool dump_matrices = false);

void derivative_check(const std::function<double(const Eigen::Ref<const Eigen::VectorXd>)> &func,
                      const std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd>)> &derivative,
                      const Eigen::Ref<const Eigen::VectorXd> var,
                      const std::vector<int> &indices_to_be_filtered,
                      bool dump_matrices = false);

void derivative_check(
    const std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd>)> &func,
    const std::function<Eigen::SparseMatrix<double>(const Eigen::Ref<const Eigen::VectorXd>)> &derivative,
    const Eigen::Ref<const Eigen::VectorXd> var,
    std::vector<int> &indices_to_be_filtered,
    bool dump_matrices = false);
