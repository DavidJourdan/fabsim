// CompositeModel.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 11/15/21

#pragma once

#include "ModelBase.h"
#include "util/typedefs.h"

#include <tuple>
#include <utility>

namespace fsim
{

/**
 * A convienent wrapper for a collection of models
 * automatically aggregates the gradient and hessian contributions from the different material models
 */
template <class... Types>
class CompositeModel
{
public:
  CompositeModel(Types &&...args) : _models(std::forward<Types>(args)...) {}

  /**
   * energy function of this material model   f : \R^n -> \R
   * @param X  a flat vector stacking all degrees of freedom
   * @return  the energy of this model evaluated at X
   */
  double energy(const Eigen::Ref<const Eigen::VectorXd> X) const;

  /**
   * gradient of the energy  \nabla f : \R^n -> \R^n
   * @param X  a flat vector stacking all degrees of freedom
   * @param Y  gradient (or sum of gradients) vector in which we will add the gradient of energy evaluated at X
   * @return Y
   */
  void gradient(const Eigen::Ref<const Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> Y) const;
  Eigen::VectorXd gradient(const Eigen::Ref<const Eigen::VectorXd> X) const;

  /**
   * hessian of the energy  \nabla^2 f : \R^n -> \R^{n \times n}
   * @param X  a flat vector stacking all degrees of freedom
   * @return  hessian of the energy stored in a sparse matrix representation
   */
  Eigen::SparseMatrix<double> hessian(const Eigen::Ref<const Eigen::VectorXd> X) const;

  /**
   * (row, column, value) triplets used to build the sparse hessian matrix
   * @param X  a flat vector stacking all degrees of freedom
   * @return  all the triplets needed to build the hessian
   */
  std::vector<Eigen::Triplet<double>> hessianTriplets(const Eigen::Ref<const Eigen::VectorXd> X) const;

  template <int k>
  auto &getModel() { return std::get<k>(_models); }

  template <int k>
  const auto &getModel() const { return std::get<k>(_models); }

private:
  std::tuple<Types...> _models;
};

namespace
{
// Compile-time tuple for-each
// see https://stackoverflow.com/questions/1198260/how-can-you-iterate-over-the-elements-of-an-stdtuple
template <std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
tuple_for_each(const std::tuple<Tp...> &, FuncT) // Unused arguments are given no names.
{}

template <std::size_t I = 0, typename FuncT, typename... Tp>
    inline typename std::enable_if < I<sizeof...(Tp), void>::type tuple_for_each(const std::tuple<Tp...> &t, FuncT f)
{
  f(std::get<I>(t));
  tuple_for_each<I + 1, FuncT, Tp...>(t, f);
}
} // namespace

template <class... Types>
double CompositeModel<Types...>::energy(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  double res = 0;
  tuple_for_each(_models, [&X, &res](auto &model) { res += model.energy(X); });
  return res;
}

template <class... Types>
void CompositeModel<Types...>::gradient(const Eigen::Ref<const Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> Y) const
{
  tuple_for_each(_models, [&X, &Y](auto &model) { model.gradient(X, Y); });
}

template <class... Types>
Eigen::VectorXd CompositeModel<Types...>::gradient(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  VectorXd Y = VectorXd::Zero(X.size());
  gradient(X, Y);
  return Y;
}

template <class... Types>
std::vector<Eigen::Triplet<double>>
CompositeModel<Types...>::hessianTriplets(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  std::vector<Eigen::Triplet<double>> triplets;
  tuple_for_each(_models, [&X, &triplets](auto &model) {
    auto tripletsI = model.hessianTriplets(X);
    triplets.insert(triplets.end(), tripletsI.begin(), tripletsI.end());
  });

  return triplets;
}

template <class... Types>
Eigen::SparseMatrix<double> CompositeModel<Types...>::hessian(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  std::vector<Triplet<double>> triplets = hessianTriplets(X);

  SparseMatrix<double> hess(X.size(), X.size());
  hess.setFromTriplets(triplets.begin(), triplets.end());
  return hess;
}

} // namespace fsim
