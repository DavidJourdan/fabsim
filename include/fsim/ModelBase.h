// ModelBase.h
//
// Base class for (almost) all material models in fabsim, based on the AoS (array of structures) layout
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 08/27/19

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <vector>

namespace fsim
{

/*
 * Base class for (almost) all material models in fabsim, based on the AoS (array of structures) layout
 * Iterates through every Element to add their energy, gradient and hessian contributions
 */
template <class Element>
class ModelBase
{
public:
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

  std::vector<Element> get_elements() { return _elements; };

protected:
  std::vector<Element> _elements;
};

template <class Element>
double ModelBase<Element>::energy(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  double result = 0;
  for(auto &element: _elements)
  {
    result += element.energy(X);
  }

  return result;
}

template <class Element>
void ModelBase<Element>::gradient(const Eigen::Ref<const Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> Y) const
{
  using namespace Eigen;

  for(auto &element: _elements)
  {
    auto grad = element.gradient(X);

    int nV = element.nbVertices();
    for(int j = 0; j < nV; ++j)
      Y.segment<3>(3 * element.idx(j)) += grad.template segment<3>(3 * j);

    for(int j = 0; j < element.idx.size() - nV; ++j)
      Y(element.idx(nV + j)) += grad(3 * nV + j);
  }
}

template <class Element>
Eigen::VectorXd ModelBase<Element>::gradient(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  VectorXd Y = VectorXd::Zero(X.size());
  gradient(X, Y);
  return Y;
}

template <class Element>
Eigen::SparseMatrix<double> ModelBase<Element>::hessian(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  std::vector<Triplet<double>> triplets = hessianTriplets(X);

  SparseMatrix<double> hess(X.size(), X.size());
  hess.setFromTriplets(triplets.begin(), triplets.end());
  return hess;
}

template <class Element>
std::vector<Eigen::Triplet<double>> ModelBase<Element>::hessianTriplets(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  std::vector<Triplet<double>> triplets;

  triplets.reserve(Element::NB_DOFS * (Element::NB_DOFS + 1) / 2 * _elements.size());
  for(auto &e: _elements)
  {
    auto hess = e.hessian(X);

    int nV = e.nbVertices();
    for(int j = 0; j < nV; ++j)
      for(int k = 0; k < nV; ++k)
        if(e.idx(j) <= e.idx(k))
          for(int l = 0; l < 3; ++l)
            for(int m = 0; m < 3; ++m)
              triplets.emplace_back(3 * e.idx(j) + l, 3 * e.idx(k) + m, hess(3 * j + l, 3 * k + m));

    for(int j = 0; j < e.idx.size() - nV; ++j)
      for(int k = 0; k < nV; ++k)
        if(3 * e.idx(k) < e.idx(3 + j))
          for(int l = 0; l < 3; ++l)
          {
            triplets.emplace_back(e.idx(nV + j), 3 * e.idx(k) + l, hess(3 * nV + j, 3 * k + l));
            triplets.emplace_back(3 * e.idx(k) + l, e.idx(nV + j), hess(3 * k + l, 3 * nV + j));
          }

    for(int j = 0; j < e.idx.size() - nV; ++j)
      for(int k = 0; k < e.idx.size() - nV; ++k)
        if(e.idx(3 + j) <= e.idx(3 + k))
          triplets.emplace_back(e.idx(nV + j), e.idx(nV + k), hess(3 * nV + j, 3 * nV + k));
  }

  return triplets;
}

} // namespace fsim
