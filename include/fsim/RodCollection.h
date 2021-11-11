// RodCollection.h
//
// Implementation of the Discrete Elastic Rods model of Bergou et al. as presented in
// "Discrete Viscous Threads" (https://doi.org/10.1145/1778765.1778853),
// see also  "A discrete, geometrically exact method for simulating nonlinear, elastic or
// non-elastic beams"  (https://hal.archives-ouvertes.fr/hal-02352879v1)
//
// Please note that, at the moment this implementation assumes a rectangular cross-section whose
// dimensions are given by the normal and binormal widths variables
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 01/09/20

#pragma once

#include "ElasticRod.h"
#include "RodStencil.h"
#include "Spring.h"
#include "util/typedefs.h"

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <vector>

namespace fsim
{

class RodCollection : public ElasticRod
{
public:
  RodCollection(const Eigen::Ref<const Mat3<double>> V,
                const std::vector<std::vector<int>> &indices,
                const Eigen::Ref<const Mat2<int>> C,
                const Eigen::Ref<const Mat3<double>> N,
                const std::vector<double> &thicknesses,
                const std::vector<double> &widths,
                double young_modulus,
                double mass = 0,
                CrossSection c = CrossSection::Circle);

  RodCollection(const Eigen::Ref<const Mat3<double>> V,
                const std::vector<std::vector<int>> &indices,
                const Eigen::Ref<const Mat2<int>> C,
                const Eigen::Ref<const Mat3<double>> N,
                const RodParams &p);

  /**
   * energy function of this material model   f : \R^n -> \R
   * @param X  a flat vector stacking all degrees of freedom
   * @return  the energy of this model evaluated at X
   */
  double energy(const Eigen::Ref<const Eigen::VectorXd> X) const override;

  /**
   * gradient of the energy  \nabla f : \R^n -> \R^n
   * @param X  a flat vector stacking all degrees of freedom
   * @param Y  gradient (or sum of gradients) vector in which we will add the gradient of energy evaluated at X
   * @return Y
   */
  void gradient(const Eigen::Ref<const Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> Y) const override;
  Eigen::VectorXd gradient(const Eigen::Ref<const Eigen::VectorXd> X) const override;

  /**
   * (row, column, value) triplets used to build the sparse hessian matrix
   * @param X  a flat vector stacking all degrees of freedom
   * @return  all the triplets needed to build the hessian
   */
  std::vector<Eigen::Triplet<double>> hessianTriplets(const Eigen::Ref<const Eigen::VectorXd> X) const override;

private:
  // contains cross-section stiffnesses and number of elements per rod
  std::vector<std::tuple<double, double, int>> rodData;
};

} // namespace fsim
