// ElasticMembrane.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 01/19/20

#pragma once

#include "IncompressibleNeoHookeanElement.h"
#include "ModelBase.h"
#include "NeoHookeanElement.h"
#include "StVKElement.h"
#include "util/typedefs.h"

namespace fsim
{

/**
 * template class for isotropic membrane models (e.g. StVK, neohookean...)
 */
template <class Element>
class ElasticMembrane : public ModelBase<Element>
{
public:
  /**
   * constructor for ElasticMembrane
   * @param V  nV by 2 list of vertex positions (initial position in the 2D plane)
   * @param F  nF by 3 list of face indices
   * @param young_modulus  membrane's Young's modulus, controls the resistance to bending
   * @param poisson_ratio  membrane's Poisson's ratio
   * @param thickness  membrane's thickness
   * @param mass  membrane's mass (defaults to 0 to disable gravity)
   */
  ElasticMembrane(const Eigen::Ref<const Mat3<double>> V,
                  const Eigen::Ref<const Mat3<int>> F,
                  double thickness,
                  double young_modulus,
                  double poisson_ratio,
                  double mass = 0);

  ElasticMembrane(const Eigen::Ref<const Mat3<double>> V,
                  const Eigen::Ref<const Mat3<int>> F,
                  const std::vector<double> &thicknesses,
                  double young_modulus,
                  double poisson_ratio,
                  double mass = 0);

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

  // number of degrees of freedom
  int nbDOFs() const { return 3 * nV; }

  // set Poisson ratio (between 0 and 0.5)
  void setPoissonRatio(double poisson_ratio);
  double getPoissonRatio() const { return _nu; }

  // set Young's modulus (positive coefficient)
  void setYoungModulus(double E);
  double getYoungModulus() const { return _E; }

  // set thickness of the membrane (controls the amount of stretching and the total weight)
  // negative values are not allowed
  void setThickness(double t);
  double getThickness() const { return _thickness; }

  void setMass(double mass);
  double getMass() const { return _mass; }

private:
  int nV, nF;
  double _E;
  double _nu;
  double _lambda;
  double _mu;
  double _mass;
  double _thickness = -1;
};

using StVKMembrane = ElasticMembrane<StVKElement>;
using NeoHookeanMembrane = ElasticMembrane<NeoHookeanElement>;
using IncompressibleNeoHookeanMembrane = ElasticMembrane<IncompressibleNeoHookeanElement>;

} // namespace fsim
