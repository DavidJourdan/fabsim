// OrthotropicStVKMembrane.h
//
// StVK version of "Stable Orthotropic Materials" by Li and Barbič (https://doi.org/10.1109/tvcg.2015.2448105)
// Parameterizes the elasticity tensor with 2 Young's moduli and 1 Poisson's ratio
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 30/10/21

#include "ModelBase.h"
#include "OrthotropicStVKElement.h"

#include <array>

namespace fsim
{

class OrthotropicStVKMembrane : public ModelBase<OrthotropicStVKElement>
{
public:
  /**
   * constructor for OrthotropicStVKMembrane
   * @param V  nV by 3 list of vertex positions
   * @param F  nF by 3 list of face indices
   * @param thickness  membrane's thickness (constant)
   * @param E1  0 degree Young's modulus
   * @param E2  90 degrees Young's modulus
   * @param poisson_ratio  membrane's Poisson's ratio
   * @param mass  membrane's mass (defaults to 0 to disable gravity)
   */
  OrthotropicStVKMembrane(const Eigen::Ref<const Mat2<double>> V,
                          const Eigen::Ref<const Mat3<int>> F,
                          double thickness,
                          double E1,
                          double E2,
                          double poisson_ratio,
                          double mass = 0);
  /**
   * Alternate constructor for varying thickness
   * @param V  nV by 3 list of vertex positions
   * @param F  nF by 3 list of face indices
   * @param thicknesses  size nF list of membrane's thicknesses
   * @param E1  0 degree Young's modulus
   * @param E2  90 degrees Young's modulus
   * @param poisson_ratio  membrane's Poisson's ratio
   * @param mass  membrane's mass (defaults to 0 to disable gravity)
   */
  OrthotropicStVKMembrane(const Eigen::Ref<const Mat2<double>> V,
                          const Eigen::Ref<const Mat3<int>> F,
                          const std::vector<double> &thicknesses,
                          double E1,
                          double E2,
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
  double getPoissonRatio() const { return _poisson_ratio; }

  // set Young's moduli (E1 and E2 respectively control the horizontal and vertical stiffness)
  void setYoungModuli(double E1, double E2);
  std::array<double, 2> getYoungModuli() const { return {_E1, _E2}; }

  void setMass(double mass);
  double getMass() const { return _mass; }

private:
  int nV;
  double _poisson_ratio;
  double _E1;
  double _E2;
  double _mass;
};

} // namespace fsim
