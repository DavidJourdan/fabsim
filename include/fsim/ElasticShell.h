// ElasticShell.h
//
// Implementation of the Discrete Shells material model of Grinspun et al. as presented in
// "Discrete shells" (https://doi.org/10.5555/846276.846284), see also "Discrete bending forces and
// their Jacobians" (https://doi.org/10.1016/j.gmod.2013.07.001) for the full derivation of the derivatives
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 09/11/19

#pragma once

#include "HingeElement.h"
#include "ModelBase.h"
#include "util/typedefs.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace fsim
{

/**
 * Implementation of the classic Discrete Shells model
 * @tparam Formulation  Either use the tangent or the square of the bending angle as a penalty
 * @tparam fullHess  Wether to compute the full matrix of second derivatives or use a quadratic approximation
 * If true, the Newton solver will converge faster when it's close to the solution but the hessian might be non-SPD
 */
template <class Formulation = TanAngleFormulation, bool fullHess = true>
class ElasticShell : public ModelBase<HingeElement<Formulation, fullHess>>
{
public:
  /**
   * ElasticShell constructor
   * @param V  nV by 3 list of vertex positions
   * @param F  nF by 3 list of face indices
   * @param thickness  membrane's thickness
   * @param young_modulus  membrane's Young's modulus, controls the resistance to bending
   * @param poisson_ratio  membrane's Poisson's ratio
   */
  ElasticShell(const Eigen::Ref<const Mat3<double>> V,
               const Eigen::Ref<const Mat3<int>> F,
               double thickness,
               double young_modulus,
               double poisson_ratio);
  ElasticShell() = default;

  // number of vertices
  int nbVertices() const { return nV; }
  // number of edges
  int nbEdges() const { return nE; }
  // number of faces
  int nbFaces() const { return nF; }
  // number of degrees of freedom
  int nbDOFs() const { return 3 * nV; }

  // set Young's modulus (positive coefficient)
  void setYoungModulus(double young_modulus);
  double getYoungModulus() { return _young_modulus; }

  // set Poisson ratio (between 0 and 0.5)
  void setPoissonRatio(double poisson_ratio);
  double getPoissonRatio() { return _poisson_ratio; }

  // set thickness of the membrane (controls the amount of bending) negative values are not allowed
  void setThickness(double thickness);
  double getThickness() { return _thickness; }

  /**
   * Generates indices for the diamond stencil structure from the list of faces
   * @param F  nF by 3 list of face indices
   * @return  nE by 4 list of hinge indices, first 2 indices represent the shared edge between two faces
   */
  static Mat4<int> hingeIndices(const Eigen::Ref<const Mat3<double>> V, const Eigen::Ref<const Mat3<int>> F);

private:
  int nE, nV, nF;
  double _young_modulus;
  double _poisson_ratio;
  double _thickness;
};

} // namespace fsim
