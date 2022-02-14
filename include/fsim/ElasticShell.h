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
#include "fsim/CompositeModel.h"
#include "fsim/ElasticMembrane.h"
#include "util/typedefs.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace fsim
{

/**
 * Implementation of the classic Discrete Shells model: only controls bending
 * @tparam Formulation  Either use the tangent or the square of the bending angle as a penalty
 */
template <class Formulation = TanAngleFormulation>
class DiscreteShell : public ModelBase<HingeElement<Formulation>>
{
public:
  /**
   * DiscreteShell constructor
   * @param V  nV by 3 list of vertex positions
   * @param F  nF by 3 list of face indices
   * @param thickness  shell's thickness
   * @param young_modulus  shell's Young's modulus, controls the resistance to bending
   * @param poisson_ratio  shell's Poisson's ratio
   */
  DiscreteShell(const Eigen::Ref<const Mat3<double>> V,
                const Eigen::Ref<const Mat3<int>> F,
                double thickness,
                double young_modulus,
                double poisson_ratio);

  /**
   * Alternate constructor for varying thickness
   * @param V  nV by 3 list of vertex positions
   * @param F  nF by 3 list of face indices
   * @param thicknesses  list of shell's thicknesses (can be of size nV, nF, or nE)
   * @param young_modulus  shell's Young's modulus, controls the resistance to bending
   * @param poisson_ratio  shell's Poisson's ratio
   */
  DiscreteShell(const Eigen::Ref<const Mat3<double>> V,
                const Eigen::Ref<const Mat3<int>> F,
                const std::vector<double> &thicknesses,
                double young_modulus,
                double poisson_ratio);

  // number of vertices
  int nbVertices() const { return nV; }
  // number of edges (excluding edges on the boundary)
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
};

/**
 * An elastic shell material model: penalizes both bending with the DiscreteShell model, 
 * and stretching with a specified MembraneModel
 * @tparam MembraneModel  material model to use for the membrane energy
 * @tparam Formulation  Either use the tangent or the square of the bending angle as a penalty
 */
template <class MembraneModel = StVKMembrane, class Formulation = TanAngleFormulation>
class ElasticShell : public CompositeModel<DiscreteShell<Formulation>, MembraneModel>
{
public:
  /**
   * ElasticShell constructor
   * @param V  nV by 3 list of vertex positions
   * @param F  nF by 3 list of face indices
   * @param thickness  shell's thickness (constant)
   * @param young_modulus  shell's Young's modulus, controls the resistance to bending
   * @param poisson_ratio  shell's Poisson's ratio
   */
  ElasticShell(const Eigen::Ref<const Mat3<double>> V,
               const Eigen::Ref<const Mat3<int>> F,
               double thickness,
               double young_modulus,
               double poisson_ratio,
               double mass = 0)
      : CompositeModel<DiscreteShell<Formulation>, MembraneModel>(
            DiscreteShell<Formulation>(V, F, thickness, young_modulus, poisson_ratio),
            MembraneModel(V, F, thickness, young_modulus, poisson_ratio, mass))
  {}

  /**
   * Alternate constructor for varying thickness
   * @param V  nV by 3 list of vertex positions
   * @param F  nF by 3 list of face indices
   * @param thicknesses  size nF list of shell's thicknesses
   * @param young_modulus  shell's Young's modulus, controls the resistance to bending
   * @param poisson_ratio  shell's Poisson's ratio
   */
  ElasticShell(const Eigen::Ref<const Mat3<double>> V,
               const Eigen::Ref<const Mat3<int>> F,
               const std::vector<double> &thicknesses,
               double young_modulus,
               double poisson_ratio,
               double mass = 0)
      : CompositeModel<DiscreteShell<Formulation>, MembraneModel>(
            DiscreteShell<Formulation>(V, F, thicknesses, young_modulus, poisson_ratio),
            MembraneModel(V, F, thicknesses, young_modulus, poisson_ratio, mass))
  {
    assert(thicknesses.size() == F.rows());
  }

  // number of vertices
  int nbVertices() const { return bendingModel().nbVertices(); }
  // number of edges (excluding edges on the boundary)
  int nbEdges() const { return bendingModel().nbEdges(); }
  // number of faces
  int nbFaces() const { return bendingModel().nbFaces(); }
  // number of degrees of freedom
  int nbDOFs() const { return bendingModel().nbDOFs(); }

  DiscreteShell<Formulation> &bendingModel() { return this->template getModel<0>(); }
  const DiscreteShell<Formulation> &bendingModel() const { return this->template getModel<0>(); }

  MembraneModel &membraneModel() { return this->template getModel<1>(); }
  const MembraneModel &membraneModel() const { return this->template getModel<1>(); }

  // set Young's modulus (positive coefficient)
  void setYoungModulus(double young_modulus)
  {
    bendingModel().setYoungModulus(young_modulus);
    membraneModel().setYoungModulus(young_modulus);
  }
  double getYoungModulus() { return this->template getModel<0>().getYoungModulus(); }

  // set Poisson ratio (between 0 and 0.5)
  void setPoissonRatio(double poisson_ratio)
  {
    bendingModel().setPoissonRatio(poisson_ratio);
    membraneModel().setPoissonRatio(poisson_ratio);
  }
  double getPoissonRatio() { return this->template getModel<0>().getPoissonRatio(); }
};

} // namespace fsim
