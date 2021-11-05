// OrthotropicStVKElement.h
//
// StVK version of "Stable Orthotropic Materials" by Li and Barbič (https://doi.org/10.1109/tvcg.2015.2448105)
// Parameterizes the elasticity tensor with 2 Young's moduli and 1 Poisson's ratio
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 30/10/21

#include "OrthotropicStVKElement.h"
#include "ElasticMembraneModel.h"

#include <iostream>

template <int id = 0>
class OrthotropicStVKModel : public ElasticMembraneModel<OrthotropicStVKElement<id>>
{
public:
  /**
   * constructor for OrthotropicStVKModel
   * @param V  nV by 3 list of vertex positions
   * @param F  nF by 3 list of face indices
   * @param thicknesses  membrane's thickness (per-triangle value)
   * @param E1  0 degree Young's modulus
   * @param E2  90 degrees Young's modulus
   * @param poisson_ratio  membrane's Poisson's ratio
   * @param stress  membrane's amount of stretch in length (e.g. 1.3 if it has been stretched from 10cm long to 13cm long)
   */
  OrthotropicStVKModel(const Eigen::Ref<const fsim::Mat3<double>> V,
                      const Eigen::Ref<const fsim::Mat3<int>> F,
                      const std::vector<double> &thicknesses,
                      double E1,
                      double E2,
                      double poisson_ratio,
                      double stress = 1)
  {
    using namespace Eigen;

    if(OrthotropicStVKElement<id>::C.norm() == 0)
      std::cerr << "Warning: overwriting elasticity tensor. Please declare your different instances as OrthotropicStVKModel<0>, OrthotropicStVKModel<1>, etc.\n";

    OrthotropicStVKElement<id>::_C << E1, poisson_ratio * sqrt(E1 * E2), 0,
                                     poisson_ratio * sqrt(E1 * E2), E2, 0,
                                     0, 0, 0.5 * sqrt(E1 * E2) * (1 - poisson_ratio);

    OrthotropicStVKElement<id>::_C /= (1 - std::pow(poisson_ratio, 2));                                

    MatrixX3d smallerV = V / stress;

    int nF = F.rows();
    this->_elements.reserve(nF);
    for(int i = 0; i < nF; ++i)
    {
      this->_elements.emplace_back(smallerV, F.row(i), thicknesses[i]);
    }
  }
};

using OrthotropicStVKMembrane = OrthotropicStVKModel<0>;