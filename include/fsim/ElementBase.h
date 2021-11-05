// ElementBase.h
//
// This class describes the basic structure of a stencil element, all stencil classes in fabsim
// derive from it, as ModelBase expect its elements to follow this structure
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 08/27/19

#pragma once

#include <Eigen/Dense>

/**
 * Base class for all stencil elements
 * @tparam nb_vtx  number of vertices of the stencil
 * @tparam additional_dofs  number of additional degrees of freedom (if any)
 */
template <int nb_vtx, int additional_dofs = 0>
class ElementBase
{
public:
  static const int NB_DOFS = 3 * nb_vtx + additional_dofs;
  static const int NB_VERTICES = nb_vtx;

  typedef Eigen::Matrix<double, NB_DOFS, 1> LocalVector;
  typedef Eigen::Matrix<double, NB_DOFS, NB_DOFS> LocalMatrix;

  Eigen::Matrix<int, nb_vtx + additional_dofs, 1> idx;

  constexpr int nb_dofs() const { return NB_DOFS; }
  constexpr int nb_vertices() const { return NB_VERTICES; }
};
