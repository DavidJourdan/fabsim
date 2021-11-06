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
                const Eigen::MatrixX2i &C,
                const Eigen::Ref<const Mat3<double>> N,
                const std::vector<std::vector<double>> &normal_widths,
                const std::vector<std::vector<double>> &binormal_widths,
                double young_modulus,
                double incompressibility);

  RodCollection(const Eigen::Ref<const Mat3<double>> V,
                const std::vector<std::vector<int>> &indices,
                const Eigen::MatrixX2i &C,
                const Eigen::Ref<const Mat3<double>> N,
                const std::vector<double> &normal_widths,
                const std::vector<double> &binormal_widths,
                double young_modulus,
                double incompressibility);

  RodCollection(const Eigen::Ref<const Mat3<double>> V,
                const std::vector<std::vector<int>> &indices,
                const Eigen::MatrixX2i &C,
                const Eigen::Ref<const Mat3<double>> N,
                double w_n,
                double w_b,
                double young_modulus,
                double incompressibility);
  RodCollection() = default;
};

} // namespace fsim
