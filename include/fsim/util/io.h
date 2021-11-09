// io.h
// helper functions for loading and saving mesh and rod data
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 12/11/19

#pragma once

#include "fsim/util/typedefs.h"

#include <Eigen/Dense>

#include <vector>

namespace fsim
{

/**
 * Read OFF file and store vertex and face information into V and F
 * @param file  path to file where V and F information is stored
 * @return V  nV by 3 matrix of vertices
 * @return F  nF by 3 matrix of face indices
 */
void readOFF(const std::string &file, Mat3<double> &V, Mat<int> &F);

/**
 * Read .rod file and store indices into rod_indices
 * @param file  path to .rod file where rod indices information is stored
 * @return rod_indices  nested vector: each vector represents a rod and contains its indices
 */
void readROD(const std::string &file, std::vector<std::vector<int>> &rod_indices);

/**
 * save vertex and face information in OFF format
 * @param file  path to .off file where V and F information will be stored
 * @return V  nV by 3 matrix of vertices
 * @return F  nF by 3 matrix of face indices
 */
void saveOFF(const std::string &file, const Eigen::Ref<const Mat3<double>> V, const Eigen::Ref<const Mat<int>> F);

/**
 * save rod indices information in .rod file
 * @param file  path to .rod file where rod indices information will be stored stored
 * @param rod_indices  nested vector: each vector represents a rod and contains its indices
 */
void saveROD(const std::string &file, const std::vector<std::vector<int>> &rod_indices);

/**
 * save both mesh and line geometry in Wavefront OBJ format
 * @param file_name  basename of .off file for V and F, and .rod file for rod_indices
 * @param V  nV by 3 matrix of vertices
 * @param F  nF by 3 matrix of face indices
 * @param rod_indices  nested vector: each vector represents a rod and contains its indices
 */
void saveOBJ(const std::string &file,
             const Eigen::Ref<const Mat3<double>> V,
             const Eigen::Ref<const Mat<int>> F,
             const std::vector<std::vector<int>> &rod_indices);

} // namespace fsim
