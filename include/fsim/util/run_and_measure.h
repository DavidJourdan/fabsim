// run_and_measure.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 04/10/18

#pragma once

#include <chrono>
#include <iostream>

template <typename TFunc>
void run_and_measure(TFunc &&f, const char *name)
{
  auto start = std::chrono::system_clock::now();
  f();
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << name << " elapsed seconds: " << elapsed_seconds.count() << "s\n";
}
