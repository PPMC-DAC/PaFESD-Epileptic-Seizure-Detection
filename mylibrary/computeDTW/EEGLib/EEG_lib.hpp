#pragma once

#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <random>
#include <cassert>
#include <string.h>
#include <memory>
#include <omp.h>

using data_t = float;  //definition of datatype

constexpr float maxval = 0x7f800000;  //positive infinity in IEEE754

struct blockRange{
    size_t eId;     // Index of the first epoch to process
    size_t nEpo;    // Number of epochs to process
    size_t pId;     // Index of the first pattern to process
    size_t nPat;    // Number of patterns to process
};


using FloatVector = std::vector<data_t>;

// z-normalized DTW matrix computation with partial euclidean distance (no sqrt)
FloatVector get_distance_matrix(const data_t* S, const size_t nS, const data_t* Q, const size_t nQ, 
                                const size_t sEpoPat, const size_t stride, const size_t w, const int verbose);

// Un_z_normalized DTW version with full euclidean distance (with sqrt) and pattern=symmetric2
FloatVector get_distance_matrix_u(const data_t* S, const size_t nS, const data_t* Q, const size_t nQ, 
                                const size_t sEpoPat, const size_t stride, const size_t w, const int verbose);

// Un_z_normalized DTW version with partial euclidean distance (without sqrt) and pattern=symmetric2
FloatVector get_distance_matrix_u_ns(const data_t* S, const size_t nS, const data_t* Q, const size_t nQ, 
                                const size_t sEpoPat, const size_t stride, const size_t w, const int verbose);
