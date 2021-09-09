/**
 * Copyright 2021 Budapest Quantum Computing Group
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef Torontonian_H
#define Torontonian_H

#include "matrix.h"
#include "matrix32.h"
#include "PicRealM.hpp"


namespace pic {

Complex16 determinant_byLU_decomposition( matrix& mtx );

/**
@brief Class to calculate the torontonian of a matrix.

Calculation is based on Cholesky decomposition.
To perform the calculation the given matrix has to be positive definite and selfadjoint with eigenvalues between 0 and 1.
*/
class Torontonian {

protected:
    /// The input matrix. Must be selfadjoint positive definite matrix with eigenvalues between 0 and 1.
    matrix mtx_orig;
    /** The scaled input matrix for which the calculations are performed.
    If the mean magnitude of the matrix elements is one, the treshold of quad precision can be set to higher values.
    */
    matrix32 mtx;
    /// The scale factors derived from the scaling of the input matrix used to scale the calculated determinants
    std::vector<long double> scale_factors;


public:


/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
Torontonian();

/**
@brief Constructor of the class.
@param mtx_in A selfadjoint matrix for which the torontonian is calculated. This matrix has to be positive definite matrix with eigenvalues between 0 and 1 (for example a covariance matrix of the Gaussian state.)
@return Returns with the instance of the class.
*/
Torontonian( matrix &mtx_in );

/**
@brief Default destructor of the class.
*/
virtual ~Torontonian();

/**
@brief Method to calculate the torontonian of a complex selfadjoint positive definite matrix with eigenvalues between 0 and 1.
@return Returns with the calculated hafnian
*/
//virtual Complex16 calculate();
virtual double calculate();


/**
@brief Call to update the memory address of the matrix mtx
@param mtx_in Input matrix defined by
*/
virtual void Update_mtx( matrix &mtx_in);

}; //Torontonian


} // PIC

#endif // Torontonian_H
