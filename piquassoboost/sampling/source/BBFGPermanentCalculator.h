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

#ifndef BBFGPermanentCalculator_H
#define BBFGPermanentCalculator_H

#include "PicState.h"
#include "PicVector.hpp"
#include "matrix.h"
#include "matrix32.h"



namespace pic {


/**
@brief Constructor of the class.
@return Returns with the instance of the class.
*/
Complex16 product_reduction( const matrix& mtx );
Complex32 product_reduction( const matrix32& mtx );



class BBFGPermanentCalculator  {


protected:

    /// The input matrix. 
    matrix mtx;

public:

/**
@brief Constructor of the class.
@return Returns with the instance of the class.
*/
BBFGPermanentCalculator( );


/**
@brief Default destructor of the class.
*/
virtual ~BBFGPermanentCalculator();

/**
@brief Call to calculate the hafnian of a complex matrix
@param use_extended Logical variable to indicate whether use extended precision for cholesky decomposition (default), or not.
@return Returns with the calculated hafnian
*/
virtual Complex16 calculate(matrix& mtx_in, bool use_extended=false, bool use_inf=false);

/**
@brief Call to update the memory address of the matrix mtx
@param mtx_in Input matrix defined by
*/
virtual void Update_mtx( matrix &mtx_in);


}; //BBFGPermanentCalculator




} // PIC

#endif
