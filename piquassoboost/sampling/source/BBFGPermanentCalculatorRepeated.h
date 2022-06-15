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

#ifndef BBFGPermanentCalculatorRepeated_H
#define BBFGPermanentCalculatorRepeated_H

#include "PicState.h"
#include "PicVector.hpp"
#include "matrix.h"
#include "matrix32.h"



namespace pic {



class BBFGPermanentCalculatorRepeated  {


protected:

public:

/**
@brief Constructor of the class.
@return Returns with the instance of the class.
*/
BBFGPermanentCalculatorRepeated( );


/**
@brief Default destructor of the class.
*/
virtual ~BBFGPermanentCalculatorRepeated();


/**
@brief Call to calculate the hafnian of a complex matrix
@param use_extended Logical variable to indicate whether use extended precision for cholesky decomposition (default), or not.
@return Returns with the calculated hafnian
*/
virtual Complex16 calculate(matrix& mtx, PicState_int64& col_mult64, PicState_int64& row_mult64, bool use_extended=false, bool use_inf=false);

/**
@brief Call to calculate the hafnian of a complex matrix
@param use_extended Logical variable to indicate whether use extended precision for cholesky decomposition (default), or not.
@return Returns with the calculated hafnian
*/
virtual Complex16 calculate(matrix& mtx, PicState_int& col_mult, PicState_int& row_mult, bool use_extended=false, bool use_inf=false);



}; //BBFGPermanentCalculatorRepeated




} // PIC

#endif
