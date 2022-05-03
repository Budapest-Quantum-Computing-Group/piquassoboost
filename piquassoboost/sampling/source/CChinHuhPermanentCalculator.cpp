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

#include <iostream>
#include "CChinHuhPermanentCalculator.h"
#include <tbb/scalable_allocator.h>
#include "tbb/tbb.h"
#include "common_functionalities.h"
#include <math.h>

namespace pic {




/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
CChinHuhPermanentCalculator::CChinHuhPermanentCalculator() {}




/**
@brief Call to calculate the permanent of the effective scattering matrix. Assuming that input state, output state and the matrix are
defined correctly (that is we've got m x m matrix, and vectors of with length m) this calculates the
permanent of an effective scattering matrix related to probability of obtaining output state from given
input state.
@param mtx_in The effective scattering matrix of a boson sampling instance
@param input_state_in The input state
@param output_state_in The output state
@return Returns with the calculated permanent
*/
Complex16
CChinHuhPermanentCalculator::calculate(matrix &mtx, PicState_int64 &input_state, PicState_int64 &output_state) {


    // determine the number of v_vectors
    int v_vectors_num = 1;
    PicVector<int> input_state_inidices; // The vector of indices corresponding to values greater than 0 in the input state
    input_state_inidices.reserve( input_state.size());
    for ( size_t idx=0; idx<input_state.size(); idx++) {
        if ( input_state[idx] > 0 ) {
            input_state_inidices.push_back(idx);
            v_vectors_num = v_vectors_num*(input_state[idx]+1);
        }
    }
    if (input_state_inidices.size() == 0) return Complex16(1.0, 0.0);

    // calculate the permanent
    tbb::task_group tg;


    // thread local storage for partial permanent
    tbb::combinable<Complex16> priv_addend{[](){return Complex16(0,0);}};


    // creating the root of the iteration containers for tasks spawned by task_group
    for (int idx=0; idx<=input_state[input_state_inidices[0]]; idx++) {

        PartialPermanentTask* calc_task = new PartialPermanentTask();
        calc_task->iter_value = PicVector<int>( input_state_inidices.size(), 0);
        calc_task->iter_value[0] = idx;

        // spawn the task for the current iter value
        tg.run([&tg, &mtx, &input_state, &input_state_inidices, &output_state, &priv_addend, calc_task]() {
            calc_task->execute(mtx, input_state, input_state_inidices, output_state, priv_addend, tg);
            delete calc_task;
        });

    }


    // wait for all task to be completed
    tg.wait();


    Complex16 permanent( 0.0, 0.0 );
    priv_addend.combine_each([&](Complex16 a) {
        permanent = permanent + a;
    });

    double factor = pow(2.0, sum(input_state));
    permanent = permanent/factor;



    return permanent;


}








/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
PartialPermanentTask::PartialPermanentTask() {

}


/**
@brief Call to execute the task to calculate the partial permanent, and spwans additional tasks to calculate partial permanents. The calculated partial permanents are stored within
thread local storages.
@param The current tbb::task_group object from which tasks are spawned
*/
void
PartialPermanentTask::execute(matrix &mtx, PicState_int64 &input_state, PicVector<int>& input_state_inidices, PicState_int64 &output_state, tbb::combinable<Complex16>& priv_addend, tbb::task_group &tg) {


            // creating the v_vector
            PicVector<int> v_vector(input_state.size(),0);
            size_t idx_max = 0;
            for ( size_t idx=0; idx<iter_value.size(); idx++) {
                if ( iter_value[idx] > 0 ) {
                    v_vector[input_state_inidices[idx]] = iter_value[idx];
                    idx_max = idx;
                }

            }


            int v_sum = sum(v_vector);
            Complex16 addend(pow(-1.0, v_sum), 0.0);

            // Binomials calculation
            for (size_t idx=0; idx<input_state.size(); idx++) { //} i in range(len(v_vector)):
                double tmp = (double)binomialCoeff( input_state[idx], v_vector[idx]);
                //std::cout << "tmp: " << tmp << std::endl;
                addend.real(addend.real()*tmp);
            }


            // product calculation
            Complex16 product(1.0, 0.0);
            for ( size_t idx=0; idx<input_state.size(); idx++) {
                if (output_state[idx] == 0 ) { // There's no reason to calculate the sum if t_j = 0
                    continue;
                }

                // Otherwise we calculate the sum
                Complex16 product_part(0.0, 0.0);
                for ( size_t jdx=0; jdx<input_state.size(); jdx++) {
                    size_t mtx_offset = idx*mtx.stride + jdx;
                    Complex16 element = mtx[mtx_offset];
                    double coeff = (double) (input_state[jdx] - 2 * v_vector[jdx]);
                    product_part.real( product_part.real() + coeff*element.real()); //(input_state_loc[jdx] - 2 * v_vector[jdx]) * self.__matrix[j][i]
                    product_part.imag( product_part.imag() + coeff*element.imag());
                    //std::cout << "product_part:" << product_part.real << " +i*" << product_part.imag << std::endl;
                }
                product_part = std::pow(product_part, output_state[idx]);

                product = product*product_part;
            }

            addend = addend*product;
            Complex16 &addend_priv = priv_addend.local();
            addend_priv = addend_priv + addend;


            // spawning new v_vectors to the do cycle
            for ( size_t idx=idx_max+1; idx<iter_value.size(); idx++) {
                for ( int jdx=1; jdx<=input_state[input_state_inidices[idx]]; jdx++) {

                    PartialPermanentTask* calc_task = new PartialPermanentTask();
                    calc_task->iter_value = iter_value;
                    calc_task->iter_value[idx] = jdx;

                    // spawn the task for the current iter value
                    tg.run([&tg, &mtx, &input_state, &input_state_inidices, &output_state, &priv_addend, calc_task]() {
                        calc_task->execute(mtx, input_state, input_state_inidices, output_state, priv_addend, tg);
                        delete calc_task;
                    });

                }
            }



}











} // PIC
