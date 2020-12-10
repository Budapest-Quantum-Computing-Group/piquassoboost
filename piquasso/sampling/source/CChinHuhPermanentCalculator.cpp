#include <iostream>
#include "CChinHuhPermanentCalculator.h"
#include <tbb/scalable_allocator.h>
#include <math.h>

namespace pic {




/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
CChinHuhPermanentCalculator::CChinHuhPermanentCalculator() {}

/**
@brief Constructor of the class.
@param C_in Input matrix defined by
@param G_in Input matrix defined by
@param mean_in Input matrix defined by
@return Returns with the instance of the class.
*/
CChinHuhPermanentCalculator::CChinHuhPermanentCalculator( matrix &mtx_in, PicState_int64 &input_state_in, PicState_int64 &output_state_in) {

    Update( mtx_in, input_state_in, output_state_in);
}




/**
@brief Call to update the memroy addresses of the stored matrices
@param C_in Input matrix defined by
@param G_in Input matrix defined by
@param mean_in Input matrix defined by
*/
void
CChinHuhPermanentCalculator::Update( matrix &mtx_in, PicState_int64 &input_state_in, PicState_int64 &output_state_in) {

    mtx = mtx_in;
    input_state = input_state_in;
    output_state = output_state_in;

}



/**
@brief Call to calculate the permanent of the effective scattering matrix. Assuming that input state, output state and the matrix are
defined correctly (that is we've got m x m matrix, and vectors of with length m) this calculates the
permanent of an effective scattering matrix related to probability of obtaining output state from given
input state.
@return Returns with the calculated permanent
*/
Complex16
CChinHuhPermanentCalculator::calculate() {


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

    // calculate the permanent

    // creating the root of the iteration containers for parallel_do
    vectors<int> iter_values;
    iter_values.reserve( input_state[input_state_inidices[0]]+1 );
    for (int idx=0; idx<=input_state[input_state_inidices[0]]; idx++) {
        PicVector<int> value( input_state_inidices.size(), 0);
        value[0] = idx;
        iter_values.push_back( value );
    }

    // number of columns
    size_t col_num = mtx.cols;
    // raw pointer to the stored data in matrix mtx
    Complex16* mtx_data = mtx.get_data();
//mtx.print_matrix();


    // parameters to be captured by the lambda function
    PicState_int64 &input_state_loc = input_state;
    PicState_int64 &output_state_loc = output_state;

    tbb::combinable<Complex16> priv_addend{[](){return Complex16(0,0);}};

    //tbb::spin_mutex my_mutex;
    tbb::parallel_do( iter_values, [&](PicVector<int>& iter_value, tbb::parallel_do_feeder<PicVector<int>>& feeder) {

            // creating the v_vector
            PicVector<int> v_vector(input_state_loc.size(),0);
            size_t idx_max = 0;
            for ( size_t idx=0; idx<iter_value.size(); idx++) {
                if ( iter_value[idx] > 0 ) {
                    v_vector[input_state_inidices[idx]] = iter_value[idx];
                    idx_max = idx;
                }

            }

/*
        {
            tbb::spin_mutex::scoped_lock my_lock{my_mutex};
            print_state(v_vector);
        }
*/

            int v_sum = sum(v_vector);
            Complex16 addend(pow(-1.0, v_sum), 0.0);

            // Binomials calculation
            for (size_t idx=0; idx<input_state_loc.size(); idx++) { //} i in range(len(v_vector)):
                double tmp = (double)binomialCoeff( input_state_loc[idx], v_vector[idx]);
                //std::cout << "tmp: " << tmp << std::endl;
                addend.real(addend.real()*tmp);
            }


            // product calculation
            Complex16 product(1.0, 0.0);
            for ( size_t idx=0; idx<input_state_loc.size(); idx++) {
                if (output_state_loc[idx] == 0 ) { // There's no reason to calculate the sum if t_j = 0
                    continue;
                }

                // Otherwise we calculate the sum
                Complex16 product_part(0.0, 0.0);
                for ( size_t jdx=0; jdx<input_state_loc.size(); jdx++) {
                    size_t mtx_offset = idx*col_num + jdx;
                    Complex16 element = mtx_data[mtx_offset];
                    double coeff = (double) (input_state_loc[jdx] - 2 * v_vector[jdx]);
                    product_part.real( product_part.real() + coeff*element.real()); //(input_state_loc[jdx] - 2 * v_vector[jdx]) * self.__matrix[j][i]
                    product_part.imag( product_part.imag() + coeff*element.imag());
                    //std::cout << "product_part:" << product_part.real << " +i*" << product_part.imag << std::endl;
                }
                product_part = pow(product_part, output_state_loc[idx]);
/*
        {
            tbb::spin_mutex::scoped_lock my_lock{my_mutex};
            std::cout << product_part << std::endl;
        }
*/
                product = product*product_part;
            }

            addend = addend*product;
            Complex16 &addend_priv = priv_addend.local();
            addend_priv = addend_priv + addend;


            // adding new v_vectors to the do cycle
            for ( size_t idx=idx_max+1; idx<iter_value.size(); idx++) {
                for ( int jdx=1; jdx<=input_state_loc[input_state_inidices[idx]]; jdx++) {
                //for ( int jdx=1; jdx<=1; jdx++) {
                    PicVector<int> iter_value_next = iter_value;
                    iter_value_next[idx] = jdx;
                    feeder.add(iter_value_next);
                }
            }




        } // parallel_do body


    ); // parallel_do

    Complex16 permanent( 0.0, 0.0 );
    priv_addend.combine_each([&](Complex16 a) {
        permanent = permanent + a;
    });

    double factor = pow(2.0, sum(input_state));
    permanent = permanent/factor;



    return permanent;


}


/**
@brief Call to calculate sum of integers stored in a container
@param vec a container if integers
@return Returns with the sum of the elements of the container
*/
template <typename scalar>
inline int
sum( PicVector<scalar> vec) {

    int ret = 0;
    for (auto it=vec.begin(); it!=vec.end(); it++) {
        if ( *it == 0) {
            continue;
        }
        ret = ret + *it;
    }
    return ret;
}

/**
@brief Call to calculate sum of integers stored in a container
@param vec a PicState_int64 instance
@return Returns with the sum of the elements of the container
*/
inline int
sum( PicState_int64 vec) {

    int ret = 0;
    for (size_t idx=0; idx<vec.size(); idx++) {
        if ( vec[idx] == 0) {
            continue;
        }
        ret = ret + vec[idx];
    }
    return ret;
}



/**
@brief Call to calculate the Binomial Coefficient C(n, k)
@param n The integer n
@param k The integer k
@return Returns with the Binomial Coefficient C(n, k).
*/
int binomialCoeff(int n, int k) {
   int C[k+1];
   memset(C, 0, sizeof(C));
   C[0] = 1;
   for (int i = 1; i <= n; i++) {
      for (int j = std::min(i, k); j > 0; j--)
         C[j] = C[j] + C[j-1];
   }
   return C[k];

}




} // PIC
