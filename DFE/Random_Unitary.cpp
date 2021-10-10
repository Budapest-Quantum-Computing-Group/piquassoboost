/*
Created on Fri Jun 26 14:13:26 2020
Copyright (C) 2020 Peter Rakyta, Ph.D.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
*/
/*! \file Random_Unitary.cpp
    \brief A class and methods to cerate random unitary matrices
*/


#include "Random_Unitary.h"
#include <tbb/scalable_allocator.h>
#include "dot.h"

namespace pic {

/**
@brief Multiply the elements of matrix b by a scalar a.
@param a A complex scalar.
@param b A square shaped matrix.
*/
void mult( Complex16 a, matrix& b ) {

    size_t element_num = b.size();

    for (size_t idx=0; idx<element_num; idx++) {
        Complex16 tmp = b[idx];
        b[idx].real( a.real()*tmp.real() - a.imag()*tmp.imag() );
        b[idx].imag( a.real()*tmp.imag() + a.imag()*tmp.real() );
    }

    return;

}


/**
@brief Call to create an identity matrix
@param matrix_size The number of rows in the resulted identity matrix
@return Returns with an identity matrix.
*/
matrix create_identity( int matrix_size ) {

    matrix mtx = matrix(matrix_size, matrix_size);
    memset(mtx.get_data(), 0, mtx.size()*sizeof(Complex16) );

    // setting the diagonal elelments to identity
    for(int idx = 0; idx < matrix_size; ++idx)
    {
        int element_index = idx*matrix_size + idx;
            mtx[element_index].real( 1.0 );
    }

    return mtx;

}


/**
@brief Constructor of the class.
@param dim_in The number of rows in the random unitary to be ceated.
@return An instance of the class
*/
Random_Unitary::Random_Unitary( int dim_in ) {

        if (dim_in < 2) {
            throw "wrong dimension";
        }

        // number of qubits
        dim = dim_in;

}


/**
@brief Call to create a random unitary
@return Returns with a pointer to the created random unitary
*/
matrix
Random_Unitary::Construct_Unitary_Matrix() {

    // create array of random parameters to construct random unitary
    double* vartheta = (double*) scalable_aligned_malloc( int(dim*(dim-1)/2)*sizeof(double), 64);
    double* varphi = (double*) scalable_aligned_malloc( int(dim*(dim-1)/2)*sizeof(double), 64);
    double* varkappa = (double*) scalable_aligned_malloc( (dim-1)*sizeof(double), 64);

    // initialize random seed:
    srand (time(NULL));

    for (int idx=0; idx<dim*(dim-1)/2; idx++) {
        vartheta[idx] = (2*double(rand())/double(RAND_MAX)-1)*2*M_PI;
    }


    for (int idx=0; idx<dim*(dim-1)/2; idx++) {
        varphi[idx] = (2*double(rand())/double(RAND_MAX)-1)*2*M_PI;
    }


    for (int idx=0; idx<(dim-1); idx++) {
        varkappa[idx] = (2*double(rand())/double(RAND_MAX)-1)*2*M_PI;
    }

    matrix Umtx = Construct_Unitary_Matrix( vartheta, varphi, varkappa );

    scalable_aligned_free(vartheta);
    scalable_aligned_free(varphi);
    scalable_aligned_free(varkappa);
    vartheta = NULL;
    varphi = NULL;
    varkappa = NULL;

    return Umtx;

}


/**
@brief Generates a unitary matrix from parameters vartheta, varphi, varkappa according to arXiv:1303:5904v1
@param vartheta array of dim*(dim-1)/2 elements
@param varphi array of dim*(dim-1)/2 elements
@param varkappa array of dim-1 elements
@return Returns with a pointer to the generated unitary
*/
matrix
Random_Unitary::Construct_Unitary_Matrix( double* vartheta, double* varphi, double* varkappa ) {


        matrix ret = create_identity(dim);

        for (int varalpha=1; varalpha<dim; varalpha++) { // = 2:obj.dim
           for (int varbeta = 0; varbeta<varalpha; varbeta++) {//   1:varalpha-1

               double theta_loc = vartheta[ convert_indexes(varalpha, varbeta) ];
               double phi_loc = varphi[ convert_indexes(varalpha, varbeta) ];


               // Eq (26)
               Complex16 a;
               a.real( cos( theta_loc )*cos(phi_loc) );
               a.imag( cos( theta_loc )*sin(phi_loc) );

               // Eq (28) and (26)
               double varepsilon = varkappa[varalpha-1]*kronecker( varalpha-1, varbeta);
               Complex16 b;
               b.real( sin( theta_loc )*cos(varepsilon));
               b.imag( sin( theta_loc )*sin(varepsilon));

               a.real( -a.real());
               b.imag( -b.imag());
               matrix Omega_loc = Omega( varalpha, varbeta, a, b );
               matrix ret_tmp = dot( ret, Omega_loc); //   ret * Omega_loc

               ret = ret_tmp;
           }
        }


        Complex16 gamma_loc;
        gamma_loc.real( gamma());
        gamma_loc.imag( 0 );

        for ( int idx=0; idx<dim*dim; idx++) {
            ret[idx] = ret[idx]*gamma_loc;
        }

        return ret;



}


/**
@brief Calculates an index from paramaters varalpha and varbeta
@param varalpha An integer
@param varbeta An integer
@return Returns with the calculated index.
*/
int
Random_Unitary::convert_indexes( int varalpha, int varbeta ) {
     int ret = varbeta + (varalpha-1)*(varalpha-2)/2;
     return ret;
}


/**
@brief Generates a unitary matrix from parameters parameters according to arXiv:1303:5904v1
@param parameters array of (dim+1)*(dim-1) elements
@return The constructed unitary
*/
matrix Random_Unitary::Construct_Unitary_Matrix(double* parameters ) {
   return Construct_Unitary_Matrix( parameters, parameters+int(dim*(dim-1)/2), parameters+int(dim*(dim-1)));
}


/**
@brief Eq (6) of arXiv:1303:5904v1
@param varalpha An integer
@param varbeta An integer
@param x A complex number
@param y A complex number
@return Return with a pointer to the calculated Omega matrix of Eq. (6) of arXiv:1303:5904v1
*/
matrix
Random_Unitary::Omega(int varalpha, int varbeta, Complex16 x, Complex16 y )   {

        matrix ret = I_alpha_beta( varalpha, varbeta );


        matrix Mloc;

        if (varalpha + varbeta != (3 + kronecker( dim, 2 )) ) {
            Mloc = M( varalpha, varbeta, x, y );

        }
        else {
            y.imag( -y.imag() );
            Mloc = M( varalpha, varbeta, x, gamma()*y );
        }


        //#pragma omp parallel for
        for ( int idx=0; idx<dim*dim; idx++ ) {
            ret[idx] = ret[idx] + Mloc[idx];
        }

        return ret;


}


/**
@brief Implements Eq (8) of arXiv:1303:5904v1
@param varalpha An integer
@param varbeta An integer
@param s A complex number
@param t A complex number
@return Return with a pointer to the calculated M matrix of Eq. (8) of arXiv:1303:5904v1
*/
matrix
Random_Unitary::M( int varalpha, int varbeta, Complex16 s, Complex16 t )   {

        matrix Qloc = Q( s, t);

        matrix ret1 = E_alpha_beta( varbeta, varbeta );
        matrix ret2 = E_alpha_beta( varbeta, varalpha );
        matrix ret3 = E_alpha_beta( varalpha, varbeta );
        matrix ret4 = E_alpha_beta( varalpha, varalpha );


        mult(Qloc[0], ret1);
        mult(Qloc[1], ret2);
        mult(Qloc[2], ret3);
        mult(Qloc[3], ret4);

        matrix ret = matrix(dim, dim);

        for ( int idx=0; idx<dim*dim; idx++ ) {
            ret[idx] = ret1[idx] + ret2[idx] + ret3[idx] + ret4[idx];
        }

        return ret;

}


/**
@brief Implements Eq (9) of arXiv:1303:5904v1
@param u1 A complex number
@param u2 A complex number
@return Return with a pointer to the calculated Q matrix of Eq. (9) of arXiv:1303:5904v1
*/
matrix
Random_Unitary::Q(  Complex16 u1, Complex16 u2 )   {

    matrix ret = matrix(2, 2);
    ret[0] = u2;
    ret[1] = u1;
    ret[2].real( -u1.real() );
    ret[2].imag( u1.imag() );
    ret[3].real( u2.real() );
    ret[3].imag( -u2.imag() );

    return ret;

}


/**
@brief Implements matrix I below Eq (7) of arXiv:1303:5904v1
@param varalpha An integer
@param varbeta An integer
@return Return with a pointer to the calculated E matrix of Eq. (7) of arXiv:1303:5904v1
*/
matrix
Random_Unitary::E_alpha_beta( int varalpha, int varbeta )   {

    matrix ret = matrix(dim, dim);
    memset( ret.get_data(), 0, dim*dim*sizeof(Complex16));
    ret[varalpha*dim+varbeta].real( 1 );

    return ret;

}

/**
@brief Implements matrix I below Eq (7) of arXiv:1303:5904v1
@param varalpha An integer
@param varbeta An integer
@return Return with a pointer to the calculated I matrix of Eq. (7) of arXiv:1303:5904v1
*/
matrix
Random_Unitary::I_alpha_beta( int varalpha, int varbeta ) {


   matrix ret = create_identity(dim);

   ret[varalpha*dim+varalpha].real( 0 );
   ret[varbeta*dim+varbeta].real( 0 );

   return ret;

}


/**
@brief Implements Eq (11) of arXiv:1303:5904v1
@return Returns eith the value of gamma
*/
double
Random_Unitary::gamma() {

    double ret = pow(-1, 0.25*(2*dim-1+pow(-1,dim)));//(-1)^(0.25*(2*dim-1+(-1)^dim));
    return ret;

}

/**
@brief Kronecker delta
@param a An integer
@param b An integer
@return Returns with the Kronecker delta value of a and b.
*/
double
Random_Unitary::kronecker( int a, int b ) {

        if (a == b) {
            return 1;
        }
        else {
            return 0;
        }

}



}




