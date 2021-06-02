#ifndef CGaussianState_H
#define CGaussianState_H

#include "matrix.h"
#include "PicState.h"
#include <vector>



namespace pic {

/**
@brief Class representing a Gaussian State
*/
class CGaussianState {

protected:
    /// The matrix which is defined by
    matrix C;
    /// The matrix which is defined by
    matrix G;
    /// The displacement of the Gaussian state
    matrix m;

public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
CGaussianState();


/**
@brief Constructor of the class.
@param C_in Input matrix defined by
@param G_in Input matrix defined by
@param m_in Input matrix defined by
@return Returns with the instance of the class.
*/
CGaussianState( matrix &C_in, matrix &G_in, matrix &m_in);

/**
@brief Call to update the memory addresses of the stored matrices
@param C_in Input matrix defined by
@param G_in Input matrix defined by
@param m_in Input matrix defined by
*/
void Update( matrix &C_in, matrix &G_in, matrix &m_in);

/**
@brief Call to update the memory address of the matrix C
@param C_in Input matrix defined by
*/
void Update_C( matrix &C_in);


/**
@brief Call to update the memory address of the matrix G
@param G_in Input matrix defined by
*/
void Update_G(matrix &G_in);


/**
@brief Call to update the memory address of the vector containing the displacements
@param m_in The new displacement vector
*/
void Update_m(matrix &m_in);



/**
@brief Applies the matrix T to the C and G.
@param T The matrix of the transformation.
@param modes The modes, on which the matrix should operate.
@return Returns with 0 in case of success.
*/
int apply_to_C_and_G( matrix &T, std::vector<size_t> modes );




}; //CGaussianState


} // PIC

#endif
