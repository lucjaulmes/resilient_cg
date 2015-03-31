#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED

#include <stdio.h>

// just matrix stuff, should be blackbox

// types we are going to use
typedef struct 
{
	int n, m, nnz, *c, *r;
	double *v;
} Matrix;

// 3 useful functions that don't fit somewhere else
double scalar_product( const int n, const double *v, const double *w );
double norm( const int n, const double *v);
void daxpy( const int n, const double a, const double *x, const double *y, double *z);

// general matrix-vector multiplication, row major ( W = A x V )
// dense, sparse, single block and task-blocks versions
void mult( const Matrix *A, const double *V, double *W );

int find_in_mat( const int row, const int col, const Matrix *A );

// debug function to show a matrix on screen
void print( const Matrix *A );

// read the matrix data from a Matrix Market file (header already parsed)
void read_matrix( const int n, const int m, const int nnz, const int symmetric, Matrix *A, FILE* input_file );

// memory utility functions
void allocate_matrix(const int rows, const int cols, const int nnz, Matrix *A);
void deallocate_matrix(Matrix *A);

// get the submatrix with rows 'rows', with all columns but 'cols'
void get_submatrix( const Matrix *A, const int *rows, const int nr, const int *cols, const int nc, const int bs, Matrix *B );

#endif // MATRIX_H_INCLUDED

