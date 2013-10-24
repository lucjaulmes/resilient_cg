#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED

#include <stdio.h>

// just matrix stuff, should be blackbox

// types we are going to use
typedef struct DenseMatrix
{
	int n, m;
	double **v;
} DenseMatrix;

typedef struct SparseMatrix
{
	int n, m, nnz, *c, *r;
	double *v;
} SparseMatrix;

typedef void (*MultFunction)(const void*, const double*, double*);

void daxpy( const int n, const double a, const double *x, const double *y, double *z);

// general matrix-vector multiplication, row major ( W = A x V )
void mult_dense ( const void *mat, const double *V, double *W );
void mult_sparse ( const void *mat, const double *V, double *W );

// transposed versions W = t(A) * V
void mult_dense_transposed ( const void *mat, const double *V, double *W );
void mult_sparse_transposed ( const void *mat, const double *V, double *W );

// matrix-matrix multiplication
void mult_dense_matrix ( const DenseMatrix *A, const DenseMatrix *B, DenseMatrix *C );

// debug function to show a matrix on screen
void print_dense( const DenseMatrix *A );
void print_sparse( const SparseMatrix *A );

void transpose_dense_matrix ( const DenseMatrix *A, DenseMatrix *B );

// read the matrix data from a Matrix Market file (header already parsed)
void read_dense_Matrix( const int n, const int m, const int nnz, const int symmetric, DenseMatrix *A, FILE* input_file );

// transform a dense matrix into a sparse matrix
void dense_to_sparse_Matrix( const DenseMatrix *B, SparseMatrix *A );

// memory utility functions
void allocate_dense_matrix(const int rows, const int cols, DenseMatrix *A);
void allocate_sparse_matrix(const int rows, const int cols, const int nnz, SparseMatrix *A);

void deallocate_dense_matrix(DenseMatrix *A);
void deallocate_sparse_matrix(SparseMatrix *A);

// get the submatrix with rows 'rows', with all columns but 'cols'
void submatrix_dense( const void *A, int *rows, int *cols, DenseMatrix *B );
void submatrix_sparse_to_dense( const void *A, int *rows, int *cols, DenseMatrix *B );
void submatrix_sparse( const void *A, int *rows, int *cols, SparseMatrix *B );

typedef void (*SubmatrixFunction)(const void*, int*, int*, DenseMatrix*);

#endif // MATRIX_H_INCLUDED

