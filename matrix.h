#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED

#include <stdio.h>

#include "global.h"

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


double scalar_product( const int n, const double *v, const double *w );
void daxpy( const int n, const double a, const double *x, const double *y, double *z);

// general matrix-vector multiplication, row major ( W = A x V )
void mult_dense ( const DenseMatrix *A, const double *V, double *W );
void mult_sparse ( const SparseMatrix *A, const double *V, double *W );

// transposed versions W = t(A) * V
void mult_dense_transposed ( const DenseMatrix *A, const double *V, double *W );
void mult_sparse_transposed ( const SparseMatrix *A, const double *V, double *W );

// matrix-matrix multiplication
void mult_dense_matrix ( const DenseMatrix *A, const DenseMatrix *B, DenseMatrix *C );

// debug function to show a matrix on screen
void print_dense( const DenseMatrix *A );
void print_sparse( const SparseMatrix *A );

void transpose_dense_matrix ( const DenseMatrix *A, DenseMatrix *B );

// read the matrix data from a Matrix Market file (header already parsed)
void read_sparse_Matrix( const int n, const int m, const int nnz, const int symmetric, SparseMatrix *A, FILE* input_file );
void read_dense_Matrix( const int n, const int m, const int nnz, const int symmetric, DenseMatrix *A, FILE* input_file );

// transform a dense matrix into a sparse matrix
void dense_to_sparse_Matrix( const DenseMatrix *B, SparseMatrix *A );

// memory utility functions
void allocate_dense_matrix(const int rows, const int cols, DenseMatrix *A);
void allocate_sparse_matrix(const int rows, const int cols, const int nnz, SparseMatrix *A);

void deallocate_dense_matrix(DenseMatrix *A);
void deallocate_sparse_matrix(SparseMatrix *A);

// get the submatrix with rows 'rows', with all columns but 'cols'
void submatrix_dense( const DenseMatrix *A, const int *rows, const int nr, const int *cols, const int nc, const int bs, DenseMatrix *B );
void submatrix_sparse_to_dense( const SparseMatrix *A, const int *rows, const int nr, const int *cols, const int nc, const int bs, DenseMatrix *B );
void submatrix_sparse( const SparseMatrix *A, const int *rows, const int nr, const int *cols, const int nc, const int bs, SparseMatrix *B );

// now depending on which we use, create aliases
#ifdef MATRIX_DENSE // using dense matrices
	#define Matrix DenseMatrix
	#define mult(A, V, W) mult_dense((DenseMatrix*)A, V, W);
	#define allocate_matrix(r, c, nnz, A) allocate_dense_matrix(r, c, (DenseMatrix*)A)
	#define deallocate_matrix(A) deallocate_dense_matrix((DenseMatrix*)A)

	#define get_submatrix(A, rows, nr, cols, nc, bs, B) submatrix_dense((DenseMatrix*)A, rows, nr, cols, nc, bs, (DenseMatrix*)B)
	#define get_dense_submatrix(A, rows, nr, cols, nc, bs, B) submatrix_dense((DenseMatrix*)A, rows, nr, cols, nc, bs, (DenseMatrix*)B)
#else // by default : using sparse matrices
	#define Matrix SparseMatrix
	#define mult(A, V, W) mult_sparse((SparseMatrix*)A, V, W);
	#define allocate_matrix(r, c, nnz, A) allocate_sparse_matrix(r, c, nnz, (SparseMatrix*)A)
	#define deallocate_matrix(A) deallocate_sparse_matrix((SparseMatrix*)A)

	#define get_submatrix(A, rows, nr, cols, nc, bs, B) submatrix_sparse((SparseMatrix*)A, rows, nr, cols, nc, bs, (SparseMatrix*)B)
	#define get_dense_submatrix(A, rows, nr, cols, nc, bs, B) submatrix_sparse_to_dense((SparseMatrix*)A, rows, nr, cols, nc, bs, (DenseMatrix*)B)
#endif

#endif // MATRIX_H_INCLUDED

