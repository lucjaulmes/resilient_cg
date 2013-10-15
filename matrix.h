#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED

#include <stdio.h>

// just matrix stuff, should be blackbox

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

// matrix-vector multiplication, row major ( W = A x V )
void mult_dense ( const void *mat, const double *V, double *W );
void mult_sparse ( const void *mat, const double *V, double *W );

void mult_dense_matrix ( const DenseMatrix *A, const DenseMatrix *B, DenseMatrix *C );

void mult_dense_transposed ( const DenseMatrix *A, const double *V, double *W );
void mult_sparse_transposed ( const SparseMatrix *A, const double *V, double *W );

void print_dense( const DenseMatrix *A );
void print_sparse( const SparseMatrix *A );

void transpose_dense_matrix ( const DenseMatrix *A, DenseMatrix *B );

void read_dense_Matrix( const int n, const int m, const int nnz, const int symmetric, DenseMatrix *A, FILE* input_file );

void dense_to_sparse_Matrix( const DenseMatrix *B, SparseMatrix *A );

void allocate_dense_matrix(const int rows, const int cols, DenseMatrix *A);
void allocate_sparse_matrix(const int rows, const int cols, const int nnz, SparseMatrix *A);

void deallocate_dense_matrix(DenseMatrix *A);
void deallocate_sparse_matrix(SparseMatrix *A);

void submatrix_dense( const void *A, int *rows, int *cols, DenseMatrix *B );
void submatrix_sparse_to_dense( const void *A, int *rows, int *cols, DenseMatrix *B );
void submatrix_sparse( const void *A, int *rows, int *cols, SparseMatrix *B );

typedef void (*SubmatrixFunction)(const void*, int*, int*, DenseMatrix*);

#endif // MATRIX_H_INCLUDED

