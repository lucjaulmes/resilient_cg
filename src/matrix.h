#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED

#include <stdio.h>

typedef struct Matrix
{
	int n, m, *c, *r;
	long nnz;
	double *v;
} Matrix;

typedef enum 
{
	FROM_FILE = 0,
	POISSON3D
} matrix_type;


// general matrix-vector multiplication, row major (W = A x V)
// dense, sparse, single block and task-blocks versions
void mult(const Matrix *A, const double *V, double *W);

// transposed versions W = t(A) * V
void mult_transposed(const Matrix *A, const double *V, double *W);

//
int find_in_matrix(const int row, const int col, const Matrix *A);

// read the matrix data from a Matrix Market file (header already parsed)
void read_matrix(const int n, const int m, const int nnz, const int symmetric, Matrix *A, FILE* input_file, const int offset);
// visual representation of matrix, with columns given in absolute or relative to diagonal (in which case similar rows are only written once)
void print_matrix_abs(FILE* f, const Matrix *A);
void print_matrix_rel(FILE* f, const Matrix *A);

void generate_Poisson3D(Matrix *A, const int p, const int stencil_points, const int start_row, const int n_rows);

// memory utility functions
void allocate_matrix(const int n, const int m, const long nnz, Matrix *A, int align_bytes);

void deallocate_matrix(Matrix *A);

// get the submatrix with rows 'rows', with all columns but 'cols'
void get_submatrix(const Matrix *A, const int *rows, const int nr, const int *cols, const int nc, const int bs, Matrix *B);

// 3 useful functions that don't fit somewhere else
static inline double norm(const int n, const double *v)
{
	int i;
	double r = 0;

	for(i=0; i<n; i++)
		r += v[i] * v[i];

	return r;
}

static inline double scalar_product(const int n, const double *v, const double *w)
{
	int i;
	double r = 0;

	for(i=0; i<n; i++)
		r += v[i] * w[i];

	return r;
}

static inline void daxpy(const int n, const double a, const double *x, const double *y, double *z)
{
	int i;
	for(i=0; i<n; i++)
		z[i] = a * x[i] + y[i];
}

#endif // MATRIX_H_INCLUDED

