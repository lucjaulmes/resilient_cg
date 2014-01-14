#include <stdlib.h>
#include <stdio.h>

#include "global.h"
#include "debug.h"

#include "matrix.h"

// 2 useful vector functions
double scalar_product( const int n, const double *v, const double *w )
{
	int i;
	double r = 0;

	for(i=0; i<n; i++)
		r += v[i] * w[i];

	return r;
}

void daxpy( const int n, const double a, const double *x, const double *y, double *z)
{
	int i;
	for(i=0; i<n; i++)
		z[i] = a * x[i] + y[i];
}

// matrix-vector multiplication, row major ( W = A x V )
void mult_dense ( const DenseMatrix *A , const double *V, double *W )
{
	int i, j;

	for(i=0; i < A->n; i++)
	{
		W[i] = 0;

		for(j=0; j < A->m; j++)
		    W[i] += A->v[i][j] * V[j];
	}
}

// matrix-vector multiplication, row major ( W = A x V )
void mult_sparse ( const SparseMatrix *A , const double *V, double *W )
{
	int i, j;

	for(i=0; i < A->n; i++)
	{
		W[i] = 0;

		for(j=A->r[i]; j < A->r[i+1]; j++)
		    W[i] += A->v[j] * V[ A->c[j] ];
	}
}

// matrix-vector multiplication ( W = t(V) x A = t( t(A) x V ) )
void mult_dense_transposed ( const DenseMatrix *A , const double *V, double *W )
{
	int i, j;

	for(i=0; i < A->m; i++)
	{
		W[i] = 0;
		for(j=0; j < A->n; j++)
		    W[i] += A->v[j][i] * V[j];
	}
}

// matrix-vector multiplication ( W = t(V) x A = t( t(A) x V ) )
void mult_sparse_transposed ( const SparseMatrix *A , const double *V, double *W )
{
	int i, j, col;

	for(i=0; i < A->m; i++)
		W[i] = 0;

	for(i=0; i < A->n; i++)
	{
		for(j=A->r[i]; j < A->r[i+1]; j++)
		{
		    col = A->c[j];

		    W[col] += A->v[j] * V[ col ];
		}
	}
}

void reorder_sparse_mat( const int n, int *rows, int *cols, double *vals, int *rows_cp, int *cols_cp, double *vals_cp )
{
	if( n <= 1 )
		return;
	else if( n == 2 )
	{
		if( rows[1] < rows[0] || (rows[0] == rows[1] && cols[1] < cols[0] ) )
		{
			int s;
			s = rows[0];
			rows[0] = rows[1];
			rows[1] = s;

			s = cols[0];
			cols[0] = cols[1];
			cols[1] = s;

			double t;
			t = vals[0];
			vals[0] = vals[1];
			vals[1] = t;
		}
		return ;
	}

	// two Halves, one rounded Up and one rounded Down
	int hu = (n+1)/2, hd=n/2;
	
	reorder_sparse_mat( hu, rows, cols, vals, rows_cp, cols_cp, vals_cp );
	reorder_sparse_mat( hd, &rows[hu], &cols[hu], &vals[hu] , &rows_cp[hu], &cols_cp[hu], &vals_cp[hu] );

	// since both halves are sorted and we risk to have lots of near-sorted things, let's make this shortcut :
	if( rows[hu-1] < rows[hu] || (rows[hu-1] == rows[hu] && cols[hu-1] < cols[hu]) )
		return;


	int i = 0, j = hu, k;

	// copy the first half into *_cp
	for(k=0; k<hu; k++)
	{
		rows_cp[k] = rows[k];
		cols_cp[k] = cols[k];
		vals_cp[k] = vals[k];
	}
	
	for(k=0; k<n && k<j; k++)
	{
		// if some i are left and row_i,col_i is before row_j,col_j or all j are finished
		if( i < hu && (j == n || rows_cp[i] < rows[j] || (rows_cp[i] == rows[j] && cols_cp[i] < cols[j])) )
		{
			rows[k] = rows_cp[i];
			cols[k] = cols_cp[i];
			vals[k] = vals_cp[i];
			i++;
		}
		else
		{
			rows[k] = rows[j];
			cols[k] = cols[j];
			vals[k] = vals[j];
			j++;
		}
	}
}

// sometimes we can't do any other way : a dense matrix would be too big
void read_sparse_Matrix( const int n, const int m, const int nnz, const int symmetric, SparseMatrix *A, FILE* input_file )
{
	int X, Y, i, *rows, *cols, *rows_cp, *cols_cp, pos = 0;
	double val, *vals, *vals_cp;

	rows = (int*)malloc( nnz * (1+symmetric) * sizeof(int) );
	cols = (int*)malloc( nnz * (1+symmetric) * sizeof(int) );
	vals = (double*)malloc( nnz * (1+symmetric) * sizeof(double) );

	A->n = n;
	A->m = m;

	int ctr = 0, thres = (nnz+19)/20;
	log_out("Reading file ...");

	for (i=0; i<nnz; i++)
	{
		if( i > 0 && i % thres == 0 )
		{
			ctr += 5;
			log_out(" %d%%", ctr);
		}

		fscanf(input_file, "%d %d %lg\n", &X, &Y, &val);
		X--;  /* adjust from 1-based to 0-based */
		Y--;

		// for debug purposes
		if( X >= n || Y >= m )
			continue;

		vals[pos] = val;
		cols[pos] = Y;
		rows[pos] = X;
		pos ++;

		if(symmetric && X != Y)
		{
			vals[pos] = val;
			cols[pos] = X;
			rows[pos] = Y;
			pos++;
		}
	}

	A->nnz = pos;

	log_out(" 100%%, reordering... ");

	rows_cp = (int*)malloc( A->nnz * sizeof(int) );
	cols_cp = (int*)malloc( A->nnz * sizeof(int) );
	vals_cp = (double*)malloc( A->nnz * sizeof(double) );

	// but no ordering is implied in MM format, plus symmetric values are not repeated so never appear at the right moment
	// -> let's reorder it all
	reorder_sparse_mat(A->nnz, rows, cols, vals, rows_cp, cols_cp, vals_cp);
	int lastrow = 0;
	pos = 0;

	A->r[lastrow] = pos;

	for (i=0; i<A->nnz; i++)
	{
		if( rows[i] > lastrow )
		{
			lastrow++;

			A->r[lastrow] = pos;
		}

		if( rows[i] > lastrow )
			fprintf(stderr,"!!!!!!!!! ERROR Empty row %d !\n", i);

		if( i > 0 && (rows[i] < rows[i-1] || (rows[i] == rows[i-1] && cols[i] < cols[i-1]) ) )
			fprintf(stderr, "!!!!!!!!! ERROR Elements not ordered : %d [%d,%d] > %d [%d, %d]", i-1, rows[i-1], cols[i-1], i, rows[i], cols[i]);

		A->v[pos] = vals[i];
		A->c[pos] = cols[i];
		pos++;
	}

	lastrow++;
	A->r[lastrow] = pos;

	log_out("done.\n");

	free(rows);
	free(cols);
	free(vals);
	free(rows_cp);
	free(cols_cp);
	free(vals_cp);
}


void read_dense_Matrix( const int n, const int m, const int nnz, const int symmetric, DenseMatrix *A, FILE* input_file )
{
	int X, Y, i;
	double val;

	A->n = n;
	A->m = m;
	// A->v should be calloc'd thus all zeroes...

	int ctr = 0, thres = (nnz+19)/20;
	log_out("Reading file ...");

	for (i=0; i<nnz; i++)
	{
		if( i > 0 && i % thres == 0 )
		{
			ctr += 5;
			log_out(" %d%%", ctr);
		}

		fscanf(input_file, "%d %d %lg\n", &X, &Y, &val);
		X--;  /* adjust from 1-based to 0-based */
		Y--;

		// for debug purposes
		if( X >= n || Y >= m )
			continue;

		A->v[X][Y] = val;

		if(symmetric && X != Y)
			A->v[Y][X] = val;
	}

	log_out("done.\n");
}

void dense_to_sparse_Matrix( const DenseMatrix *B, SparseMatrix *A )
{
	int i,j,pos = 0;

	for(i=0; i<B->n; i++)
	{
		A->r[i] = pos;

		for(j=0; j<B->n; j++)
		    if(B->v[i][j] != 0)
		    {
		        A->v[pos] = B->v[i][j];
		        A->c[pos] = j;
		        pos++;
		    }
	}

	A->n = B->n;
	A->m = B->m;
	A->r[B->n] = pos;
	A->nnz = pos;
}

// matrix multiplication A x B = C (suppose all matrices to have right sizes)
void mult_dense_matrix ( const DenseMatrix *A, const DenseMatrix *B, DenseMatrix *C )
{
	// A->n == C->n
	// B->m == C->m
	// A->m == B->n

	int i, j, k;
	for(i=0; i < C->n; i++)
		for(j=0; j < C->m; j++)
		{
		    C->v[i][j] = 0;
		    for(k=0; k<A->m; k++)
		        C->v[i][j] += A->v[i][k] * B->v[k][j];
		}
}

// matrix multiplication
void transpose_dense_matrix ( const DenseMatrix *A, DenseMatrix *B )
{
	// A->m == B->n
	// A->n == B->m

	int i, j;
	for(i=0; i < A->n; i++)
		for(j=0; j < A->m; j++)
		    B->v[j][i] = A->v[i][j];
}

void print_dense( const DenseMatrix *A )
{
	int i, j;
	for(i=0; i < A->n; i++)
	{
		log_out("| ");

		for(j=0; j < A->m; j++)
			if( A->v[i][j] == 0 )
				log_out("  0		" );
			else
				log_out(" % 1.2e ", A->v[i][j] );

		log_out(" |\n");
	}
}

void print_sparse( const SparseMatrix *A )
{
	int i, j;
	for(i=0; i < A->n; i++)
	{
		log_out("%4d   |  ", i);

		for( j= A->r[i]; j < A->r[i+1]; j++)
			log_out(" [%4d ] % 1.2e ", A->c[j], A->v[j]);

		log_out("\n");
	}
}


void allocate_dense_matrix(const int n, const int m, DenseMatrix *A)
{
	A->n = n;
	A->m = m;

	A->v = (double**) malloc( n * sizeof(double*) );
	// contiguous
	A->v[0] = (double*) calloc( n * m, sizeof(double) );

	int i;
	for(i=1; i<n; i++)
		A->v[i] = A->v[0] + i * m ;
}

void deallocate_dense_matrix(DenseMatrix *A)
{
	free(A->v[0]);
	free(A->v);
}


void allocate_sparse_matrix(const int n, const int m, const int nnz, SparseMatrix *A)
{
	A->n = n;
	A->m = m;
	A->nnz = nnz;

	A->r = (int*)malloc( (n+1) * sizeof(int) );
	A->c = (int*)calloc( nnz, sizeof(int) );
	A->v = (double*)calloc( nnz, sizeof(double) );
}

void deallocate_sparse_matrix(SparseMatrix *A)
{
	free(A->r);
	free(A->c);

	free(A->v);
}

void submatrix_dense( const DenseMatrix *A , const int *rows, const int nr, const int *cols, const int nc, const int bs, DenseMatrix *B )
{
	int i, ii, j, jj, k, l;

	for(i=0, k=0; i<nr; i++)
		for(ii=rows[i]; ii < rows[i] + bs && ii < A->n ; ii++, k++)

			for(j=0, l=0; j<nc; j++)
				for(jj=cols[j]; jj < cols[j] + bs && jj < A->m ; jj++, l++)

					B->v[k][l] = A->v[ii][jj];
}

void submatrix_sparse_to_dense( const SparseMatrix *A , const int *rows, const int nr, const int *cols, const int nc, const int bs, DenseMatrix *B )
{
	// nc = Number of Columns, nr = Number of Rows, bs = Block Size
	int i, ii, j, jj, k;

	// i iterates each block of rows, ii each row in A that needs to be copied. Parallelly, k iterates each row in B corresponding to ii.
	for(i=0, k=0; i<nr; i++)
		for(ii=rows[i]; ii < rows[i] + bs && ii < A->n && k < B->n ; ii++, k++)
		{
			// now j iterates over each element in A, and jj over each block of columns
			// if j is found to be in a block [ cols[jj], cols[jj] + bs ], we compute the corresponding column in B and copy the value
			for(j=A->r[ii], jj = 0; j < A->r[ii+1]; j++)
			{
				while( jj < nc && A->c[j] >= cols[jj] + bs )
					jj++;

				// from here on, we are sure that A->c[j] < cols[jj] + bs

				// if we did all the blocks for row ii, go to next row
				if( jj >= nc )
					break;

				if( A->c[j] >= cols[jj] )
				{
					int col_in_B = jj * bs + (A->c[j] - cols[jj]);

					if( col_in_B > B->m )
						break;

					B->v[k][col_in_B] = A->v[j];
				}
			}
		}
}

void submatrix_sparse( const SparseMatrix *A , const int *rows, const int nr, const int *cols, const int nc, const int bs, SparseMatrix *B )
{
	// nb = Number of Blocks, bs = Block Size
	int i, ii, j, jj, k, p = 0;

	B->r[0] = 0;

	// i iterates each block of rows, ii each row in A that needs to be copied. Parallelly, k iterates each row in B corresponding to ii.
	for(i=0, k=0; i<nr; i++)
		for(ii=rows[i]; ii < rows[i] + bs && ii < A->n && k < B->n ; ii++, k++)
		{
			
			// now j iterates over each element in A, and jj over each block of columns
			// if j is found to be in a block [ cols[jj], cols[jj] + bs ], we compute the corresponding column in B and copy the value
			for(j=A->r[ii], jj = 0; j < A->r[ii+1]; j++)
			{
				while( jj < nc && A->c[j] >= cols[jj] + bs )
					jj++;

				// from here on, we are sure that A->c[j] < cols[jj] + bs

				// if we did all the blocks for row ii, go to next row
				if( jj >= nc )
					break;

				if( A->c[j] >= cols[jj] )
				{
					int col_in_B = jj * bs + (A->c[j] - cols[jj]);

					if( col_in_B > B->m )
						break;

					B->v[p] = A->v[j];
					B->c[p] = col_in_B;
					p++;
				}
			}

			B->r[k+1] = p;
		}
}


