#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "global.h"
#include "debug.h"

#include "matrix.h"

// 3 useful vector functions
double norm( const int n, const double *v )
{
	int i;
	double r = 0;

	for(i=0; i<n; i++)
		r += v[i] * v[i];

	return r;
}

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
void mult ( const Matrix *A,  const double *V, double *W )
{
	int i, j;

	for(i=0; i < A->n; i++)
	{
		W[i] = 0;

		for(j=A->r[i]; j < A->r[i+1]; j++)
			W[i] += A->v[j] * V[ A->c[j] ];
	}
}


void mult_dense_blockrow ( const int m, const int bs, const char *dep UNUSED, const int stride, const double *mat, const double *V, double *W )
{
	int i, j;

	for(i=0; i<bs; i++)
	{
		W[i] = 0;
		for(j=0; j<m; j++)
			W[i] += mat[i * stride + j] * V[j];
	}

	log_err(SHOW_TASKINFO, "Blockrow matmult [%d x %d] %p = [%p..%p -1] x %p finished : %e\n", bs, m, W, mat, &mat + (i+1)*stride, V, scalar_product(bs, W, W));
}

// slightly overlapping in's (from rows) should not be a problem
void mult_blockrow( const int m UNUSED, const int bs, const char *dep, const int *rows, const int *endrows, const double *vals, const int *cols, const double *V, double *W )
{
	int i, j;

	for(i=0; i < bs; i++)
	{
		W[i] = 0;

		for(j=rows[i]; j < endrows[i]; j++)
			W[i] += vals[j] * V[ cols[j] ];
	}

	if( dep != NULL && dep[0] == 'i' )
		log_out("doing the direct computation for g\n");

	log_err(SHOW_TASKINFO, "Blockrow matmult [%d x %d] %p = [%p..%p -1] x %p finished %e\n", bs, m, W, vals+rows[0], vals+endrows[bs-1], V, scalar_product(bs, W, W));
}

// matrix-vector multiplication ( W = t(V) x A = t( t(A) x V ) )
void mult_transposed ( const Matrix *A , const double *V, double *W )
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

void write( FILE* f, const Matrix *A )
{
	int i, j;
	for(i=0; i < A->n; i++)
	{
		for( j= A->r[i]; j < A->r[i+1]; j++)
			if( i > A->c[j] )
				continue;
			else
				fprintf(f, "%d %d %12.7lf\n", A->c[j]+1, i+1, A->v[j]);

		log_out("\n");
	}
}

// return pos in matrix ( so then you only have to take A->v[pos] ), -1 if does not exist
int find_in_mat( const int row, const int col, const Matrix *A )
{
	if( row > A->n || col > A->m )
		return -1;
	
	int low = A->r[row], upp = A->r[row+1]-1, mid;

	if(A->c[low] == col)
		return low;
	if(A->c[upp] == col)
		return upp;

	while( low+1 < upp )
	{
		mid = (low + upp) / 2;

		if( A->c[mid] > col )
			upp = mid;
		else if( A->c[mid] < col )
			low = mid;
		else
			return mid;
	}

	return -1;
}


void read_matrix( const int n, const int m, const int nnz, const int symmetric, Matrix *A, FILE* input_file )
{
	int Y, prevY = -1, X, i, j, k, pos = 0, *nb_subdiagonals;
	double val;

	if( symmetric )
		nb_subdiagonals = (int*)calloc( n, sizeof(int) );

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
		if( Y >= n || X >= m )
			continue;

		if( Y > prevY )
		{
			A->r[Y] = pos;

			// leave space for the subdiagonals elements
			if( symmetric )
				pos += nb_subdiagonals[Y];

			prevY = Y;
		}

		A->v[pos] = val;
		A->c[pos] = X;
		pos ++;

		if( symmetric && X > Y )
			nb_subdiagonals[X]++;
	}

	A->nnz = pos;
	A->r[A->n] = pos;

	log_out(" 100%%, filling symmetric part...");
	
	if( symmetric )
	{
		// now let's fill in the subdiagonal part
		int *fill_row = malloc( n * sizeof(int) );

		for( j=0; j<A->n; j++ )
			fill_row[j] = A->r[j];

		for( i=0; i<A->n; i++ )
			for( k = A->r[i] + nb_subdiagonals[i] ; k < A->r[i+1] ; k++ )
			{
				if( i == A->c[k] )
					continue;

				j = A->c[k];
				// now put (i,j) in (j,i)

				pos = fill_row[j];
				A->c[pos] = i;
				A->v[pos] = A->v[k];

				fill_row[j]++;
			}

		free(nb_subdiagonals);
		free(fill_row);
		log_out(" done.\n");
	}
}

void print( const Matrix *A )
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

void allocate_matrix(const int n, const int m, const int nnz, Matrix *A)
{
	A->n = n;
	A->m = m;
	A->nnz = nnz;


	A->r = (int*)malloc( (n+1) * sizeof(int) );

	A->c = (int*)calloc( nnz, sizeof(int) );
	A->v = (double*)calloc( nnz, sizeof(double) );

	if( ! A->v || ! A->c || ! A->r )
	{
		fprintf(stderr, "Allocating sparse matrix of size %d rows and %d non-zeros failed !\n", n, nnz);
		exit(2);
	}

	//int align = get_alignment(), potential_padding = n * align;
	//posix_memalign((void**)&A->c, align, nnz * sizeof(int) + potential_padding );
	//posix_memalign((void**)&A->v, align, nnz * sizeof(double) + potential_padding );

	//memset(A->c, '\0', nnz * sizeof(int) + potential_padding );
	//memset(A->v, '\0', nnz * sizeof(double) + potential_padding );
}

void deallocate_matrix(Matrix *A)
{
	free(A->r);
	free(A->c);

	free(A->v);
}

void get_submatrix( const Matrix *A , const int *rows, const int nr, const int *cols, const int nc, const int bs, Matrix *B )
{
	// nb = Number of Blocks, bs = Block Size
	int i, ii, j, jj, k, p = 0;

	// i iterates each block of rows, ii each row in A that needs to be copied. Parallelly, k iterates each row in B corresponding to ii.
	for(i=0, k=0; i<nr; i++)
		for(ii=rows[i]; ii < rows[i] + bs && ii < A->n && k < B->n ; ii++, k++)
		{
			B->r[k] = p;
			
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
		}
	
	B->r[k] = p;
	B->nnz = p;
}


