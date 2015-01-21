#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "global.h"
#include "debug.h"

#include "matrix.h"

// matrix-vector multiplication, row major ( W = A x V )
void mult( const Matrix *A,  const double *V, double *W )
{
	int i, j;

	for(i=mpi_zonestart[mpi_rank]; i<mpi_zonestart[mpi_rank]+mpi_zonesize[mpi_rank]; i++)
	{
		W[i] = 0;

		for(j=A->r[i]; j < A->r[i+1]; j++)
			W[i] += A->v[j] * V[ A->c[j] ];
	}
}

// matrix-vector multiplication ( W = t(V) x A = t( t(A) x V ) )
void mult_transposed ( const Matrix *A , const double *V, double *W )
{
	int i, j, col;

	for(i=0; i < A->m; i++)
		W[i] = 0;

	for(i=mpi_zonestart[mpi_rank]; i<mpi_zonestart[mpi_rank]+mpi_zonesize[mpi_rank]; i++)
	{
		for(j=A->r[i]; j < A->r[i+1]; j++)
		{
			col = A->c[j];

			W[col] += A->v[j] * V[ col ];
		}
	}
}

void print_matrix( FILE* f, const Matrix *A )
{
	int i, j;
//	for(i=mpi_zonestart[mpi_rank]; i<mpi_zonestart[mpi_rank]+mpi_zonesize[mpi_rank]; i++)
//	{
//		printf("%4d   |  ", i);
//
//		for( j= A->r[i]; j < A->r[i+1]; j++)
//			fprintf(f, " [%4d ] % 1.2e ", A->c[j], A->v[j]);
//
//		printf("\n");
//	}

	// hoping less than 100 items / line
	int c[100], n=0, k;
	double v[100];

	for(i=mpi_zonestart[mpi_rank]; i < mpi_zonestart[mpi_rank] + mpi_zonesize[mpi_rank]; i++)
	{
		int same = n == A->r[i+1] - A->r[i];

		for( j = A->r[i], k=0; same && j < A->r[i+1] && k < 100; j++, k++)
			same &= (A->c[j]-i == c[k] && A->v[j] == v[k]);

		if( ! same )
		{
			fprintf(f, "%5d -- ", i);

			n = A->r[i+1] - A->r[i];
			for( j = A->r[i], k=0; j < A->r[i+1] && k < 100; j++, k++)
			{
				fprintf(f, " [%5d] % 1.2e ", A->c[j]-i, A->v[j]);
				c[k] = A->c[j]-i;
				v[k] = A->v[j];
			}

			fprintf(f, "\n");
		}
	}
	fprintf(f, "%5d -- end\n", i);
}

// return pos in matrix ( so then you only have to take A->v[pos] ), -1 if does not exist
int find_in_matrix( const int row, const int col, const Matrix *A )
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
	int Y, prevY = -1, X, i, j, k, pos = 0, *nb_subdiagonals = NULL;
	double val;

	if( symmetric )
		nb_subdiagonals = (int*)calloc( n, sizeof(int) );

	A->n = n;
	A->m = m;

	int ctr = 0, thres = (nnz+19)/20;
	{}//log_out("Reading file ...");

	for (i=0; i<nnz; i++)
	{
		if( i > 0 && i % thres == 0 )
		{
			ctr += 5;
			{}//log_out(" %d%%", ctr);
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

	{}//log_out(" 100%%, filling symmetric part...");
	
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
		{}//log_out(" done.\n");
	}
}

// finite-difference method for a 3D Poisson's equation gives a SPD matrix with -6 on the diagonal, 
// and 1s on the diagonal+1, diagonal+p and diagonal+p²
void generate_Poisson3D(Matrix *A, const int p, const int stencil_points, const int start_row, const int end_row)
{
	int p2 = p * p, i, j=0, pos=0;

	const int    *stenc_c;
	const double *stenc_v;

	const int    stenc_c7[]  = { -p2,  -p,  -1,   0,   1,   p,  p2};
	const double stenc_v7[]  = { 1.0, 1.0, 1.0,-6.0, 1.0, 1.0, 1.0};

	const double r = 1.0;
	const int    stenc_c19[] =
	{
		       -p2-p,          -p2-1,  -p2+0, -p2+1,          -p2+p,
		 -p-1,    -p,    -p+1,    -1,      0,     1,     p-1,     p,     p+1,   
		        p2-p,           p2-1,   p2+0,  p2+1,           p2+p
	};
	const double stenc_v19[] =
	{
		     1+r,      1+r,    8*r-4,   1+r,      1+r,
		2, 6-2*r, 2, 6-2*r, -32-16*r, 6-2*r, 2, 6-2*r, 2, 
		     1+r,      1+r,    8*r-4,   1+r,      1+r
	};

	const int    stenc_c27[] =
	{
		-p2-p-1, -p2-p, -p2-p+1, -p2-1,  -p2+0, -p2+1, -p2+p-1, -p2+p, -p2+p+1,
		   -p-1,    -p,    -p+1,    -1,      0,     1,     p-1,     p,     p+1,   
		 p2-p-1,  p2-p,  p2-p+1,  p2-1,   p2+0,  p2+1,  p2+p-1,  p2+p,  p2+p+1
	};
	const double stenc_v27[] =
	{
		   2+r,  8-10*r,    2+r,  8-10*r,   100*r-40,  8-10*r,    2+r,  8-10*r,    2+r,
		20-2*r, 80-20*r, 20-2*r, 80-20*r, -400-200*r, 80-20*r, 20-2*r, 80-20*r, 20-2*r,
		   2+r,  8-10*r,    2+r,  8-10*r,   100*r-40,  8-10*r,    2+r,  8-10*r,    2+r 
	};

	if( stencil_points == 7 )
	{
		stenc_c = stenc_c7;
		stenc_v = stenc_v7;
	}
	else if( stencil_points == 19 )
	{
		stenc_c = stenc_c19;
		stenc_v = stenc_v19;
	}
	else if( stencil_points == 27 )
	{
		stenc_c = stenc_c27;
		stenc_v = stenc_v27;
	}
	else
		// this should be impossible, but silences compiler warnings
		return;


	// let's only do the part here.
	for(j=start_row; j<end_row; j++)
	{
		A->r[j] = pos;
		for(i=0; i<stencil_points; i++)
			if( j + stenc_c[i] > 0 && j + stenc_c[i] < A->n )
			{
				A->c[pos] = j + stenc_c[i];
				A->v[pos] = stenc_v[i];
				pos++;
			}
	}

	// point to just beyond last element
	A->r[j] = pos;
}

void allocate_matrix(const int n, const int m, const long nnz, Matrix *A, int align_bytes )
{
	A->n = n;
	A->m = m;
	A->nnz = nnz;


	A->r = (int*)aligned_calloc( align_bytes, (n+1) * sizeof(int));

	A->c = (int*)aligned_calloc( align_bytes, nnz * sizeof(int));
	A->v = (double*)aligned_calloc( align_bytes, nnz * sizeof(double));

	if( ! A->v || ! A->c || ! A->r )
	{
		fprintf(stderr, "Allocating sparse matrix of size %d rows and %ld non-zeros failed !\n", n, nnz);
		exit(2);
	}
}

void deallocate_matrix(Matrix *A)
{
	free(A->r);
	free(A->c);

	if( A->v )
		free(A->v);
}

void get_submatrix( const Matrix *A , const int *rows, const int nr, const int *cols, const int nc, const int bs, Matrix *B )
{
	// nb = Number of Blocks, bs = Block Size
	int i, ii, j, jj, k, p = 0;

	// i iterates each block of rows, ii each row in A that needs to be copied. Parallelly, k iterates each row in B corresponding to ii.
	for(i=0, k=0; i<nr; i++)
		for(ii=rows[i]; ii < rows[i] + bs && ii < mpi_zonestart[mpi_rank]+mpi_zonesize[mpi_rank] && k < B->n ; ii++, k++)
		{
			B->r[k] = p;
			
			// now j iterates over each element in A, and jj over each block of columns
			// if j is found to be in a block [ cols[jj], cols[jj] + bs ], we compute the corresponding column in B and copy the value
			for(j=A->r[ii], jj = 0; j < A->r[ii+1]; j++)
			{
				/*
				// remove above-diagonals, if we need just half the matrix
				if( ii > A->c[j] )
					continue;
				*/
			
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


