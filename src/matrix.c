#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "global.h"
#include "debug.h"

#include "matrix.h"

// matrix-vector multiplication, row major (W = A x V)
// takes in local W but global V
void mult(const Matrix *A, const double *V, double *W)
{
	int i, j;

	for(i=0; i<mpi_zonesize[mpi_here]; i++)
	{
		W[i] = 0;

		for(j=A->r[i]; j < A->r[i+1]; j++)
			W[i] += A->v[j] * V[ A->c[j] ];
	}
}

// matrix-vector multiplication ( W = t(V) x A = t(t(A) x V ))
// takes in global W but local V
void mult_transposed(const Matrix *A , const double *V, double *W)
{
	int i, j, col;

	for(i=0; i < A->m; i++)
		W[i] = 0;

	for(i=0; i<mpi_zonesize[mpi_here]; i++)
	{
		for(j=A->r[i]; j < A->r[i+1]; j++)
		{
			col = A->c[j];

			W[col] += A->v[j] * V[ col ];
		}
	}
}

void print_matrix_abs(FILE *f, const Matrix *A)
{
	int i, j;
	for(i=0; i<block_bounds[nb_blocks]; i++)
	{
		fprintf(f, "%4d   |  ", mpi_zonestart[mpi_here]+i);

		for(j= A->r[i]; j < A->r[i+1]; j++)
			fprintf(f, " [%4d ] % 1.2e ", A->c[j], A->v[j]);

		fprintf(f, "\n");
	}
}

void print_matrix_rel(FILE* f, const Matrix *A)
{
	// hoping less than 100 items / line
	int c[100], n=0, i, j, k;
	double v[100];

	for(i=0; i<block_bounds[nb_blocks]; i++)
	{
		// check if this row is the same as previous : compare lengths, then all columns (relative to diagonal) and values
		int same = n == A->r[i+1] - A->r[i];

		for( j = A->r[i], k=0; same && j < A->r[i+1] && k < 100; j++, k++)
			same &= (A->c[j]-i == c[k] && A->v[j] == v[k]);

		// if different, print new line number and contents
		if( ! same )
		{
			fprintf(f, "%5d -- ", mpi_zonestart[mpi_here]+i);

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

	// print last line
	fprintf(f, "%5d -- end\n", i);
}

// return pos in matrix (so then you only have to take A->v[pos] ), -1 if does not exist
int find_in_matrix(const int row, const int col, const Matrix *A)
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


void read_matrix(const int n, const int m, const int nnz, const int symmetric, Matrix *A, FILE* input_file, const int offset UNUSED)
{
	// n is number of local rows, m number of cols (and number of total rows, thus m == A->n == A->m )
	int Y, prevY = -1, X, i, j, k, pos = 0, pos_subdiag = 0, row, col, *nb_subdiag = NULL;
	double val;
	Matrix subdiag;

	if( symmetric )
	{
		// for symmetric matrices; first loop, gathering elements below diagonal into subdiag, but only for the elements
		// a) with columns such that they will end up in our block of rows once transposed b) in rows before our block of rows

		allocate_matrix(m, n, nnz, &subdiag, sizeof(double));
		nb_subdiag  = (int*)calloc(n, sizeof(int));

		while( fscanf(input_file, "%d %d %lg\n", &X, &Y, &val) == 3 )
		{
			if( Y-1 >= mpi_zonestart[mpi_here] )
				break;

			// file is 1-based
			X--;
			Y--;

			if( Y > prevY )
			{
				// on new rows, update row pointer
				prevY = Y;
				subdiag.r[Y] = pos_subdiag;
			}

			// transpose X -> row -> offset row, Y -> col
			col = X - mpi_zonestart[mpi_here];

			if( col >= 0 && col < n )
			{
				subdiag.c[pos_subdiag] = col;
				subdiag.v[pos_subdiag] = val;
				pos_subdiag ++;
				nb_subdiag[col]++;
			}
		}
	}
	else
		// alternatively, on non-symmetric matrices, skip everything that is before our block of rows
		while( fscanf(input_file, "%d %d %lg\n", &X, &Y, &val) == 3 && Y-1 < mpi_zonestart[mpi_here] )
			;

	// start at our block of rows : first value already read in X, Y, val
	prevY = -1;
	do
	{
		// coordinates in file are 1-based. Shift Y to local representation in row.
		X --;
		Y --;
		row = Y - mpi_zonestart[mpi_here];

		if( row > prevY )
		{
			// on new rows, update row pointers
			prevY = row;
			A->r[row] = pos;

			if( symmetric )
			{
				pos += nb_subdiag[row];
				subdiag.r[Y] = pos_subdiag;
			}
		}

		A->c[pos] = X;
		A->v[pos] = val;
		pos++;

		// transpose X -> row -> offset row, Y -> col
		col = X - mpi_zonestart[mpi_here];

		if( symmetric && X > Y && col < n )
		{
			subdiag.c[pos_subdiag] = col;
			subdiag.v[pos_subdiag] = val;
			pos_subdiag++;
			nb_subdiag[col]++;
		}
	}
	// exit if new row is beyond the local block of rows, or at end of file
	while( fscanf(input_file, "%d %d %lg\n", &X, &Y, &val) == 3 && Y-1 < mpi_zonestart[mpi_here] + n);

	// mark the end of regions read by setting the pointer past considered rows
	A->r[row+1] = pos;
	if( symmetric )
		subdiag.r[mpi_zonestart[mpi_here]+row+1] = pos_subdiag;
	
	if( symmetric )
	{
		// now let's fill in the subdiagonal part
		int *fill_row = malloc( n * sizeof(int) );

		for( j=0; j<n; j++ )
			fill_row[j] = A->r[j];

		for(i=0; i<mpi_zonestart[mpi_here] + n; i++ )
			for( k = subdiag.r[i] ; k < subdiag.r[i+1] ; k++ )
			{
				j = subdiag.c[k];
				// now put (i,j) in (j,i)

				pos = fill_row[j];
				A->c[pos] = i;
				A->v[pos] = subdiag.v[k];

				fill_row[j]++;
			}

		free(fill_row);
		deallocate_matrix(&subdiag);
		free(nb_subdiag);
	}

	fclose(input_file);
}

// finite-difference method for a 3D Poisson's equation gives a SPD matrix with -6 on the diagonal, 
// and 1s on the diagonal+1, diagonal+p and diagonal+p²
void generate_Poisson3D(Matrix *A, const int p, const int stencil_points, const int start_row, const int n_rows)
{
	int p2 = p * p, i, j=0, pos, diag;

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


	pos = 0;
	for(j=0; j<n_rows; j++)
	{
		// row j is really row start_row+j, and though rows are locally offset, columns are numbered globally 
		// so the diagonal element at current row is [diag,diag], represented by [diag-offset = j, diag]
		A->r[j] = pos;
		diag = start_row + j;

		for(i=0; i<stencil_points; i++)
			if( diag + stenc_c[i] > 0 && diag + stenc_c[i] < A->n )
			{
				A->c[pos] = diag + stenc_c[i];
				A->v[pos] = stenc_v[i];
				pos++;
			}
	}

	// point to just beyond last element
	A->r[j] = pos;
}

void allocate_matrix(const int n, const int m, const long nnz, Matrix *A, int align_bytes)
{
	A->n = n;
	A->m = m;
	A->nnz = nnz;


	A->r = (int*)aligned_calloc(align_bytes, (n+1) * sizeof(int));

	A->c = (int*)aligned_calloc(align_bytes, nnz * sizeof(int));
	A->v = (double*)aligned_calloc(align_bytes, nnz * sizeof(double));

	if( ! A->v || ! A->c || ! A->r )
	{
		fprintf(stderr, "Allocating sparse matrix of size %dx%d and %ld non-zeros failed !\n", n, m, nnz);
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

void get_submatrix(const Matrix *A , const int *rows, const int nr, const int *cols, const int nc, const int bs, Matrix *B)
{
	// nb = Number of Blocks, bs = Block Size
	int i, ii, j, jj, k, p = 0;

	// i iterates each block of rows, ii each row in A that needs to be copied. Parallelly, k iterates each row in B corresponding to ii.
	for(i=0, k=0; i<nr; i++)
		for(ii=rows[i]; ii < rows[i] + bs && ii < mpi_zonesize[mpi_here] && k < B->n ; ii++, k++)
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


