#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <setjmp.h>
#include <errno.h>
#include <err.h>

#ifdef _OMPSS
	#include <nanos_omp.h>
#endif

#include "global.h"
#include "matrix.h"
#include "failinfo.h"
#include "debug.h"
#include "csparse.h"
#include "cg.h"

#include "recover.h"

void prepare_x_decorrelated(double *x, const int n, const int *lost_blocks, const int nb_lost)
{
	int b, i;

	for (b = 0; b < nb_lost; b++)
	{
		for (i = lost_blocks[b] * failblock_size_dbl; i < lost_blocks[b+1] * failblock_size_dbl && i < n; i++)
			x[i] = 0;
	}
}

void prepare_x_uncorrelated(double *x, const double *initial_x, const int n, const int *lost_blocks, const int nb_lost)
{
	int b, i;

	for (b = 0; b < nb_lost; b++)
	{
		for (i = lost_blocks[b] * failblock_size_dbl; i < lost_blocks[b+1] * failblock_size_dbl && i < n; i++)
			x[i] = initial_x[i];
	}
}

void recover_direct(const Matrix *A, const int sgn, const double *u, const double *v, double *w, int lost_block)
{
	// all this can be inside a task on a per-block fashion presumably, if needed
	// if A u = v - w then depending if we want v or w :
	// v = w + A u
	// w = v - A u
	// so s is +/- 1, we get w = v + sgn(A u)
	int i, lost = lost_block * failblock_size_dbl;

	if (lost > A->n)
	{
		fprintf(stderr, "Cannot interpolate since block starts at %d but matrix has size %d.\n"
						"Check whether there are %d blocks.\n", lost, A->n, lost_block);
		return ;
	}

	Matrix local;
	local.m = A->m;
	local.n = failblock_size_dbl;

	if (lost + failblock_size_dbl > A->n)
		local.n = A->n - lost;

	local.v = A->v;
	local.c = A->c;
	local.r = &(A->r[lost]);

	mult(&local, u, &(w[lost]));

	if (v != NULL && sgn < 0)
	{
		for (i = lost; i < lost+local.n; i++)
			w[i] = v[i] - w[i];
	}
	else if (v != NULL && sgn >= 0)
	{
		for (i = lost; i < lost+local.n; i++)
			w[i] = v[i] + w[i];
	}
	else if (sgn < 0)
	{
		for (i = lost; i < lost+local.n; i++)
			w[i] = - w[i];
	}
}

void recover_inverse(const Matrix *A, const double *b, const double *g, double *x, int *lost_blocks, const int nb_lost)
{
	if (nb_lost == 1)
		do_interpolation(A, b, g, x, lost_blocks, 1);

	else if (get_strategy() != MULTFAULTS_GLOBAL)
	{
		/* strategies from the paper of Langou et al. to recover x blocks independently: prepare x */
		if (get_strategy() == MULTFAULTS_UNCORRELATED)
			prepare_x_uncorrelated(x, b, A->n, lost_blocks, nb_lost);

		else
			prepare_x_decorrelated(x, A->n, lost_blocks, nb_lost);

		/* recover blocks independently */
		int i;
		for (i = 0; i < nb_lost; i++)
			do_interpolation(A, b, g, x, lost_blocks + i, 1);
	}
	else // nb_lost > 1 && strategy == MULTFAULTS_GLOBAL
	{
		/* cluster failed blocks depending on neighbouring blocks, to recover them jointly */
		int id, i, j, set[nb_lost];

		for (id = 0; id < nb_lost; id++)
		{
			if (lost_blocks[id] < 0)
				continue;

			/* get in 'set' the block lost_blocks[id] and all its neighbours. */
			int set_size = get_failed_neighbourset(lost_blocks + id, nb_lost - id, set, nb_lost);

			/* Remove from lost_blocks all blocks that are in the neighbour set of lost_blocks[id].
			 * Remember both lost_blocks and set are sorted, and we've done all lost_blocks with i < id. */
			lost_blocks[id] = -1;
			for (i = id + 1, j = 0; i < nb_lost; i++)
			{
				if (lost_blocks[i] < 0)
					continue;

				for (; j < set_size; j++)
				{
					if (set[j] == lost_blocks[i])
						lost_blocks[i] = -1;
					else if (set[j] > lost_blocks[i])
						break;
				}
			}

			log_err(SHOW_FAILINFO, "MULTFAULTS_GLOBAL clusetered recovery with %d/%d errors\n", set_size, nb_lost);

			do_interpolation(A, b, g, x, set, set_size);
		}
	}
}

void do_interpolation(const Matrix *A, const double *b, const double *g, double *x, const int *lost_blocks, const int nb_lost)
{
	int i, total_lost = nb_lost * failblock_size_dbl, lost[nb_lost];

	// change from block number to first row in block number
	for (i = 0; i < nb_lost; i++)
		lost[i] = lost_blocks[i] * failblock_size_dbl;

	if (lost[nb_lost -1] > A->n)
	{
		fprintf(stderr, "Cannot interpolate since block starts at %d but matrix has size %d.\n"
						"Check whether there are %d blocks.\n", lost[nb_lost-1], A->n, lost_blocks[nb_lost-1]);
		return ;
	}

	if (lost[nb_lost -1] + failblock_size_dbl > A->n)
		total_lost -= (lost[nb_lost -1] + failblock_size_dbl - A->n);

	Matrix recup;
	double *rhs = (double*)aligned_calloc(failblock_size_bytes, total_lost * sizeof(double));

	int nnz = 0;
	for (i = 0; i < nb_lost; i++)
	{
		int max = lost[i] + failblock_size_dbl;
		if (max > A->n)
			max = A->n ;
		nnz += A->r[max] - A->r[lost[i]];
	}

	allocate_matrix(total_lost, total_lost, nnz, &recup, failblock_size_bytes);

	// get the submatrix for those lines
	get_submatrix(A, lost, nb_lost, lost, nb_lost, failblock_size_dbl, &recup);

	// fill in the rhs with the part we need
	get_rhs(nb_lost, lost, nb_lost, lost, failblock_size_dbl, A, b, g, x, rhs);

	// from csparse
	cs *submatrix = cs_calloc(1, sizeof(cs));
	if (submatrix == NULL)
		err(1, "Failed to allocate submatrix for inversion");
	submatrix->m = recup.m;
	submatrix->n = recup.n;
	submatrix->nzmax = recup.nnz;
	submatrix->nz = -1;
	// don't work with triplets, so has to be compressed column even though we work with compressed row
	// but since here the matrix is symmetric they are interchangeable
	submatrix->p = recup.r;
	submatrix->i = recup.c;
	submatrix->x = recup.v;

	cs_cholsol(submatrix, rhs, 0);

	// and update the x values we interpolated, that are returned in rhs
	int j, k;
	for (i = 0, k = 0; i < nb_lost; i++)
		for (j = lost[i]; j < lost[i] + failblock_size_dbl && j < A->n; j++, k++)
			x[j] = rhs[k];

	cs_free(submatrix);

	deallocate_matrix(&recup);
	free(rhs);
}

// give rows to but in rhs, and cols to avoid
void get_rhs(const int n, const int *rows, const int m, const int *except_cols, const int bs, const Matrix *A, const double *b, const double *g, const double *x, double *rhs)
{
	int i, ii, j, jj, k;

	if (b != NULL)
		for (i = 0, k = 0; i < n; i++)
		{
			for (ii = rows[i]; ii < rows[i] + bs && ii<A->n; ii++, k++)
				rhs[k] = b[ii] ;
		}

	if (g != NULL)
		for (i = 0, k = 0; i < n; i++)
		{
			for (ii = rows[i]; ii < rows[i] + bs && ii<A->n; ii++, k++)
				rhs[k] -= g[ii] ;
		}

	for (i = 0, k = 0; i < n; i++)
		for (ii = rows[i]; ii < rows[i] + bs && ii<A->n; ii++, k++)
		{
			// for each lost line ii, start with b_ii
			// and remove contributions A_ii,j * x_j
			// from all rows j that are not lost
			for (j = A->r[ii], jj = 0; j<A->r[ii+1]; j++)
			{
				// update jj so that except_cols[jj] + bs > A->c[j]
				while(jj < m && except_cols[jj] + bs <= A->c[j])
					jj++;


				// if the column of item j is not in the [except_cols[jj],except_cols[jj]+bs-1] set
				if (jj >= m || A->c[j] < except_cols[jj])
					rhs[k] -= A->v[j] * x[A->c[j]];
			}
		}
}

// remember definitions of recover_direct/inverse
// recover_inverse(A,b,g,x,..) : recovering x using b - g = A * x(g may be NULL then b = A * x, e.g. use for Ap = A * p)
// recover_direct(A,sgn,u,v,w,..) : w = v + sgn *(A u)

// these functions return 0 if all is good -1 for impossible(so didn't try) and > 0 for number(at least) of blocks still failed

int recover_g_recompute(magic_pointers *mp, double *g, int block)
{
	// g = b -1 *(A * x)
	// safe to use for old_g or new_g after new_g = b - A new_x(so that we are sure that the whole new_x is finished)
	double *x = mp->x;
	int r = -1;

	if (!has_skipped_blocks(MASK_ITERATE))
	{
		recover_direct(mp->A, -1, x, mp->b, g, block);
		r = check_recovery_errors();
	}

	log_err(SHOW_FAILINFO, "\tRecomputing block %d of g : %d\n", block, r);

	return r;
}

int recover_Ap(magic_pointers *mp, double *Ap, const double *p, int block)
{
	// Ap = 0 + 1 *(A * p)
	int r = -1;

	if (!has_skipped_blocks(1 << get_data_vectptr(p)))
	{
		recover_direct(mp->A, 1, p, NULL, Ap, block);
		r = check_recovery_errors();
	}

	log_err(SHOW_FAILINFO, "\tRecomputing block %d of Ap : %d\n", block, r);

	return r;
}

int recover_Ax(magic_pointers *mp, double *Ax, int block)
{
	// Ax = 0 +1 *(A * x)
	double *x = mp->x;
	int r = -1;

	if (!has_skipped_blocks(MASK_ITERATE))
	{
		recover_direct(mp->A, 1, x, NULL, Ax, block);

		r = check_recovery_errors();
	}

	log_err(SHOW_FAILINFO, "\tRecomputing block %d of Ax : %d\n", block, r);

	return r;
}

int recover_g_update(magic_pointers *mp, double *g, int block)
{
	// g = g - alpha * Ap
	// use when previous is impossible
	double *Ap = mp->Ap;
	int r = -1;

	if (!is_skipped_block(block, MASK_A_P))
	{
		int recover_size = failblock_size_dbl, blockpos = block * failblock_size_dbl;
		if (blockpos + recover_size > mp->A->n)
			recover_size = mp->A->n - blockpos;

		daxpy(recover_size, -(*mp->alpha), Ap + blockpos, g + blockpos, g + blockpos);

		r = check_recovery_errors();
	}

	log_err(SHOW_FAILINFO, "\tRepeating update for block %d of g : %d\n", block, r);

	return r;
}

int recover_p_repeat(magic_pointers *mp, double *p, const double *old_p, int block)
{
	// p = beta * g + p
	double *g = mp->g;
	int r = -1;

	if (!is_skipped_block(block, 1 << get_data_vectptr(old_p)) && !is_skipped_block(block, MASK_GRADIENT))
	{
		int recover_size = failblock_size_dbl, blockpos = block * failblock_size_dbl;
		if (blockpos + recover_size > mp->A->n)
			recover_size = mp->A->n - blockpos;

		daxpy(recover_size, *(mp->beta), old_p + blockpos, g + blockpos, p + blockpos);

		r = check_recovery_errors();
	}

	log_err(SHOW_FAILINFO, "\tRepeating update for block %d of p[%d] %d with beta %e\n", block, get_data_vectptr(p), r, *(mp->beta));

	return r;
}

int recover_x_lossy(magic_pointers *mp, double *x)
{
	// yay, can't fail ! A and b from "safe backup store"
	int *lost_blocks, nb_lost = get_all_failed_blocks(MASK_ITERATE, &lost_blocks);

	recover_inverse(mp->A, mp->b, NULL, x, lost_blocks, nb_lost);

	if (nb_lost)
		free(lost_blocks);

	clear_failed(MASK_ITERATE);

	#if VERBOSE >= SHOW_FAILINFO
	int i;char str[60 + 6 * nb_lost];
	sprintf(str, "\tInterpolating %d blocks of x lossy-style : %d  {", nb_lost, 0);
	for (i = 0; i < nb_lost-1; i++) sprintf(str + strlen(str), "%d, ", lost_blocks[i]);
	log_err(SHOW_FAILINFO, "%s%d}\n", str, lost_blocks[i]);
	#endif

	return 0;
}

int recover_full_xk(magic_pointers *mp, double *x, const int mark_clean)
{
	// x = A^-1(b - g - sum(x))
	double *g = mp->g;
	int r = -1, nb_lost = 0, *lost_blocks;

	if (!overlapping_faults(MASK_GRADIENT, MASK_ITERATE))
	{
		nb_lost = get_all_failed_blocks(MASK_ITERATE, &lost_blocks);

		recover_inverse(mp->A, mp->b, g, x, lost_blocks, nb_lost);

		r = check_recovery_errors();

		if (!r && mark_clean)
		{
			int i;
			for (i = 0; i < nb_lost; i++)
				mark_corrected(lost_blocks[i], MASK_ITERATE);
		}

		if (nb_lost)
			free(lost_blocks);
	}

	#if VERBOSE >= SHOW_FAILINFO
	int i;char str[50 + 6 * nb_lost];
	sprintf(str, "\tInterpolating %d blocks of xk : %d  {", nb_lost, r);
	for (i = 0; i < nb_lost; i++) sprintf(str+strlen(str),  "%d, ", lost_blocks[i]);
	str[strlen(str) - 2] = '}';str[strlen(str) - 1] = '\n';
	log_err(SHOW_FAILINFO, str);
	#endif

	return r;
}

int recover_full_p_invert(magic_pointers *mp, double *p, const int mark_clean)
{
	// p = A^-1(Ap - sum(A * p))
	double *Ap = mp->Ap;
	int i, r = -1, nb_lost = 0, *lost_blocks, mask = 1 << get_data_vectptr(p);

	if (!overlapping_faults(MASK_A_P, mask))
	{
		nb_lost = get_all_failed_blocks(mask, &lost_blocks);

		recover_inverse(mp->A, Ap, NULL, p, lost_blocks, nb_lost);

		r = check_recovery_errors();

		if (!r && mark_clean)
		{
			for (i = 0; i < nb_lost; i++)
				mark_corrected(lost_blocks[i], mask);
		}

		if (nb_lost)
			free(lost_blocks);
	}

	#if VERBOSE >= SHOW_FAILINFO
	char str[50 + 6 * nb_lost];
	sprintf(str, "\tInterpolating %d blocks of p[%d] : %d  {", nb_lost, get_data_vectptr(p), r);
	for (i = 0; i < nb_lost-1; i++) sprintf(str+strlen(str), "%d, ", lost_blocks[i]);
	log_err(SHOW_FAILINFO, "%s%d}\n", str, lost_blocks[i]);
	#endif

	return r;
}


int recover_full_old_p_invert(magic_pointers *mp, double *old_p, const int mark_clean)
{
	// old_p = A^-1(old_Ap - sum(A * old_p)) ... we(normally) saved old_Ap into old_p where it was failed
	// and marked the first item of the block with NAN if it was impossible
	double *old_Ap = old_p;
	int i, r = 0, nb_lost = 0, mask = 1 << get_data_vectptr(old_p), *lost_blocks;

	nb_lost = get_all_failed_blocks_vect(old_p, &lost_blocks);

	for (i = 0; i < nb_lost; i++)
		if (old_Ap[lost_blocks[i] * failblock_size_dbl] == NAN)
		{
			r = -1;
			break;
		}

	if (r == 0)
	{
		recover_inverse(mp->A, old_Ap, NULL, old_p, lost_blocks, nb_lost);

		r = check_recovery_errors();

		if (!r && mark_clean)
		{
			for (i = 0; i < nb_lost; i++)
				mark_corrected(lost_blocks[i], mask);
		}
	}

	#if VERBOSE >= SHOW_FAILINFO
	char str[50 + 6 * nb_lost];
	sprintf(str, "\tInterpolating %d blocks of old_p[%d] : %d  {", nb_lost, get_data_vectptr(old_p), r);
	for (i = 0; i < nb_lost; i++) sprintf(str+strlen(str),  "%d, ", lost_blocks[i]);
	str[strlen(str) - 2] = '}';str[strlen(str) - 1] = '\n';
	log_err(SHOW_FAILINFO, str);
	#endif

	return r;
}

void save_oldAp_for_old_p_recovery(magic_pointers *mp, double *old_p, const int s, const int e)
{
	const int mask = 1 << get_data_vectptr(old_p);
	int start_block, end_block, i;

	#if VERBOSE >= SHOW_FAILINFO
	char str[500];
	sprintf(str, "\tSaving failed blocks of old_Ap for later recovery of old_p[%d] : {  ", get_data_vectptr(old_p));
	#endif

	for (i = s / failblock_size_dbl; i < (e + failblock_size_dbl - 1) / failblock_size_dbl; i++)
	{
		if (!is_skipped_block(i, mask))
			continue;

		// compute task-local part of this fail block and copy that
		start_block = i * failblock_size_dbl;
		end_block = (i + 1) * failblock_size_dbl;

		if (start_block < s)
			start_block = s;
		if (end_block > e)
			end_block = e;

		if (!is_skipped_block(i, MASK_A_P))
		{
			memcpy(&(old_p[start_block]), &(mp->Ap[start_block]), (end_block - start_block) * sizeof(double));

			#if VERBOSE >= SHOW_FAILINFO
			sprintf(str+strlen(str),  "%d, ", i);
			#endif
		}
		// unrecoverable. Should we set to nan ?
		else
			old_p[start_block] = NAN;
	}

	#if VERBOSE >= SHOW_FAILINFO
	str[strlen(str) - 2] = '}';str[strlen(str) - 1] = '\n';
	log_err(SHOW_FAILINFO, str);
	#endif
}

int recover_full_p_repeat(magic_pointers *mp, double *p, const double *old_p, const int mark_clean)
{
	// p = beta * old_p + g
	int i, r = 0, rr;
	const int mask = 1 << get_data_vectptr(p);

	for (i = 0; i < nb_failblocks; i++)
	{
		if (!is_skipped_block(i, mask))
			continue;

		rr = recover_p_repeat(mp, p, old_p, i);

		if (mark_clean && !rr)
			mark_corrected(i, mask);

		r += abs(rr);
	}

	return r;
}

int recover_full_g_recompute(magic_pointers *mp, double *g, const int mark_clean)
{
	int i, r = 0, rr;

	for (i = 0; i < nb_failblocks; i++)
	{
		if (!is_skipped_block(i, MASK_GRADIENT))
			continue;

		rr = recover_g_recompute(mp, g, i);

		if (mark_clean && !rr)
			mark_corrected(i, MASK_GRADIENT);

		r += abs(rr);
	}

	return r;
}

int recover_full_g_update(magic_pointers *mp, double *g, const int mark_clean)
{
	int i, r = 0, rr;

	for (i = 0; i < nb_failblocks; i++)
	{
		if (!is_skipped_block(i, MASK_GRADIENT))
			continue;

		rr = recover_g_update(mp, g, i);

		if (mark_clean && !rr)
			mark_corrected(i, MASK_GRADIENT);

		r += abs(rr);
	}

	return r;
}

int recover_mvm_skips_g(magic_pointers *mp, double *g, const int mark_clean)
{
	int i, r = 0, rr;

	for (i = 0; i < nb_failblocks; i++)
		if (is_skipped_not_failed_block(i, MASK_GRADIENT))
		{

			rr = recover_g_update(mp, g, i);

			if (mark_clean && !rr)
				mark_corrected(i, MASK_GRADIENT);

			r += abs(rr);
		}

	return r;
}

int recover_full_Ap(magic_pointers *mp, double *Ap, const double *p, const int mark_clean)
{
	int i, r = 0, rr;

	for (i = 0; i < nb_failblocks; i++)
	{
		if (!is_skipped_block(i, MASK_A_P))
			continue;

		rr = recover_Ap(mp, Ap, p, i);

		if (mark_clean && !rr)
			mark_corrected(i, MASK_A_P);

		r += abs(rr);
	}

	return r;
}

