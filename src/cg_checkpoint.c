
#if CKPT == CKPT_TO_DISK
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>
#endif

void force_checkpoint(const int n, detect_error_data *err_data, double *iterate, double *gradient, double *p, double *Ap)
{
	int *behaviour = &(err_data->error_detected), *prev_error = &(err_data->prev_error);

	#if SDC == SDC_ORTHO
		double *err_sq = mp.err_sq;
		#define PRAGMA_CHECKPOINT STRINGIFY(omp task out(*behaviour) inout(*err_data, *err_sq) label(force_checkpoint) no_copy_deps)
	#elif SDC == SDC_GRADIENT
		double *old_err_sq = mp.old_err_sq;
		#define PRAGMA_CHECKPOINT STRINGIFY(omp task out(*behaviour) inout(*err_data, *old_err_sq) label(force_checkpoint) no_copy_deps)
	#else //if SDC == SDC_ALPHA || (!SDC && DUE == ROLLBACK)
		double *old_err_sq = mp.old_err_sq, *alpha = mp.alpha;
		#define PRAGMA_CHECKPOINT STRINGIFY(omp task out(*behaviour) inout(*err_data, *alpha, *old_err_sq) label(force_checkpoint) no_copy_deps)
	#endif

	_Pragma( PRAGMA_CHECKPOINT )
    #undef PRAGMA_CHECKPOINT
	{
		#if SDC == SDC_ORTHO
		*err_data->save_err_sq = *err_sq;
		#elif SDC == SDC_GRADIENT
		*err_data->save_err_sq = *old_err_sq;
		#else //if SDC == SDC_ALPHA
		*err_data->save_err_sq = *old_err_sq;
		*err_data->save_alpha = *alpha;
		#endif
		*behaviour = SAVE_CHECKPOINT;
		log_err(SHOW_DBGINFO, "FORCING TO SAVE CHECKPOINT\n");

		*prev_error = 0;
	}

	checkpoint_vectors(n, err_data, behaviour, iterate, gradient, p, Ap);
}

void due_checkpoint(const int n, detect_error_data *err_data, double *iterate, double *gradient, double *p, double *Ap)
{
	int *behaviour = &(err_data->error_detected);

	double *old_err_sq = mp.old_err_sq, *alpha = mp.alpha;

	// reuse pragma check
	#pragma omp task out(*behaviour) inout(*err_data, *old_err_sq) label(due_checkpoint) no_copy_deps
	{
		if( !aggregate_skips() )
		{
			log_err(SHOW_DBGINFO, "SAVING CHECKPOINT\n");

			*behaviour = SAVE_CHECKPOINT;
			*err_data->save_err_sq = *old_err_sq;
			*err_data->save_alpha = *alpha;
		}
		//else if( *prev_error ) -- we don't care about this in due_checkpoint : no sdc, thus no prev_error
		else //if( *behaviour == RELOAD_CHECKPOINT )
		{
			log_err(SHOW_DBGINFO, "LOADING CHECKPOINT\n");

			*behaviour = RELOAD_CHECKPOINT;
			*old_err_sq = *err_data->save_err_sq;
			*alpha = *err_data->save_alpha;
		}
	}

	checkpoint_vectors(n, err_data, behaviour, iterate, gradient, p, Ap);
}

void force_rollback(const int n, detect_error_data *err_data, double *iterate, double *gradient, double *p, double *Ap)
{
	int *behaviour = &(err_data->error_detected);
	#if SDC
	int *prev_error = &(err_data->prev_error);
	#endif

	#if SDC == SDC_ORTHO
		double *err_sq = mp.err_sq;
		#define PRAGMA_CHECKPOINT STRINGIFY(omp task out(*behaviour) inout(*err_data, *err_sq) label(force_rollback) no_copy_deps)
	#elif SDC == SDC_GRADIENT
		double *old_err_sq = mp.old_err_sq;
		#define PRAGMA_CHECKPOINT STRINGIFY(omp task out(*behaviour) inout(*err_data, *old_err_sq) label(force_rollback) no_copy_deps)
	#else //if SDC == SDC_ALPHA || (!SDC && DUE == ROLLBACK)
		double *old_err_sq = mp.old_err_sq, *alpha = mp.alpha;
		#define PRAGMA_CHECKPOINT STRINGIFY(omp task out(*behaviour) inout(*err_data, *alpha, *old_err_sq) label(force_rollback) no_copy_deps)
	#endif

	_Pragma( PRAGMA_CHECKPOINT )
    #undef PRAGMA_CHECKPOINT
	{
		#if SDC == SDC_ORTHO
		*err_sq = *err_data->save_err_sq;
		#endif

		#if SDC
		// if twice we're stuck : restart
		if( *prev_error )
		{
			#if SDC == SDC_GRADIENT
			*old_err_sq = INFINITY;
			#elif SDC == SDC_ORTHO
			#else //if SDC == SDC_ALPHA
			*old_err_sq = INFINITY;
			*alpha = 0.0;
			#endif
			*behaviour = RESTART_CHECKPOINT;
			log_err(SHOW_DBGINFO, "FORCED ROLLBACK + RESTART\n");
		}
		else
		#endif
		{
			#if SDC == SDC_GRADIENT
			*old_err_sq = *err_data->save_err_sq;
			#elif SDC == SDC_ORTHO
			#else //if SDC == SDC_ALPHA || (!SDC && DUE == ROLLBACK)
			*old_err_sq = *err_data->save_err_sq;
			*alpha = *err_data->save_alpha;
			#endif
			*behaviour = RELOAD_CHECKPOINT;
			log_err(SHOW_DBGINFO, "FORCED ROLLBACK\n");
		}

		#if SDC
		*prev_error = !*prev_error;
		#endif
	}

	checkpoint_vectors(n, err_data, behaviour, iterate, gradient, p, Ap);
}

void checkpoint_vectors(const int n, detect_error_data *err_data, int *behaviour, double *iterate, double *gradient, double *p, double *Ap UNUSED)
{
	int i;
	for(i=0; i < nb_blocks; i ++ )
	{
		int s = get_block_start(i), e = get_block_end(i);
		if( e > n )
			e = n;

		#if SDC == SDC_ORTHO
			#define PRAGMA_CKPT_VECT STRINGIFY(omp task in(*behaviour) inout(iterate[s:e-1], gradient[s:e-1], p[s:e-1]) firstprivate(i, s, e) label(checkpoint_vectors) priority(100) no_copy_deps)
		#else
			#define PRAGMA_CKPT_VECT STRINGIFY(omp task in(*behaviour) inout(iterate[s:e-1], gradient[s:e-1], p[s:e-1], Ap[s:e-1]) firstprivate(i, s, e) label(checkpoint_vectors) priority(100) no_copy_deps)
		#endif

		_Pragma( PRAGMA_CKPT_VECT )
		{
		#if CKPT == CKPT_IN_MEMORY
			if( *behaviour == SAVE_CHECKPOINT )
			{
				memcpy(err_data->save_x+s,  iterate+s,  (e-s) * sizeof(double));
				memcpy(err_data->save_g+s,  gradient+s, (e-s) * sizeof(double));
				memcpy(err_data->save_p+s,  p+s,        (e-s) * sizeof(double));
				#if SDC != SDC_ORTHO
				memcpy(err_data->save_Ap+s, Ap+s,       (e-s) * sizeof(double));
				#endif
			}
			else if( *behaviour != DO_NOTHING )
			{
				memcpy(iterate+s,  err_data->save_x+s,  (e-s) * sizeof(double));
				memcpy(gradient+s, err_data->save_g+s,  (e-s) * sizeof(double));

				if( *behaviour == RELOAD_CHECKPOINT )
				{
					// not restarting, just going back to last checkpoint
					memcpy(p+s,    err_data->save_p+s,  (e-s) * sizeof(double));
					#if SDC != SDC_ORTHO
					memcpy(Ap+s,   err_data->save_Ap+s, (e-s) * sizeof(double));
					#endif
				}
				#if SDC == SDC_ORTHO
				else // RESTART_CHECKPOINT
				{
					memcpy(p+s,    err_data->save_g+s,  (e-s) * sizeof(double));
					// if possible we should cancel update_it and update_g here
					// or cancel everything before beta and replace with a recompute version ?
				}
				#endif

				clear_failed_blocks(~0, s, e);
			}
		#elif CKPT == CKPT_TO_DISK
			char path[250];
			int ckpt_fd;
			sprintf(path, "%s%d", err_data->checkpoint_path, i);

			if( *behaviour == SAVE_CHECKPOINT )
			{
				ckpt_fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
				log_err(SHOW_FAILINFO, "Open file %s to checkpoint : %d\n", path, ckpt_fd);

				if( ckpt_fd < 0 )
				{
					fprintf(stderr, "ERROR Unable to open file %s to write checkpoint.\n", path);
					perror("open() error message is ");
				}


				write(ckpt_fd,  iterate+s,  (e-s) * sizeof(double));
				write(ckpt_fd,  gradient+s, (e-s) * sizeof(double));
				write(ckpt_fd,  p+s,        (e-s) * sizeof(double));
				#if SDC != SDC_ORTHO
				write(ckpt_fd, Ap+s,        (e-s) * sizeof(double));
				#endif

				fsync(ckpt_fd);
				close(ckpt_fd);
			}
			else if( *behaviour != DO_NOTHING )
			{
				ckpt_fd = open(path, O_RDONLY);
				log_err(SHOW_FAILINFO, "Open file %s to rollback : %d\n", path, ckpt_fd);

				if( ckpt_fd < 0 )
				{
					*(mp.err_sq) = 0.0; // fail
					fprintf(stderr, "ERROR No checkpoint file %s or unable to open : error %d. Exiting.\n", path, errno);
					perror("open() error message is ");
					return;
				}

				read(ckpt_fd,  iterate+s,  (e-s) * sizeof(double));
				read(ckpt_fd, gradient+s,  (e-s) * sizeof(double));

				if( *behaviour == RELOAD_CHECKPOINT )
				{
					// not restarting, just going back to last checkpoint
					read(ckpt_fd,    p+s,  (e-s) * sizeof(double));
					#if SDC != SDC_ORTHO
					read(ckpt_fd,   Ap+s,  (e-s) * sizeof(double));
					#endif
				}
				#if SDC == SDC_ORTHO
				else // RESTART_CHECKPOINT
				{
					memcpy(p+s,    gradient+s,  (e-s) * sizeof(double));
					// if possible we should cancel update_it and update_g here
					// or cancel everything before beta and replace with a recompute version ?
				}
				#endif
				close(ckpt_fd);

				clear_failed_blocks(~0, s, e);
			}
		#endif
		}
	}
}

