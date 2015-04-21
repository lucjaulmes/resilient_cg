
#if CKPT == CKPT_TO_DISK
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>
#endif

void force_checkpoint(checkpoint_data *ckpt_data, double *iterate, double *gradient, double *p, double *Ap)
{
	int *behaviour = &(ckpt_data->instructions);
	double *old_err_sq = mp.old_err_sq, *alpha = mp.alpha;

	#pragma omp task out(*behaviour) inout(*ckpt_data, *alpha, *old_err_sq) label(force_checkpoint) no_copy_deps
	{
		*ckpt_data->save_err_sq = *old_err_sq;
		*ckpt_data->save_alpha = *alpha;
		*behaviour = SAVE_CHECKPOINT;
		log_err(SHOW_DBGINFO, "FORCING TO SAVE CHECKPOINT\n");
	}

	checkpoint_vectors(ckpt_data, behaviour, iterate, gradient, p, Ap);
}

void due_checkpoint(checkpoint_data *ckpt_data, double *iterate, double *gradient, double *p, double *Ap)
{
	int *behaviour = &(ckpt_data->instructions);
	double *old_err_sq = mp.old_err_sq, *alpha = mp.alpha;

	// reuse pragma check
	#pragma omp task out(*behaviour) inout(*ckpt_data, *old_err_sq) label(due_checkpoint) no_copy_deps
	{
		if(!aggregate_skips())
		{
			log_err(SHOW_DBGINFO, "SAVING CHECKPOINT\n");

			*behaviour = SAVE_CHECKPOINT;
			*ckpt_data->save_err_sq = *old_err_sq;
			*ckpt_data->save_alpha = *alpha;
		}
		else
		{
			log_err(SHOW_DBGINFO, "LOADING CHECKPOINT\n");

			*behaviour = RELOAD_CHECKPOINT;
			*old_err_sq = *ckpt_data->save_err_sq;
			*alpha = *ckpt_data->save_alpha;
		}
	}

	checkpoint_vectors(ckpt_data, behaviour, iterate, gradient, p, Ap);
}

void force_rollback(checkpoint_data *ckpt_data, double *iterate, double *gradient, double *p, double *Ap)
{
	int *behaviour = &(ckpt_data->instructions);
	double *old_err_sq = mp.old_err_sq, *alpha = mp.alpha;

	#pragma omp task out(*behaviour) inout(*ckpt_data, *alpha, *old_err_sq) label(force_rollback) no_copy_deps
	{
		*old_err_sq = *ckpt_data->save_err_sq;
		*alpha = *ckpt_data->save_alpha;
		*behaviour = RELOAD_CHECKPOINT;
		log_err(SHOW_DBGINFO, "FORCED ROLLBACK\n");
	}

	checkpoint_vectors(ckpt_data, behaviour, iterate, gradient, p, Ap);
}

void checkpoint_vectors(checkpoint_data *ckpt_data, int *behaviour, double *iterate, double *gradient, double *p, double *Ap UNUSED)
{
	int i;
	for(i=0; i < nb_blocks; i ++)
	{
		int s = get_block_start(i), e = get_block_end(i);

		#pragma omp task in(*behaviour) inout(iterate[s:e-1], gradient[s:e-1], p[s:e-1], Ap[s:e-1]) firstprivate(i, s, e) label(checkpoint_vectors) priority(100) no_copy_deps
		{
		#if CKPT == CKPT_IN_MEMORY
			if(*behaviour == SAVE_CHECKPOINT)
			{
				memcpy(ckpt_data->save_x+s,  iterate+s,  (e-s) * sizeof(double));
				memcpy(ckpt_data->save_g+s,  gradient+s, (e-s) * sizeof(double));
				memcpy(ckpt_data->save_p+s,  p+s,        (e-s) * sizeof(double));
				memcpy(ckpt_data->save_Ap+s, Ap+s,       (e-s) * sizeof(double));
			}
			else if(*behaviour != DO_NOTHING)
			{
				memcpy(iterate+s,  ckpt_data->save_x+s,  (e-s) * sizeof(double));
				memcpy(gradient+s, ckpt_data->save_g+s,  (e-s) * sizeof(double));

				if(*behaviour == RELOAD_CHECKPOINT)
				{
					// not restarting, just going back to last checkpoint
					memcpy(p+s,    ckpt_data->save_p+s,  (e-s) * sizeof(double));
					memcpy(Ap+s,   ckpt_data->save_Ap+s, (e-s) * sizeof(double));
				}

				clear_failed_blocks(~0, s, e);
			}
		#elif CKPT == CKPT_TO_DISK
			char path[250];
			int ckpt_fd;
			sprintf(path, "%s%d", ckpt_data->checkpoint_path, i);

			if(*behaviour == SAVE_CHECKPOINT)
			{
				ckpt_fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
				log_err(SHOW_FAILINFO, "Open file %s to checkpoint : %d\n", path, ckpt_fd);

				if(ckpt_fd < 0)
				{
					fprintf(stderr, "ERROR Unable to open file %s to write checkpoint.\n", path);
					perror("open() error message is ");
				}


				write(ckpt_fd,  iterate+s,  (e-s) * sizeof(double));
				write(ckpt_fd,  gradient+s, (e-s) * sizeof(double));
				write(ckpt_fd,  p+s,        (e-s) * sizeof(double));
				write(ckpt_fd, Ap+s,        (e-s) * sizeof(double));

				fsync(ckpt_fd);
				close(ckpt_fd);
			}
			else if(*behaviour != DO_NOTHING)
			{
				ckpt_fd = open(path, O_RDONLY);
				log_err(SHOW_FAILINFO, "Open file %s to rollback : %d\n", path, ckpt_fd);

				if(ckpt_fd < 0)
				{
					*(mp.err_sq) = 0.0; // fail
					fprintf(stderr, "ERROR No checkpoint file %s or unable to open : error %d. Exiting.\n", path, errno);
					perror("open() error message is ");
					return;
				}

				read(ckpt_fd,  iterate+s,  (e-s) * sizeof(double));
				read(ckpt_fd, gradient+s,  (e-s) * sizeof(double));

				if(*behaviour == RELOAD_CHECKPOINT)
				{
					// not restarting, just going back to last checkpoint
					read(ckpt_fd,    p+s,  (e-s) * sizeof(double));
					read(ckpt_fd,   Ap+s,  (e-s) * sizeof(double));
				}
				close(ckpt_fd);

				clear_failed_blocks(~0, s, e);
			}
		#endif
		}
	}
}

