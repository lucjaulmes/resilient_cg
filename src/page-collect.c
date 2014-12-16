//
// slightly forked page-collect.c from PageMapTools, in order to get just
// one PID's virtual memory adresses (instead of scanning all + hw adresses)
// see http://www.eqware.net/Articles/CapturingProcessMemoryUsageUnderLinux/
// original copyright notice follows :
//

/* page-collect.c -- collect a snapshot each of of the /proc/pid/maps files, 
 *      with each VM region interleaved with a list of physical addresses 
 *      which make up the virtual region.
 * Copyright C2009 by EQware Engineering, Inc.
 *
 *    page-collect.c is part of PageMapTools.
 *
 *    PageMapTools is free software: you can redistribute it and/or modify
 *    it under the terms of version 3 of the GNU General Public License
 *    as published by the Free Software Foundation
 *
 *    PageMapTools is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with PageMapTools.  If not, see http://www.gnu.org/licenses. 
 */
#define _LARGEFILE64_SOURCE

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <errno.h>
#include <string.h>
#include <inttypes.h> // gives SCNuPTR

#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <ctype.h>

#define FILENAMELEN         256
#define LINELEN             256

#define PROC_DIR_NAME       "/proc"
#define MAPS_NAME           "maps"


static void init_static(char *m_name, int *page_size)
{
	int pid = getpid();
	*page_size = sysconf(_SC_PAGESIZE);

	char dirname[FILENAMELEN];
	sprintf(dirname, "%s/%d", PROC_DIR_NAME, pid);
	sprintf(m_name, "%s/%d/%s", PROC_DIR_NAME, pid, MAPS_NAME);
	
    struct stat buf;

    int n = stat(dirname, &buf);
    int ok = (n == 0 && (buf.st_mode & S_IFDIR) != 0);

	if( !ok || access((m_name), R_OK) != 0 )
	{
		fprintf(stderr, "ERROR %d is not a valid pid right now !\n", pid);
		// No point in going on. Sigh. Harakiri.
		exit(EXIT_FAILURE);
	}
}

// get virtual memory ranges : for each range i put the start and end adresses in start[i],end[i] 
// and the number of pages in the ranges k=0..i (i included) in range_end[i]
int retrieve_vm_ranges(intptr_t *start, intptr_t *end, int *range_end, int *total_pages, intptr_t *free_pass, int nb_free)
{
	static int page_size = 0;
	static char m_name[FILENAMELEN] = "";

	if( page_size == 0 )
		init_static(m_name, &page_size);

	int nb_range = 0, nb_vulnerable = 0;
	FILE *m      = NULL;
	*total_pages = 0;

	char line[LINELEN];

	m = fopen(m_name, "r");
	if (m == NULL)
	{
		fprintf(stderr, "Unable to open \"%s\" for reading (errno=%d).\n", m_name, errno);
		return -1;
	}

	// get lines of map file 
	while(fgets(line, LINELEN, m) != NULL)
	{
		unsigned long s, e, inode;
		char read, write, exec, shared;
		int range_vulnerable;

		// each line starts with a range of virtual adresses. Ignore offset in mapped file, get the rest.
		int dev_maj, dev_min, n = sscanf(line, "%lX-%lX %c%c%c%c %*X %d:%d %lX", 
				&s, &e, &read, &write, &exec, &shared, &dev_maj, &dev_min, &inode);

		if (n != 9)
		{
			fprintf(stderr, "Invalid line read from \"%s\": %s\n", m_name, line);
			continue;
		}

		if(inode != 0 && !(dev_maj == 0 && dev_min == 4))
			// file mapped, and not from /dev/zero : vulnerable iff( write != '-' )
			range_vulnerable = (write == 'w');
		else
		{
			// anon (private|shared) : always vulnerable
			range_vulnerable = 1;

			// BUT let's remove the pages to which we give a "free pass"
			// NB. to help identify, they are r--p, which should be rather rare
			// Also, they have been allocated with boundary alignment on pages, so checking vma_start is enough
			if( read == 'r' && write == '-' && exec == '-' && shared == 'p' )
			{	
				int i;
				for(i=0; i<nb_free; i++)
					if( free_pass[i] == (intptr_t)s )
					{
						range_vulnerable = 0;
						break;
					}
			}
		}

		// each non-zero range, keep it, count pages
		int num_pages = (e - s) / page_size;

		if( num_pages <= 0 )
			continue;

		(*total_pages) += num_pages;

		if (range_vulnerable)
		{
			nb_vulnerable		+= num_pages;
			start[nb_range] 	 = s;
			end[nb_range]		 = e;
			range_end[nb_range]	 = nb_vulnerable;
			nb_range++;
		}
skip:
	}

	fclose(m);

	return nb_range;
}


