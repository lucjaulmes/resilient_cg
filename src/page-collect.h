#ifndef PAGECOLLECT_H_INCLUDED
#define PAGECOLLECT_H_INCLUDED

#define FILENAMELEN         256
#define LINELEN             256

#define PROC_DIR_NAME       "/proc"
#define MAPS_NAME           "maps"

void mempage_file(char *m_name);
int retrieve_vm_ranges(intptr_t *start, intptr_t *end, int *range_end, int *total_pages, intptr_t *free_pass, int nb_free);

#endif // PAGECOLLECT_H_INCLUDED

