#ifndef PAGECOLLECT_H_INCLUDED
#define PAGECOLLECT_H_INCLUDED

int retrieve_vm_ranges(intptr_t *start, intptr_t *end, int *range_end, int *total_pages, intptr_t *free_pass, int nb_free);

#endif // PAGECOLLECT_H_INCLUDED

