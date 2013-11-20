#ifndef GLOBAL_H_INCLUDED
#define GLOBAL_H_INCLUDED

#define NOFAULT 0
#define SINGLEFAULT 1
#define MULTFAULTS_GLOBAL 2
#define MULTFAULTS_UNCORRELATED 3
#define MULTFAULTS_DECORRELATED 4

extern char fault_strat;
extern int BS;

// defining a few aliases for matrices and functions
#ifdef MATRIX_DENSE // using dense matrices
	#define Matrix DenseMatrix
	#define mult(A, V, W) mult_dense((DenseMatrix*)A, V, W);
	#define mult_task(A, V, W) mult_dense_task((DenseMatrix*)A, V, W);
	#define mult_task_dep(A, V, W, p) mult_dense_task((DenseMatrix*)A, V, W, p);
	#define get_rhs(n, rows, m, cols, A, b, x, rhs) get_rhs_dense(n, rows, m, cols, (DenseMatrix*)A, b, x, rhs)
	#define get_submatrix(A, rows, cols, B) submatrix_dense((DenseMatrix*)A, rows, cols, B)
	#define deallocate_matrix(A) deallocate_dense_matrix((DenseMatrix*)A)
#else // by default : using sparse matrices
	#define Matrix SparseMatrix
	#define mult(A, V, W) mult_sparse((SparseMatrix*)A, V, W);
	#define mult_task(A, V, W) mult_sparse_task((SparseMatrix*)A, V, W);
	#define mult_task_dep(A, V, W, p) mult_sparse_task_dep((SparseMatrix*)A, V, W, p);
	#define get_rhs(n, rows, m, cols, A, b, x, rhs) get_rhs_sparse(n, rows, m, cols, (SparseMatrix*)A, b, x, rhs)
	#define get_submatrix(A, rows, cols, B) submatrix_sparse_to_dense((SparseMatrix*)A, rows, cols, B)
	#define deallocate_matrix(A) deallocate_sparse_matrix((SparseMatrix*)A)
#endif 


double scalar_product( const int n, const double *v, const double *w );

void start_measure();
double stop_measure();


#endif // GLOBAL_H_INCLUDED

