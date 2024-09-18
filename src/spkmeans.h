#ifndef UNTITLED10_SPKMEANS_H
#define UNTITLED10_SPKMEANS_H

#define PY_SSIZE_T_CLEAN

#include <Python.h>

typedef struct
{
    double val;
    int idx;
    double *vector;
} eigenval;

void print_matrix(double **matrix, int n, int m);
PyObject *make_me_a_python_matrix(double **mat, int n, int m);
double **make_me_a_k_matrix(int n, int k, eigenval *sortedEigan);
void norm_the_matrix(double **matrix, int n, int k);
PyObject *spkmeans(char *filename, char *goal, int k, int source); /*  */
void free_matrix(double **matrix, int n);
int eigengap(eigenval *sortedEigen, int n);
double *getDiag(double **matrix, int n);
eigenval *doubleToEigen(double **arr, int n, double *eiganvals);
int compare_eigan(const void *p1, const void *p2);
double **jacob(double **matrix, int n);     /*  */
double convergence(double **matrix, int n); /*  */
double **idmatrix(int n);
void rotationP(double **matrix, int n, double **V);                /*  */
void v_mul_p(double **V, double s, double c, int n, int i, int j); /*  */
int sign(double x);
int *largestOffDiag(double **A, int n);
int is_diagnol(double **matrix, int n);
void a_to_a_tag(double **A, double c, int n, double s, int i, int j); /*  */
double **lnorm(double **d, double **w, int n);
void dsquare(double **matrix, int n);
double **ddm(double **matrix, int n);          /*  */
double **wam(double **matrix, int n, int dim); /*  */
double norm(double *a, double *b, int dim);    /*  */
void print_vector(double *a, int n);
double **get_all_point(FILE *ifp, int dim, int rows, int longest);
int valid_number(char *s);
int dimensionOfVector(FILE *ifp);
int numOfRowInFile(FILE *ifp, int *longesst_line);
int validate_goal(char *goal);
void print_matrix(double **matrix, int n, int m);
double *get_i_col(double **V, int n, int i);
PyObject *make_me_a_python_matrix(double **mat, int n, int dim);
PyObject *kmeans2_py(int n, int k, int dim, PyObject *points_py, PyObject *points_to_cluster_py);
void assert_int_arr(const int *arr);
void assert_double_mat(double **mat);
void assert_eigen_arr(eigenval *arr);
void assert_fp(FILE *fp);
void assert_double_arr(double *arr);
void assert_char_arr(char *arr);

#endif /*UNTITLED10_SPKMEANS_H*/