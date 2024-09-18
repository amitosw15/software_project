#define PY_SSIZE_T_CLEAN
/* #include <Python.h>
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
/* #include "kmeans.h"*/
#include "kmeans2.h"
#include "spkmeans.h"

/* TO DO
1. Graph Representation
2. The Weighted Adjacency Matrix
3. The Diagonal Degree Matrix
4. The Normalized Graph Laplacian
5. Finding Eigenvalues and Eigenvectors
6. Jacobi algorithm
7. The Eigengap Heuristic
8,    #include <python3.7/Python.h>

 */

void print_matrix(double **matrix, int n, int m)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < m; j++)
        {
            if (j != m - 1)
            {
                printf("%f , ", matrix[i][j]);
            }
            else
            {
                printf("%f \n", matrix[i][j]);
            }
        }
    }
    return;
}

int validate_goal(char *goal)
{
    /* if valid return 1 */
    int valid, i;
    char *goals[] = {"wam", "ddg", "lnorm", "jacobi"};
    valid = 0;
    for (i = 0; i < 4; i++)
    {
        if (0 == strcmp(goal, goals[i]))
        {
            valid = 1;
        }
    }
    return valid;
}

int numOfRowInFile(FILE *ifp, int *longesst_line)
{
    char c;
    int count = 0;
    int max = 0;
    int count_len = 0;
    if (ifp != NULL)
    {
        while ((c = fgetc(ifp)) != EOF)
        {
            count_len += 1;
            if (c == '\n')
            {
                if (count_len > max)
                {
                    max = count_len;
                }
                count += 1;
                count_len = 0;
            }
        }
        *longesst_line = max;
    }
    fseek(ifp, 0, 0);
    return count;
}

int dimensionOfVector(FILE *ifp)
{
    int count = 0;
    char c;
    while ((c = fgetc(ifp)) != EOF)
    {
        if (c == '\n')
        {
            break;
        }
        if (c == ',')
        {
            count += 1;
        }
    }
    fseek(ifp, 0, 0);
    return count + 1;
}

int valid_number(char *s)
{ /* return 1 if input is a valid integer */
    while (*s != '\0')
    {
        if (*s < '0' || *s > '9')
            return 0;
        s++;
    }
    return 1;
}

double **get_mat_from_file(FILE *fp, int N, int dim)
{
    double n1;
    char c;
    int i, j;
    double **data_points;
    double *block;

    j = 0;
    block = calloc(N * dim, sizeof(double));
    assert_double_arr(block);
    
    data_points = calloc(N, sizeof(double *));
    assert_double_mat(data_points);
    for (i = 0; i < N; i++)
    {
        data_points[i] = block + i * dim;
    }
    i = 0;
    while (fscanf(fp, "%lf%c", &n1, &c) == 2)
    {
        data_points[i][j] = n1;
        j++;
        if (c == '\n')
        {
            i++;
            j = 0;
        }
    }
    /*data_points[i][j] = n1; */
    fclose(fp);
    /*printf("\n finish get_mat_from_file\n");*/
    return data_points;
}

double **get_all_point(FILE *ifp, int dim, int rows, int longest)
{
    int line_count;
    int dim_count, i;
    double *cur_point;
    char *cur_line;
    char *s_cur_point;
    double **matrix = (double **)calloc(rows, sizeof(double *));
    assert_double_mat(matrix);
    for (i = 0; i < rows; i++)
    {
        matrix[i] = (double *)calloc(dim, sizeof(double));
        assert_double_arr(matrix[i]);
    }
    cur_point = (double *)calloc(1, sizeof(double));
    assert_double_arr(cur_point);
    cur_line = (char *)calloc(longest + 1, sizeof(char));
    assert_char_arr(cur_line);
    s_cur_point = (char *)calloc(30, sizeof(char));
    assert_char_arr(s_cur_point);
    if (cur_point == NULL || cur_line == NULL || s_cur_point == NULL)
    {
        printf("An Error Has Occurred");
        exit(1);
    }
    line_count = 0;
    dim_count = 0;
    while (line_count < rows)
    {
        if (fscanf(ifp, "%s", cur_line) == 0)
        {
            break; /* cur_line contain cur vector*/
        }
        s_cur_point = strtok(cur_line, ",");
        while (dim_count != dim)
        {
            *cur_point = atof(s_cur_point);
            matrix[line_count][dim_count] = *cur_point;
            s_cur_point = strtok(NULL, ",");
            dim_count += 1;
        }
        dim_count = 0;
        line_count += 1;
    }
    free(cur_line);
    free(cur_point);
    free(s_cur_point);
    return matrix;
}

void print_row_vector(double *a, int n)
{
    int i;
    for (i = 0; i < n - 1; i++)
    {
        printf("%f,", a[i]);
    }
    printf("%f\n", a[i]);
}

double norm(double *a, double *b, int dim)
{
  
    int i;
    double res,sum = 0;
    for (i = 0; i < dim; i++)
    {
        sum += pow(a[i] - b[i], 2);
    }
    res = sqrt(sum);
    /*printf("res=%lf\n",res);*/
    return res;
}


double **wam(double **matrix, int n, int dim)
{
    int i, j;
    /*printf("n size id: %d",n);*/
    double **adjMatrix = (double **)calloc(n, sizeof(double *));
    assert_double_mat(adjMatrix);
    /*printf("\n finish 1 assert");*/
    for (i = 0; i < n; i++)
    {
        adjMatrix[i] = (double *)calloc(n, sizeof(double));
        assert_double_arr(adjMatrix[i]);
    }
    /*printf("\nfinish assert\n");*/
    j=0;
    for (i = 0; i < n; i++)
    {
        /*printf("enter for, i:%d\n",i);*/
        while (j < n)

        {
            if (i != j)
            {
                adjMatrix[i][j] = exp(-(norm(matrix[i], matrix[j], dim) / 2));
                adjMatrix[j][i] = adjMatrix[i][j];
                /*printf("\n adj entry %f",adjMatrix[i][j]);*/
            }
            j++;
        }
        j = i + 1;
    }
    /*print_matrix(adjMatrix,n,n);*/
    return adjMatrix;
}

double **ddm(double **matrix, int n)
{

    int i, z;
    double sum;
    double **diagMatrix = (double **)calloc(n, sizeof(double *));
    assert_double_mat(diagMatrix);
    for (i = 0; i < n; i++)
    {
        diagMatrix[i] = (double *)calloc(n, sizeof(double));
        assert_double_arr(diagMatrix[i]);
    }
    for (i = 0; i < n; i++)
    {
        sum = 0;
        for (z = 0; z < n; z++)
        {
            sum += matrix[i][z];
        }
        diagMatrix[i][i] = sum;
    }
    return diagMatrix;
}

void dsquare(double **matrix, int n)
{
    int i;
    for (i = 0; i < n; i++)
    {
        matrix[i][i] = (1 / (pow(matrix[i][i], 0.5)));
    }
}

double **lnorm(double **d, double **w, int n)
{
    /*
    input: matrix d- from ddm (and dsquare), matrix w- wam, n: size of the input and ouput matrixes
    returns a n*n matrix equals to I-d*w*d

     */
    int i, j;
    double **lapli = (double **)calloc(n, sizeof(double *));
    assert_double_mat(lapli);
    for (i = 0; i < n; i++)
    {
        lapli[i] = (double *)calloc(n, sizeof(double));
        assert_double_arr(lapli[i]);
    }
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            lapli[i][j] = d[i][i] * w[i][j];
        }
    }
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            lapli[i][j] = d[j][j] * lapli[i][j];
        }
    }
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (i == j)
            {
                lapli[i][j] = 1 - lapli[i][j];
            }
            else
            {
                lapli[i][j] = (0 - (lapli[i][j]));
            }
        }
    }
    return lapli;
}

int is_diagnol(double **matrix, int n)

{
    /* return 0 if thr matrix isnt diag */
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (i != j && matrix[i][j] != 0)
            {
                return 0;
            }
        }
    }
    return 1;
}

void a_to_a_tag(double **A, double c, int n, double s, int i, int j)
{ /*  to be completed
     printf("before\n");
     print_matrix(A,n,n);
     printf("\n"); */
    int k;
    double **a_tag = (double **)calloc(n, sizeof(double *));
    assert_double_mat(a_tag);
    for (k = 0; k < n; k++)
    {
        a_tag[k] = (double *)calloc(n, sizeof(double));
        assert_double_arr(a_tag[k]);
    }
    a_tag[i][i] = ((pow(c, 2) * (A[i][i])) - 2 * s * c * A[i][j] + pow(s, 2) * A[j][j]);
    a_tag[j][j] = ((pow(s, 2) * (A[i][i])) + 2 * s * c * A[i][j] + (pow(c, 2) * A[j][j]));
    for (k = 0; k < n; k++)
    {
        if (k != i && k != j)
        {
            a_tag[k][i] = c * A[k][i] - s * A[k][j];
            a_tag[k][j] = c * A[k][j] + s * A[k][i];
            a_tag[i][k] = a_tag[k][i];
            a_tag[j][k] = a_tag[k][j];
        }
    }
    for (k = 0; k < n; k++)
    {
        A[i][k] = a_tag[i][k];
        A[k][i] = a_tag[k][i];
        A[j][k] = a_tag[j][k];
        A[k][j] = a_tag[k][j];
    }
    A[i][j] = 0.0;
    A[j][i] = 0.0;
    free_matrix(a_tag, n);
    // printf("after\n");
    // print_matrix(A,n,n);
}

int *largestOffDiag(double **A, int n)
{
    /* return [i,j] of largest elem off the diag in the matrix */
    int i, j;
    double val;
    int *max_i_j = (int *)calloc(2, sizeof(int));
    assert_int_arr(max_i_j);
    // check if n is bigger than 1
    val = fabs(A[0][1]);
    max_i_j[0] = 0;
    max_i_j[1] = 1;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (i != j && fabs(A[i][j]) > val)
            {
                val = fabs(A[i][j]);
                max_i_j[0] = i;
                max_i_j[1] = j;
            }
        }
    }
    return max_i_j;
}

int sign(double x)
{
    if (x < 0)
    {
        return -1;
    }
    else
    {
        return 1;
    }
}

void v_mul_p(double **V, double s, double c, int n, int i, int j)
{
    int k;
    double vi, vj;

    for (k = 0; k < n; k++)
    {
        vi = V[k][i];
        vj = V[k][j];
        V[k][i] = (c * vi) - (s * vj);
        V[k][j] = (s * vi) + (c * vj);
    }
    /* for (i=0;i<n;i++){
        mat[i]=(double*)calloc(n,sizeof(double));
        assert(mat[i]);
    }
    for (i=0;i<n;i++){
        for (j=0;j<n;j++){
            mat[i][j]=0;
            for (k=0;k<n;k++){
                mat[i][j]+=(V[i][k]*p[k][j]);
            }
        }
    }
    for (i=0;i<n;i++){
        for (j=0;j<n;j++){
            V[i][j]=mat[i][j];
        }
    }*/

/*      free_matrix(mat,n);
 */}

void rotationP(double **matrix, int n, double **V)
{

    int i, j;
    double teta, t, c, s;
    int *i_j_max;
    i_j_max = largestOffDiag(matrix, n);
    i = i_j_max[0];
    j = i_j_max[1];
    free(i_j_max);
    teta = (matrix[j][j] - matrix[i][i]) / (2 * matrix[i][j]);
    t = sign(teta) / (fabs(teta) + pow(pow(teta, 2) + 1, 0.5));
    c = 1 / (pow(pow(t, 2) + 1, 0.5));
    s = t * c;
    v_mul_p(V, s, c, n, i, j);
    a_to_a_tag(matrix, c, n, s, i, j);
}
double **idmatrix(int n)
{
    int i;
    double **V = (double **)calloc(n, sizeof(double *));
    assert_double_mat(V);
    for (i = 0; i < n; i++)
    {
        V[i] = (double *)calloc(n, sizeof(double));
        assert_double_arr(V[i]);
        V[i][i] = 1;
    }
    return V;
}

double convergence(double **matrix, int n)
{
    int i, j;
    double sum = 0;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (i != j)
            {
                sum += pow(matrix[i][j], 2);
            }
        }
    }
    return sum;
}

double **jacob(double **matrix, int n)
{
    double conA, conA_tag;
    double eps = 0.00001;
    double diff = 1;
    double **V = idmatrix(n);
    int count = 0;
    while (0 == is_diagnol(matrix, n) && (diff > eps) && count < 100)
    {
        count++;
        conA = convergence(matrix, n);
        printf("in jacob %d\n", count);
        rotationP(matrix, n, V); /*  should change this function such that it wouldnt return any value and free the memory */
        conA_tag = convergence(matrix, n);
        diff = conA - conA_tag;
    }
    printf("chachaboomboom");
    return V;
}


int compare_eigan(const void *p1, const void *p2)
{
    /* compare two object of eigaenval- first by value then bu index (decreasing order) */
    eigenval *q1 = (eigenval*)p1, *q2 = (eigenval*)p2;
    if (q1->val==q2->val){
        return (q1->idx)-(q2->val);
    }
    if (q1->val>q2->val){
        return 1;
    }
    return -1;

/*     printf("\n q1 val is=%f, q2 val is=%f\n",q1->val,q2->val);
 */    
}

void printeigen(eigenval *sorted, int n){
    int i;
    for (i=0;i<n;i++){
        eigenval cur=sorted[i];
        printf("eigen num %d val is %f the vector is:\n",cur.idx,cur.val);
        print_vector(cur.vector,n);

    }
}
eigenval *doubleToEigen(double **jacobi, int n, double *eiganvals) /*  might be wrong function */
{
    /* from a double array to eiganvector with the same vals in the correspondening positions */
    int i;
    // printf("**** double to eigen *****");
    eigenval *eigen_arr = (eigenval *)calloc(n, sizeof(eigenval));
    assert_eigen_arr(eigen_arr);
    // print_vector(eiganvals,n);
    for (i = 0; i < n; i++)
    {
        eigenval item;
        item.val = eiganvals[i];
        item.idx = i;
        item.vector = get_i_col(jacobi, n, i);
        eigen_arr[i]=item;
        /* printf("double to eigen iteration idx: %d, val: %f, vector:\n",item.idx,item.val); */
        /*print_vector(item.vector,n);*/

    }
    // printf("**** double to eigen *****");
    return eigen_arr;
}
double *getDiag(double **matrix, int n)
{
    int i;
    double *diag = (double *)calloc(n, sizeof(double));
    assert_double_arr(diag);
    for (i = 0; i < n; i++)
    {
        diag[i] = matrix[i][i];
    }
    return diag;
}

void printeigen(eigenval *sorted, int n){
    int i;
    for (i=0;i<n;i++){
        eigenval cur=sorted[i];
        printf("eigen num %d val is %f the vector is:\n",cur.idx,cur.val);
        print_vector(cur.vector,n);

    }
}

void print_vector(double *mat, int n)
{
    int i;
    for (i = 0; i < n - 1; i++)
    {
        printf("%f,", mat[i]);
    }
    printf("%f\n", mat[n - 1]);
}

int eigengap(eigenval *sortedEigen, int n)
{
    /* the Eigangap heuristic*/
    int k, i;
    double delta;
    double max_gap = -1;
    eigenval e1,e2;
    delta = 0;
    k = 0;
    for (i = 0; i < n / 2; i++)
    {
        e1=sortedEigen[i];
        e2=sortedEigen[i+1];
        delta = fabs(e1.val-e2.val);
        if (delta > max_gap)
        {
            k = i;
            max_gap = delta;
        }
    }
    k += 1;
    return k;
}
void free_matrix(double **matrix, int n)
{
    int i;
    for (i = 0; i < n; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
    return;
}

double *get_i_col(double **V, int n, int i)
{
    int k;
    double *vector = (double *)calloc(n, sizeof(double));
    assert_double_arr(vector);
    for (k = 0; k < n; k++)
    {
        vector[k] = V[k][i];
    }
    return vector;
}

PyObject *spkmeans(char *filename, char *goal, int k, int source) /*  should change */
{
    int dim, n, longest;
    double **wamm, **damm, **lnormi, **jacobi, *eigenvalues, **U;
    eigenval *sortedEigen;
    PyObject *res;

    FILE *fp;
    fp = fopen(filename, "r");
    assert_fp(fp);
    dim = dimensionOfVector(fp);
    n = numOfRowInFile(fp, &longest);
    double **points = get_all_point(fp, dim, n, longest);
    res = PyLong_FromLong(-1);
    fclose(fp);
    if (0 == strcmp(goal, "wam"))
    {
        wamm = wam(points, n, dim);
        print_matrix(wamm, n, n);
        free_matrix(wamm, n);
    }
    else if (0 == strcmp(goal, "ddg"))
    {
        wamm = wam(points, n, dim);
        damm = ddm(wamm, n);
        print_matrix(damm, n, n);
        free_matrix(wamm, n);
        free_matrix(damm, n);
    }

    else if (0 == strcmp(goal, "lnorm"))
    {
        wamm = wam(points, n, dim);
        damm = ddm(wamm, n);
        dsquare(damm, n);
        lnormi = lnorm(damm, wamm, n);
        print_matrix(lnormi, n, n);
        free_matrix(wamm, n);
        free_matrix(damm, n);
        free_matrix(lnormi, n);
    }
    else if (0 == strcmp(goal, "jacobi"))
    {
        jacobi = jacob(points, n);
        eigenvalues = getDiag(points, n);
        print_vector(eigenvalues, n);
        print_matrix(jacobi, n, n);
        free_matrix(jacobi, n);
        free(eigenvalues);
    }

    else if (0 == strcmp(goal, "spk"))
    {
        printf("\npoints:\n");
        print_matrix(points,n,n);
        wamm = wam(points, n, dim);
        damm = ddm(wamm, n);
        dsquare(damm, n);
        lnormi = lnorm(damm, wamm, n);
        printf("\nlnormi:\n");
        print_matrix(lnormi,n,n);
        jacobi = jacob(lnormi, n);
        eigenvalues = getDiag(lnormi, n);
        sortedEigen = doubleToEigen(jacobi, n, eigenvalues);
        qsort(sortedEigen, n, sizeof(eigenval), compare_eigan);
        if (k == 0)
        {
            k = eigengap(sortedEigen, n);
        }
        U = make_me_a_k_matrix(n, k, sortedEigen);
        norm_the_matrix(U, n, k);
        res = make_me_a_python_matrix(U, n, k); /*converts U to Pyobject*/
        free_matrix(U, n);
        free_matrix(points, n); ///
        free_matrix(jacobi, n);
        free_matrix(wamm, n);
        free_matrix(damm, n);
        free_matrix(lnormi, n);
        free(eigenvalues);
        free(sortedEigen); /* check */
        return res;        // back to python
    }
    free_matrix(points, n);
    return res;
}
double **make_me_a_k_matrix(int n, int k, eigenval *sortedEigan)
{
    /* returns a n*k matrix with first k columns of a given matrix*/
    int i, j;
    double **U = (double **)calloc(n, sizeof(double *));
    assert_double_mat(U);
    for (i = 0; i < n; i++)
    {
        U[i] = (double *)calloc(k, sizeof(double));
        assert(U[i]);
    }

    for (i = 0; i < k; i++)
    {
        
        double *eigen_vector = sortedEigan[i].vector;
        for (j = 0; j < n; j++)
        {
            U[j][i] = eigen_vector[j];/*check*/
        }
    }
    return U;
}
PyObject *make_me_a_python_matrix(double **mat, int n, int dim)
{
    Py_ssize_t i, j, rows = n, columns = dim;
    PyObject *res = PyList_New(n);

    for (i = 0; i < rows; i++)
    {
        PyObject *item = PyList_New(dim);
        for (j = 0; j < columns; j++)
            PyList_SET_ITEM(item, j, PyFloat_FromDouble(mat[i][j]));
        PyList_SET_ITEM(res, i, item);
    }
    return res;
}
void norm_the_matrix(double **matrix, int n, int k)
{
    /* normalize the matrix in size of n rows and k columns */
    int i, j;
    double sum;
    for (i = 0; i < k; i++)
    {
        sum = 0;
        for (j = 0; j < n; j++)
        {
            sum += (matrix[j][i] * matrix[j][i]);
        }
        sum = pow(sum, 0.5);
        for (j = 0; j < n; j++)
        {
            matrix[j][i] = matrix[j][i] / sum;
        }
    }
}
PyObject *kmeans2_py(int n, int k, int dim, PyObject *points_py, PyObject *points_to_cluster_py)
{
    return k_means2(n, k, dim, points_py, points_to_cluster_py);
}
int main(int varc, char *argv[])
{
    char *filename;
    char *goal = argv[1];

    if (varc != 3)
    {
        printf("Invalid Input!"); /*  check */
    }
    if (validate_goal(goal) == 0)
    {
        printf("Invalid Input!"); /*  check */
    }
    /* make check for opening file */
    filename = argv[2];
    spkmeans(filename, goal, 0, 1); /* last arg: 1- from c ; 0-python */
    return 0;                       /*  check */
}

void assert_int_arr(const int *arr)
{
    if (arr == NULL)
    {
        printf("An Error Has Occured");
        exit(0);
    }
}

void assert_double_mat(double **mat)
{
    if (mat == NULL)
    {
        printf("An Error Has Occured");
        exit(0);
    }
}

void assert_double_arr(double *arr)
{
    if (arr == NULL)
    {
        printf("An Error Has Occured");
        exit(0);
    }
}

void assert_char_arr(char *arr)
{
    if (arr == NULL)
    {
        printf("An Error Has Occured");
        exit(0);
    }
}
void assert_eigen_arr(eigenval *arr)
{
    if (arr == NULL)
    {
        printf("An Error Has Occured");
        exit(0);
    }
}

void assert_fp(FILE *fp)
{
    if (fp == NULL)
    {
        printf("An Error Has Occured");
        exit(0);
    }
}