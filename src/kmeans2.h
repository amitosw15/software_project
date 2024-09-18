#ifndef UNTITLED10_KMEANS2_H
#define UNTITLED10_KMEANS2_H

#include <Python.h>
/* static PyObject *fit(PyObject *, PyObject *) */
double MAX(double arg1, double arg2)
{
    if (arg1 > arg2)
    {
        return arg1;
    }
    return arg2;
}
double cal_dis(int dim, double *vector1, double *vector2)
{
    /*calc dis between two vectors*/
    int i;
    double sum = 0;
    for (i = 0; i < dim; i++)
    {
        sum = sum + pow((vector1[i] - vector2[i]), 2);
    }
    sum = pow(sum, 0.5);
    return sum;
}

int find_closets_vector(int dim, int k, double *vector, double **cluster)
{
    /* returns index of the closest vector to *vector in **cluster*/
    double min_dis = cal_dis(dim, vector, cluster[0]);
    int closest = 0;
    int i;
    for (i = 0; i < k; i++)
    {
        double cur_dis = cal_dis(dim, vector, cluster[i]);
        if (cur_dis < min_dis)
        {
            min_dis = cur_dis;
            closest = i;
        }
    }
    return closest;
}

/* static PyMethodDef capiMethods[] = {
    {"fit", (PyCFunction)fit, METH_VARARGS, PyDoc_STR("function for some function")},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    NULL,
    -1,
    capiMethods};

PyMODINIT_FUNC
PyInit_mykmeanssp(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m)
    {
        return NULL;
    }
    return m;
}
 */
static PyObject *k_means2(int N, int k, int dim, PyObject *data_points_py, PyObject *k_cluster_py)
{
    int i, j, *amount_each_index, count_iter = 0, max_iter = 300;
    PyObject *item;
    double eps, **pointer_to_points, **centers, **pointer_to_cluster, *points, *k_cluster, *pointer_to_center, max_change;
    pointer_to_points = (double **)calloc(N + 1, sizeof(double *));
    points = (double *)calloc(N * dim + 1, sizeof(double));
    eps = 0.00001;
    if (points == NULL || pointer_to_points == NULL)
    {
        printf("An Error Has Occurred\n");
        exit(1);
    }
    for (i = 0; i < N; i++)
    {
        pointer_to_points[i] = points + i * dim;
    }
    for (i = 0; i < N * dim; i++)
    {
        item = PyList_GetItem(data_points_py, i);
        points[i] = PyFloat_AsDouble(item);
    }
    k_cluster = (double *)calloc(k * dim + 1, sizeof(double));
    pointer_to_cluster = (double **)calloc(k + 1, sizeof(double *));
    if (k_cluster == NULL || pointer_to_cluster == NULL)
    {
        printf("An Error Has Occurred\n");
        exit(1);
    }
    for (i = 0; i < k; i++)
    {
        pointer_to_cluster[i] = k_cluster + i * dim;
    }
    for (i = 0; i < k * dim; i++)
    { /*k cluster containing first k vectors of the file*/
        item = PyList_GetItem(k_cluster_py, i);
        k_cluster[i] = PyFloat_AsDouble(item);
    }
    amount_each_index = (int *)calloc(k, sizeof(int));
    pointer_to_center = (double *)calloc(k * dim, sizeof(double));
    centers = (double **)calloc(k, sizeof(double *));
    if (amount_each_index == NULL || pointer_to_center == NULL || centers == NULL)
    {
        printf("An Error Has Occurred\n");
        exit(1);
    }
    for (i = 0; i < k; i++)
    {
        centers[i] = pointer_to_center + i * dim;
    }
    for (i = 0; i < k; i++)
    {
        memset(centers[i], 0, dim); /*centers init to zero*/
        amount_each_index[i] = 0;
    }
    while (count_iter < max_iter)
    { /*calc the clusters within max iter attemps*/
        count_iter += 1;
        for (i = 0; i < N; i++)
        { /* calc for each vector the closest vector the closest cluster availiable*/
            int closest = find_closets_vector(dim, k, pointer_to_points[i], pointer_to_cluster);
            amount_each_index[closest] += 1;
            for (j = 0; j < dim; j++)
            {
                centers[closest][j] += pointer_to_points[i][j];
            }
        }
        max_change = -1;
        for (i = 0; i < k; i++)
        { /* find the center of each cluster*/
            for (j = 0; j < dim; j++)
            {
                centers[i][j] /= amount_each_index[i];
            }
        }
        for (i = 0; i < k; i++)
        {
            /*calc the change of each center,
             then assign the new center and
              reset the old them for the next iter */
            max_change = MAX(max_change, cal_dis(dim, pointer_to_cluster[i], centers[i]));
            for (j = 0; j < dim; j++)
            {
                pointer_to_cluster[i][j] = centers[i][j];
                centers[i][j] = 0;
            }
            amount_each_index[i] = 0;
        }
        if (max_change < eps && max_change > -1)
        {
            break;
        }
    }
    PyObject *result = PyList_New(dim * k);
    for (i = 0; i < k * dim; i++)
    {
        PyList_Insert(result, i, PyFloat_FromDouble(k_cluster[i]));
    }
    free(pointer_to_center);
    free(k_cluster);
    free(points);
    free(centers);
    free(pointer_to_cluster);
    free(pointer_to_points);
    return result;
}

/* static PyObject *fit2(PyObject *self, PyObject *args)
{
    int N, k, dim;
    PyObject *data_points_py, *k_cluster_py;
    if (!PyArg_ParseTuple(args, "iiiOO:fit", &N, &k, &dim, &k_cluster_py, &data_points_py))
    {
        return NULL;
    }
    return Py_BuildValue("O", k_means(N, k, dim, data_points_py, k_cluster_py));
} */


#endif
