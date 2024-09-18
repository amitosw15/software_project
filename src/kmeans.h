#ifndef UNTITLED10_KMEANS_H
#define UNTITLED10_KMEANS_H

double cal_dis(int dim, double *vector1, double *vector2);
int find_closets_vector(int dim, int k, double *vector, double **cluster);

/* double cal_dis(int dim, double *vector1, double *vector2)
{
    int i;
    double sum = 0;
    for (i = 0; i < dim; i++)
    {
        sum = sum + pow((vector1[i] - vector2[i]), 2);
    }
    sum = pow(sum, 0.5);
    return sum;
}
 */
/* int find_closets_vector(int dim, int k, double *vector, double **cluster)
{
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
 */

#endif /*UNTITLED10_SPKMEANS_H*/