import numpy as np
import pandas as pd
import sys
import myspkmeans


## TO DO ##
# kmeans++

def final():
    # check input
    def input_validation():
        try:
            if len(sys.argv) != 4:  # check if 3 or 4
                print("Invalid Input!")
                exit()  # change later######
            k = int(sys.argv[1])  # check
            if k < 0:
                print("Invalid Input!")
                exit()
            allowed = ["spk", "wam", "ddg", "lnorm", "jacobi"]
            goal = sys.argv[2]
            filename = sys.argv[3]
            if goal not in allowed:
                print("Invalid Input!")
                exit()
        except ValueError:
            print("Invalid Input!")
            exit()
        except IndexError:
            print("Invalid Input!")
            exit()
    input_validation()
    k = int(sys.argv[1])
    goal = sys.argv[2]
    filename = sys.argv[3]
    print(k, goal, filename)
    data = myspkmeans.fit(filename, goal, k, 1)
    print(k, goal, filename)
    if goal != "spk":
        exit()
    df = pd.DataFrame(data)
    dim = len(df.columns)
    k = dim
    eps = 0.00001
    numpy_data_point = df.to_numpy()
    n = len(pd_data_points)
    m = len(pd_data_points.columns)

    def k_means_pp(data_points, n, k):
        np.random.seed(0)
        first_vector_index = np.random.choice(n)
        k_vectors_indices = [first_vector_index]
        k_vectors = [numpy_data_point[first_vector_index]]
        for i in range(k-1):
            probability_chart = []
            d = []
            sum_d = 0
            for l in range(n):
                cur = numpy_data_point[l]
                min_dis = float('inf')
                for j in range(len(k_vectors)):
                    center = np.array(k_vectors[j])
                    dist = (np.linalg.norm(cur-center))**2
                    if (min_dis > dist):
                        min_dis = dist
                d.append(min_dis)
                sum_d += min_dis
            for l in range(n):
                probability_chart.append(d[l]/sum_d)
            selected_vector_index = np.random.choice(
                n, None, True, p=probability_chart)
            k_vectors_indices.append(selected_vector_index)
            # print(selected_vector_index)
            new_center = data_points[selected_vector_index]
            k_vectors.append(new_center)
        # print(*k_vectors_indices,sep=',')
        # return k_vectors

    def print_points(init_points, n, k, m, data_points):
        max_iter = 300
        eps = 0.00001
        list_of_k_init_cluster = [
            i for sublist in init_points for i in sublist]
        flat_list_of_points = [i for sublist in data_points for i in sublist]
        centeroids_out = mykmeanssp.fit(n, k, m, max_iter,
                                        eps, list_of_k_init_cluster, flat_list_of_points)[:k*m]
        centeroids_out = np.asarray(centeroids_out).round(4)
        centeroids_out = np.array_split(centeroids_out, k)
        centeroids_out = [centeroids_out[i].tolist() for i in range(k)]
        for c in centeroids_out:
            print(','.join('%.4f' % j for j in c))
    init_points = k_means_pp(numpy_data_point, n, k)
    print_points(init_points, n, k, m, numpy_data_point)


if __name__ == "__main__":
    final()
