import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans

retained_set = set()
discard_set_dict = dict()
compression_set_dict = dict()
threshold_dis = 0


def preprocess_data(input_file_path):
    """
    load data from text file
    :param input_file_path: path of input file
    :return: numpy array of float type
    """
    with open(input_file_path, "r") as input_file:
        input_data = np.array(input_file.readlines())

    data = np.array([np.array(item.strip('\n').split(',')) for item in input_data])
    return data.astype(np.float)


def get_clusters_dict(labels_):
    """
    create mapping of cluster id and point index in features
    :param labels_: cluster labels returned by kmeans
    :return: dict of cluster_id and point index in features
    """
    clusters = defaultdict(list)
    for i, cluster_id in enumerate(labels_):
        clusters[cluster_id].append(i)
    return clusters


def get_retained_set(clusters):
    """
    create retained set
    :param clusters: dict
    :return: retained set of point indices
    """
    rs = set()
    for point_indices in clusters.values():
        if len(point_indices) == 1:
            rs.add(point_indices[0])
    return rs


def create_discard_set_dict(clusters, features):
    """
    create dict of discard set statistics
    :param clusters:
    :param features:
    :return: discard set
    """
    for cluster_id, point_indices in clusters.items():
        num_of_points = len(point_indices)
        sum_of_coords = np.sum(features[point_indices, 2:], axis=0)
        sum_sq_of_coords = np.sum(np.square(features[point_indices, 2:]), axis=0)
        centroid = sum_of_coords / num_of_points
        standard_deviation = np.sqrt(np.subtract(sum_sq_of_coords / num_of_points, np.square(centroid)))
        points = np.asarray(features[point_indices, 0], dtype=int).tolist()

        discard_set_dict[cluster_id] = []
        discard_set_dict[cluster_id].append(num_of_points)
        discard_set_dict[cluster_id].append(sum_of_coords)
        discard_set_dict[cluster_id].append(sum_sq_of_coords)
        discard_set_dict[cluster_id].append(centroid)
        discard_set_dict[cluster_id].append(standard_deviation)
        discard_set_dict[cluster_id].append(points)
    # print('discard_set_dict: ', discard_set_dict)


def create_compression_set_dict(clusters, features):
    """
    create dict of compression set statistics
    :param clusters:
    :param features:
    :return: compression set
    """
    for cluster_id, point_indices in clusters.items():
        if len(point_indices) <= 1:
            continue
        num_of_points = len(point_indices)
        sum_of_coords = np.sum(features[point_indices, 2:], axis=0)
        sum_sq_of_coords = np.sum(np.square(features[point_indices, 2:]), axis=0)
        centroid = sum_of_coords / num_of_points
        standard_deviation = np.sqrt(np.subtract(sum_sq_of_coords / num_of_points, np.square(centroid)))
        points = np.asarray(features[point_indices, 0], dtype=int).tolist()

        compression_set_dict[cluster_id] = []
        compression_set_dict[cluster_id].append(num_of_points)
        compression_set_dict[cluster_id].append(sum_of_coords)
        compression_set_dict[cluster_id].append(sum_sq_of_coords)
        compression_set_dict[cluster_id].append(centroid)
        compression_set_dict[cluster_id].append(standard_deviation)
        compression_set_dict[cluster_id].append(points)
    # print('compression_set_dict: ', compression_set_dict)


def get_mahalanobis_distance(point_features, set_dict):
    """
    Calculate minimum mahalanobis distance
    :param point_features:
    :param set_dict:
    :return: mahalanobis distance
    """
    mahalanobis_distance = float('inf')
    md_cluster_id = -1

    for cluster_id, stats in set_dict.items():
        centroid = stats[3]
        standard_deviation = stats[4]
        y_square = np.square(np.divide(np.subtract(point_features, centroid), standard_deviation))
        md = np.sqrt(np.sum(y_square, axis=0))
        if md < mahalanobis_distance:
            mahalanobis_distance = md
            md_cluster_id = cluster_id

    return mahalanobis_distance, md_cluster_id


def update_set_dicts(set_dict, point_tl_features, cluster_id):
    """
    update dicts
    :param set_dict:
    :param point_tl_features:
    :param cluster_id:
    :return:
    """
    stats = set_dict[cluster_id]
    num_of_points = stats[0] + 1
    sum_of_coords = np.add(stats[1], point_tl_features[2:])
    sum_sq_of_coords = np.add(stats[2], np.square(point_tl_features[2:]))
    centroid = sum_of_coords / num_of_points
    standard_deviation = np.sqrt(np.subtract(sum_sq_of_coords / num_of_points, np.square(centroid)))
    points = stats[5]
    points.append(int(point_tl_features[0]))

    set_dict[cluster_id][0] = num_of_points
    set_dict[cluster_id][1] = sum_of_coords
    set_dict[cluster_id][2] = sum_sq_of_coords
    set_dict[cluster_id][3] = centroid
    set_dict[cluster_id][4] = standard_deviation
    set_dict[cluster_id][5] = points
    # print('set_dict[', cluster_id, ']: ', set_dict[cluster_id][5])


def find_merge_clusters(dict_1, dict_2):
    """
    Find which clusters to merge
    :param dict_1:
    :param dict_2:
    :return: dict{cluster_id: closest cluster_id, ..}
    """
    close_clusters_dict = dict()
    for cluster_id_1, stats_1 in dict_1.items():
        mahalanobis_distance = threshold_dis
        md_cluster_id = -1
        for cluster_id_2, stats_2 in dict_2.items():
            if cluster_id_1 == cluster_id_2:
                continue
            centroid_1 = stats_1[3]
            standard_deviation_1 = stats_1[4]
            centroid_2 = stats_2[3]
            standard_deviation_2 = stats_2[4]
            num_1 = np.subtract(centroid_1, centroid_2)
            num_2 = np.subtract(centroid_2, centroid_1)
            y_square_1 = np.square(
                np.divide(num_1, standard_deviation_2, out=np.zeros_like(num_1), where=standard_deviation_2 != 0))
            y_square_2 = np.square(
                np.divide(num_2, standard_deviation_1, out=np.zeros_like(num_2), where=standard_deviation_1 != 0))
            md_1 = np.sqrt(np.sum(y_square_1, axis=0))
            md_2 = np.sqrt(np.sum(y_square_2, axis=0))

            md = min(md_1, md_2)
            if md < mahalanobis_distance:
                mahalanobis_distance = md
                md_cluster_id = cluster_id_2
        close_clusters_dict[cluster_id_1] = md_cluster_id

    return close_clusters_dict


def merge_clusters(dict_1, dict_2, cluster_id_1, cluster_id_2):
    """
    merge clusters and update dicts
    :param dict_1:
    :param dict_2:
    :param cluster_id_1:
    :param cluster_id_2:
    :return:
    """
    stats_1 = dict_1[cluster_id_1]
    stats_2 = dict_2[cluster_id_2]
    num_of_points = stats_1[0] + stats_2[0]
    sum_of_coords = np.add(stats_1[1], stats_2[1])
    sum_sq_of_coords = np.add(stats_1[2], stats_2[2])
    centroid = sum_of_coords / num_of_points
    standard_deviation = np.sqrt(np.subtract(sum_sq_of_coords / num_of_points, np.square(centroid)))
    points = stats_1[5]
    points.extend(stats_2[5])

    dict_2[cluster_id_2][0] = num_of_points
    dict_2[cluster_id_2][1] = sum_of_coords
    dict_2[cluster_id_2][2] = sum_sq_of_coords
    dict_2[cluster_id_2][3] = centroid
    dict_2[cluster_id_2][4] = standard_deviation
    dict_2[cluster_id_2][5] = points


def write_intermediate_result(output_file_path, iter_num):
    """
    write intermediate result to output file
    :param iter_num: Iteration number
    :param output_file_path:
    :return: nothing
    """
    num_discard_points = sum([value[0] for value in discard_set_dict.values()])
    num_compression_points = sum([value[0] for value in compression_set_dict.values()])

    if iter_num == 1:
        with open(output_file_path, "w") as f:
            f.write('The intermediate results:\n')
            f.write('Round ' + str(iter_num) + ': ' + str(num_discard_points) + ',' + str(len(compression_set_dict))
                    + ',' + str(num_compression_points) + ',' + str(len(retained_set)) + '\n')
    else:
        with open(output_file_path, "a") as f:
            f.write('Round ' + str(iter_num) + ': ' + str(num_discard_points) + ',' + str(len(compression_set_dict))
                    + ',' + str(num_compression_points) + ',' + str(len(retained_set)) + '\n')


def write_final_clustering_result(output_file_path):
    """
    write final clustering result to output file
    :param output_file_path:
    :return:
    """
    final_clusters_dict = dict()
    for cluster_id, stats in discard_set_dict.items():
        for point in stats[5]:
            final_clusters_dict[point] = cluster_id
    for cluster_id, stats in compression_set_dict.items():
        for point in stats[5]:
            final_clusters_dict[point] = -1
    for point in retained_set:
        final_clusters_dict[point] = -1
    # print('final_clusters_dict: ', final_clusters_dict)

    with open(output_file_path, "a") as f:
        f.write('\nThe clustering results:\n')
        for point_id in sorted(final_clusters_dict.keys(), key=int):
            f.write(str(point_id) + ',' + str(final_clusters_dict[point_id]) + '\n')


def execute_task1():
    global retained_set
    global threshold_dis
    if len(sys.argv) > 3:
        input_file_path = sys.argv[1]
        n_cluster = int(sys.argv[2])
        output_file_path = sys.argv[3]
    else:
        # input_file_path = './data/data_small.txt'
        input_file_path = './data/hw6_clustering.txt'
        n_cluster = 10  # 10
        output_file_path = './output/output_task.txt'

    num_partitions = 5

    # 1. Load 20% of the data randomly
    # todo: scale features (normalise)
    data = preprocess_data(input_file_path)
    np.random.shuffle(data)  # randomise data
    features = np.array_split(data, num_partitions)  # split data into 5 parts (20% each)

    # 2. Run K-Means (e.g., from sklearn) with a large K (e.g., 5 times of the number of the input clusters) on the
    # data in memory using the Euclidean distance as the similarity measurement.
    features_1 = features[0]
    kmeans_1 = KMeans(n_clusters=5 * n_cluster).fit(features_1[:, 2:])

    # 3. In the K-Means result from Step 2, move all the clusters that contain only one point to RS (outliers).
    # print(kmeans_1.labels_)
    clusters = get_clusters_dict(kmeans_1.labels_)
    retained_set = get_retained_set(clusters)
    # print('retained_set: ', retained_set)

    # 4. Run K-Means again to cluster the rest of the data points with K = the number of input clusters.
    features_ds = np.delete(features_1, list(retained_set), axis=0)
    kmeans_2 = KMeans(n_clusters=n_cluster).fit(features_ds[:, 2:])
    clusters = get_clusters_dict(kmeans_2.labels_)
    # print('clusters: ', clusters)

    # 5. Use the K-Means result from Step 4 to generate the DS clusters (i.e., discard their points and generate
    # statistics).
    create_discard_set_dict(clusters, features_ds)

    # 6. Run K-Means on the points in the RS with a large K to generate CS (clusters with more than one points) and
    # RS (clusters with only one point).
    features_rs = features_1[list(retained_set), :]
    if len(retained_set) >= 5 * n_cluster:
        kmeans_3 = KMeans(n_clusters=5 * n_cluster).fit(features_rs[:, 2:])
        clusters = get_clusters_dict(kmeans_3.labels_)
        retained_set = get_retained_set(clusters)
        create_compression_set_dict(clusters, features_rs)
        # print(retained_set)

    write_intermediate_result(output_file_path, 1)

    # d = number of features available for a point
    threshold_dis = 2 * np.sqrt(features_1.shape[1] - 2)
    # print('threshold_dis: ', threshold_dis)
    for iter_num in range(2, num_partitions + 1):
        # 7. Load another 20% of the data randomly.
        true_labels_features = features[iter_num - 1]
        # features_part = features[iter_num][:, 2:]

        for idx, point_tl_features in enumerate(true_labels_features):
            point_features = point_tl_features[2:]
            # 8. For the new points, compare them to each of the DS using the Mahalanobis Distance and assign them to
            # the nearest DS clusters if the distance is < 2√𝑑.
            mahalanobis_distance, cluster_id = get_mahalanobis_distance(point_features, discard_set_dict)
            if cluster_id != -1 and mahalanobis_distance < threshold_dis:
                # assign to ds cluster
                # print('discard set')
                update_set_dicts(discard_set_dict, point_tl_features, cluster_id)
            else:
                # 9. For the new points that are not assigned to DS clusters, using the Mahalanobis Distance and assign
                # the points to the nearest CS clusters if the distance is < 2√𝑑
                mahalanobis_distance, cluster_id = get_mahalanobis_distance(point_features, compression_set_dict)
                if cluster_id != -1 and mahalanobis_distance < threshold_dis:
                    # assign to cs cluster
                    # print('compression set')
                    update_set_dicts(compression_set_dict, point_tl_features, cluster_id)
                else:
                    # 10. For the new points that are not assigned to a DS cluster or a CS cluster, assign them to RS.
                    retained_set.add(idx)
                    # print('retained_set: ', retained_set)

        # 11. Run K-Means on the RS with a large K (e.g., 5 times of the number of the input clusters) to generate
        # CS (clusters with more than one points) and RS (clusters with only one point).
        features_rs = true_labels_features[list(retained_set), :]
        if len(retained_set) >= 5 * n_cluster:
            kmeans_4 = KMeans(n_clusters=5 * n_cluster).fit(features_rs[:, 2:])
            clusters = get_clusters_dict(kmeans_4.labels_)
            retained_set = get_retained_set(clusters)
            create_compression_set_dict(clusters, features_rs)
            # print('retained_set: ', retained_set)

        # 12. Merge CS clusters that have a Mahalanobis Distance < 2√𝑑.
        close_clusters_dict = find_merge_clusters(compression_set_dict, compression_set_dict)
        # print('close_clusters_dict: ', close_clusters_dict)
        for cluster_id_1, cluster_id_2 in close_clusters_dict.items():
            if cluster_id_1 != cluster_id_2 and cluster_id_1 in compression_set_dict \
                    and cluster_id_2 in compression_set_dict:
                merge_clusters(compression_set_dict, compression_set_dict, cluster_id_1, cluster_id_2)
                # print('popped cluster_id_2: ', cluster_id_2)
                compression_set_dict.pop(cluster_id_2)

        # merge_clusters('compression_set_dict', 'compression_set_dict', threshold_dis)

        # 13. If this is the last run , merge CS clusters with DS clusters that have a Mahalanobis Distance < 2√𝑑.
        if iter_num == num_partitions:
            # merge_clusters('compression_set_dict', 'discard_set_dict', threshold_dis)
            close_clusters_dict = find_merge_clusters(compression_set_dict, discard_set_dict)
            for cluster_id_cs, cluster_id_ds in close_clusters_dict.items():
                if cluster_id_cs in compression_set_dict and cluster_id_ds in discard_set_dict:
                    merge_clusters(compression_set_dict, discard_set_dict, cluster_id_cs, cluster_id_ds)
                    compression_set_dict.pop(cluster_id_cs)

        write_intermediate_result(output_file_path, iter_num)

    # since retained set contains indices wrt the split data, finally get unique indices from data (features)
    if len(retained_set) > 0:
        features_rs = features[num_partitions-1][list(retained_set), 0]
        retained_set = set([int(num) for num in features_rs])

    write_final_clustering_result(output_file_path)


if __name__ == '__main__':
    start_time = time.time()
    execute_task1()
    print('Execution time: ', time.time() - start_time)
