import numpy as np
import random
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
import time
import fastdtw
import math

def create_window(frac, len1, len2):
    window = []
    mult = 1.0 - frac
    c = int(len1 - math.sqrt(mult*len1*len1))
    for k in range(c):
        for i in range(len1):
            for j in range(len2):
                if i == j + k:
                    window.append((i,j))

    for k in range(c):
        for i in range(len1):
            for j in range(len2):
                if j == i + k and i != j:
                    window.append((i,j))
    return window

def create_distance_matrix(input_pts, num_input_pts, window_frac):
    print "CREATING DTW DISTANCE MATRIX"
    start_time = time.time()
    window = create_window(window_frac, input_pts[0].shape[0], input_pts[0].shape[0])
    mat = np.ones((num_input_pts, num_input_pts))*(-1)
    for i in range(num_input_pts):
        for j in range(num_input_pts):
            print "Getting dtw distance for (%s, %s)" % (i, j)
            if mat[i, j] == -1:
                mat[i, j] = get_dtw_dist(input_pts[i], input_pts[j], window)
                mat[j, i] = mat[i, j]

    print "FINISHING DTW DIST MATRIX"
    print "------ %s elapsed minutes ---------" % ((time.time() - start_time)/60.0)
    return mat


def k_medoids_dtw_cluster(input_pts, centroid_labels_vec, k=4, window_frac=0.1):
    start_time = time.time()
    MAX_ITER = 100
    # MAX_ITER = 5
    # m = input_pts.shape[0] # number of points
    m = len(input_pts)
    # Pick k random medoids:
    curr_medoids = np.array([-1]*k)
    while not len(np.unique(curr_medoids)) == k:
        curr_medoids = np.array([random.randint(0, m - 1) for _ in range(k)])
    old_medoids = np.array([-1]*k) # Doesn't matter what we initialize these to.
    new_medoids = np.array([-1]*k)

    # Create window matrix for fastdtw:
    # window_frac = 0.05

    distance_matrix = create_distance_matrix(input_pts, m, window_frac)

    # Until the medoids stop updating, do the following:
    iter_num = 0
    while not ((old_medoids == curr_medoids).all()):
        # Assign each point to cluster with closest medoid.
        centroid_labels_vec = assign_points_to_clusters_random(curr_medoids, input_pts, centroid_labels_vec, distance_matrix)

        # Update cluster medoids to be lowest cost point.
        for curr_medoid in curr_medoids:
            print 'In update medoids - on %d' % curr_medoid
            # clusters = np.where(clusters == curr_medoid)[0]
            cluster, cluster_pts_id = get_cluster(input_pts, curr_medoid, centroid_labels_vec)
            curr_medoid_val = input_pts[curr_medoid]
            new_medoids[curr_medoids == curr_medoid] = compute_new_medoid_pam_like_random(curr_medoid, curr_medoid_val, cluster, cluster_pts_id, distance_matrix)

        old_medoids[:] = curr_medoids[:]
        curr_medoids[:] = new_medoids[:]
        iter_num += 1
        print 'FINISHING ITERATION NUMBER %d' % iter_num

        if iter_num == MAX_ITER:
            break
    print('time to run k-medoids clustering was %f' % (time.time() - start_time))
    return centroid_labels_vec, curr_medoids

def get_cluster(input_pts, curr_medoid, centroid_labels_vec):
    start_time = time.time()
    # put all pts in cluster with given centorid label:
    cluster = []
    cluster_pts_id = []
    for i in range(len(input_pts)):
        centroid_label = centroid_labels_vec[i]
        if(centroid_label == curr_medoid):
            cluster.append(input_pts[i])
            cluster_pts_id.append(i)
    # print('time to run get cluster fct was %f', time.time() - start_time)
    return cluster, cluster_pts_id

def get_dtw_dist(pt1, pt2, window):
    dist_obj = fastdtw.dtw(pt1, pt2, window)
    dist = dist_obj[0]
    # print('time to run get_dtw_dist fct was %f', time.time() - start_time)
    return dist


def assign_points_to_clusters(medoids, input_pts, centroid_labels_vec, window):
    # distances_to_medoids = input_pts[:,medoids]
    # find dist from each pt to medoids and assign
    # pt to closest medoid
    start_time = time.time()
    print 'in assign points to clusters'
    for i in range(len(input_pts)):
        curr_pt = input_pts[i]
        # curr_closest_medoid = 0
        curr_smallest_dist = float('inf')
        for j in range(len(medoids)):
            curr_medoid_num = medoids[j]
            curr_medoid = input_pts[curr_medoid_num]
            curr_dist = get_dtw_dist(curr_pt, curr_medoid, window)
            if(curr_dist < curr_smallest_dist):
                centroid_labels_vec[i] = curr_medoid_num
                curr_smallest_dist = curr_dist
    # clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    # clusters[medoids] = medoids
    print 'leaving assign points to clusters'
    print('time to run assign_points_to_clusters fct was %f' % (time.time() - start_time))
    return centroid_labels_vec

def assign_points_to_clusters_random(medoids, input_pts, centroid_labels_vec, distance_matrix):
    # distances_to_medoids = input_pts[:,medoids]
    # find dist from each pt to medoids and assign
    # pt to closest medoid
    start_time = time.time()
    print 'in assign points to clusters'
    for i in range(len(input_pts)):
        print "Assigning point %s to a cluster" % i
        if i in medoids:
            print "FOUND A POINT AS A MEDOID"
            centroid_labels_vec[i] = i
            continue
        curr_pt = input_pts[i]
        # curr_closest_medoid = 0
        curr_smallest_dist = float('inf')
        frac = 0.5
        num_medoids_to_search = int(len(medoids)*frac)
        if num_medoids_to_search <= 1:
            num_medoids_to_search = 2
        # print num_medoids_to_search
        rand_medoids = np.random.randint(len(medoids), size=num_medoids_to_search)
        # for j in range(len(medoids)):
        for j in rand_medoids:
            curr_medoid_num = medoids[j]
            curr_medoid = input_pts[curr_medoid_num]
            curr_dist = distance_matrix[i, curr_medoid_num]
            # curr_dist = get_dtw_dist(curr_pt, curr_medoid, window)
            if(curr_dist < curr_smallest_dist):
                centroid_labels_vec[i] = curr_medoid_num
                curr_smallest_dist = curr_dist
    # clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    # clusters[medoids] = medoids
    print 'leaving assign points to clusters'
    print('time to run assign_points_to_clusters fct was %f' % (time.time() - start_time))
    return centroid_labels_vec

def compute_new_medoid(cluster, cluster_pts_input_pts_id, window):
    # mask = np.ones(input_pts.shape)
    start_time = time.time()
    print 'compute new medoids start'
    chosen_centroid_id = 0
    curr_min_total_dist = float('inf')
    for i in range(len(cluster)):
        curr_test_centroid = cluster[i]
        total_dist = 0
        for j in range(len(cluster)):
            if i != j:
                cluster_pt = cluster[j]
                total_dist += get_dtw_dist(curr_test_centroid, cluster_pt, window)
        if(total_dist < curr_min_total_dist):
            chosen_centroid_id = cluster_pts_input_pts_id[i]
            curr_min_total_dist = total_dist
    print 'compute new medoids end'
    print('time to run compute_new_medoid fct was %f', time.time() - start_time)
    return chosen_centroid_id
    # mask = np.ones(len(input_pts))
    # mask[np.ix_(cluster,cluster)] = 0.
    # cluster_input_pts = np.ma.masked_array(data=input_pts, mask=mask, fill_value=10e9)
    # costs = cluster_input_pts.sum(axis=1)
    # return costs.argmin(axis=0, fill_value=10e9)

def compute_new_medoid_pam_like(curr_medoid_id, curr_medoid_val, cluster, cluster_pts_input_pts_id, window):
    # mask = np.ones(input_pts.shape)
    start_time = time.time()
    print 'compute new medoids start'
    max_iter = 3
    holder_curr_best_id = curr_medoid_id
    holder_curr_best_val = curr_medoid_val
    for i in range(max_iter):
        test_centroid_cluster_id = random.randint(0, len(cluster) - 1)
        test_centroid_val = cluster[test_centroid_cluster_id]
        test_centroid_global_id = cluster_pts_input_pts_id[test_centroid_cluster_id]

        # get cumulative dist within cluster from test centroid:
        total_dist_test_pt = 0
        for j in range(len(cluster)):
            cluster_pt = cluster[j]
            total_dist_test_pt += get_dtw_dist(test_centroid_val, cluster_pt, window)
        total_dist_curr_medoid = 0
        for j in range(len(cluster)):
            cluster_pt = cluster[j]
            total_dist_curr_medoid += get_dtw_dist(holder_curr_best_val, cluster_pt, window)
        if total_dist_test_pt < total_dist_curr_medoid:
            holder_curr_best_id = test_centroid_global_id
            holder_curr_best_val = test_centroid_val
    print 'compute new medoids end'
    print('time to run compute_new_medoid fct was %f' % (time.time() - start_time))
    return holder_curr_best_id

def compute_new_medoid_pam_like_random(curr_medoid_id, curr_medoid_val, cluster, cluster_pts_input_pts_id, distance_matrix):
    # mask = np.ones(input_pts.shape)
    start_time = time.time()
    print 'compute new medoids start'
    max_iter = 3
    holder_curr_best_id = curr_medoid_id
    holder_curr_best_val = curr_medoid_val
    for i in range(max_iter):
        print "LENGTH OF CLUSTER IS %s" % len(cluster)
        test_centroid_cluster_id = random.randint(0, len(cluster) - 1)
        test_centroid_val = cluster[test_centroid_cluster_id]
        test_centroid_global_id = cluster_pts_input_pts_id[test_centroid_cluster_id]

        # get cumulative dist within cluster from test centroid:
        total_dist_test_pt = 0
        frac = 0.5
        num_pts_to_search = int(len(cluster)*frac)
        if num_pts_to_search == 0:
            num_pts_to_search = 1
        rand_pts = np.random.randint(len(cluster), size=num_pts_to_search)
        # for j in range(len(cluster)):
        for j in rand_pts:
            # cluster_pt = cluster[j]
            # total_dist_test_pt += get_dtw_dist(test_centroid_val, cluster_pt, window)
            curr_medoid_num_id = cluster_pts_input_pts_id[j]
            total_dist_test_pt += distance_matrix[test_centroid_global_id, curr_medoid_num_id]
        total_dist_curr_medoid = 0
        # for j in range(len(cluster)):
        for j in rand_pts:
            cluster_pt = cluster[j]
            # total_dist_curr_medoid += get_dtw_dist(holder_curr_best_val, cluster_pt, window)
            curr_medoid_num_id = cluster_pts_input_pts_id[j]
            total_dist_curr_medoid += distance_matrix[test_centroid_global_id, curr_medoid_num_id]
        if total_dist_test_pt < total_dist_curr_medoid:
            holder_curr_best_id = test_centroid_global_id
            holder_curr_best_val = test_centroid_val
    print 'compute new medoids end'
    print('time to run compute_new_medoid fct was %f' % (time.time() - start_time))
    return holder_curr_best_id

