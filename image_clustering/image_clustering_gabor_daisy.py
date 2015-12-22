from skimage.feature import daisy
from skimage.io import imread
from skimage.filters import gabor_filter
import os
import resize_image
import numpy as np
from sklearn.decomposition import TruncatedSVD
import shutil
from sklearn.cluster import KMeans, MiniBatchKMeans
import math
import time
from sklearn import metrics
import argparse
import gabor_features

# import csv
# from cartification_clustering import CartificationClustering
# from skimage import data
# from sklearn.preprocessing import Normalizer
# from sklearn.pipeline import make_pipeline
# import matplotlib.pyplot as plt


def get_gabor_filter_features(im, ind, min_freq, max_freq):
    im_shape = im.shape
    # print 'The length of the shape is %s' % len(im_shape)
    num_dim_image = len(im_shape)
    if num_dim_image == 3:
        im_slice = im[:,:,ind]
    else:
        im_slice = im
    real_gab = gabor_filter(im_slice, min_freq)[0].flatten()
    # print real_gab.shape
    img_gab = gabor_filter(im_slice, min_freq)[1].flatten()
    # print img_gab.shape
    curr_flat_features = np.hstack((real_gab, img_gab))
    curr_freq = min_freq

    num_steps = int(math.log10(max_freq))
    for i in range(num_steps):
        curr_freq *= 10
        real_gab = gabor_filter(im_slice, curr_freq)[0].flatten()
        img_gab = gabor_filter(im_slice, curr_freq)[1].flatten()
        next_flat_features = np.hstack((real_gab, img_gab))
        curr_flat_features = np.hstack((curr_flat_features, next_flat_features))
        # curr_flat_features = np.hstack((curr_flat_features, real_gab))
        if curr_freq > max_freq:
            break
    return curr_flat_features

def resize_image_and_get_features(dir_of_file_full_path, filename, num_rows, num_cols, recursive_file_search, dir_upstream_of_edited):
    if recursive_file_search:
        resized_filename = resize_image.resize_image_recursive_file_search(dir_of_file_full_path, filename, num_rows, num_cols, dir_upstream_of_edited)
    else:
        resized_filename = resize_image.resize_image(dir_of_file_full_path, filename, num_rows, num_cols)
    resized_filename_camera = resize_image(curr_path, file, num_rows, num_cols)
    im = imread(resized_filename)
    index = 0
    min_frequency = 10
    max_frequency = 1000
    gabor_filter_features = get_gabor_filter_features(im, index, min_frequency, max_frequency)
    print(gabor_filter_features.shape)
    # return gabor_filter_features
    im_shape = im.shape
    # print 'The length of the shape is %s' % len(im_shape)
    num_dim_image = len(im_shape)
    if num_dim_image == 3:
        im1 = im[:,:,0]
        im2 = im[:,:,1]
        im3 = im[:,:,2]
    else:
        im1 = im

    descs1, descs_img = daisy(im1, step=180, radius=58, rings=2, histograms=6,
                             orientations=8, visualize=True)
    # descs2, descs_img2 = daisy(im1, step=180, radius=58, rings=2, histograms=6,
    #                          orientations=8, visualize=True)
    # descs3, descs_img3 = daisy(im1, step=180, radius=58, rings=2, histograms=6,
    #                          orientations=8, visualize=True)

    # FULL feature set using gabor, daisy, and image features:
    # descs = np.hstack((descs1, descs2, descs3))
    # flat_descs = descs.flatten()
    # im_flat = np.hstack((im1, im2, im3)).flatten()
    # im_descs_stack = np.hstack((im_flat, flat_descs))
    # flat_features_full = im_descs_stack.flatten()

    # descs = np.hstack((descs1, descs2, descs3))
    flat_descs = descs1.flatten()
    im_flat = im1.flatten()
    im_descs_stack = np.hstack((im_flat, flat_descs))
    flat_features_full = im_descs_stack.flatten()
    # flat_features = descs1.flatten()
    # return flat_features
    return np.hstack((gabor_filter_features, flat_descs))

def resize_image_and_get_full_gabor_features(dir_of_file_full_path, filename, num_rows, num_cols, recursive_file_search, dir_upstream_of_edited):
    if recursive_file_search:
        resized_filename = resize_image.resize_image_recursive_file_search(dir_of_file_full_path, filename, num_rows, num_cols, dir_upstream_of_edited)
    else:
        resized_filename = resize_image.resize_image(dir_of_file_full_path, filename, num_rows, num_cols)

    num_levels = 5
    num_orientations = 5
    return gabor_features.get_gabor_features(resized_filename, num_levels, num_orientations)


def get_first_file_name(directory_full_path):
    for subdir, dirs, files in os.walk(dir_path):
        for file_i in files:
            return file_i
            # return os.path.join(subdir, file_i)

def get_first_file_name_recursive_file_search(directory_full_path):
    for subdir, dirs, files in os.walk(dir_path):
        # print "SUBDIR IS %s" % subdir
        pos_slash = subdir.rfind('/')
        # dir_of_file_full_path = subdir[:pos_slash+1]
        curr_dir = subdir[pos_slash+1::]
        if curr_dir == 'edited':
            continue
        for file_i in files:
            # print os.path.join(subdir, file_i)
            return os.path.join(subdir, file_i)

def cluster_images_in_dir_recursive_file_search(dir_path, output_file_name, num_rows, num_cols, num_desired_features, num_clusters_user, recursive_file_search):
    curr_path = os.getcwd()
    # print dir_path
    dir_name = dir_path
    dir_path_sample = os.path.join(curr_path, dir_name)
    # sample_img = 'space_wolf_dreadnought.jpg'
    sample_img = get_first_file_name_recursive_file_search(dir_path_sample)
    pos_slash = sample_img.rfind('/')
    dir_of_file_full_path = sample_img[:pos_slash+1]
    sample_img = sample_img[pos_slash+1::]
    sample_feat = resize_image_and_get_full_gabor_features(dir_of_file_full_path, sample_img, num_rows, num_cols, recursive_file_search, dir_name)
    size_feat = sample_feat.shape[0]
    temp_mat = np.zeros(size_feat)
    temp_mat = temp_mat.reshape(1, size_feat)
    # with file(output_file_name, 'w') as outfile:
    for subdir, dirs, files in os.walk(dir_path):
        num_samples = len(files)
        pos_slash = subdir.rfind('/')
        # dir_of_file_full_path = subdir[:pos_slash+1]
        curr_dir = subdir[pos_slash+1::]
        # print 'Current dir is %s' % curr_dir
        if curr_dir == 'edited':
            continue
        for file_i in files:
            # print file_i
            img_features = resize_image_and_get_full_gabor_features(subdir, file_i, num_rows, num_cols, recursive_file_search, dir_name)
            img_features = img_features.reshape(1, img_features.shape[0])
            temp_mat = np.vstack((temp_mat, img_features))
        # break
    temp_mat = temp_mat[1:, :]
    # perform SVD:
    num_reduced_features = num_desired_features
    svd = TruncatedSVD(num_reduced_features)
    temp_mat = svd.fit_transform(temp_mat)

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print()
    np.savetxt(output_file_name, temp_mat, fmt='%f')
    num_features = num_desired_features
    # NUM_CLUSTERS = int(num_samples/float(num_clusters_user))
    NUM_CLUSTERS = num_clusters_user
    frac_of_neighbors_knn = 0.1
    putative_num_clusters = NUM_CLUSTERS
    input_file = output_file_name

    # cart_clustering = CartificationClustering(input_file, num_samples, num_features, frac_of_neighbors_knn, putative_num_clusters)
    # cluster_labels = cart_clustering.cart_cluster()
    km = MiniBatchKMeans(n_clusters=putative_num_clusters, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose=True)
    km.fit(temp_mat)
    cluster_labels = km.labels_

    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(temp_mat, cluster_labels, sample_size=1000))
    return cluster_labels

def cluster_images_in_dir(dir_path, output_file_name, num_rows, num_cols, num_desired_features, num_clusters_user):
    curr_path = os.getcwd()
    dir_name = dir_path
    dir_path_sample = os.path.join(curr_path, dir_name)
    # sample_img = 'space_wolf_dreadnought.jpg'
    sample_img = get_first_file_name(dir_path_sample)
    dir_of_file_full_path = dir_path

    recursive_file_search = False


    sample_feat = resize_image_and_get_full_gabor_features(dir_path_sample, sample_img, num_rows, num_cols, recursive_file_search, dir_name)
    size_feat = sample_feat.shape[0]
    temp_mat = np.zeros(size_feat)
    temp_mat = temp_mat.reshape(1, size_feat)
    # with file(output_file_name, 'w') as outfile:

    for subdir, dirs, files in os.walk(dir_path):
        num_samples = len(files)
        for file_i in files:
            # print dirs
            img_features = resize_image_and_get_full_gabor_features(subdir, file_i, num_rows, num_cols, recursive_file_search, dir_name)
            img_features = img_features.reshape(1, img_features.shape[0])
            temp_mat = np.vstack((temp_mat, img_features))
        break
    temp_mat = temp_mat[1:, :]
    # perform SVD:
    num_reduced_features = num_desired_features
    svd = TruncatedSVD(num_reduced_features)
    temp_mat = svd.fit_transform(temp_mat)

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print()
    np.savetxt(output_file_name, temp_mat, fmt='%f')
    num_features = num_desired_features
    # NUM_CLUSTERS = int(num_samples/float(num_clusters_user))
    NUM_CLUSTERS = num_clusters_user
    frac_of_neighbors_knn = 0.1
    putative_num_clusters = NUM_CLUSTERS
    input_file = output_file_name

    # cart_clustering = CartificationClustering(input_file, num_samples, num_features, frac_of_neighbors_knn, putative_num_clusters)
    # cluster_labels = cart_clustering.cart_cluster()
    km = MiniBatchKMeans(n_clusters=putative_num_clusters, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose=True)
    km.fit(temp_mat)
    cluster_labels = km.labels_

    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(temp_mat, cluster_labels, sample_size=1000))

    return cluster_labels


def partition_image_set(cluster_labels, dir_path):

#     create dir for each cluster:
    clusters_classes = np.unique(cluster_labels)
    print(clusters_classes)
    for i in range(len(clusters_classes)):
#         make dir:
        cluster_dir = 'cluster_' + str(i)
        dir = os.path.join(dir_path, cluster_dir)
        if os.path.exists(dir):
            shutil.rmtree(dir)
        # if not os.path.exists(dir):
        os.makedirs(dir)

    for subdir, dirs, files in os.walk(dir_path):
        i = 0
        for file_i in files:
        #     copy file_i into cluster folder i
            cluster_dir = 'cluster_' + str(int(cluster_labels[i]))
            dir = os.path.join(dir_path, cluster_dir)
            dir = os.path.join(dir, file_i)
            filename_path = os.path.join(subdir, file_i)
            shutil.copy2(filename_path, dir)
            i += 1
        break

def partition_image_set_recursive_file_search(cluster_labels, dir_path):

#     create dir for each cluster:
    clusters_classes = np.unique(cluster_labels)
    print(clusters_classes)
    for i in range(len(clusters_classes)):
#         make dir:
        cluster_dir = 'cluster_' + str(i)
        dir = os.path.join(dir_path, cluster_dir)
        if os.path.exists(dir):
            shutil.rmtree(dir)
        # if not os.path.exists(dir):
        os.makedirs(dir)

    i = 0
    for subdir, dirs, files in os.walk(dir_path):

        for file_i in files:
            pos_slash = subdir.rfind('/')
            # dir_of_file_full_path = subdir[:pos_slash+1]
            curr_dir = subdir[pos_slash+1::]
            # print 'Current dir is %s' % curr_dir
            if curr_dir == 'edited':
                continue
            if 'cluster' in curr_dir:
                continue
        #     copy file_i into cluster folder i
            cluster_dir = 'cluster_' + str(int(cluster_labels[i]))
            cluster_dir_path = os.path.join(dir_path, cluster_dir)
            dir = os.path.join(cluster_dir_path, file_i)
            filename_path = os.path.join(subdir, file_i)

            arr = file_i.split('.')
            j = 0
            while os.path.isfile(dir):
                file_i_new = arr[0] + str(j) + '.' + arr[-1]
                dir = os.path.join(cluster_dir_path, file_i_new)
                j += 1
            shutil.copy2(filename_path, dir)
            i += 1
        # break

def convert_cluster_labels_to_simple_range(cluster_labels):
    converted_cluster_labels = np.zeros(cluster_labels.shape[0])
    unique_labels = np.unique(cluster_labels)
    converted_label = 0
    for i in range(len(unique_labels)):
        label_i = unique_labels[i]
        for j in range(len(cluster_labels)):
            if cluster_labels[j] == label_i:
                converted_cluster_labels[j] = converted_label
        converted_label += 1
    return converted_cluster_labels

def str_to_bool(string):
    str_lower = string.lower()
    if str_lower == 'true':
        return True
    elif str_lower == 'false':
        return False
    else:
        # not proper input:
        raise ValueError("Cannot convert {} to a bool".format(string))

if __name__ == '__main__':
    start_time = time.time()
    curr_path = os.getcwd()

    desc='Export attachments from ElassticSearch.'
    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=desc)

    parser.add_argument("--num_clusters", help="Number of clusters to use", default=5)
    parser.add_argument("--num_desired_features", help="Enter dimensionality of desired feature space", default=8)
    parser.add_argument("--dir_name", help="name of directory containing images. Target directory must be in same directory as this file", default='images_set')
    parser.add_argument("--num_rows", help="number of rows to use in resized image", default=200)
    parser.add_argument("--num_cols", help="number of cols to use in resized image", default=200)
    # parser.add_argument("--recursive_file_search", help="Enter True or False for a recursive search for image files", default='false')
    parser.add_argument("--recursive_file_search", help="Enter True or False for a recursive search for image files", default='false')


    args = parser.parse_args()
    print('Clustering with %s clusters in %s dimensional feature space' % (args.num_clusters, args.num_desired_features))

    # dir_name = 'images_set'
    dir_name = args.dir_name
    dir_path = os.path.join(curr_path, dir_name)
    output_file_name = 'cluster_results.csv'
    num_rows = int(args.num_rows)
    num_cols = int(args.num_cols)
    # num_rows = num_cols = 200
    num_desired_features = int(args.num_desired_features)
    num_clusters_user = int(args.num_clusters)

    recursive_file_search = str_to_bool(args.recursive_file_search)
    if recursive_file_search:
        cluster_labels = cluster_images_in_dir_recursive_file_search(dir_path, output_file_name, num_rows, num_cols, num_desired_features, num_clusters_user, recursive_file_search)
    else:
        cluster_labels = cluster_images_in_dir(dir_path, output_file_name, num_rows, num_cols, num_desired_features, num_clusters_user)
    cluster_labels = convert_cluster_labels_to_simple_range(cluster_labels).astype(int)
    print cluster_labels
    if recursive_file_search:
        partition_image_set_recursive_file_search(cluster_labels, dir_path)
    else:
        partition_image_set(cluster_labels, dir_path)
    stop_time = time.time()
    delta_t_in_min = (stop_time - start_time)/60.0
    print('------ %s minutes -------' % delta_t_in_min)




