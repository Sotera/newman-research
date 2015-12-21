import numpy as np
import matplotlib.pyplot as plt
# import rpy2.robjects.numpy2ri
# from rpy2.robjects.packages import importr
import kmedoids_minibatch_dist_matrix
import csv

def convert_y_labels_to_positive_definite_interval(data_y):
    for i in range(len(data_y)):
        if data_y[i] == -1:
            data_y[i] = 1
        else:
            data_y[i] = 2
    return data_y

def get_rss_of_model(centroid_labels_vec, data_y):
#     find majority label of each centroid:
    assigned_centroid_labels_set = set(centroid_labels_vec)
    print 'CENTROID LABELS VEC IS:'
    print centroid_labels_vec
    print assigned_centroid_labels_set
    num_classes = len(set(data_y))
    centroid_class_vec = []
    for i in assigned_centroid_labels_set:
        class_counts_vec = np.array([0]*num_classes)
        for j in range(len(centroid_labels_vec)):
            if centroid_labels_vec[j] == i:
                class_for_pt = data_y[j]
                # print class_for_pt
                class_counts_vec[class_for_pt-1] += 1
        # chosen_class = arg_max(class_counts_vec) + 1
        chosen_class = np.argmax(class_counts_vec) + 1
        centroid_class_vec.append(chosen_class)

#   Find error between y and y_hat (true labels vs generated labels):
    num_misclass = 0
    m = 0
    print 'Centroid class vec is:'
    print centroid_class_vec
    for i in assigned_centroid_labels_set:
        for j in range(len(centroid_labels_vec)):
            if centroid_labels_vec[j] == i and centroid_class_vec[m] != data_y[j]:
                num_misclass += 1
        m += 1
    return num_misclass

# time_series_name = 'wafer'
time_series_name = 'yoga'
# time_series_data_file = '/home/chris/Time_Series_Datasets/UCR_TS_Archive_2015/'+time_series_name+'/'+time_series_name+'_TRAIN'
time_series_data_file = 'UCR_time_series_archive_2015/'+time_series_name+'/'+time_series_name+'_TRAIN'

fd = open(time_series_data_file)
data_x = []
data_y = []
# data_y.astype(int)
i = 0
# for line in fd:
reader = csv.reader(fd)
for line in reader:
    # print type(line)
    line = np.asarray(line)
    arr = line.astype(np.float)
    temp_x = arr[1::]
    print 'NUM FEATURES == %s' %len(temp_x)
    data_x.append(temp_x)
    data_y.append(int(arr[0]))
    # print data_y[i]
    # print line[0]
    i += 1

# data_y = convert_y_labels_to_positive_definite_interval(data_y)
# data_y = data_y[:50]
# data_x = data_x[:50]
# Initialization of centroid_labels_vector:
centroid_labels_vec = []
for i in range(len(data_x)):
    centroid_labels_vec.append(0)

# k = 5
k = int(0.1*len(data_x))
window_frac = 0.1
# Clustering of time series data:
centroid_labels_vec, curr_medoids = kmedoids_minibatch_dist_matrix.k_medoids_dtw_cluster(data_x, centroid_labels_vec, k)
rss_of_model = get_rss_of_model(centroid_labels_vec, data_y)
error_rate = rss_of_model/float(len(data_y))
print rss_of_model
print error_rate
print data_y
# plt.plot(data_x[14] )
# plt.show(data_x)


