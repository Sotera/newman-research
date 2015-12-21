import numpy as np
import subprocess
import numpy.matlib

#use Java CartiClus package to perform the Cartiflus: Cartification-based
#  subspace cluster finder

# or just for cartification:
# cart_command = ['java','-jar' ,'carticlus.jar' ,cart.CartifierDriver, data_file, k, minsup]
# subprocess.call(cart_command)

class CartificationClustering(object):
    def __init__(self, input_file, num_samples, num_features, frac_of_neighbors_knn, putative_num_clusters):
        self.input_file = input_file
        self. num_samples = num_samples
        self.num_features = num_features
        self.frac_of_neighbors_knn = frac_of_neighbors_knn
        self.putative_num_clusters = putative_num_clusters
        self.output_file = 'cart_cluster_output.csv'
        self.cartlog = 'cartlog.txt'

    def get_cluster_labels_vec(self):
        num_samples = self.num_samples
        cluster_file = self.output_file
        labels_vec = np.zeros(num_samples)
        fd = open(cluster_file, 'r+')
        # row_count = sum(1 for row in fd)
        # labels_mat = np.matlib.zeros((max_row_num, max_col_num))
        curr_cluster_label = 0
        for line in fd:
            arr = line.split(' ')
            j = 0
            for i in range(len(arr)):
                if arr[i][0] == '[':
                    j = i+1
                    break
            for i in range(j, len(arr)):
                # print('i = %d' % i)
                # print('len arr = %d' %len(arr))
                # print('data point is %d' % int(arr[i]))
                labels_vec[int(arr[i])] = curr_cluster_label
            curr_cluster_label += 1

        fd.close()
        return labels_vec

    def cart_cluster(self):
        data_file = self.input_file
        numOfdimensions = self.num_features
        N = self.num_samples
        frac = self.frac_of_neighbors_knn
        putative_num_clusters = self.putative_num_clusters

        frac_minsup = 1/float(putative_num_clusters*2)
        minsup = int(N*frac_minsup)
        # k = int(N*frac)
        k = int(2*N/float(putative_num_clusters))
        output_file = 'cart_cluster_output.csv'
        cartlog = 'cartlog.txt'
        # cart_cluster_command = ['java','-jar' ,'carticlus.jar' ,data_file, k, minsup, numOfdimensions,[cartLog],[outputfile]]
        cart_cluster_command_with_file = ['java','-jar' ,'carticlus.jar', data_file, str(k), str(minsup), str(numOfdimensions), cartlog, output_file]
        cart_cluster_command_without_file = ['java','-jar' ,'carticlus.jar', data_file, str(k), str(minsup), str(numOfdimensions)]
        subprocess.call(cart_cluster_command_with_file)
        labels_vec = self.get_cluster_labels_vec()
        return labels_vec

if __name__ == '__main__':
    data_file = "edited.txt"
    numOfdimensions = 40
    # k = int(numOfdimensions/1.0)
    N = 1000
    frac = 0.2
    putative_num_clusters = 15
    frac_minsup = 1/float(putative_num_clusters)
    # minsup = int(N*frac_minsup)
    minsup = 5
    k = int(N*frac)
    output_file = 'cart_cluster_output.csv'
    cartlog = 'cartlog.txt'
    # cart_cluster_command = ['java','-jar' ,'carticlus.jar' ,data_file, k, minsup, numOfdimensions,[cartLog],[outputfile]]
    cart_cluster_command_with_file = ['java','-jar' ,'carticlus.jar', data_file, str(k), str(minsup), str(numOfdimensions), cartlog, output_file]
    cart_cluster_command_without_file = ['java','-jar' ,'carticlus.jar', data_file, str(k), str(minsup), str(numOfdimensions)]
    subprocess.call(cart_cluster_command_with_file)

    def get_cluster_labels_vec(num_samples, cluster_file):
        labels_vec = np.zeros(num_samples)
        fd = open(cluster_file, 'r+')
        # row_count = sum(1 for row in fd)
        # labels_mat = np.matlib.zeros((max_row_num, max_col_num))
        curr_cluster_label = 0
        for line in fd:
            arr = line.split(' ')
            j = 0
            for i in range(len(arr)):
                if arr[i][0] == '[':
                    j = i+1
                    break
            for i in range(j, len(arr)):
                labels_vec[int(arr[i])] = curr_cluster_label
            curr_cluster_label += 1

        fd.close()
        return labels_vec

    labels_vec = get_cluster_labels_vec(N, output_file)




