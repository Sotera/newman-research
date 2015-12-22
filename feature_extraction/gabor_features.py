import os
import numpy as np
import matplotlib.pyplot as plt
from LogGabor import LogGabor
# from SLIP import Image, imread
from SLIP import imread


def get_shape_of_filtered_image(image_file_path):
    image = imread(image_file_path)
    lg = LogGabor('default_param.py')
    lg.set_size(image)
    i_level = 0
    theta = 0
    params = {'sf_0':1./(2**i_level), 'B_sf':lg.pe.B_sf, 'theta':theta, 'B_theta':lg.pe.B_theta}
    FT_lg = lg.loggabor(0, 0, **params)
    filtered_img_mat = lg.FTfilter(image, FT_lg, full=True)
    return filtered_img_mat.shape

# lg = LogGabor('default_param.py')
def get_gabor_features(image_file_path, num_levels, num_orientations):
    image = imread(image_file_path)
    opts= {'vmin':0., 'vmax':1., 'interpolation':'nearest', 'origin':'upper'}

    lg = LogGabor('default_param.py')
    lg.set_size(image)
    # num_features = num_levels*num_orientations*2
    # feature_vec = np.zeros((num_features,1))
    # phi = (np.sqrt(5) +1.)/2. # golden number
    # fig = plt.figure(figsize=(fig_width, fig_width/phi))
    # xmin, ymin, size = 0, 0, 1.
    i = 0
    for i_level in range(num_levels):
        # for theta in np.linspace(0, np.pi, num_orientations, endpoint=False):
        for theta in np.linspace(0, np.pi, num_orientations):
            params = {'sf_0':1./(2**i_level), 'B_sf':lg.pe.B_sf, 'theta':theta, 'B_theta':lg.pe.B_theta}
            # loggabor takes as args: u, v, sf_0, B_sf, theta, B_theta)
            FT_lg = lg.loggabor(0, 0, **params)
            filtered_img_mat = lg.FTfilter(image, FT_lg, full=True)
            # print "SHAPE OF FILTERED IMAGE IS (%s, %s)" % filtered_img_mat.shape
            im_abs_feature = np.absolute(filtered_img_mat).flatten()
            # im_abs_feature = np.sum(np.absolute(filtered_img_mat))
            # im_sqr_feature = np.sum(np.square(np.real(filtered_img_mat)))
            # im_sqr_feature = np.sum(np.square(filtered_img_mat))
            if i == 0:
                feature_vec = im_abs_feature
            else:
                feature_vec = np.hstack((feature_vec, im_abs_feature))
            i += 1
    print
    return feature_vec

if __name__ == '__main__':
    curr_path = os.getcwd()
    filename = 'images_set/desert.jpg'
    full_file_path = os.path.join(curr_path, filename)
    print full_file_path
    num_levels = 5
    num_orientations = 8
    feature_vec = get_gabor_features(full_file_path, num_levels, num_orientations)
    print "The feature vector is:"
    print feature_vec
    print "with dimensions (%s, %s)" % feature_vec.shape
    # x = np.array([[1,2],[3,4]])
    # print np.sum(x)