from PIL import Image
import pytesseract as PT
import os
from skimage.io import imread
import resize_image
from textblob import TextBlob
import unicodedata
from skimage.io import imsave
from skimage.filters import threshold_isodata


import numpy as np
import matplotlib.pyplot as plt
from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2

def image_processing(dir_of_file_full_path, filename, num_rows, num_cols):
    resized_filename = resize_image.resize_image_ocr(dir_of_file_full_path, filename, num_rows, num_cols)
    # image_full_path = os.path.join(dir_of_file_full_path, filename)
    im = imread(resized_filename)

    # size_slice_im = im[:,:,0].shape
    # im_adaptive = np.zeros((size_slice_im[0], size_slice_im[1], 3))
    # for i in range(3):
    #     threshold = threshold_isodata(im[:,:,i])
    #     print threshold
    #     im_adaptive_i = im[:,:,i] > threshold
    #     im_adaptive[:,:,i] = im_adaptive_i

    astro = color.rgb2gray(im)
    num_iter = 1
    for i in range(num_iter):
        ax_len = 3
        psf = np.ones((ax_len, ax_len)) / float(ax_len*ax_len)
        astro = conv2(astro, psf, 'same')
        astro += 0.1 * astro.std() * np.random.standard_normal(astro.shape)

        deconvolved, _ = restoration.unsupervised_wiener(astro, psf)

        # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
        #
        # plt.gray()
        #
        # ax[0].imshow(astro, vmin=deconvolved.min(), vmax=deconvolved.max())
        # ax[0].axis('off')
        # ax[0].set_title('Data')
        #
        # ax[1].imshow(deconvolved)
        # ax[1].axis('off')
        # ax[1].set_title('Self tuned restoration')
        #
        # fig.subplots_adjust(wspace=0.02, hspace=0.2,
        #                     top=0.9, bottom=0.05, left=0, right=1)
        # plt.show()

        astro = deconvolved
    arr = resized_filename.split('.')
    im_adaptive_filename = arr[0] + '_thresholded.' + arr[-1]
    imsave(im_adaptive_filename, deconvolved)
    # image_file = Image.open(im_adaptive_filename)
    # image_file = image_file.convert('1')
    # image_file.save(im_adaptive_filename)
    return im_adaptive_filename


curr_path = os.getcwd()
# dir_of_file = 'images_set'
dir_of_file = 'receipt_class_images'
dir_full_path = os.path.join(curr_path, dir_of_file)
# filename = 'tesseract-opencv-test.png'
filename = 'receipt2.jpg'
# filename = 'img7.jpg'
num_rows = num_cols = 900
# num_rows = 555
# num_cols = 110
im_post_process_filepath = image_processing(dir_full_path, filename, num_rows, num_cols)
im = Image.open(im_post_process_filepath)
# im = Image.open(os.path.join(dir_full_path, 'ocr/img7_ocr.jpg'))
text = PT.image_to_string(im, lang='eng', boxes=False)
print '========ouput========'
print text
# t_str = unicodedata.normalize('NFKD', text).encode('ascii','ignore')
print '=========correction======='
b = TextBlob(unicode(text, 'utf-8'))
print b.correct()