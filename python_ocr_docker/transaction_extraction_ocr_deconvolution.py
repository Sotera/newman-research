import PIL
from PIL import Image
import pytesseract as PT
import os
from skimage.io import imread
import resize_image
from skimage.io import imsave
import argparse
import numpy as np
from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2

# from textblob import TextBlob
# import unicodedata
# from skimage.filters import threshold_isodata
# import matplotlib.pyplot as plt


def image_processing(dir_of_file_full_path, filename, num_rows, num_cols):
    resized_filename = resize_image.resize_image_ocr(dir_of_file_full_path, filename, num_rows, num_cols)
    # image_full_path = os.path.join(dir_of_file_full_path, filename)
    im = imread(resized_filename)
    astro = color.rgb2gray(im)
    num_iter = 1
    for i in range(num_iter):
        ax_len = 3
        psf = np.ones((ax_len, ax_len)) / float(ax_len*ax_len)
        astro = conv2(astro, psf, 'same')
        astro += 0.1 * astro.std() * np.random.standard_normal(astro.shape)

        deconvolved, _ = restoration.unsupervised_wiener(astro, psf)
        astro = deconvolved
    arr = resized_filename.split('.')
    im_adaptive_filename = arr[0] + '_thresholded.' + arr[-1]
    imsave(im_adaptive_filename, deconvolved)
    # image_file = Image.open(im_adaptive_filename)
    # image_file = image_file.convert('1')
    # image_file.save(im_adaptive_filename)
    return im_adaptive_filename

if __name__ == '__main__':
    desc='OCR image'
    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=desc)

    parser.add_argument("file_path", help="Path to image file to OCR")
    args = parser.parse_args()

    # filename = 'receipt2.jpg'
    # filename = 'arabic3.jpg'
    # filename = 'img7.jpg'

    file_path = args.file_path
    dir_full_path = os.path.dirname(os.path.abspath(file_path))
    filename = file_path.rsplit('/', 1)[-1]
    num_rows = num_cols = 900
    # num_rows = 555
    # num_cols = 110
    im_post_process_filepath = image_processing(dir_full_path, filename, num_rows, num_cols)
    im = Image.open(im_post_process_filepath)
    # im = Image.open(os.path.join(dir_full_path, 'ocr/img7.jpg'))
    lang_english = 'eng'
    lang_arabic = 'ara'
    text = PT.image_to_string(im, lang=lang_english, boxes=False)
    print '========ouput========'
    print text
    outfile = 'ocr_output.txt'
    fd = open(outfile, 'w')
    for line in text:
        fd.write(line)

    # t_str = unicodedata.normalize('NFKD', text).encode('ascii','ignore')
    # print '=========correction======='
    # b = TextBlob(unicode(text, 'utf-8'))
    # corrected_text = b.correct()
    # print corrected_text

