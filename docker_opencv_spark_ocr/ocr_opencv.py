from PIL import Image
# from pillow import Image

import pytesseract as PT
import os
import resize_image
from skimage.io import imsave
import argparse
import numpy as np
import cStringIO

from skimage.io import imread
from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2
import codecs
# from textblob import TextBlob

# OpenCV:
import cv2

# def create_opencv_image_from_stringio(img_stream, cv2_img_flag=0):
#     #img_stream.seek(0)
#     img_array = np.asarray(img_stream.read(), dtype=np.uint16)
#     return cv2.imdecode(img_array, cv2_img_flag)

def image_processing_cstringio(cstring_object, num_rows, num_cols):
    img = resize_image.resize_image_ocr_cstringio(cstring_object, num_rows, num_cols)
    #img = create_opencv_image_from_stringio(resized_filename)
    img = cv2.medianBlur(img,3)
    # print img

    # Using adaptive methods:
    # ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    # th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #             cv2.THRESH_BINARY,11,2)
    # th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #             cv2.THRESH_BINARY,11,2)
    # w_vec = [0.3, 0.3, 0.4]
    # im = (th3*w_vec[0] + th2*w_vec[1] + th1*w_vec[2])/3
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, 0)
    except:
        print 'Image is already grayscaled'
    (thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return Image.fromarray(im_bw)

    # return cStringIO.StringIO(im_bw)
    # im_adaptive_filename = cStringIO.StringIO()
    # imsave(im_adaptive_filename, im_bw)
    # return im_adaptive_filename

def image_processing(dir_of_file_full_path, filename, num_rows, num_cols):
    resized_filename = resize_image.resize_image_ocr(dir_of_file_full_path, filename, num_rows, num_cols)

    #OpenCV addition:
    # ----------------------
    img = cv2.imread(resized_filename,0)
    img = cv2.medianBlur(img,3)

    # Using adaptive methods:
    # ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    # th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #             cv2.THRESH_BINARY,11,2)
    # th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #             cv2.THRESH_BINARY,11,2)
    # w_vec = [0.3, 0.3, 0.4]
    # im = (th3*w_vec[0] + th2*w_vec[1] + th1*w_vec[2])/3

    (thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # ----------------------
    # astro = im_bw
    # astro = color.rgb2gray(im)

    deconvolved = im_bw
    arr = resized_filename.split('.')
    im_adaptive_filename = arr[0] + '_thresholded.' + arr[-1]
    imsave(im_adaptive_filename, deconvolved)
    return im_adaptive_filename

def ocr_image(dir_full_path, filename, num_rows, num_cols):
    im_post_process_filepath = image_processing(dir_full_path, filename, num_rows, num_cols)
    im = Image.open(im_post_process_filepath)

    # text = PT.image_to_string(im, lang=lang_english, boxes=False)
    text = PT.image_to_string(im, lang=args.lang, boxes=False)
    # text = PT.image_to_string(im, lang=lang_arabic, boxes=False)
    print '========ouput========'
    print text
    return text

def ocr_image_cstringio(cstring_object, num_rows, num_cols):
    lang_eng = 'eng'
    im = image_processing_cstringio(cstring_object, num_rows, num_cols)
    text = PT.image_to_string(im, lang_eng, boxes=False)
    # text = PT.image_to_string(im, lang=lang_arabic, boxes=False)
    print '========ouput========'
    print text
    return text

if __name__ == '__main__':
    desc='OCR image'
    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=desc)
    default_file = 'receipt_class_images/images1.jpeg'
    default_file_2 = 'receipt_class_images/receipt2.jpg'
    default_file_3 = 'receipt_class_images/textArea01.png'
    default_file_4_ara = 'receipt_class_images/arabic1.png'

    lang_english = 'eng'
    lang_arabic = 'ara'
    default_resize = 900
    # parser.add_argument("--file_path", help="Path to image file to OCR", default=default_file_4_ara)
    parser.add_argument("--file_path", help="Path to image file to OCR", default=default_file_3)
    parser.add_argument("--lang", help="Language of text in image", default=lang_english)
    parser.add_argument("--resize_dim", help="Dimensions to give resized image", default=default_resize)
    args = parser.parse_args()

    # filename = 'receipt2.jpg'
    # filename = 'arabic3.jpg'
    # filename = 'img7.jpg'

    file_path = args.file_path
    dir_full_path = os.path.dirname(os.path.abspath(file_path))
    filename = file_path.rsplit('/', 1)[-1]
    num_rows = num_cols = int(args.resize_dim)
    # num_rows = 555
    # num_cols = 110
    im_post_process_filepath = image_processing(dir_full_path, filename, num_rows, num_cols)
    im = Image.open(im_post_process_filepath)

    # text = PT.image_to_string(im, lang=lang_english, boxes=False)
    text = PT.image_to_string(im, lang=args.lang, boxes=False)
    # text = PT.image_to_string(im, lang=lang_arabic, boxes=False)
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

