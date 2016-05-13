#!/usr/bin/env python
from PIL import Image, ImageChops
import os
import numpy

# import StringIO
# import cStringIO
# import io
# from skimage.io import imread


def resize_image(dir_of_file_full_path, filename, num_rows, num_cols):
    image_file_full_path = dir_of_file_full_path + '/' + filename
    F_IN = image_file_full_path
    # F_IN = "/path/to/image_in.jpg"
    arr = filename.split('.')
    # F_OUT = dir_of_file_full_path + '/edited/' + arr[0] + '_edited.' + arr[-1]
    dir = dir_of_file_full_path + '/edited'
    if not os.path.exists(dir):
        os.makedirs(dir)
    F_OUT = dir_of_file_full_path + '/edited/' + arr[0] + '_edited.' + arr[-1]
    print(F_OUT)
    size = (num_rows,num_cols)

    image = Image.open(F_IN)
    image.thumbnail(size, Image.ANTIALIAS)
    image_size = image.size

    thumb = image.crop( (0, 0, size[0], size[1]) )

    offset_x = max( (size[0] - image_size[0]) / 2, 0 )
    offset_y = max( (size[1] - image_size[1]) / 2, 0 )

    thumb = ImageChops.offset(thumb, offset_x, offset_y)
    # thumb2 = image.resize((num_rows, num_cols))
    thumb.save(F_OUT)
    edited_filename = F_OUT
    return edited_filename

def resize_image_recursive_file_search(dir_of_file_full_path, filename, num_rows, num_cols, upstream_of_edited_dir_path):
    image_file_full_path = dir_of_file_full_path + '/' + filename
    F_IN = image_file_full_path
    # F_IN = "/path/to/image_in.jpg"
    arr = filename.split('.')
    # F_OUT = dir_of_file_full_path + '/edited/' + arr[0] + '_edited.' + arr[-1]
    dir = upstream_of_edited_dir_path + '/edited'
    if not os.path.exists(dir):
        os.makedirs(dir)
    F_OUT = upstream_of_edited_dir_path + '/edited/' + arr[0] + '_edited.' + arr[-1]
    i = 0
    while os.path.isfile(F_OUT):
    # if os.path.isfile(F_OUT):
        F_OUT = upstream_of_edited_dir_path + '/edited/' + arr[0] + str(i) + '.' + arr[-1]
        i += 1
    print(F_OUT)
    size = (num_rows,num_cols)

    image = Image.open(F_IN)
    image.thumbnail(size, Image.ANTIALIAS)
    image_size = image.size

    thumb = image.crop( (0, 0, size[0], size[1]) )

    offset_x = max( (size[0] - image_size[0]) / 2, 0 )
    offset_y = max( (size[1] - image_size[1]) / 2, 0 )

    thumb = ImageChops.offset(thumb, offset_x, offset_y)
    # thumb2 = image.resize((num_rows, num_cols))
    thumb.save(F_OUT)
    edited_filename = F_OUT
    return edited_filename

def resize_image_ocr(dir_of_file_full_path, filename, num_rows, num_cols):
    image_file_full_path = dir_of_file_full_path + '/' + filename
    F_IN = image_file_full_path
    # F_IN = "/path/to/image_in.jpg"
    arr = filename.split('.')
    # F_OUT = dir_of_file_full_path + '/edited/' + arr[0] + '_edited.' + arr[-1]
    dir = dir_of_file_full_path + '/ocr'
    if not os.path.exists(dir):
        os.makedirs(dir)
    F_OUT = dir_of_file_full_path + '/ocr/' + arr[0] + '_ocr.' + arr[-1]
    print(F_OUT)
    size = (num_rows,num_cols)

    image = Image.open(F_IN)
    image.thumbnail(size, Image.ANTIALIAS)
    thumb2 = image.resize((num_rows, num_cols))
    thumb2.save(F_OUT)
    edited_filename = F_OUT
    return edited_filename


def resize_image_ocr_cstringio(cstring_object, num_rows, num_cols):
    F_IN = cstring_object
    # F_OUT = io.StringIO()
    size = (num_rows,num_cols)

    image = Image.open(F_IN)
    image.thumbnail(size, Image.ANTIALIAS)
    thumb2 = image.resize((num_rows, num_cols))
    # format_im = u'PNG'
    #thumb2.save(F_OUT, format_im)
    open_cv_image = numpy.array(thumb2) 
    # Convert RGB to BGR 
    #open_cv_image = open_cv_image[:, :, ::-1].copy()
    open_cv_image = open_cv_image[:,:].copy()
    return open_cv_image
    # thumb_str = io.StringIO(unicode(thumb2.tobytes(), 'utf16', errors='ignore'),)
    # #F_OUT.write(unicode(thumb_str))
    # return F_OUT

if __name__ == '__main__':
    dir_of_file_full_path = os.getcwd()
    filename = 'dreadnought.jpg'
    num_rows = num_cols = 100
    resized_filename = resize_image(dir_of_file_full_path, filename, num_rows, num_cols)
