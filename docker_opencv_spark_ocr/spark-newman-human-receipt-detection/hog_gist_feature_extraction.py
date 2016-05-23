from skimage.io import imread
import os
import cStringIO
import time
import argparse
import numpy as np
import pandas as pd
from skimage.io import imsave
import skimage.io as skimage_io
import json

# custom scripts:
import gabor_features
import resize_image
import base64

# OpenCV:
import cv2

# from pyspark import SparkContext, SparkConf
from pysparkling import Context


def image_processing_cstringio(cstring_object, num_rows, num_cols):
    resized_filename = resize_image.resize_image_ocr_cstringio(cstring_object, num_rows, num_cols)


    #OpenCV addition:
    # ----------------------
    img = cv2.imread(resized_filename,0)
    img = cv2.medianBlur(img,3)

    (thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    deconvolved = im_bw

    im_adaptive_filename = cStringIO.StringIO()
    imsave(im_adaptive_filename, deconvolved)
    return im_adaptive_filename


def resize_image_and_get_full_gabor_features(cstring_image_obj, num_rows, num_cols):
    resized_filename = resize_image.resize_image_cstringio(cstring_image_obj, num_rows, num_cols)
    # resized_filename = skimage.io.imread(cstring_image_obj)
    # resized_filename = Image.open(cstring_image_obj)
    # resized_filename = np.array(resized_filename)
    # resized_filename = resized_filename[:,:].copy()

    num_levels = 3
    num_orientations = 8
    gabor_features_vec = gabor_features.get_gabor_features_texture_classification(resized_filename, num_levels, num_orientations)
    print 'Dimension of gabor feature vec is %d' % gabor_features_vec.shape[1]
    # print type(gabor_features_vec)
    print gabor_features_vec.shape
    return gabor_features_vec

def get_hog_features(cstring_image_obj):
    # file_full_path = cstring_image_obj.read()
    bin_n = 16

    # img = Image.open(cstring_image_obj)
    # read_original = cv2.imread(img)
    # img = cv2.imread(file_full_path, 0)
    img = skimage_io.imread(cstring_image_obj)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    # quantizing binvalues in (0...16)
    bins = np.int32(bin_n*ang/(2*np.pi))

    # Divide to 4 sub-squares
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    print hist.shape
    return hist

def serialize_and_make_df(image_dir_path):
    data = []
    count = 0
    for subdir, dirs, files in os.walk(image_dir_path):
        for file_i in files:
            # if(count == 1):
            #     break
            file_full_path = os.path.join(subdir, file_i)
            f = open(file_full_path, 'r').read()
            # byte_data = bytearray(f)
            byte_data = base64.b64encode(f)
            # file_path = os.path.join(subdir, file_i)
            dict = {}
            dict["name"] = file_full_path
            dict["bytes"] = byte_data
            # line = '{\"name\":' + file_full_path
            # line += ',\"bytes\":' + str(byte_data) + '}'
            # print line
            print dict
            data.append(json.dumps(dict))
            print count
            count += 1
    df = pd.DataFrame(data)
    return df, data

def dump(x):
    return json.dumps(x)

def get_hog_and_gist_feats(bytes_str, num_rows, num_cols):
    # cstring_image_obj = cStringIO.StringIO(bytes_str)
    # cstring_image_obj = bytes_str
    img_data = str(base64.b64decode(bytes_str))
    # print img_data
    cstring_image_obj = cStringIO.StringIO(img_data)
    # img = Image.open(cstring_image_obj)
    # cstring_image_obj = img_data
    # try:
    hog_features_vec = get_hog_features(cstring_image_obj)
    gabor_features_vec = resize_image_and_get_full_gabor_features(cstring_image_obj, num_rows, num_cols)
    gabor_features_vec = gabor_features_vec.reshape(gabor_features_vec.shape[1], 1)
    supp_feat_vec = np.vstack((gabor_features_vec, hog_features_vec.reshape(hog_features_vec.shape[0], 1)))
    # print 'Full vec shape is:'
    # print supp_feat_vec.shape

    # supp_feat_vec = gabor_features_vec
    return supp_feat_vec.flatten().tolist()
    # except Exception as e:
    #     print e
    #     print("ERROR reading image file %s" % str(bytes_str))
    #     print
    #     return None

def get_hog_gist_feats_cstringio(cstring_image_obj, num_rows, num_cols):
    try:
        hog_features_vec = get_hog_features(cstring_image_obj)
        gabor_features_vec = resize_image_and_get_full_gabor_features(cstring_image_obj, num_rows, num_cols)
        gabor_features_vec = gabor_features_vec.reshape(gabor_features_vec.shape[1], 1)
        supp_feat_vec = np.vstack((gabor_features_vec, hog_features_vec.reshape(hog_features_vec.shape[0], 1)))
        # print 'Full vec shape is:'
        # print supp_feat_vec.shape

        # supp_feat_vec = gabor_features_vec
        return supp_feat_vec.flatten().tolist()
    except:
        return None


def get_features(image_json_txt_obj):
    print '---------------PROCESSING IMAGE----------------'
    # print image_json_txt_obj
    image_json_dict = json.loads(image_json_txt_obj)
    # print image_json_dict
    bytes_str = str(image_json_dict["bytes"])
    # print bytes_str
    num_rows = num_cols = 100
    features = get_hog_and_gist_feats(bytes_str, num_rows, num_cols)
    # features = 'features'
    image_json_dict["features"] = features
    # print features[1]
    return image_json_dict

def run_feature_extraction():
    start_time = time.time()
    desc='Feature Extraction for Images'
    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=desc)
    default_path = '/media/chris/cschulze_external_4tb/receipt_classifier_images/nonreceipts/train2014'
    # default_path = '/media/chris/cschulze_external_4tb/elliot_data/train_nonpill'
    # default_path = '/train_nonpill'

    parser.add_argument("--input_dir", help="input directory", default=default_path)
    parser.add_argument("--output", help="output file", default='image_features')
    args = parser.parse_args()
# serialize and put all images in rdd:
# use json schema:
#     "image_name": "",
#     "bytes": ""
#     "features": "array[]"
    image_dir_path = args.input_dir
    df, data_arr = serialize_and_make_df(image_dir_path)
    print df.head()
    print df.info()

    # df to df_cvs:
    csv_df_file = 'dataframe_csv_file.csv'
    json_df_file = 'dataframe_csv_file.json'
    df.to_csv(csv_df_file, header=False, index=False)
    # df.to_json(json_df_file)

    # rdd from df_csv
    # pysparkling:
    sc = Context()

    # pyspark:
    # conf = SparkConf().setAppName("HOG and GIST ETL")
    # sc = SparkContext(conf=conf)

    # rdd = sc.textFile(json_df_file)
    num_parts = 4
    rdd = sc.parallelize(data_arr, num_parts)
    # submit image rdd to processing
    rdd_features = rdd.map(get_features).coalesce(1)
    # save as txt file:
    rdd_features.map(dump).saveAsTextFile(args.output)
    print "------------------ %f minutes elapsed ------------------------" % ((time.time() - start_time)/60.0)


if __name__ == '__main__':
    run_feature_extraction()
