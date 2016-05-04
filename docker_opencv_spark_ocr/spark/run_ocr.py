import os
import sys
# import exifread
import base64
import cStringIO
import argparse
import json
from pyspark import SparkContext, SparkConf
# OCR script
import ocr_opencv

def OCR(img):
    # image resize defaults:
    num_rows = num_cols = 1000
    txt = ocr_opencv.ocr_image_cstringio(img, num_rows, num_cols)
    return txt

def process_image(byte64_jpeg, filename):
    attch_data = str(base64.b64decode(byte64_jpeg))
    buf = cStringIO.StringIO(attch_data)
    txt = OCR(buf)
    return txt

def process_attachment(attachment):
    if attachment:
        attachment["image_analytics"] = {}
        # replace the below 2 lines - allow for ocr of all image types (gif, tiff, bmp, etc.)
        if "contents64" in attachment and "extension" in attachment \
                and (attachment["extension"] == ".jpg" or attachment["extension"] == ".jpeg" or attachment["extension"] == ".png"):
            # text = process_image(attachment["contents64"], attachment["filename"])
            try:
                text = process_image(attachment["contents64"], attachment["filename"])
                if text:
                    attachment["image_analytics"]["ocr_output"] = text
            except:
                print "FAILED:  File: %s, text from ocr:%s"%(attachment["filename"],'-1NotHere')
                print "ERROR:", sys.exc_info()[0]
    return attachment

def process_email(email):
    for attachment in email["attachments"]:
        process_attachment(attachment)
    return email

def dump(x):
    return json.dumps(x)

# def read_json_file():
#     with open('cresentful.json') as data_file:
#         data = json.load(data_file)
#         return data

if __name__ == '__main__':
    desc='exif extract'
    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=desc)

    parser.add_argument("input_attachment_content_path", help="input attachments")
    parser.add_argument("output_attachments_gps", help="output attachments enriched with geo / gps tagged images when applicable")
    args = parser.parse_args()

    conf = SparkConf().setAppName("Newman ocr images")
    sc = SparkContext(conf=conf)
    rdd_emails = sc.textFile(args.input_attachment_content_path).map(lambda x: json.loads(x))
    rdd_emails.map(lambda email : process_email(email)).map(dump).saveAsTextFile(args.output_attachments_gps)

