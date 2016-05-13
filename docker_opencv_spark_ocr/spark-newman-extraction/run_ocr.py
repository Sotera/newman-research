import os
import sys
import base64
import cStringIO
import argparse
import json
from pyspark import SparkContext, SparkConf

def process_partition(email_iter):
    sys.path.append(".")
    import ocr_opencv

    # Add supported mime types here
    _OCR_CONTENT_TYPES = ["image/jpeg", "image/png"]

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


    for email in email_iter:
        attachment_results = []
        for attachment in email["attachments"]:

            if attachment:
                #TODO replace the below 2 lines - allow for ocr of all image types (gif, tiff, bmp, etc.)
                #replaced extension with "content_type"  which is the MIME type entry from the server -- this will solve the problem where the attachment extensions are sometimes not provided
                if "contents64" in attachment and "content_type" in attachment and attachment["content_type"].lower() in _OCR_CONTENT_TYPES:
                    # text = process_image(attachment["contents64"], attachment["filename"])
                    try:
                        text = process_image(attachment["contents64"], attachment["filename"])
                        if text.strip():
                            attachment_results.append({"guid" : attachment["guid"], "image_analytics": {"ocr_output" : text }})

                    except Exception as e:
                        print "FAILED:  File: {0}, text from ocr:{1}, {2}".format(attachment["filename"],'-1NotHere',e)
                        print "ERROR:", sys.exc_info()[0]

        # Only need to output if there were ocr results
        if attachment_results:
            yield (email["id"], attachment_results)

def dump_tab_delimitted(x):
    return x[0]+"\t"+json.dumps(x[1])

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
    rdd_emails.mapPartitions(lambda email_iter : process_partition(email_iter)).map(dump_tab_delimitted).saveAsTextFile(args.output_attachments_gps)

