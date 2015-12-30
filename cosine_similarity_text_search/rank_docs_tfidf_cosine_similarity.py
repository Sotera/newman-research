import numpy as np
import time
import os
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy.linalg as LA
import json

# To find document similarity, the vector space model is used, with tfidf as features to represent
# each doc:

def rank_docs(input_list_file, docs_part_file, top_N):
    # get text from each doc and append to doc vec
    doc_vec = get_doc_vec(input_list_file, docs_part_file)

    # get tf-idf features of doc vec
    tf_idf_mat = get_tfidf_matrix(doc_vec)

    # get vector of value of cosine(theta), where theta is angle between input list tfidf features
    # and each doc tfidf features
    cos_similarity_vec = get_cosine_similarity_vec(tf_idf_mat)

    # print rankings of docs according to input list and write found relevant docs in decending order
    # to 'cos_sim_ranking_output_file' file:
    print_rankings_and_write_to_file(cos_similarity_vec, doc_vec, top_N)

def get_cosine_similarity_vec(tf_idf_mat):
    query_vec = tf_idf_mat[0,::]
    num_elem_in_mat = tf_idf_mat.shape[0]
    cos_sim_vec_elem_num = num_elem_in_mat - 1
    cos_sim_vec = np.zeros((cos_sim_vec_elem_num, 1))
    norm_query_vec = LA.norm(query_vec.todense())

    for i in range(cos_sim_vec_elem_num):
        cos_sim_vec[i, 0] = get_cosine_similarity(query_vec, tf_idf_mat[i+1, :], norm_query_vec)
    return cos_sim_vec

def get_cosine_similarity(vec_1, vec_2, norm_vec_1):
    # Returns -2.0 or between -1.0 and 1.0
    # represent empty document with -2.0 as cosine similarity
    norm_vec_2 = LA.norm(vec_2.todense())
    if norm_vec_2 == 0:
        return -2.0
    dot_prod = vec_1.dot(vec_2.T)[0,0]
    cos_theta = dot_prod/float(norm_vec_1*norm_vec_2)
    return cos_theta

def get_top_results_indices(vec, N):
    order = sorted(range(len(vec)), key=lambda i:vec[i])
    return list(reversed(order[-N:]))

def print_rankings_and_write_to_file(cos_similarity_vec, doc_vec, top_N):
    order = get_top_results_indices(cos_similarity_vec, top_N)
    cos_sim_ranking_output_file = 'cos_sim_ranking_output_file'
    fd = open(cos_sim_ranking_output_file, 'w')
    j = 1
    for i in order:
        relevant_text_rank_str = 'Relevant text rank = %s' % j
        print relevant_text_rank_str
        print "Cosine similarity is %s" % cos_similarity_vec[i]
        print "Text body:"
        print doc_vec[i + 1]
        print "\n"
        fd.write(relevant_text_rank_str+'\n')
        fd.write(doc_vec[i + 1])
        fd.write('\n\n')
        j += 1

def get_doc_vec(input_list_file, docs_part_file):
    num_elements = 1 + get_num_docs(docs_part_file)
    doc_vec = np.empty([num_elements,], dtype=object)

    # get input list of search terms and append to doc vec
    input_list_str = get_str_from_input_file(input_list_file)

    # print input_list_str
    doc_vec[0] = input_list_str
    i = 1
    fd = open(docs_part_file, 'r')
    for line in fd:
        doc = json.loads(''.join(line))
        body = doc["body"]
        body = re.sub(r'[^\x00-\x7F]',' ', body)
        body = body.replace("[:newline:]", "           ")
        body = body.replace('\n','')
        body = body.replace('\t','')
        body = body.replace('\"','')
        body = body.replace('>','')
        body = body.replace('>>','')
        body = body.replace('<','')
        doc_vec[i] = body
        i += 1
    return doc_vec

def get_tfidf_matrix(doc_vec):
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
    X = vectorizer.fit_transform(doc_vec)
    return X

def get_num_docs(docs_part_file):
    fd = open(docs_part_file, 'r')
    count = 0
    for line in fd:
        count += 1
    return count

# Used to make single string from input query file:
# Format of input query file is a document containing lines of words.
# Each line can have a single word or many words.
# If implementer wants to use a user string query from command prompt, then
# have this function return the user input as a string
def get_str_from_input_file(input_list_file):
    with open(input_list_file, 'r') as fd:
        input_str = fd.read().replace('\n', ' ')
    return input_str

if __name__ == '__main__':
    start_time = time.time()
    curr_path = os.getcwd()

    desc='Find similar documents to query'
    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=desc)

    # Client input parser:
    help_str = "Input file name. If not in same dir as .py script, specify full path of input file."
    parser.add_argument("input_file_name", help=help_str)
    parser.add_argument("docs_part_file", help="Enter doc part file name")
    parser.add_argument("--top_N", help="Enter desired number of top results from full similarity measure list", default=10)
    args = parser.parse_args()
    rank_docs(args.input_file_name, args.docs_part_file, int(args.top_N))
    print '-----elapsed minutes = %s -----' % ((time.time() - start_time)/60.0)

    # Debugging:
    # temp_input_file_name = 'input_list.txt'
    # temp_docs_part_file = 'doc_part_files/part-00000'
    # top_N = 10
    # rank_docs(temp_input_file_name, temp_docs_part_file, top_N)
    # print '-----elapsed minutes = %s -----' % ((time.time() - start_time)/60.0)
