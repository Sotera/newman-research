import os
import argparse

def combine_part_files(source_dir, destination_file_path):
    fd = open(destination_file_path, 'w')
    for subdir, dirs, files in os.walk(source_dir):
        # num_samples = len(files)
        # pos_slash = subdir.rfind('/')
        # # dir_of_file_full_path = subdir[:pos_slash+1]
        # curr_dir = subdir[pos_slash+1::]
        # # print 'Current dir is %s' % curr_dir
        # if curr_dir == 'edited':
        #     continue
        for file_i in files:
            if 'part' in file_i and 'crc' not in file_i:
                write_all_lines_to_file(os.path.join(subdir,file_i), fd)

def write_all_lines_to_file(source_file, dest_file_fd):
    fd_src = open(source_file, 'r')
    for line in fd_src:
        dest_file_fd.write(line)


if __name__ == '__main__':
    desc='Find similar documents to query'
    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=desc)

    # USER input parser:
    help_str = "Input directory name. If not in same dir as .py script, specify full path of input file."
    parser.add_argument("input_dir_name", help=help_str)
    parser.add_argument("output_part_file", help="Enter doc part file name")
    args = parser.parse_args()
    combine_part_files(args.input_dir_name, args.output_part_file)