import utix.mpex as mpex
import utix.ioex as ioex

if __name__ == '__main__':
    mpex.freeze_support()
    src_file = r'/Users/zgchen/experiments/dfsv1f/source_data/main_data/augmented_data_version_rephrase_nlu_shuf_debug.csv'
    lines1 = ioex.read_all_lines(src_file)
    lines2 = mpex.mp_read_from_files(num_p=4, input_path=src_file, chunk_size=1000)

    print(lines1 == lines2)

    lengths1 = tuple(len(x) for x in lines1)
    lengths2 = mpex.mp_read_from_files(num_p=4, target=mpex.MPTarget(target=len, common_func=True, data_from_files=True), input_path=src_file, chunk_size=1000)
    print(lengths1 == lengths2)