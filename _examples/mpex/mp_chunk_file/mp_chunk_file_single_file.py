import utix.mpex as mpex
import utix.ioex as ioex

if __name__ == '__main__':
    mpex.freeze_support()
    src_file = r'E:\Data\dfsv1f\datasets\sanity_check1\train\train.csv'
    trg = r'./tmp2'
    mpex.mp_chunk_file(src_file, trg, chunk_size=100, num_p=4)
    lines1 = sorted(ioex.read_all_lines(src_file))
    lines2 = sorted(ioex.iter_all_lines_from_all_sub_dirs(input_path=trg, pattern='*.csv'))
    print(lines1 == lines2)
