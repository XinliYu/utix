import utix.mpex as mpex
import utix.ioex as ioex
import utix.pathex as paex

if __name__ == '__main__':
    mpex.freeze_support()
    src = paex.get_files_by_pattern(dir_or_dirs='tmp2', pattern='*.csv')
    trg = r'./tmp3'
    paex.ensure_dir_existence(trg, clear_dir=True)
    mpex.mp_chunk_file(src, trg, chunk_size=33, num_p=4)
    lines1 = sorted(ioex.read_all_lines_from_all_files(src))
    lines2 = sorted(ioex.iter_all_lines_from_all_sub_dirs(input_path=trg, pattern='*.csv'))
    print(lines1 == lines2)
