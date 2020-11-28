from functools import partial

import utix.mpex as mpex
import utix.ioex as ioex
import utix.pathex as paex
import utix.strex as strex
import utix.timex as timex

if __name__ == '__main__':
    src = r'E:\Data\dfsv1f\source_data\main_data\features_2020223\slot_value_features.json'
    trg = r'./tmp1.txt'

    timex.tic('with mp iter')
    it = mpex.mp_read(data_iter=[src],
                      provider=mpex.MPProvider(create_iterator=partial(ioex.iter_all_lines_from_all_files, use_tqdm=True), chunk_size=1000),
                      producer=mpex.MPTarget(target=strex.hash_str, pass_each_data_item=True, pass_pid=False), )
    hashes1 = list(it)
    timex.toc()

    timex.tic('no mp iter')
    hashes2 = [strex.hash_str(x) for x in ioex.iter_all_lines_from_all_files(src)]
    timex.toc()

    print(hashes1.sort() == hashes2.sort())