import faiss
import numpy as np
from os import path

from tqdm import tqdm

import utix.general as gx
import utix.ioex as ioex
import utix.pathex as pathex
import utix.npex as npex
from utix.timex import tic, toc
import utix.datau as datau
import os


def get_embeds_from_ids(index, id_arr, default_embed_id=0):
    return np.concatenate(
        [
            # ! **faiss** may return -1; for convenience of embedding reconstruction, we could replace -1 by a default id like 0.
            np.concatenate([index.reconstruct(int(top_id) if top_id != -1 else default_embed_id)[np.newaxis, :] for top_id in top_ids])[np.newaxis, :]
            for top_ids in id_arr
        ]
    )

    # top_embeds = []
    # curr_top_embeds = []
    # for top_ids in top_id_arr:
    #     for top_id in top_ids:
    #         curr_top_embeds.append(index.reconstruct(int(top_id))[np.newaxis, :])
    #     top_embeds.append(np.concatenate(curr_top_embeds)[np.newaxis, :])
    #     curr_top_embeds.clear()
    # return np.concatenate(top_embeds)


def iter_embeds(embeds_path, format='labeled_numpy', read_embeds=True, read_labels=True, use_tqdm: bool = True, tqdm_msg: str = None, sort=True, **kwargs):
    if format == 'labeled_numpy':
        yield from datau.iter_labeled_numpy_batch(dir_path=embeds_path,
                                                  np_file_pattern=kwargs.get('np_file_pattern'),
                                                  label_file_pattern=kwargs.get('label_file_pattern'),
                                                  read_embeds=read_embeds,
                                                  read_labels=read_labels,
                                                  use_tqdm=use_tqdm,
                                                  tqdm_msg=tqdm_msg,
                                                  sort=sort)
    else:
        raise NotImplementedError('the embedding file format is not supported')


def load_embeds(embeds_path, format='labeled_numpy', read_embeds=True, read_labels=True, use_tqdm: bool = True, tqdm_msg: str = None, sort=True, **kwargs):
    if tqdm_msg is None:
        if read_embeds and read_labels:
            tqdm_msg = f'loading embeds with labels at {embeds_path}'
        elif read_embeds:
            tqdm_msg = f'loading embeds at {embeds_path}'
        elif read_labels:
            tqdm_msg = f'loading labels at {embeds_path}'
        else:
            return
    embeds_it = iter_embeds(embeds_path=embeds_path, format=format, read_embeds=read_embeds, read_labels=read_labels, use_tqdm=use_tqdm, tqdm_msg=tqdm_msg, sort=sort, **kwargs)
    tic('Load embeddings ...')
    if format == 'labeled_numpy':
        output = list(embeds_it)
        if read_embeds and read_labels:
            embeds_list, labels_list = gx.unzip(output)
            gx.hprint_message(f"Total number of embedding batches at {embeds_path} to index", len(embeds_list))
            output = (embeds_list, labels_list)
        elif read_embeds or read_labels:
            gx.hprint_message(f"Total number of embedding batches at {embeds_path} to index", len(output))
    else:
        raise NotImplementedError('the embedding file format is not supported')

    toc(msg=f'Done!')
    return output


def build_index(embeds_path, output_path, num_clusters=65536, use_gpu=False, train_ratio=1.0, embeds_format='labeled_numpy', sort=True, **kwargs):
    # embeds_file_paths = pathex.get_sorted_files_from_all_sub_dirs__(embeds_path, full_path=True)

    # gx.write_all_lines(path.join(output_dir, f'{EMBEDS_INDEX_FILE_PREFIX}_{embeds_key}_files.txt'), embeds_file_paths)
    # text_file_path = path.join(output_dir, f'{EMBEDS_INDEX_FILE_PREFIX}_{embeds_key}.txt')
    # index_file_path = path.join(output_dir, f'{EMBEDS_INDEX_FILE_PREFIX}_{embeds_key}.idx')

    embeds_list, _ = load_embeds(embeds_path=embeds_path, format=embeds_format, sort=sort, **kwargs)

    tic('Initializing index ...')
    if not num_clusters:
        num_clusters = len(embeds_list) // 100
    index = faiss.index_factory(embeds_list[0].shape[-1], f"IVF{num_clusters},Flat", faiss.METRIC_INNER_PRODUCT)
    if use_gpu:
        index = faiss.index_cpu_to_all_gpus(index)

    tic('Concatenating embeddings ...')
    if 0 < train_ratio < 1:
        gx.hprint_message(f"will sample subset for training with ratio {train_ratio}...")

    all_embeds = np.concatenate(embeds_list if train_ratio == 1 else list(gx.sampled_iter(embeds_list, train_ratio)))
    toc(msg=f'Initialization done!')

    tic(f'Training embeddings of shape {all_embeds.shape} ...')
    index.train(all_embeds)
    if use_gpu:
        index = faiss.index_gpu_to_cpu(index)
    toc(msg='Index training done!')

    tic('Add embeddings to index ...')
    del all_embeds
    embed_index_start = 0

    for embeds in tqdm(embeds_list):
        embed_count = embeds.shape[0]
        index.add_with_ids(embeds, np.arange(embed_index_start, embed_index_start + embed_count))
        embed_index_start += embed_count

    # with open(text_file_path, 'w+') as wf:
    #     for embeds, batch in embeds_iter(embeds_file_paths=embeds_file_paths, embeds_key=embeds_key, sample_file=sample_file, sample_ratio=train_ratio, embeds_idx=embeds_idx, use_tqdm=True, yield_batch=True):
    #         write_all_lines_to_stream(wf=wf, iterable=batch[embeds_txt_key], use_tqdm=False)
    #         embed_count = embeds.shape[0]
    #         index.add_with_ids(embeds, np.arange(embed_index_start, embed_index_start + embed_count))
    #         embed_index_start += embed_count

    if path.exists(output_path):
        os.remove(output_path)
    gx.hprint_message('saving indexed embeddings to', output_path)
    faiss.write_index(index, output_path)
    toc(msg='Indexing done!')
    return index


def check_index_recon(embeds_path, index_or_index_path, embeds_format='labeled_numpy', sort=True, **kwargs):
    index = faiss.read_index(index_or_index_path) if isinstance(index_or_index_path, str) else index_or_index_path
    faiss.downcast_index(index).make_direct_map()
    embeds_list, _ = load_embeds(embeds_path=embeds_path, format=embeds_format, sort=sort, **kwargs)

    # tic("Gathering targets ...")
    # all_tgt_embeds = []
    # for file_path in Tqdm.tqdm(embeds_paths):
    #     embeds_group, batch_group = pickle_load(file_path)
    #     for embeds, batch in zip(embeds_group[embeds_key], batch_group):
    #         all_tgt_embeds.append(embeds[embeds_idx])
    #
    # toc("Done!")

    tic("Checking embedding reconstruction difference ...")
    all_embeds = np.concatenate(embeds_list)
    all_embeds_recon = index.reconstruct_n(0, len(all_embeds))
    embeds_diff = np.linalg.norm(all_embeds - all_embeds_recon)
    toc("Passed embedding reconstruction difference check.") \
        if embeds_diff == 0 else toc(f"Embedding reconstruction difference: {embeds_diff}.")


def search_index(query_embeds_path, target_labels_path, index_or_index_path, output_path, top, use_gpu=False, query_embeds_format='labeled_numpy', query_sort=True, target_sort=True, **kwargs):
    target_labels = sum(load_embeds(embeds_path=target_labels_path, read_embeds=False, sort=query_sort, **kwargs), [])
    index = faiss.read_index(index_or_index_path) if isinstance(index_or_index_path, str) else index_or_index_path
    if use_gpu:
        index = faiss.index_cpu_to_all_gpus(index)

    def _jobj_iter():
        for query_embeds, queries in iter_embeds(embeds_path=query_embeds_path, format=query_embeds_format, use_tqdm=True, tqdm_msg='searching index', sort=target_sort, **kwargs):
            retrieved_scores, retrieved_indices = index.search(query_embeds, top)
            for query, idxes, scores in zip(queries, retrieved_indices, retrieved_scores):
                yield {'query': query, 'retrievals': [target_labels[idx] for idx in idxes], 'scores': [str(score) for score in scores]}
    ioex.write_json_objs(_jobj_iter(), output_path=output_path, use_tqdm=True)


# def search_index(src_embeds_dir, tgt_embeds_dir, index_dir, eval_output_dir,
#                  query_embeds_key, query_txt_key,
#                  target_embeds_key, target_txt_key,
#                  embeds_idx=0, retrieval_top=20, output_top=20,
#                  ranking_similarity='cos', sanity_check=False):
#     embeds_file_pattern = f'*{OUTPUT_FILE_EXTENSION_NAME}'
#     src_embeds_file_paths = get_files_by_pattern(dir_or_dirs=src_embeds_dir, pattern=embeds_file_pattern, sort=True)
#     tgt_embeds_file_paths = get_files_by_pattern(dir_or_dirs=tgt_embeds_dir, pattern=embeds_file_pattern, sort=True)
#     ensure_dir_existence(eval_output_dir)
#
#     index_file_path = path.join(index_dir, f'{EMBEDS_INDEX_FILE_PREFIX}_{target_embeds_key}.idx')
#     index = faiss.read_index(index_file_path)
#     faiss.downcast_index(index).make_direct_map()
#     time_stamp = int(time())
#     logger = get_logger(name=f'log_{time_stamp}', log_dir_path=eval_output_dir)
#
#     tic("Gathering targets ...")
#     all_tgt_txt, all_tgt_domain, all_tgt_intent, all_tgt_entities, all_tgt_hypos = [], [], [], [], []
#     all_tgt_src_txt, all_tgt_src_domain, all_tgt_src_intent, all_tgt_src_entities, all_tgt_src_hypos = [], [], [], [], []
#     if sanity_check:
#         all_tgt_embeds = []
#
#     for embeds, batch in embeds_iter(embeds_file_paths=tgt_embeds_file_paths, embeds_key=target_embeds_key, embeds_idx=embeds_idx, use_tqdm=True, yield_batch=True):
#         all_tgt_txt.extend(batch[target_txt_key])  # TODO generalize these in the future
#         all_tgt_domain.extend(batch['tgt_domain'])
#         all_tgt_intent.extend(batch['tgt_intent'])
#         all_tgt_entities.extend(batch['tgt_entities'])
#         all_tgt_hypos.extend(batch['tgt_hypo'])
#
#         all_tgt_src_txt.extend(batch['src_txt'])  # TODO generalize these in the future
#         all_tgt_src_domain.extend(batch['src_domain'])
#         all_tgt_src_intent.extend(batch['src_intent'])
#         all_tgt_src_entities.extend(batch['src_entities'])
#         all_tgt_src_hypos.extend(batch['src_hypo'])
#
#         if sanity_check:
#             all_tgt_embeds.append(embeds)
#
#     tgt_txt_dict = index_dict(all_tgt_txt)
#     tgt_hypo_dict = index_dict(all_tgt_hypos)
#     toc("All targets gathered!")
#
#     if sanity_check:
#         hprint_message("Checking target embedding reconstruction difference.")
#         all_tgt_embeds = np.concatenate(all_tgt_embeds)
#         all_tgt_embeds_recon = index.reconstruct_n(0, len(all_tgt_embeds))
#         tgt_embeds_diff = np.linalg.norm(all_tgt_embeds - all_tgt_embeds_recon)
#         toc("Passed target embedding reconstruction difference check.") \
#             if tgt_embeds_diff == 0 else toc(f"Target embedding reconstruction difference: {tgt_embeds_diff}.")
#         score_histogram_bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
#
#     def _output_bath_iter():
#         label = 0
#         if sanity_check:
#             total_num_dot_largest_at_top = total_num_cos_largest_at_top = total_num_norm_largest_at_top = total_top_scores = total_label_scores = 0
#         hprint_message("Test begins ...")
#
#         search_outputs, src_batch_group = [], []
#         file_idx = 0
#
#         for src_embeds, batch, batch_file_path, batch_idx in embeds_iter(embeds_file_paths=src_embeds_file_paths, embeds_key=query_embeds_key, embeds_idx=embeds_idx, use_tqdm=True, yield_batch=True, yield_batch_info=True):
#             if batch_idx == 0:
#                 logger.info(f"Process query file {file_idx} at path {batch_file_path} ...")
#                 if len(src_batch_group) != 0:
#                     yield search_outputs, src_batch_group
#                     search_outputs.clear()
#                     src_batch_group.clear()
#                     file_idx += 1
#             logger.info(f"Process batch {batch_idx} ...")
#
#             # search for near-by embeddings
#             D, top_id_arr = index.search(src_embeds, retrieval_top)
#
#             # region label construction
#             batch_size = len(batch['src_txt'])
#             src_tgt_txts, src_tgt_domains, src_tgt_intents, src_tgt_entities = batch[target_txt_key], batch['tgt_domain'], batch['tgt_intent'], batch['tgt_entities']
#             labels = [0] * batch_size
#             for j in range(batch_size):
#                 src_tgt_txt = src_tgt_txts[j]
#                 if src_tgt_txt in tgt_txt_dict:
#                     labels[j] = tgt_txt_dict[src_tgt_txt]
#                 else:
#                     src_tgt_hypo = '|'.join((src_tgt_domains[j], src_tgt_intents[j], src_tgt_entities[j]))
#                     labels[j] = tgt_hypo_dict[src_tgt_hypo]
#             batch['tgt_labels'] = labels
#             # endregion
#
#             batch['tgt_txt'], batch['tgt_domain'], batch['tgt_intent'], batch['tgt_entities'], batch['tgt_hypo'] = all_tgt_txt, all_tgt_domain, all_tgt_intent, all_tgt_entities, all_tgt_hypos
#             batch['tgt_src_txt'], batch['tgt_src_domain'], batch['tgt_src_intent'], batch['tgt_src_entities'], batch['tgt_src_hypo'] = all_tgt_src_txt, all_tgt_src_domain, all_tgt_src_intent, all_tgt_src_entities, all_tgt_src_hypos
#
#             if ranking_similarity:
#                 top_embeds = get_embeds_from_ids(index, top_id_arr)
#                 src_embeds_tensor = torch.from_numpy(src_embeds).unsqueeze(dim=-2)
#                 top_tgt_embeds_tensor = torch.from_numpy(top_embeds)
#
#                 if sanity_check:
#                     dot_sim_scores = get_embedding_similarities(src_embeds_tensor, top_tgt_embeds_tensor, 'dot')
#                     num_dot_largest_at_top = count_largest_at_top(dot_sim_scores)
#                     cos_sim_scores = get_embedding_similarities(src_embeds_tensor, top_tgt_embeds_tensor, 'cos')
#                     num_cos_largest_at_top = count_largest_at_top(cos_sim_scores)
#                     norm_sim_scores = get_embedding_similarities(src_embeds_tensor, top_tgt_embeds_tensor, 'l2')
#                     num_norm_largest_at_top = count_largest_at_top(norm_sim_scores)
#                     logger.debug(f"actual top rate by dot product: {num_dot_largest_at_top / batch_size}")
#                     logger.debug(f"actual top rate by cosine similarity: {num_cos_largest_at_top / batch_size}")
#                     logger.debug(f"actual top rate by 2-norm: {num_norm_largest_at_top / batch_size}")
#                     total_num_dot_largest_at_top += num_dot_largest_at_top
#                     total_num_cos_largest_at_top += num_cos_largest_at_top
#                     total_num_norm_largest_at_top += num_norm_largest_at_top
#                     if ranking_similarity == 'dot':
#                         ranking_scores = dot_sim_scores
#                     elif ranking_similarity == 'cos':
#                         ranking_scores = cos_sim_scores
#                     elif ranking_similarity == 'l2':
#                         ranking_scores = norm_sim_scores
#                 else:
#                     ranking_scores = get_embedding_similarities(src_embeds_tensor, top_tgt_embeds_tensor, ranking_similarity)
#
#                 ranking_scores = ranking_scores.numpy()
#                 sort_idxes = np.argsort(-ranking_scores, axis=-1)
#                 _all_rows = np.arange(batch_size)[:, None]
#                 D = ranking_scores = ranking_scores[_all_rows, sort_idxes]
#                 top_embeds = top_embeds[_all_rows, sort_idxes]
#                 top_id_arr = top_id_arr[_all_rows, sort_idxes]
#
#                 label_scores = None
#                 if sanity_check:
#                     tgt_labels = np.array(batch['tgt_labels'])
#                     label_embeds_original = all_tgt_embeds[tgt_labels]
#                     tgt_labels = tgt_labels[:, np.newaxis]
#                     label_embeds_from_index = get_embeds_from_ids(index, tgt_labels).squeeze()
#                     label_embedding_diff = np.linalg.norm(label_embeds_from_index - label_embeds_original)
#                     logger.debug("passed label embedding difference check") \
#                         if label_embedding_diff == 0 else logger.debug(f"label embedding difference: {label_embedding_diff}")
#
#                     label_embeds_tensor = torch.from_numpy(label_embeds_original).unsqueeze(dim=1)
#                     label_ranking_scores: np.ndarray = get_embedding_similarities(src_embeds_tensor, label_embeds_tensor, ranking_similarity).numpy()
#                     top_ranking_scores = ranking_scores[:, 0]
#                     logger.debug(f"top score histogram {np.histogram(top_ranking_scores, bins=score_histogram_bins)[0]}")
#                     logger.debug(f"label score histogram {np.histogram(label_ranking_scores.flatten(), bins=score_histogram_bins)[0]}")
#
#                     top_ranking_score_sum = float(np.sum(top_ranking_scores))
#                     label_ranking_score_sum = float(np.sum(label_ranking_scores))
#                     total_top_scores += top_ranking_score_sum
#                     total_label_scores += label_ranking_score_sum
#                     logger.debug(f"mean top score {top_ranking_score_sum / batch_size}")
#                     logger.debug(f"mean label score {label_ranking_score_sum / batch_size}")
#
#                     label_scores = label_ranking_scores.flatten()
#
#             search_outputs.append((D, top_id_arr, label_scores))
#             src_batch_group.append(batch)
#             label += batch_size
#
#         if len(src_batch_group) != 0:
#             yield search_outputs, src_batch_group
#             file_idx += 1
#
#         if sanity_check:
#             total_num_embeds = all_tgt_embeds.shape[0]
#             metrics['avg-top-score'] = total_top_scores / total_num_embeds
#             metrics['avg-label-score'] = total_label_scores / total_num_embeds
#             metrics['dot-sanity'] = total_num_dot_largest_at_top / total_num_embeds
#             metrics['cos-sanity'] = total_num_cos_largest_at_top / total_num_embeds
#             metrics['norm-sanity'] = total_num_norm_largest_at_top / total_num_embeds
#
#     eval_output_file_path = path.join(eval_output_dir, f'eval_{time_stamp}')
#     analyzer = TargetRankingAnalyzer(src_txt_field_name=query_txt_key, tgt_txt_field_name=target_txt_key, output_top_k=output_top)
#     metrics = {}
#     analyzer(output_file=eval_output_file_path, outputs_and_batches=_output_bath_iter(), metrics=metrics)
#     allen_util.dump_metrics(path.join(eval_output_dir, f'metrics_{time_stamp}.json'), metrics)
#     if sanity_check:
#         print(metrics)
#     toc("Experiment completed.")


if __name__ == '__main__':
    src_embeds_path = '/Users/zgchen/experiments/dfsv1f/results/main_data_features/train/rs0-dmr100B-e99-lr400-d0-msl12-b512-bcgs64/run_159081569951/embeds_1590815699/src'
    trg_embeds_path = '/Users/zgchen/experiments/dfsv1f/results/main_data_features/train/rs0-dmr100B-e99-lr400-d0-msl12-b512-bcgs64/run_159081569951/embeds_1590815699/trg'
    index_paht = './test.index'
    np_file_pattern = r'batch([0-9]+)\.np\.list.npy'
    label_file_pattern = 'batch{}.label.list'
    output_path = './test.txt'
    sort = 'index'
    build_index(embeds_path=trg_embeds_path,
                np_file_pattern=np_file_pattern,
                label_file_pattern=label_file_pattern,
                output_path=index_paht,
                num_clusters=None,
                sort=sort)
    check_index_recon(embeds_path=trg_embeds_path, index_or_index_path=index_paht, np_file_pattern=np_file_pattern, label_file_pattern=label_file_pattern, sort=sort)
    search_index(query_embeds_path=src_embeds_path,
                 target_labels_path=trg_embeds_path,
                 index_or_index_path=index_paht,
                 output_path=output_path,
                 top=25,
                 query_sort=sort,
                 target_sort=sort,
                 np_file_pattern=np_file_pattern,
                 label_file_pattern=label_file_pattern)
    pass
