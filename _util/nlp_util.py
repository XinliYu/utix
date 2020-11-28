import shutil
import warnings
from enum import IntEnum
from itertools import islice
from os import path
from typing import Union, Iterator, Callable, List, Mapping

import numpy as np
from nltk import ngrams

from utix.dictex import kvswap, sort_by_values
from utix._util.io_ext import write_dict_as_text, read_dict_from_text
from utix.pathex import ensure_dir_existence
from utix.dictex import xfdict, SlotsDict
from utix.general import xmean_, iterable, unzip, hprint_message
from utix.listex import slide_window__, find_sub_list, seg_tags, find_first_larger_than


class TruncationMode(IntEnum):
    NoTruncation = 0
    HardTruncation = 1
    WholeWord = 2
    WholeWord_AllowExceedingWindowSize = 3


class TokenIndexResult(SlotsDict):
    __slots__ = ('tokens', 'offsets', 'clause_ids', 'head_len', 'tail_len', 'long_sequence')

    def __init__(self, tokens, offsets, clause_ids, head_len, tail_len, long_sequence):
        self.tokens, self.offsets, self.clause_ids, self.head_len, self.tail_len, self.long_sequence = tokens, offsets, clause_ids, head_len, tail_len, long_sequence


def rouge_n(hypothesis: Union[str, Iterator], reference: Union[str, Iterator], n: Union[int, Iterator[int]] = 2, out: dict = None, ignore_tokens=None):
    """
    Computes the rouge-n scores, which are the bag-of-ngrams precision/recall/F1-score between the hypothesis and the reference.
    First, we compute the bag-of-ngrams overlap between the hypothesis and the reference, then
    1) the rouge-n-p is the overlap size divided by the number of hypothesis n-grams;
    2) the rouge-n-r is the overlap size divided by the number of reference n-grams;
    3) and the rouge-n-f is the F1-score between the rouge-n-p and the rouge-n-r.

    :param hypothesis: the hypothesis text, or a list of hypothesis tokens.
    :param reference: the reference text, or a list of reference tokens.
    :param n: an integer or a list of integers; we will compute n-grams for both the hypothesis and the reference for each of the specified `n`.
    :param out: provides an optional dictionary; the computed scores will be written into this dictionary.
    :param ignore_tokens: ignore tokens specified in this parameter when computing the precision or recall.
    :return: a mapping contains the scores.
    """
    if isinstance(hypothesis, str):
        hypothesis = hypothesis.split()
    if isinstance(reference, str):
        reference = reference.split()

    if ignore_tokens is not None:
        if iterable(ignore_tokens):
            hypothesis = tuple(x for x in hypothesis if x not in ignore_tokens)
            reference = tuple(x for x in reference if x not in ignore_tokens)
        else:
            hypothesis = tuple(x for x in hypothesis if x != ignore_tokens)
            reference = tuple(x for x in reference if x != ignore_tokens)

    if out is None:
        out = {}

    def _rouge_n(n):
        hyp_ngrams = set(ngrams(hypothesis, n))
        ref_ngrams = set(ngrams(reference, n))
        hyp_len = len(hyp_ngrams)
        ref_len = len(ref_ngrams)
        overlap_len = len(hyp_ngrams.intersection(ref_ngrams))

        out[f"rouge_{n}_p"] = precision = 0.0 if hyp_len == 0 else overlap_len / hyp_len
        out[f"rouge_{n}_r"] = recall = 0.0 if ref_len == 0 else overlap_len / ref_len
        out[f"rouge_{n}_f"] = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

    if isinstance(n, int):
        _rouge_n(n)
    else:
        for _n in set(n):
            _rouge_n(_n)

    return xfdict(out)


def rouge_n_batch(hypotheses: Iterator[Union[str, Iterator]], references: Iterator[Union[str, Iterator]], n: Union[int, Iterator[int]] = 2, ignore_tokens=None, agg_func: Callable = xmean_):
    """
    Computes an aggregated rough-n score for a batch. See also `rouge_n` function.
    :param hypotheses: a list of hypothesis.
    :param references: a list of reference.
    :param n: an integer or a list of integers; we will compute n-grams for both pair of hypothesis and reference for each of the specified `n`.
    :param ignore_tokens: ignore tokens specified in this parameter when computing the precision or recall.
    :param agg_func: the aggregation function.
    :return: the batch rough-n scores.
    """
    return agg_func(rouge_n(hypothesis=hyp, reference=ref, n=n, ignore_tokens=ignore_tokens) for hyp, ref in zip(hypotheses, references))


def wordpiece_tokenize(text: str, vocab, unk_token='[UNK]', max_word_len=100, offset_start=0, offsets_out: list = None, wordpiece_ids_out: list = None):
    """
    An implementation of wordpiece tokenizer.
    :param text: the text to tokenize.
    :param vocab: the vocabulary.
    :param unk_token: the symbol for an unknown token.
    :param max_word_len: the maximum length for a word to tokenize.
    :param offset_start: the starting offset
    :param offsets_out: outputs the offset of each word (i.e. the `offset_start` plus the start position of each word in the returned wordpieces) into this list.
    :param wordpiece_ids_out: outputs wordpiece ids into this list.
    :return: the wordpieces of the `text`.
    """
    output_tokens = []

    # region setup for the add-token functions
    if offsets_out is not None and wordpiece_ids_out is not None:
        def _add_tokens(_sub_tokens):
            offsets_out.append(offset_start + len(output_tokens))
            output_tokens.extend(_sub_tokens)
            wordpiece_ids_out.extend((vocab[_token] for _token in _sub_tokens))

        def _add_single_token(_token):

            offsets_out.append(offset_start + len(output_tokens))
            output_tokens.append(_token)
            wordpiece_ids_out.append(vocab[_token])
    elif offsets_out is not None:
        def _add_tokens(_sub_tokens):
            offsets_out.append(offset_start + len(output_tokens))
            output_tokens.extend(_sub_tokens)

        def _add_single_token(_token):
            offsets_out.append(offset_start + len(output_tokens))
            output_tokens.append(_token)
    elif wordpiece_ids_out is not None:
        def _add_tokens(_sub_tokens):
            output_tokens.extend(_sub_tokens)
            wordpiece_ids_out.extend((vocab[_token] for _token in _sub_tokens))

        def _add_single_token(_token):
            output_tokens.append(_token)
            wordpiece_ids_out.append(vocab[_token])
    else:
        def _add_tokens(_sub_tokens):
            output_tokens.extend(_sub_tokens)

        def _add_single_token(_token):
            output_tokens.append(_token)
    # endregion

    if text in vocab:
        _add_single_token(text)
        return output_tokens

    for token in text.split():
        token_len = len(token)
        if token_len > max_word_len:
            _add_single_token(unk_token)
            continue

        start, unk_flag, sub_tokens = 0, False, []
        while start < token_len:
            end = token_len
            cur_substr = None
            while start < end:
                substr = token[start:end]
                if start > 0:
                    substr = "##" + substr
                if substr in vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                unk_flag = True
                break
            sub_tokens.append(cur_substr)

            start = end

        if unk_flag:
            _add_single_token(unk_token)
        else:
            _add_tokens(sub_tokens)
    return output_tokens


def nchr_tokenize(text: str, n: int, base_tokenizer: Callable = None, startchr='^', endchr='$', token_boundry='#'):
    basetokens = text.split() if base_tokenizer is None else base_tokenizer(text)
    if len(basetokens) == 1:
        basetokens[0] = (startchr if startchr else '') + basetokens[0] + (endchr if endchr else '')
    else:
        if startchr:
            if token_boundry:
                basetokens[0] = startchr + basetokens[0] + token_boundry
            else:
                basetokens[0] = startchr + basetokens[0]
        elif token_boundry:
            basetokens[0] = token_boundry + basetokens[0] + token_boundry

        if endchr:
            if token_boundry:
                basetokens[-1] = token_boundry + basetokens[-1] + endchr
            else:
                basetokens[-1] = basetokens[-1] + endchr
        elif token_boundry:
            basetokens[-1] = token_boundry + basetokens[-1] + token_boundry

        if token_boundry:
            for i in range(1, len(basetokens) - 1):
                basetokens[i] = token_boundry + basetokens[i] + token_boundry

    return sum(([''.join(x) for x in ngrams(basetoken, n)] if len(basetoken) >= n else [basetoken] for basetoken in basetokens), [])


def repeat_tokenizer(text: str, base_tokenizer: Callable = None, num_repeat: Union[int, Callable] = None):
    basetokens = text.split() if base_tokenizer is None else base_tokenizer(text)

    if num_repeat is None:
        return sum(([basetoken] * len(basetoken) for basetoken in basetokens), [])
    elif isinstance(num_repeat, int):
        return sum(([basetoken] * num_repeat for basetoken in basetokens), [])
    else:
        return sum(([basetoken] * num_repeat(basetoken) for basetoken in basetokens), [])


def text_index(text_data: Union[str, Iterator[str]], tokenizer: Callable, head_ids, tail_ids, sep_ids, max_index_len, lowercase, lowercase_exceptions, long_sequence_truncation, verbose=__debug__):
    """
    Implements the general main logic for text indexing.
    :param text_data: the text data to index; can be a single string, or a list/iterable of strings. No matter if it is single string or multiple strings, they will be serialized into a single list of integers.
                        Typically the text data is a single sentence, or is a list of multiple sentences; but it is totally OK for them to be texts at other levels.
                        The text data can even be just a list of tokens, where the tokenizer could further tokenize them as sub tokens.
    :param tokenizer: a function that tokenizes the input text data.
    :param head_ids:
    :param tail_ids:
    :param sep_ids:
    :param max_index_len:
    :param lowercase:
    :param lowercase_exceptions:
    :param long_sequence_truncation:
    :param verbose:
    :return:
    """

    def _add_head_and_tail_ids(ids: List[int]) -> List[int]:
        return head_ids + ids + tail_ids

    def _add_head_and_tail_clause_ids(ids: List[int]) -> List[int]:
        if ids:
            return [ids[0]] * len_start_pieces + ids + [ids[-1]] * len_end_pieces
        else:
            return [0] * (len_start_pieces + len_end_pieces)

    offsets, ids_no_head_and_tail = [0], []
    len_start_pieces = len(head_ids)
    len_end_pieces = len(tail_ids)
    max_num_wordpiece_ids = max_index_len - len_start_pieces - len_end_pieces

    tokens_type = type(text_data)

    if tokens_type is str:  # allows a single string as the input
        if lowercase and text_data not in lowercase_exceptions:
            text_data = text_data.lower()
        tokenizer(text_data, offset_start=len_start_pieces, offsets_out=offsets, wordpiece_ids_out=ids_no_head_and_tail)
    else:  # allows multiple strings as the input
        text = (token.lower() if lowercase and token not in lowercase_exceptions else token for token in text_data)
        for token in text:
            tokenizer(token, offset_start=len(ids_no_head_and_tail) + len_start_pieces, offsets_out=offsets, wordpiece_ids_out=ids_no_head_and_tail)
            if len(ids_no_head_and_tail) > max_num_wordpiece_ids and long_sequence_truncation != TruncationMode.NoTruncation:
                if verbose:
                    warnings.warn(f"long-sequence truncation happens for a sequence; the last processed token/text is `{token}`")
                break

    if len(ids_no_head_and_tail) <= max_num_wordpiece_ids:
        final_ids = _add_head_and_tail_ids(ids_no_head_and_tail)
        clause_ids = _add_head_and_tail_clause_ids(seg_tags(find_sub_list(ids_no_head_and_tail, sub=sep_ids, return_sub_end=True), seq_len=len(ids_no_head_and_tail)))
        offsets.append(len(final_ids) - len_end_pieces)
        long_sequecne = False
    elif long_sequence_truncation != TruncationMode.NoTruncation:
        truncate_offset_idx = find_first_larger_than(offsets, max_num_wordpiece_ids)
        if truncate_offset_idx is None:
            # This is the edge case.
            # `truncate_offset_idx` is `None` means the `find_first_larger_than` failed;
            # This is caused by the length of `ids_no_head_and_tail` exceeding `max_num_wordpiece_ids` when the last token is indexed.
            if long_sequence_truncation == TruncationMode.HardTruncation:
                ids_no_head_and_tail = ids_no_head_and_tail[:max_num_wordpiece_ids]
            elif long_sequence_truncation == TruncationMode.WholeWord:
                ids_no_head_and_tail = ids_no_head_and_tail[:(offsets[-1] - len_start_pieces)]
                offsets = offsets[:-1]
        else:
            if truncate_offset_idx <= 1:
                warnings.warn("the long-sequence truncation happens at the first word; either the maximum number of wordpieces is set too low, or the first word is very long")
            if long_sequence_truncation == TruncationMode.HardTruncation:
                ids_no_head_and_tail = ids_no_head_and_tail[:max_num_wordpiece_ids]
                offsets = offsets[:truncate_offset_idx]
            else:
                truncate_offset_idx -= (long_sequence_truncation == TruncationMode.WholeWord)
                ids_no_head_and_tail = ids_no_head_and_tail[:(offsets[truncate_offset_idx] - len_start_pieces)]  # `offsets` are the starting positions of each token considering the starting pieces, and therefore the `offsets[...] - len_start_pieces`
                offsets = offsets[:truncate_offset_idx]
        final_ids = _add_head_and_tail_ids(ids_no_head_and_tail)
        clause_ids = _add_head_and_tail_clause_ids(seg_tags(find_sub_list(ids_no_head_and_tail, sub=sep_ids, return_sub_end=True), seq_len=len(ids_no_head_and_tail)))
        offsets.append(len(final_ids) - len_end_pieces)
        long_sequecne = True
    else:
        offsets.append(len_start_pieces + len(ids_no_head_and_tail))
        clause_ids = seg_tags(find_sub_list(ids_no_head_and_tail, sub=sep_ids, return_sub_end=True), seq_len=len(ids_no_head_and_tail))
        wordpiece_windows, token_type_windows = unzip((
            (_add_head_and_tail_ids(ids_no_head_and_tail[start:end]), _add_head_and_tail_clause_ids(clause_ids[start:end]))
            for start, end in slide_window__(seq=ids_no_head_and_tail,
                                             window_len=max_num_wordpiece_ids,
                                             step_size=max_num_wordpiece_ids // 2,
                                             offsets=None,
                                             must_not_exceed_window_size=True)
        ))
        final_ids, clause_ids = sum(wordpiece_windows, []), sum(token_type_windows, [])
        long_sequecne = True

    return TokenIndexResult(tokens=final_ids, offsets=offsets, clause_ids=clause_ids, head_len=len_start_pieces, tail_len=len_end_pieces, long_sequence=long_sequecne)


def reconstruct_texts_from_batch(indexed_tokens: np.array, vocab, trg_indexed_tokens: np.array = None, trg_vocab=None, label=None):
    def _recon(token_indexes):
        return ' '.join(vocab.get_token(token_index) for token_index in token_indexes)

    if trg_indexed_tokens is None:
        return np.apply_along_axis(_recon, -1, indexed_tokens)
    else:
        return reconstruct_texts_from_batch(indexed_tokens=indexed_tokens, vocab=vocab), reconstruct_texts_from_batch(indexed_tokens=trg_indexed_tokens, vocab=trg_vocab)[label]


class Vocabulary:
    __slots__ = ('save_path', 'min_count', 'max_size', 'pad_token', 'unk_token', '_index2token', '_token2index', '_token_count', 'build_mode', 'vocab_name', 'index_offset', '_active_tokens')

    def __init__(self, save_path, min_count: int, max_size: int = None, vocab_name=None, pad_token='[PAD]', unk_token='[UNK]', fixed_tokens: Mapping = None, build_mode=False, format=None, index_offset=0):
        self.vocab_name = vocab_name or path.basename(save_path)
        self.save_path = save_path
        self.min_count = min_count
        self.max_size = max_size
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.build_mode = build_mode
        self.index_offset = index_offset
        self._active_tokens = None
        if build_mode:
            shutil.rmtree(self.save_path)

        token2index_file = path.join(self.save_path, 'token2index.txt')
        if not path.exists(token2index_file):
            self._token2index = {pad_token: 0, unk_token: 1}
            self._index2token = {0: pad_token, 1: unk_token}
            self._token_count = {}
            self.build_mode = True
        else:
            tokencount_file = path.join(self.save_path, 'tokencount.txt')
            if path.exists(tokencount_file):
                self._token_count = read_dict_from_text(tokencount_file, valtype=int)
                self._set_active_tokens_by_counts()
                self._active_tokens.add(pad_token)
                self._active_tokens.add(unk_token)
            else:
                self._token_count = None

            self._token2index = read_dict_from_text(token2index_file, valtype=int, format=format)
            self._index2token = kvswap(self._token2index)
            self.build_mode = False

        if fixed_tokens is not None:
            if self._active_tokens is not None:
                self._active_tokens.update(fixed_tokens.keys())
            for token, idx in fixed_tokens.items():
                self._token2index[token] = idx
                self._index2token[idx] = token

        if not self.build_mode:
            hprint_message(title=f'size of vocabulary {self.vocab_name}', content=self.vocab_size())

    def __len__(self):
        return len(self._token2index)

    def __call__(self, token):
        if self.build_mode:
            if token not in self._token_count:
                self._token_count[token] = 1
            else:
                self._token_count[token] += 1
        else:
            if self._active_tokens is not None and token not in self._active_tokens:
                return 1
            return self._token2index.get(token, 1)

    def get_token(self, token_index):
        return self._index2token.get(token_index, None)

    def __repr__(self):
        return f"Vocabulary {self.vocab_name}; size: {self.__len__()}, build mode: {self.build_mode}, path: {self.save_path}"

    def _set_active_tokens_by_counts(self):
        if self.max_size is None:
            self._active_tokens = set(k for k, v in self._token_count.items() if v >= self.min_count)
        else:
            self._active_tokens = set(islice((k for k, v in self._token_count.items() if v >= self.min_count), self.max_size))

    def vocab_size(self):
        return len(self._active_tokens) if self._active_tokens is not None else len(self)

    def save(self):
        if self.build_mode:
            self._token_count = sort_by_values(self._token_count, reverse=True)
            self._set_active_tokens_by_counts()
            self._active_tokens.update(self._token2index.keys())
            for token in self._token_count:
                idx = len(self) + self.index_offset
                self._token2index[token] = idx
                self._index2token[idx] = token
            ensure_dir_existence(self.save_path, clear_dir=True)
            write_dict_as_text(self._token2index, output_path=path.join(self.save_path, 'token2index.txt'))
            write_dict_as_text(self._token_count, output_path=path.join(self.save_path, 'tokencount.txt'))
            hprint_message(title=f'size of vocabulary {self.vocab_name}', content=self.vocab_size())

            self.build_mode = False
