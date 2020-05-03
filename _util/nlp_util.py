import warnings
from enum import IntEnum
from typing import Union, Iterator, Callable, List

from nltk import ngrams

from _util.dict_ext import xfdict, dwrap, SlotsDict

from _util.general_ext import xmean_, iterable, unzip
from _util.list_ext import slide_window__, find_sub_list, seg_tags, find_first_larger_than


class TruncationMode(IntEnum):
    NoTruncation = 0
    HardTruncation = 1
    WholeWord = 2
    WholeWord_AllowExceedingWindowSize = 3


class TokenIndexResult(SlotsDict):
    __slots__ = ('tokens', 'offsets', 'clause_ids', 'head_len', 'tail_len', 'long_sequence')

    def __init__(self, tokens, offsets, clause_ids, head_len, tail_len, long_sequence):
        self.tokens, self.offsets, self.clause_ids, self.head_len, self.tail_len, self.long_sequence = tokens, offsets, clause_ids, head_len, tail_len, long_sequence


def rouge_n(hypothesis: Union[str, Iterator], reference: Union[str, Iterator], n: Union[int, Iterator[int]] = 2, out: dict = None, ignore=None):
    if isinstance(hypothesis, str):
        hypothesis = hypothesis.split()
    if isinstance(reference, str):
        reference = reference.split()

    if ignore is not None:
        if iterable(ignore):
            hypothesis = tuple(x for x in hypothesis if x not in ignore)
            reference = tuple(x for x in reference if x not in ignore)
    else:
        hypothesis = tuple(x for x in hypothesis if x != ignore)
        reference = tuple(x for x in reference if x != ignore)

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


def rouge_n_batch(hypotheses: Iterator[Union[str, Iterator]], references: Iterator[Union[str, Iterator]], n: Union[int, Iterator[int]] = 2, ignore=None, agg_func: Callable = xmean_):
    return agg_func(rouge_n(hypothesis=hyp, reference=ref, n=n, ignore=ignore) for hyp, ref in zip(hypotheses, references))


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
