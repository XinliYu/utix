import re
from collections import Mapping
from typing import Union, Iterator

import nltk


def pos_tag__(text: str, break_into_sentences=False):
    """

    :param text:
    :param break_into_sentences: `True` if to break the input `text` into senteces, and then do part-of-speech tagging for each
    :return:
    """
    if break_into_sentences:
        # tokenize the article into sentences, then tokenize each sentence into words
        token_sentences = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text)]
        # tag each tokenized sentence into parts of speech: pos_sentences
        return [nltk.pos_tag(sent) for sent in token_sentences]
    else:
        return nltk.pos_tag(nltk.word_tokenize(text))


_default_pos_map = {
    'JJ': 'j',
    "JJR": "j",
    "JJS": "j",
    "NN": "n",
    "NNS": "n",
    "NNP": "n",
    "NNPS": "n",
    "CD": "c",
    "IN": "i",
    "VBG": "v",
    "VBN": "v"
}

_default_extract_pattern = re.compile(r'((j|(nv?)|c)*ni)?(j|(nv?)|c)*n')


def iter_entities_by_pos_pattern(text_or_tokens: Union[str, Iterator[str]], pos_map: Mapping = _default_pos_map, pos_pattern=_default_extract_pattern, join_tokens=False):
    """

    Iterates through possible entities in the given text that match the specified part-of-speech pattern.
    This function uses `pos_map` to map the original part-of-speech tags to simpler characters that makes the formulation of `pos_pattern` easier.
    For example, `pos_map = {'NN': 'n', 'NNS': 'n', 'NNP': 'n', 'NNPS': 'n'}`, then the text `this is the New York City` with its original part-of-speech tags 'DT VBZ NNP NNP NNP' will be mapped to `___nnn`, where pos-tags not in `pos_map` will be mapped to `_`.
    Then you can use a simple pattern `n+` to extract the entity `New York City`.

    Using the default.
    ------------------
    >>> import nltk_util as nltkx
    >>> text = "This case total doesn't reflect the number of active cases, but rather the total number of people infected since the start of the pandemic. " \
    >>>        "That means, according to official statistics, New York City alone now has had more infections than the whole of China, which has reported 81,907 cases, according to the Chinese National Health Commission."
    >>> print(list(nltkx.iter_entities_by_pos_pattern(text,join_tokens=True)) == ['case total',
    >>>                                                                            'number of active cases',
    >>>                                                                            'total number of people',
    >>>                                                                            'start',
    >>>                                                                            'pandemic',
    >>>                                                                            'official statistics',
    >>>                                                                            'New York City',
    >>>                                                                            'more infections',
    >>>                                                                            'whole of China',
    >>>                                                                            '81,907 cases',
    >>>                                                                            'Chinese National Health Commission'])

    Only extracts consecutive Nones.
    --------------------------------
    >>> print(list(nltkx.iter_entities_by_pos_pattern(text, pos_map={'NN': 'n', 'NNS': 'n', 'NNP': 'n', 'NNPS': 'n'}, pos_pattern='n{2,3}',join_tokens=True)) == ['case total', 'New York City', 'Chinese National Health'])

    Allows passing in tokens.
    -------------------------
    >>> print(list(nltkx.iter_entities_by_pos_pattern(nltk.word_tokenize(text), pos_map={'NN': 'n', 'NNS': 'n', 'NNP': 'n', 'NNPS': 'n'}, pos_pattern='n{2,3}',join_tokens=True)) == ['case total', 'New York City', 'Chinese National Health'])

    :param text_or_tokens: can pass the text, or the tokens.
    :param pos_map: a dictionary that maps the original part-of-speech tags to single characters for easier formulation of `pos_pattern`.
    :param pos_pattern: the part-of-speech pattern.
    :param join_tokens: `True` to join extracted entity tokens; `False` to yield the tokens of each entity.
    :return: the extracted entities that match the specified part-of-speech pattern.
    """
    tokens, tags = tuple(zip(*(pos_tag__(text_or_tokens) if isinstance(text_or_tokens, str) else nltk.pos_tag(text_or_tokens))))
    tag_str = ''.join(pos_map.get(x, '_') for x in tags)
    if join_tokens:
        for match in re.finditer(pos_pattern, tag_str):
            start, end = match.span()
            yield ' '.join(tokens[start:end])
    else:
        for match in re.finditer(pos_pattern, tag_str):
            start, end = match.span()
            yield tokens[start:end]


def iter_entities_by_pos_pattern__(text_or_tokens: Union[str, Iterator[str]], pos_map: Mapping = _default_pos_map, pos_pattern=_default_extract_pattern, join_tokens=True):
    """
    The same as `iter_entities_by_pos_pattern`, but returns the a 3-tuple, the entity, the part-of-speech tags of that entity, and the entity's token index in the input `text_or_tokens`.
    """
    tokens, tags = tuple(zip(*(pos_tag__(text_or_tokens) if isinstance(text_or_tokens, str) else nltk.pos_tag(text_or_tokens))))
    tag_str = ''.join(pos_map.get(x, '_') for x in tags)
    if join_tokens:
        for match in pos_pattern.finditer(tag_str):
            start, end = match.span()
            yield ' '.join(tokens[start:end]), tags[start:end], start
    else:
        for match in pos_pattern.finditer(tag_str):
            start, end = match.span()
            yield tokens[start:end], tags[start:end], start


def create_ngram_text(txt: str, n, tokenizer=None):
    input_sequence = [i for i in nltk.ngrams(['('] + txt.split() if tokenizer is None else tokenizer(txt) + [')'], n)]
    input_sequence = ["#".join(i) for i in input_sequence]
    input_sequence = " ".join(input_sequence)
    return input_sequence
