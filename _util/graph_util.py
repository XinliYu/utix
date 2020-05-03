from collections import namedtuple
from enum import IntEnum
from itertools import product
from typing import Callable, List, Union, Tuple, Set, Mapping

import networkx as nx
import numpy as np
import matplotlib.axes

import utilx.jsonx as json
import utilx.npex as npx
import utilx.plotex as pltx
import utilx.rndex as rndx
from utilx.dictex import tup2dict__, PaddedListDict
from framework.torch.data_loading.data_entries import make_data_
from utilx.general import extra_msg
from utilx.ioex import TYPE_FILENAME_OR_STREAM
from utilx.msgex import ensure_arg_not_none_or_empty
from utilx.pathex import get_main_name_ext_name
from utilx.strex import StringFilter, NamedFieldExtractor, split__


def iter_node_attrs(g: nx.Graph):
    return g._node.values()


def iter_node_keys(g: nx.Graph):
    return g._node.keys()


def iter_nodes(g: nx.Graph):
    return g._node.items()


def get_node_attributes(g: nx.Graph, *attr_names):
    if len(attr_names) == 1:
        attr_name: str = attr_names[0]
        return [node[attr_name] for node in iter_node_attrs(g)]
    else:
        return [tuple(node[attr_name] for attr_name in attr_names) for node in iter_node_attrs(g)]


def node_degrees(g: nx.Graph):
    return np.array([val for (node, val) in g.degree()])


def degree_matrix(g: nx.Graph):
    return np.diag(node_degrees(g))


def inversed_degree_matrix(g: nx.Graph):
    return np.diag(1 / node_degrees(g))


def graph_to_files(g: nx.Graph, node_data_file: str, adj_file: TYPE_FILENAME_OR_STREAM, node_data_format='json', adj_file_format='adj'):
    # NOTE: a networkx graph supports global node data and adjacency data; it does not support local node data
    if node_data_format == 'json':
        json.dump_iter(g._node.items(), node_data_file)
    elif node_data_format == 'json_single':
        json.dump(g._node, node_data_file)

    if adj_file_format == 'json':
        json.dump_iter(g._adj.items(), node_data_file)
    elif adj_file_format == 'json_single':
        json.dump(g._adj, adj_file)
    elif adj_file_format == 'adj':
        write_adj_data(adj_file, g._adj)


class AdjDictFormat(IntEnum):
    SrcNodeAdjDict = 0
    EdgeDataDict = 1


def write_adj_data(file: TYPE_FILENAME_OR_STREAM, adj_dict: dict, node_sep: str = '\t', field_sep: str = '|', dict_format: AdjDictFormat = AdjDictFormat.SrcNodeAdjDict):
    if isinstance(file, str):
        file = open(file, 'w')
    sorted_keys = sorted(adj_dict.keys())
    if dict_format == AdjDictFormat.SrcNodeAdjDict:
        for key in sorted_keys:
            adj_str = node_sep.join((f'{adj}{field_sep}{adj_data}' if adj_data else str(adj) for adj, adj_data in adj_dict[key].items()))
            file.write(f'{key}{node_sep}{adj_str}\n')
    elif dict_format == AdjDictFormat.EdgeData:
        prev_src = sorted_keys[0][0]
        adj_list = []
        for key in sorted_keys:
            src, adj = key
            if prev_src == src:
                adj_list.append((adj, adj_dict[key]))
            else:
                adj_str = node_sep.join((f'{adj}{field_sep}{adj_data}' for adj, adj_data in adj_list))
                file.write(f'{src}{node_sep}{adj_str}\n')
    file.close()


# region stochastic block model

class SbmStyle(IntEnum):
    CorePeriphery = 0
    AnomalyClusters = 1


def sbm_core_periphery_P(k: int, block_size_range=(10, 30), intra_block_p=(0.6, 0.9), inter_block_p=(0.05, 0.02), num_noisy_p_per_block=None, noise_p=None, seed: int = None, **kwargs):
    if seed is not None:
        np.random.seed(seed)
    sizes = np.random.randint(block_size_range[0], block_size_range[1], size=k).tolist()
    P = rndx.random_square_matrix(inter_block_p, k)
    rndx.random_fill_diagonal(P, intra_block_p)
    rndx.random_matrix_replace(P, dist=noise_p, num_replaces=num_noisy_p_per_block)
    npx.make_symmetric_matrix(P)
    return P, sizes


def sbm_anomaly_clusters_P(k: int, anomaly_clusters_size_range, non_anomaly_size_range, anomaly_clusters_p=(0.6, 0.9), non_anomaly_p=(0.05, 0.02), num_noisy_p_per_block=None, noise_p=None, seed: int = None, **kwargs):
    if seed is not None:
        np.random.seed(seed)
    sizes = np.random.random_integers(*anomaly_clusters_size_range, size=k).tolist() + [np.random.random_integers(*non_anomaly_size_range)]
    P = rndx.random_square_matrix(non_anomaly_p, k + 1)
    rndx.random_fill_diagonal(P, anomaly_clusters_p, first_k=k)
    rndx.random_matrix_replace(P, dist=noise_p, num_replaces=num_noisy_p_per_block)
    npx.make_symmetric_matrix(P)
    return P, sizes


def rnd_sbm_graph(k: Union[int, tuple, list], seed: int = None, graph_style: SbmStyle = SbmStyle.CorePeriphery, directed=False, **kwargs) -> nx.Graph:
    if seed is not None:
        np.random.seed(seed)
    if type(k) is not int:
        k = round(np.random.uniform(low=k[0], high=k[1]))
    if graph_style == SbmStyle.CorePeriphery:
        p, sizes = sbm_core_periphery_P(k, seed=seed, **kwargs)
    elif graph_style == SbmStyle.AnomalyClusters:
        p, sizes = sbm_anomaly_clusters_P(k, seed=seed, **kwargs)
    else:
        raise NotImplementedError
    g = nx.stochastic_block_model(sizes=sizes,
                                  p=p,
                                  seed=seed,
                                  directed=directed,
                                  nodelist=kwargs.get('nodelist', None),
                                  selfloops=kwargs.get('selfloops', False),
                                  sparse=kwargs.get('sparse', False))
    setattr(g, 'p', p)
    return g


# endregion

# region plots

def draw_with_communities(g: nx.Graph, community_label='block', cmap=pltx.ColorMaps.Tab10, node_size=10, edge_width=0.6, pos=None):
    # TODO: allows some communities have larger node size
    nx.draw(g, node_color=pltx.get_colors(it=(x[community_label] for x in g._node.values()), cmap=cmap), node_size=node_size, width=edge_width, pos=pos)


def draw_one_community(g: nx.Graph, community, community_label='block', cmap=pltx.ColorMaps.Tab10, node_size=10, edge_width=0.6, pos=None):
    sug_g = g.subgraph([k for k, v in g._node.items() if v[community_label] == community])
    nx.draw(sug_g, node_color=pltx.get_color(color=community, cmap=cmap), node_size=node_size, width=edge_width, pos=pos(sug_g) if callable(pos) else None)


# endregion

# region graph file reading

class AdjacencyFileFormat:
    """
    Defines the format and reads a single line of adjacency/path list. In addition to the relation information, each node/edge can be associated with local data related to the relations.

    It is OK to place simple global node data (e.g. just one or two numbers) in the adjacency file as if they were local data;
    however, global node data is preferred to avoid redundancy.
    """
    __slots__ = ('source_node_format', 'adj_node_format', 'node_sep', 'field_sep', 'comment_indicator', '_source_node_format_type', '_adj_node_format_type')
    TYPE_DATA_TEXT_FORMAT = Union[Tuple, str, NamedFieldExtractor]
    PrevData = namedtuple('AdjacencyFileFormat_PrevData', 'prev_src_data, prev_adj_data')

    def _set_format(self, format_obj, format_var_name, format_type_var_name):
        if isinstance(format_obj, tuple):
            setattr(self, format_var_name, format_obj)
            setattr(self, format_type_var_name, 0)
        else:
            setattr(self, format_var_name, NamedFieldExtractor(format_obj))
            setattr(self, format_type_var_name, 1)

    def _parse_by_format2(self, key_prefix, format_obj, format_type: int, text: str) -> dict:
        if format_type == 0:
            return tup2dict__(zip(format_obj, split__(text, self.field_sep, parse=True)), key_prefix=key_prefix)
        elif format_type == 1:
            return tup2dict__(format_obj.parse(text).items(), key_prefix=key_prefix)

    def __init__(self, src_node_format: TYPE_DATA_TEXT_FORMAT = ('nid',),
                 adj_node_format: TYPE_DATA_TEXT_FORMAT = None,
                 node_sep: str = '\t',
                 field_sep: str = '|',
                 comment_indicator='#'):
        """
        :param src_node_format: an object that defines the format of the source node text. A node can have fields, for example, the node id, the node type, etc. and this format object instructs how to extract the tree fields from the text.
                                    The format object can be 1) a tuple of field names, e.g. `(nid,)`, or `('nid', 'ntype', 'attr1', 'attr2')`;
                                    2) a format string, e.g. `{nid}|{ntype}|{attr1}|{attr2}`; 3) a regular expression with named groups.
                                    If the format object is a tuple of field names, then the the `field_sep` parameter must be provided.
                                    Otherwise, the `field_sep` parameter is not effective.
                                    The field values extracted from the text will be parsed by `ast.literal_eval`.
        :param adj_node_format: an object that defines the format of an adjacent/path node text. Accepts the same type of objects as the `src_node_format` parameter.
        :param node_sep: used to separate node data in one line.
        :param field_sep: used to separate the fields.
        """
        # region sets format objs
        if isinstance(src_node_format, str) and 'node_id' not in src_node_format:
            raise ValueError("the field `node_id` must be defined in the node text format string")
        self._set_format(format_obj=src_node_format, format_var_name='source_node_format', format_type_var_name='_source_node_format_type')

        if adj_node_format:
            if isinstance(src_node_format, str) and 'node_id' not in src_node_format:
                raise ValueError("the field `node_id` must be defined in the node text format string")
            self._set_format(format_obj=adj_node_format, format_var_name='adj_node_format', format_type_var_name='_adj_node_format_type')
        else:
            self.adj_node_format = self.source_node_format
            self._adj_node_format_type = self._source_node_format_type

        # endregion

        self.node_sep = node_sep
        self.field_sep = field_sep
        self.comment_indicator = comment_indicator
        if isinstance(self.source_node_format, tuple) or isinstance(self.adj_node_format, tuple):
            ensure_arg_not_none_or_empty(arg_val=field_sep,
                                         arg_name='field_sep',
                                         extra_msg="one of the `source_node_format`, `adj_node_format`, `edge_format` is a tuple, and in this case `field_sep` must be specified")

    def parse_line(self, line: str, src_ntypes=None, adj_ntypes=None, etypes=None, prev_data: PrevData = None, verbose=__debug__):
        """
        Parses one line of a graph adjacency/path data file. Each line of this file should describe the source node and adjacency/path nodes, and the node attributes are assumed associated with the adjacency/path.
        Simple node meta attributes may be saved as node attributes for the source node; however, it is not encouraged to save complex node meta attributes not associated with the adjacency/path in the adjacency/path file.
        :param line: one line of data to parse. 1) one line consists of text representations of several nodes, separated by the `node_sep`; the first node is the source node; the other nodes are the adjacent nodes;
                                                2) the text of each node may contain multiple fields, separated by the `field_sep`; field values will be parsed as their most likely value; field names are not supported; one should specify the filed name in the node format objects.
        :param src_ntypes: the source node type filter; this method will not parse a line whose source node type is not in this `src_node_types`.
        :param adj_ntypes: the adjacency/path node type filter; this method will skip an adjacency/path node whose node type is not in this `adj_node_types`.
        :param etypes: the edge type filter; this method will skip an adjacency/path node whose edge type is not in this `adj_node_types`; requires the adjacency/path node to have an `edge_type` attribute.
        :param verbose:
        :return:
        """

        if prev_data:
            prev_src_data, prev_adj_data = prev_data
            if not line:
                if prev_adj_data:
                    prev_src_data.update(prev_adj_data)
                return prev_src_data
            line = line.lstrip()
            if line[0].startswith(self.comment_indicator):
                return None, prev_data
            prev_data = bool(prev_data)
        else:
            ensure_arg_not_none_or_empty(arg_val=line, arg_name='line', extra_msg='the input graph adjacency/path line to parse is empty')
            line = line.lstrip()
            if line[0].startswith(self.comment_indicator):
                return None

        splits = line.split(self.node_sep)
        if len(splits) <= 1:
            raise ValueError(f"the string `{line}` does not seen to describe an adjacency list or a path, while using `{self.node_sep}` as the delimiter")
        try:
            data_entry = self._parse_by_format2(key_prefix='src_', format_obj=self.source_node_format, format_type=self._source_node_format_type, text=splits[0])
            use_prev_data: bool = prev_data and prev_src_data is not None and data_entry['src_nid'] == prev_src_data['src_nid']
        except Exception as err:
            raise extra_msg(err,
                            f"a source node cannot be parsed from the string `{splits[0]}`; the whole line to parse is `{line}`")

        if src_ntypes is None or data_entry.get('src_ntype', None) in src_ntypes:  # applies the source node type filter
            adj_data = prev_adj_data if use_prev_data and prev_adj_data is not None else PaddedListDict()
            for adj_node_edge_str in splits[1:]:
                cur_adj_data = self._parse_by_format2(key_prefix='adj_', format_obj=self.adj_node_format, format_type=self._adj_node_format_type, text=adj_node_edge_str)
                if (adj_ntypes and cur_adj_data.get('adj_ntype', None) not in adj_ntypes) or (etypes and cur_adj_data.get('adj_etype', None) not in etypes):
                    continue
                adj_data += cur_adj_data

            if prev_data:
                if use_prev_data:
                    data_entry.update(prev_src_data)
                    return None, (data_entry, adj_data)
                else:
                    if prev_adj_data:
                        prev_src_data.update(prev_adj_data)
                    return prev_src_data, (data_entry, adj_data)
            else:
                if adj_data:
                    data_entry.update(adj_data)
                    return data_entry
                else:
                    return None
        else:
            if prev_data:
                if use_prev_data:
                    return None, (prev_src_data, prev_adj_data)
                else:
                    if prev_adj_data:
                        prev_src_data.update(prev_adj_data)
                    return prev_src_data, (None, None)
            else:
                return None


class ReadGraphAdjacencyFile:
    # region types
    TYPE_NODE_EDGE_FILTER = Union[StringFilter, List[str], Tuple[str, ...], Set[str]]

    # endregion

    def __init__(self, file_format: Union[str, AdjacencyFileFormat] = None,
                 src_node_types: TYPE_NODE_EDGE_FILTER = None,
                 adj_node_types: TYPE_NODE_EDGE_FILTER = None,
                 edge_types: TYPE_NODE_EDGE_FILTER = None,
                 node_edge_types_in_file_name=True,
                 file_name_part_sep: str = '-',
                 file_ext: str = None,
                 data_typing: Union[Mapping, Callable] = None,
                 multiline=False,
                 node_meta_data: Mapping = None,
                 **kwargs):
        """

        :param file_format: the file format for the adjacency file; accepts 1) 'json' or 'csv';
                                                                            2) 'simple' with optional directives; 'simple' is for parsing simple adjacency files with only node ids; the directives are 3 characters for the node-id separator, the comment indicator, and 0 or 1 about whether to remove empty entries, e.g. `simple:\t#1`;
                                                                                'simple' with the default directives works in the same way as `AdjacencyFileFormat()` (i.e. the default `AdjacencyFileFormat` object with no parameters), and is 2x faster.
                                                                            3) an object with a `parse_line(src_ntypes, adj_ntypes, etypes, prev_data)` method; the `src_ntypes`, `adj_ntypes`, `etypes` are type filters, and `prev_data` is for multi-line reading;
                                                                               its return type depends on if `prev_data` is set;
                                                                                if `prev_data` is set, then should return a 2-tuple, the first being the node data object (`None` if the data is not ready yet), and the second being the data object to be passed into the next line;
                                                                                if `prev_data` is not set, then it only returns the node data object.
        :param src_node_types:
        :param adj_node_types:
        :param edge_types:
        :param node_edge_types_in_file_name:
        :param file_name_part_sep:
        :param file_ext:
        :param data_typing:
        :param multiline:
        :param node_meta_data:
        :param kwargs:
        """
        self._src_node_types = src_node_types
        self._adj_node_types = adj_node_types
        self._edge_types = edge_types
        self._has_src_node_type_filter = bool(src_node_types)
        self._has_adj_node_type_filter = bool(adj_node_types)
        self._has_edge_type_filter = bool(edge_types)
        self._file_name_part_sep = file_name_part_sep
        self._file_format = file_format
        self._file_ext = file_ext
        self._node_edge_types_in_file_name = node_edge_types_in_file_name
        self._data_types = data_typing
        self.multiline = multiline
        self.node_meta_data = node_meta_data

    def __call__(self, file_path: str, *args, **kwargs):

        # region retrieve default node/edge types from the file name
        # default_src_node_type, default_edge_node_type, default_edge_type = positional_extract(main_name, sep=self._file_name_part_sep, n=3, sentinel=str.isdigit)
        # if not default_src_node_type:
        #     raise ValueError(msg_invalid_arg_value(arg_val=file_path, arg_name='file_path'))
        # elif not default_edge_node_type:
        #     default_edge_node_type = default_src_node_type
        #
        # if self._src_node_types:
        #     if default_src_node_type not in self._src_node_types:
        #         default_src_node_type = None
        #     if default_edge_node_type not in self._src_node_types:
        #         default_edge_node_type = None
        # if self._edge_types and default_edge_type not in self._edge_types:
        #     default_edge_type = None

        # endregion

        if isinstance(file_path, str):
            main_name, ext_name = get_main_name_ext_name(file_path)  # NOTE: `ext_name` carries the '.'
            if self._file_ext is not None and ext_name != self._file_ext:
                return
            fin = open(file_path, 'r')

        else:
            fin = file_path

        def _update_meta():
            if self.node_meta_data:
                src_meta = self.node_meta_data[data['src_nid']]
                if isinstance(src_meta, Mapping):
                    data.update({(k if k.startswith('src_') else 'src_' + k): v for k, v in src_meta.items()})
                else:
                    data['src_meta'] = src_meta

        if isinstance(self._file_format, str):
            if self._file_format == 'json':
                pass
            elif self._file_format == 'csv':
                pass
            elif self._file_format.startswith('simple'):
                # region simple adjacency/path file with only node ids

                directives: str = self._file_format[7:]
                if not directives:
                    sep, com, remm, cut_before, cut_after = '\t', '#', True, ';', None
                else:
                    sep, com, remm, cut_before, cut_after = directives[0], directives[1], bool(int(directives[2])), directives[3], directives[4]

                if self.multiline:
                    prev = adj = None
                    for line in fin:
                        line = line.strip()
                        if line[0] != com:
                            splits = split__(line, sep=sep, remove_empty_split=remm, lstrip=True, rstrip=True, parse=True, cut_before=cut_before, cut_after=cut_after)
                            src_node_id = splits[0]
                            if src_node_id != prev:
                                if prev is not None:
                                    data = {'src_nid': prev, 'adj_nid': adj}
                                    _update_meta()
                                    yield make_data_(data_dict=data, data_typing=self._data_types)
                                prev = src_node_id
                                adj = splits[1:]
                            else:
                                adj.extend((splits[1:]))
                    data = {'src_nid': src_node_id, 'adj_nid': adj}
                    _update_meta()
                    yield make_data_(data_dict=data, data_typing=self._data_types)
                else:
                    for line in fin:
                        line = line.strip()
                        if line[0] != com:
                            splits = split__(line, sep=sep, remove_empty_split=remm, lstrip=True, rstrip=True, parse=True, cut_before=cut_before, cut_after=cut_after)
                            data = {'src_nid': splits[0], 'adj_nid': splits[1:]}
                            _update_meta()
                            yield make_data_(data_dict=data, data_typing=self._data_types)
                # endregion
        else:
            # region complex adjacency/path file with adjacency/path associated attributes
            if self.multiline:
                prev_data = (None, None)
                for line in fin:
                    data, prev_data = self._file_format.parse_line(line, src_ntypes=self._src_node_types,
                                                                   adj_ntypes=self._adj_node_types,
                                                                   etypes=self._edge_types,
                                                                   prev_data=prev_data)
                    if data is not None:
                        _update_meta()
                        yield make_data_(data, data_typing=self._data_types)
                data = self._file_format.parse_line(None, src_ntypes=self._src_node_types,
                                                    adj_ntypes=self._adj_node_types,
                                                    etypes=self._edge_types,
                                                    prev_data=prev_data)
                _update_meta()
                yield make_data_(data, data_typing=self._data_types)
            else:
                for line in fin:
                    data = self._file_format.parse_line(line, src_ntypes=self._src_node_types,
                                                        adj_ntypes=self._adj_node_types,
                                                        etypes=self._edge_types)
                    _update_meta()
                    yield make_data_(data, data_typing=self._data_types)
            # endregion

        if hasattr(fin, 'close'):
            fin.close()


# endregion


# region utilities
def edge_density(com1, com2, edges):
    ecnt: int = 0
    for pair in product(com1, com2):
        ecnt += pair in edges
    return ecnt / (len(com1) * len(com2) - (len(com1) if com1 is com2 else 0))
# endregion
