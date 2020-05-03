import math
import warnings
from copy import copy
from enum import Enum
from os import path
from typing import Iterator, Union, Callable, Tuple

import matplotlib.colors as clrs
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from utilx.dictex import IndexDict
from utilx.general import is_list_or_tuple, make_range, hprint_message, iterable, value__, zip__
import utilx.npex as npx


class Markers(Enum):
    Point = '.'
    Pixel = ','
    Circle = 'o'
    TriangleDown = 'v'
    TriangleUp = '^'
    TriangleLeft = '<',
    TriangleRight = '>'
    TriDown = '1'
    TriUp = '2'
    TriLeft = '3'
    TriRight = '4'
    Octagon = '8'
    Square = 's'
    Pentagon = 'p'
    Plus = 'P'
    Star = '*'
    Hexagon1 = 'h'
    Hexagon2 = 'H'
    X = 'x'
    XFilled = 'X'
    Diamond = 'D'
    ThinkDiamond = 'd'
    VerticalBar = '|'
    HorizontalLine = '_'
    TickLeft = 0
    TickRight = 1
    TickUp = 2
    TickDown = 3
    CaretLeft = 4
    CaretRight = 5
    CaretUp = 6
    CaretDown = 7
    CaretLeftCentered = 8
    CaretRightCentered = 9
    CaretUpCentered = 10
    CaretDownCentered = 11


class ColorMaps(Enum):
    """
    A convenient enum class that lists all color maps in https://matplotlib.org/tutorials/colors/colormaps.html.
    """

    # region dark to light color
    Greys: str = 'Greys'
    Purples: str = 'Purples'
    Blues: str = 'Blues'
    Greens: str = 'Greens'
    Oranges: str = 'Oranges'
    Reds: str = 'Reds'
    YellowOrangeBrown: str = 'YlOrBr'
    YellowOrangeRed: str = 'YlOrRd'
    OrangeRed: str = 'OrRd'
    PurpleRed: str = 'PuRd'
    RedPurple: str = 'RdPu'
    BluePurple: str = 'BuPu'
    GreenBlue: str = 'GnBu'
    PurpleBlue: str = 'PuBu'
    YellowGreenBlue: str = 'YlGnBu'
    PurpleBlueGreen: str = 'PuBuGn'
    BuGn: str = 'BlueGreen'
    YellowGreen: str = 'YlGn'
    Binary: str = 'binary'
    GistYarg: str = 'gist_yarg'
    Wistia: str = 'Wistia'
    # endregion

    # region light to dark color
    GistGrey: str = 'gist_grey'
    Grey: str = 'grey'
    Bone: str = 'bone'
    Pink: str = 'pink'
    GistHeat: str = 'gist_heat'
    # endregion

    # region special continuous color
    ViriDis: str = 'viridis'
    Plasma: str = 'plasma'
    Inferno: str = 'inferno'
    Magma: str = 'magma'
    Cividis: str = 'cividis'
    Spring: str = 'spring'
    Summer: str = 'summer'
    Autumn: str = 'autumn'
    Winter: str = 'winter'
    Hot: str = 'hot'
    Cool: str = 'cool'
    AffirmativeHot: str = 'afmhot'
    Copper: str = 'copper'
    # endregion

    # region dividing colors
    DivPinkYellowishGreen: str = 'PiYG'
    DivPurpleGreen: str = 'PRGn'
    DivBrownBlueishGreen: str = 'BrBG'
    DivOrangePurple: str = 'PuOr'
    DivRedGrey: str = 'RdGy'
    DivRedBlue: str = 'RdBu'
    DivRedYellowishBlue: str = 'RdYlBu'
    DivRedYellowishGreen: str = 'RdYlGn'
    DivBlueWhiteRed: str = 'bwr'
    Spectral: str = 'spectral'
    CoolWarm: str = 'coolwarm'
    Seismic: str = 'seismic'
    # endregion

    # region mixed colors
    Twilight: str = 'twilight'
    Twilight_Shifted: str = 'twilight_shifted'
    HSV: str = 'hsv'
    Flag: str = 'flag'
    Prism: str = 'prism'
    Ocean: str = 'ocean'
    GistEarth: str = 'gist_earth'
    Terrain: str = 'terrain'
    GistStern: str = 'gist_stern'
    GnuPlot: str = 'gnuplot'
    GnuPlot2: str = 'gnuplot2'
    CMRMap: str = 'cmrmap'
    CubeHelix: str = 'cubehelix'
    Brg: str = 'brg'
    GistRainbow: str = 'gist_rainbow'
    Rainbow: str = 'rainbow'
    Jet: str = 'jet'
    NipySpectral: str = 'nipy_spectral'
    GistNcar: str = 'gist_ncar'
    # endregion

    # region discrete colors
    Pastel1: str = 'Pastel1'
    Pastel2: str = 'Pastel2'
    Paired: str = 'paired'
    Accent: str = 'Accent'
    Dark2: str = 'Dark2'
    Set1: str = 'Set1'
    Set2: str = 'Set2'
    Set3: str = 'Set3'
    Tab10: str = 'tab10'
    Tab20: str = 'tab20'
    Tab20b: str = 'tab20b'
    Tab20c: str = 'tab20c'
    # endregion


class Locs(Enum):
    Best = 0
    UpperLeft = 2
    UpperCenter = 9
    UpperRight = 1
    LowerLeft = 3
    LowerCenter = 8
    LowerRight = 4
    Right = 5
    CenterLeft = 6
    CenterRight = 7
    Center = 10


_DISCRETE_COLOR_MAPS = ('Pastel1', 'Pastel2', 'paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c')


def is_discrete_color_map(_n) -> bool:
    _nt = type(_n)

    def _solve_str(x):
        return True if x in _DISCRETE_COLOR_MAPS else type(get_cmap(x)) is clrs.ListedColormap

    if _nt is ColorMaps:
        return _solve_str(_n.value)
    elif _nt is str:
        return _solve_str(_n)
    else:
        return type(_n) is clrs.ListedColormap


def get_cmap(_n: Union[str, ColorMaps]):
    """
    :param _n:
    :return:
    """
    _nt = type(_n)
    return plt.get_cmap(_n) if _nt is str else (plt.get_cmap(_n.value) if _nt is ColorMaps else _n)


def get_color(color, cmap=ColorMaps.Tab10, cmap_norm=None):
    if cmap is None:
        return None

    if isinstance(color, clrs.Colormap):
        return color

    cmap = get_cmap(cmap)

    if cmap_norm is None:
        return cmap(color)
    else:
        return cmap(cmap_norm(color))


def get_colors(it: Iterator, cmap=ColorMaps.Tab10, cmap_norm=None):
    cmap = get_cmap(cmap)
    if cmap_norm is None:
        d = IndexDict()
        if is_discrete_color_map(cmap):
            return [cmap(d.index(x)) for x in it]
        else:
            if not is_list_or_tuple(it):
                it = list(it)
            d.add_all(it)
            d = d.get_normalized_index_dict()
            return [cmap(d[x]) for x in it]
    else:
        return [cmap(cmap_norm(x)) for x in it]


def savefig__(fname, dpi: int = 1200, format=None, clear=False, verbose=__debug__, *args, **kwargs):
    if format is None:
        format = path.splitext(path.basename(fname))[1]
        if format:
            format = format[1:]
        else:
            format = 'svg'

    plt.savefig(fname=fname, dpi=dpi, format=format, *args, **kwargs)
    if verbose:
        hprint_message('figure saved', fname)
    if clear:
        plt.clf()


class TitleInfo:
    __slots__ = ('title', 'loc', 'pad', 'size')


class AxisInfo:
    __slots__ = ('xticks', 'xticks_labels', 'xticks_labelsize', 'yticks', 'yticks_labels', 'yticks_labelsize', 'integer_xticks_range', 'integer_yticks_range', 'xlim', 'ylim', 'xlim_reversed', 'ylim_reversed',
                 'xlabel', 'ylabel', 'xlabelpad', 'ylabelpad', 'xlabelsize', 'ylabelsize')

    def __init__(self, xticks=None, xticks_labels=None, xticks_labelsize=None, yticks=None, yticks_labels=None, yticks_labelsize=None, integer_xticks_range=None, integer_yticks_range=None,
                 xlim=None, ylim=None, xlim_reversed=False, ylim_reversed=False,
                 xlabel=None, ylabel=None, xlabelpad=None, ylabelpad=None, xlabelsize=None, ylabelsize=None):
        self.xticks = xticks
        self.xticks_labels = xticks_labels
        self.xticks_labelsize = xticks_labelsize
        self.yticks = yticks
        self.yticks_labels = yticks_labels
        self.yticks_labelsize = yticks_labelsize
        self.integer_xticks_range = integer_xticks_range
        self.integer_yticks_range = integer_yticks_range
        self.xlim = xlim
        self.ylim = ylim
        self.xlim_reversed = xlim_reversed
        self.ylim_reversed = ylim_reversed
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlabelpad = xlabelpad
        self.ylabelpad = ylabelpad
        self.xlabelsize = xlabelsize
        self.ylabelsize = ylabelsize

    def __call__(self, ax):
        set_ticks(ax=ax, xticks_labels=self.xticks_labels,
                  xticks=self.xticks,
                  xticks_labelsize=self.xticks_labelsize,
                  yticks_labels=self.yticks_labels,
                  yticks=self.yticks,
                  yticks_labelsize=self.yticks_labelsize,
                  integer_xticks_range=self.integer_xticks_range,
                  integer_yticks_range=self.integer_yticks_range,
                  xlim=self.xlim,
                  xlim_reversed=self.xlim_reversed,
                  ylim=self.ylim,
                  ylim_reversed=self.ylim_reversed)
        set_labels(ax, xlabel=self.xlabel, ylabel=self.ylabel, xlabelpad=self.xlabelpad, ylabelpad=self.ylabelpad, xlabelsize=self.xlabelsize, ylabelsize=self.ylabelsize)

    def new(self, **kwargs):
        for k in self.__slots__:
            if k not in kwargs:
                kwargs[k] = getattr(self, k)
        return AxisInfo(**kwargs)


class LegendInfo:
    __slots__ = ('labels', 'loc', 'ncol', 'fontsize', 'bbox_to_anchor')

    def __init__(self, labels, loc='best', ncol=1, fontsize=None, bbox_to_anchor=None):
        self.labels = labels
        self.loc = loc
        self.ncol = ncol
        self.fontsize = fontsize
        self.bbox_to_anchor = bbox_to_anchor

    def __call__(self, ax, handles=None):
        set_legends(ax, handles=handles, labels=self.labels, loc=self.loc, ncol=self.ncol, fontsize=self.fontsize, bbox_to_anchor=self.bbox_to_anchor)


class PCAInfo:
    __slots__ = ('sample_dim', 'feature_dim', 'n_components', 'standardizer')

    def __init__(self, sample_dim=-2, feature_dim=-1, n_components=2, standardizer=True):
        self.sample_dim = sample_dim
        self.feature_dim = feature_dim
        self.n_components = n_components
        self.standardizer = standardizer

    def __call__(self, X):
        return npx.reduce_by_pca(X, n_components=self.n_components, sample_dim=self.sample_dim, feature_dim=self.feature_dim, standardizer=self.standardizer)


class ColorbarInfo:
    __slots__ = ('ticks', 'ticks_labelsize', 'shrink', 'show_colorbar')

    def __init__(self, ticks, ticks_labelsize=None, shrink=1.0, show_colorbar=True):
        self.ticks = ticks
        self.ticks_labelsize = ticks_labelsize
        self.shrink = shrink
        self.show_colorbar = show_colorbar

    def __call__(self, ax, fig=None, ref_ax=None, shrink=None):
        set_colorbar(ax, fig, ref_ax, show_colorbar=self.show_colorbar, ticks=self.ticks, ticks_labelsize=self.ticks_labelsize, shrink=shrink or self.shrink)


def set_labels(ax=plt, xlabel=None, ylabel=None, xlabelpad=None, ylabelpad=None, xlabelsize=None, ylabelsize=None):
    if xlabel is not None:
        if hasattr(ax, 'xlabel'):
            ax.xlabel(xlabel, labelpad=xlabelpad, fontsize=xlabelsize)
        else:
            ax.set_xlabel(xlabel, labelpad=xlabelpad, fontsize=xlabelsize)
    if ylabel is not None:
        if hasattr(ax, 'ylabel'):
            ax.ylabel(ylabel, labelpad=ylabelpad, fontsize=ylabelsize)
        else:
            ax.set_ylabel(ylabel, labelpad=ylabelpad, fontsize=ylabelsize)


def set_labels__(ax=plt, labels: Union[Tuple[str, str], AxisInfo] = None):
    if isinstance(labels, AxisInfo):
        set_labels(ax, xlabel=labels.xlabel, ylabel=labels.ylabel, xlabelpad=labels.xlabelpad, ylabelpad=labels.ylabelpad, xlabelsize=labels.xlabelsize, ylabelsize=labels.ylabelsize)
    elif callable(labels):
        labels(ax)
    else:
        set_labels(xlabel=labels[0], ylabel=labels[1])


def set_title(ax=plt, title=None, size=None, pad=None, loc='center'):
    if title is not None:
        _title = ax.title if hasattr(ax, 'title') and callable(ax.title) else ax.set_title
        _title(title, fontsize=size, pad=pad, loc=loc)


def set_title__(ax=plt, title: Union[str, Callable] = None):
    if isinstance(title, str):
        set_title(ax, title=title)
    elif callable(title):
        title(ax)


def set_legends(ax=plt, labels=None, loc: Union[str, int, Locs] = None, ncol: int = 1, fontsize=None, handles=None, bbox_to_anchor=None):
    if handles is None:
        if fontsize is None:
            ax.legend(labels=labels, loc=value__(loc), ncol=ncol, bbox_to_anchor=bbox_to_anchor)
        else:
            ax.legend(labels=labels, loc=value__(loc), ncol=ncol, prop=dict(size=fontsize), bbox_to_anchor=bbox_to_anchor)
    else:
        if fontsize is None:
            ax.legend(handles=handles, labels=labels, loc=value__(loc), ncol=ncol, bbox_to_anchor=bbox_to_anchor)
        else:
            ax.legend(handles=handles, labels=labels, loc=value__(loc), ncol=ncol, prop=dict(size=fontsize), bbox_to_anchor=bbox_to_anchor)


def set_legends__(ax=plt, legends=None, *args, **kwargs):
    if legends is not None:
        if callable(legends):
            legends(ax, *args, **kwargs)
        else:
            set_legends(ax, labels=legends, *args, **kwargs)


def set_ticks(ax=plt, xticks=None, xticks_labels=None, xticks_labelsize=None, yticks_labels=None, yticks=None, yticks_labelsize=None, integer_xticks_range=None, integer_yticks_range=None, xlim=None, ylim=None, xlim_reversed=False, ylim_reversed=False):
    xticks, xticks_labels, xlim = solve_integer_ticks_range(ticks=xticks, ticks_labels=xticks_labels, integer_ticks_range=integer_xticks_range, lim=xlim)
    yticks, yticks_labels, ylim = solve_integer_ticks_range(ticks=yticks, ticks_labels=yticks_labels, integer_ticks_range=integer_yticks_range, lim=ylim)

    if xticks is not None or xticks_labels is not None:
        if hasattr(ax, 'xticks'):
            ax.xticks(xticks, xticks_labels)
        else:
            if xticks is not None:
                ax.set_xticks(xticks)
            if xticks_labels is not None:
                ax.set_xticklabels(xticks_labels)
    if yticks is not None or yticks_labels is not None:
        if hasattr(ax, 'yticks'):
            ax.yticks(yticks, yticks_labels)
        else:
            if yticks is not None:
                ax.set_yticks(yticks)
            if yticks_labels is not None:
                ax.set_yticklabels(yticks_labels)
    set_xticks_labelsize(ax, xticks_labelsize)
    set_yticks_labelsize(ax, yticks_labelsize)

    # region xlim, ylim; ! must be placed at the end; otherwise strange clippings happen
    if xlim is not None:
        if xlim_reversed:
            ax.xlim(xlim[1], xlim[0])
        else:
            ax.xlim(*xlim)
    if ylim is not None:
        if ylim_reversed:
            ax.ylim(ylim[1], ylim[0])
        else:
            ax.ylim(*ylim)
    # endregion


def set_ticks__(ax=plt, ticks=None):
    if isinstance(ticks, AxisInfo):
        set_ticks(ax=ax, xticks_labels=ticks.xticks_labels,
                  xticks=ticks.xticks,
                  xticks_labelsize=ticks.xticks_labelsize,
                  yticks_labels=ticks.yticks_labels,
                  yticks=ticks.yticks,
                  yticks_labelsize=ticks.yticks_labelsize,
                  integer_xticks_range=ticks.integer_xticks_range,
                  integer_yticks_range=ticks.integer_yticks_range,
                  xlim=ticks.xlim,
                  xlim_reversed=ticks.xlim_reversed,
                  ylim=ticks.ylim,
                  ylim_reversed=ticks.ylim_reversed)
    elif callable(ticks):
        ticks(ax)
    else:
        set_ticks(ax, xticks=ticks[0], yticks=ticks[1])


def set_common_labels(fig, common_xlabel=None, common_ylabel=None, common_xlabelpad=None, common_ylabelpad=None):
    if common_xlabel or common_ylabel:
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        if common_ylabelpad is None:
            common_ylabelpad = 11
        set_labels(plt, common_xlabel, common_ylabel, xlabelpad=common_xlabelpad, ylabelpad=common_ylabelpad)
        plt.gcf().subplots_adjust(bottom=0.15)


class PlotInfo:
    __slots__ = ('title', 'markers', 'legends', 'axis_info', 'pca', 'scatter', 'cmap', 'cmap_norm', 'color', 'marker_size', 'rotation', 'reverse_xy', 'linestyle', 'fillstyle')

    def __init__(self, title: Union[str, Callable] = None, markers=None, legends=None, axis_info: Callable = None, pca=None, scatter=False, cmap=None, cmap_norm=None, color=None, marker_size=None, rotation=None, reverse_xy=None, linestyle=None, fillstyle=None):
        self.title = title
        self.markers = value__(markers)
        self.legends = legends
        self.axis_info = axis_info
        self.pca = pca
        self.scatter = scatter
        self.cmap = get_cmap(cmap)
        self.cmap_norm = cmap_norm
        self.color = color
        self.marker_size = marker_size
        self.rotation = npx.rotate2d(rotation) if rotation is not None else None
        self.reverse_xy = reverse_xy
        self.linestyle = linestyle
        self.fillstyle = fillstyle

    def __call__(self, ax, data, x=None):
        if self.pca is not None:
            data = self.pca(data)
        if self.rotation is not None:  # TODO 3d rotate
            data = data @ self.rotation

        if self.scatter:
            if len(data.shape) == 2:
                if self.reverse_xy:
                    if self.markers is not None and len(self.markers) > 1:
                        if data.shape[0] == 2:
                            lines = [ax.plot(sub_data[1], sub_data[0], marker=marker, markersize=self.marker_size, fillstyle=fillstyle, color=get_color(color or i, cmap=self.cmap, cmap_norm=self.cmap_norm))
                                     for i, (sub_data, marker, marker_size, color, fillstyle) in enumerate(zip__(data.T, self.markers, self.marker_size, self.color, self.fillstyle))]
                        elif data.shape[1] == 2:
                            lines = [ax.plot(sub_data[1], sub_data[0], marker=marker, markersize=self.marker_size, fillstyle=fillstyle, color=get_color(color or i, cmap=self.cmap, cmap_norm=self.cmap_norm))
                                     for i, (sub_data, marker, marker_size, color, fillstyle) in enumerate(zip__(data, self.markers, self.marker_size, self.color, self.fillstyle))]

                    else:
                        if data.shape[0] == 2:
                            lines = ax.scatter(data[1], data[0], marker=self.markers, cmap=self.cmap, norm=self.cmap_norm, c=self.color, s=self.marker_size)
                        elif data.shape[1] == 2:
                            lines = ax.scatter(data.T[1], data.T[0], marker=self.markers, cmap=self.cmap, norm=self.cmap_norm, c=self.color, s=self.marker_size)
                else:
                    if self.markers is not None and len(self.markers) > 1:
                        if data.shape[0] == 2:
                            lines = [ax.plot(sub_data[0], sub_data[1], marker=marker, markersize=self.marker_size, color=get_color(color or i, cmap=self.cmap, cmap_norm=self.cmap_norm))
                                     for i, (sub_data, marker, marker_size, color) in enumerate(zip__(data.T, self.markers, self.marker_size, self.color))]
                        elif data.shape[1] == 2:
                            lines = [ax.plot(sub_data[0], sub_data[1], marker=marker, markersize=self.marker_size, color=get_color(color or i, cmap=self.cmap, cmap_norm=self.cmap_norm))
                                     for i, (sub_data, marker, marker_size, color) in enumerate(zip__(data, self.markers, self.marker_size, self.color))]

                    else:
                        if data.shape[0] == 2:
                            lines = ax.scatter(data[0], data[1], marker=self.markers, cmap=self.cmap, norm=self.cmap_norm, c=self.color, s=self.marker_size)
                        elif data.shape[1] == 2:
                            lines = ax.scatter(data.T[0], data.T[1], marker=self.markers, cmap=self.cmap, norm=self.cmap_norm, c=self.color, s=self.marker_size)
            else:
                raise ValueError(f'the scatter plot does not support data of shape `{data.shape}`')
        else:
            if self.markers is not None and len(self.markers) > 1:
                if isinstance(data, (list, tuple)):
                    data = list(zip(*data))
                elif hasattr(data, 'T'):
                    data = data.T
                if x is None:
                    lines = [ax.plot(sub_data, marker=marker, markersize=self.marker_size, linestyle=linestyle, fillstyle=fillstyle, color=get_color(color or i, cmap=self.cmap, cmap_norm=self.cmap_norm))
                             for i, (sub_data, marker, marker_size, color, linestyle, fillstyle) in enumerate(zip__(data, self.markers, self.marker_size, self.color, self.linestyle, self.fillstyle))]

                elif self.reverse_xy:
                    lines = [ax.plot(sub_data, x, marker=marker, markersize=self.marker_size, linestyle=linestyle, fillstyle=fillstyle, color=get_color(color or i, cmap=self.cmap, cmap_norm=self.cmap_norm))
                             for i, (sub_data, marker, marker_size, color, linestyle, fillstyle) in enumerate(zip__(data, self.markers, self.marker_size, self.color, self.linestyle, self.fillstyle))]
                else:
                    lines = [ax.plot(x, sub_data, marker=marker, markersize=self.marker_size, linestyle=linestyle, fillstyle=fillstyle, color=get_color(color or i, cmap=self.cmap, cmap_norm=self.cmap_norm))
                             for i, (sub_data, marker, marker_size, color, linestyle, fillstyle) in enumerate(zip__(data, self.markers, self.marker_size, self.color, self.linestyle, self.fillstyle))]
            else:
                if x is None:
                    lines = ax.plot(data, marker=self.markers, markersize=self.marker_size)
                else:
                    lines = ax.plot(x, data, marker=self.markers, markersize=self.marker_size)
        set_legends__(ax, self.legends)
        set_title__(ax, self.title)

        if callable(self.axis_info):
            self.axis_info(ax)
        return lines

    def new(self, **kwargs):
        for k in self.__slots__:
            if k not in kwargs:
                kwargs[k] = getattr(self, k)
        return PlotInfo(**kwargs)


class ImShowInfo:
    __slots__ = ('title', 'markers', 'axis_info', 'cmap', 'cmap_norm', 'color', 'marker_size', 'vmin', 'vmax', 'colorbar')

    def __init__(self, title=None, markers=None, axis_info=None, cmap=None, cmap_norm=None, color=None, marker_size=None, vmin=None, vmax=None, colorbar=None):
        self.title = title
        self.markers = markers
        self.axis_info = axis_info
        self.cmap = get_cmap(cmap)
        self.cmap_norm = cmap_norm
        self.color = color
        self.marker_size = marker_size
        self.vmin = vmin
        self.vmax = vmax
        self.colorbar = colorbar

    def __call__(self, ax, data, vmin=None, vmax=None):
        im = ax.imshow(X=data, cmap=self.cmap, norm=self.cmap_norm, vmin=vmin or self.vmin, vmax=vmax or self.vmax)
        if self.colorbar is True:
            ax.colorbar()
        elif callable(self.colorbar):
            self.colorbar(ax)

        set_title__(ax, self.title)

        if callable(self.axis_info):
            self.axis_info(ax)
        return im


def get_subplots_shape(num_subplots, max_nrows=None, max_ncols=None):
    if max_nrows is None:
        if max_ncols is None:
            return math.ceil(num_subplots / 2), 2
        else:
            return math.ceil(num_subplots / max_ncols), max_ncols
    else:
        if max_ncols is None:
            return max_nrows, math.ceil(num_subplots / max_nrows)
        else:
            return math.ceil(num_subplots / max_ncols), max_ncols


def init_figure(*X, figsize=None, max_nrows=None, max_ncols=None, sharex=True, sharey=True):
    if len(X) == 1:
        if figsize is not None:
            plt.figure(figsize=figsize)
    else:
        nrows, ncols = get_subplots_shape(len(X), max_nrows, max_ncols)
        if figsize is None:
            figsize = (5 * ncols, 4 * nrows)
        return plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, figsize=figsize)


def imshow__(*data, imshow_info: ImShowInfo = None, figsize=None, max_nrows=None, max_ncols=None, sharex=True, sharey=True, shared_colorbar=None, share_vmin=True, share_vmax=True):
    fig_info = init_figure(*data, figsize=figsize, max_nrows=max_nrows, max_ncols=max_ncols, sharex=sharex, sharey=sharey)
    if len(data) == 1:
        data = data[0]
        if imshow_info is None:
            plt.imshow(data)
        else:
            imshow_info(plt, data)
    else:
        fig, axes = fig_info

        if share_vmin:
            vmin = min(np.min(x) for x in data)
        else:
            vmin = None
        if share_vmax:
            vmax = max(np.max(x) for x in data)
        else:
            vmax = None
        print(vmin)
        print(vmax)
        if imshow_info is None:
            for ax, _data in zip(axes.flat, data):
                im = ax.imshow(_data, vmin=vmin, vmax=vmax)
        elif callable(imshow_info):
            for ax, _data in zip(axes.flat, data):
                im = imshow_info(ax, _data, vmin=vmin, vmax=vmax)
        else:
            for ax, _data, _imshow_info in zip(axes.flat, data, imshow_info):
                im = _imshow_info(ax, _data, vmin=vmin, vmax=vmax)

        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="3%", pad=0.15)

        # cax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat])

        # fig.subplots_adjust(right=0.8)
        # cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # pad_fraction = 0.5
        # divider = make_axes_locatable(ax)
        # width = axes_size.AxesY(ax, aspect=1. / 20)
        # pad = axes_size.Fraction(pad_fraction, width)
        # cax = divider.append_axes("right", size=width, pad=pad)

        nrows = axes.shape[0]
        if shared_colorbar is True:
            set_colorbar(ax=plt, fig=im, ref_ax=axes, shrink=0.5 if nrows == 1 else 1.0)
        elif callable(shared_colorbar):
            shared_colorbar(plt, fig=im, ref_ax=axes, shrink=0.5 if nrows == 1 else 1.0)


def check_legends(X, legends=None):
    if legends is not None:
        if iterable(X[0]):
            if hasattr(X[0], '__len__') and len(X[0]) != len(legends):
                warnings.warn(f'plot {len(X[0])} sequences with {len(legends)} legends')
        elif len(legends) != 1:
            warnings.warn(f'plot one sequence with {len(legends)} legends')


def plot__(*data, x=None, plot_info=None, figsize=None, max_nrows=None, max_ncols=None, sharex=True, sharey=True, common_xlabel=None, common_ylabel=None, common_xlabelpad=None, common_ylabelpad=None, wspace=None, hspace=None, shared_legends=None, adjust_top=None, adjust_bottom=None, adjust_left=None, adjust_right=None):
    fig_info = init_figure(*data, figsize=figsize, max_nrows=max_nrows, max_ncols=max_ncols, sharex=sharex, sharey=sharey)

    if len(data) == 1:
        data = data[0]
        if plot_info is None:
            plt.plot(data)
        else:
            plot_info(plt, data, x)
    else:
        fig, axes = fig_info
        fig.subplots_adjust(bottom=adjust_bottom, top=adjust_top, left=adjust_left, right=adjust_right, wspace=wspace, hspace=hspace)
        if plot_info is None:
            for ax, _data in zip(axes.flat, data):
                ax.plot(_data)
        elif callable(plot_info):
            for ax, _data in zip(axes.flat, data):
                plot_info(ax, _data, x)
        else:
            for ax, _data, _plot_info in zip(axes.flat, data, plot_info):
                _plot_info(ax, _data, x)

        set_common_labels(fig=fig, common_xlabel=common_xlabel, common_ylabel=common_ylabel, common_xlabelpad=common_xlabelpad, common_ylabelpad=common_ylabelpad)
        if shared_legends is not None:
            shared_legends(fig)


# def scatter(*X, ):
#     pass


def set_colorbar(ax=plt, fig=None, ref_ax=None, show_colorbar=True, ticks=None, ticks_labelsize=None, shrink=1.0):
    if show_colorbar and hasattr(ax, 'colorbar'):
        if ticks is not None:
            cb = ax.colorbar(mappable=fig, ax=ref_ax, ticks=ticks, shrink=shrink)
        else:
            cb = ax.colorbar(mappable=fig, ax=ref_ax, shrink=shrink)
        if ticks_labelsize is not None:
            cb.ax.tick_params(labelsize=ticks_labelsize)


def solve_integer_ticks_range(ticks=None, ticks_labels=None, integer_ticks_range=None, lim=None):
    if integer_ticks_range is not None:
        if ticks_labels is None:
            ticks_labels = tuple(make_range(integer_ticks_range))
        if ticks is None:
            ticks = np.arange(0, len(ticks_labels), dtype=np.int)
        if lim is None:
            lim = (-0.5, len(ticks_labels) - 0.5)

    return ticks, ticks_labels, lim


def set_xticks_labelsize(ax=plt, labelsize=14):
    if labelsize is not None:
        ax.tick_params(axis="x", labelsize=labelsize)


def set_yticks_labelsize(ax=plt, labelsize=14):
    if labelsize is not None:
        ax.tick_params(axis="y", labelsize=labelsize)
