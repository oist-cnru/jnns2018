"""Graph class and code for bokeh graphs"""

import os
import warnings

import numpy as np

import bokeh
import bokeh.plotting as bpl
import bokeh.io as bio
from bokeh.models import FixedTicker, FuncTickFormatter
from bokeh.models import LinearColorMapper, BasicTicker, ColorBar, Label
from bokeh.models import HoverTool

import ipywidgets
from ipywidgets import widgets

from . import utils_bokeh as ubkh


TOOLS = ()


class Figure:

    def __init__(self, x_range=None, y_range=None, title='',
                       height=250, width=900,
                       tools=TOOLS, interactive=False):
        bpl.output_notebook(hide_banner=True)
        self.interactive = interactive
        self._handle = None
        self.lines = []
        self.N = 0

        self.fig = ubkh.figure(title=title, plot_width=width, plot_height=height,
                               x_range=x_range, y_range=y_range,
                               tools=['save', self._hover_tool(), 'box_zoom', 'reset'])

    def _hover_tool(self):
        return HoverTool(tooltips=[("name", "@name"),
                                   ("(x,y)", "(@x{%d}, @y{%0.1f})")],
                         formatters={'x' : 'printf', 'y': 'printf'})


    def set_x_ticks(self, fig, ticks=None):
        if ticks is None:
            ticks=(-500, 0, 500, 1000)
        fig.xaxis[0].ticker = FixedTicker(ticks=ticks)

    def set_y_ticks(self, fig, ticks=None):
        if ticks is not None: # no default value
            fig.yaxis[0].ticker = FixedTicker(ticks=ticks)

    def save(self, fig, title, ext='pdf'):
        full_title = '{}{}'.format(title, self.model_desc)
        ubkh.save_fig(fig, full_title, ext=ext, verbose=True)

    def line(self, x, y, alpha=0.5, color='black', line_width=1, label=None, name=None):
        """Just a line"""
        self.N = max(self.N, len(x))
        data  = {'x': x, 'y': y}
        if name is not None:
            data['name'] = len(x)*[name]
        line = self.fig.line(x='x', y='y', source=data,
                             alpha=alpha, color=color, line_width=line_width)
        if label is not None:
            x_label, y_label = float(x[0] - 0.025 * (x[-1] - x[0])), float(y[0])
            self.fig.add_layout(Label(x=x_label, y=y_label, text=label,
                                      text_baseline='middle', text_align='center',
                                      text_font_size='10pt', text_color=color))
        self.lines.append((x, y, line))

    def circle(self, x, y, alpha=0.5, label=None, **kwargs):
        """Just a line"""
        self.N = max(self.N, len(x))
        data  = {'x': x, 'y': y}
        line = self.fig.circle(x='x', y='y', source=data, alpha=alpha,
                               line_color=None, **kwargs)
        # if label is not None:
        #     x_label, y_label = float(x[0] - 0.025 * (x[-1] - x[0])), float(y[0])
        #     self.fig.add_layout(Label(x=x_label, y=y_label, text=label,
        #                               text_baseline='middle', text_align='center'))
        # self.lines.append((x, y, line))

    def _interact_aux(self):
        self._handle = bpl.show(self.fig, notebook_handle=True)
        start_w = widgets.IntSlider(value=0,      min=0, max=self.N)
        end_w   = widgets.IntSlider(value=self.N, min=0, max=self.N)
        interact(self.update, start=start_w, end=end_w)
        self.update(0, self.N)

    def show(self):
        if self.interactive:
            self._interactive_aux()
        else:
            bpl.show(self.fig)

    def update(self, start, end):
        end = max(end, start)
        for x, y, line in self.lines:
            line.data_source.data = {'x': x[start:end], 'y': y[start:end]}
        bio.push_notebook()


class OutputFigure(Figure):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lines = {}
        self.shown = False

    def line_history(self, x, y, alpha=0.5, color='black', line_width=1, label=None, name=None):
        """A line with y for multiple timesteps"""
        data  = {'x': x, 'y': y[max(y.keys())]}
        if name is not None:
            data['name'] = len(x)*[name]
        line = self.fig.line(x='x', y='y', source=data,
                             alpha=alpha, color=color, line_width=line_width)
        self._lines[name] = (x, y, line)
        self._interact_aux(sorted(y.keys()), name, self.update)


    def lines_history(self, x, ys, alpha=0.5, colors=['red', 'green'], line_width=1, label=None, names=None):
        """A line with y for multiple timesteps"""
        assert len(self._lines) == 0 # current limitation
        lines_data = []
        for i, y in enumerate(ys):
            data  = {'x': x, 'y': y[max(y.keys())]}
            if names is not None:
                data['name'] = len(x)*[names[i]]
            line = self.fig.line(x='x', y='y', source=data,
                                 alpha=alpha, color=colors[i], line_width=line_width)
            lines_data.append((x, y, line))
        big_name = '_'.join(names)
        self._lines[big_name] = lines_data
        self._interact_aux(sorted(y.keys()), big_name, self.updates)

    def _interact_aux(self, options, name, f):
        select = widgets.SelectionSlider(options=options, value=options[-1],
                                         description=name,
                                         continuous_update=True,
                                         orientation='horizontal')
        ipywidgets.interact(f, t=select, name=ipywidgets.fixed(name))

    def update(self, t, name):
        x, y, line = self._lines[name]
        line.data_source.data = {'x': x, 'y': y[t]}
        if self.shown:
            bio.push_notebook(self._handle)

    def updates(self, t, name):
        for x, y, line in self._lines[name]:
            line.data_source.data = {'x': x, 'y': y[t]}
        if self.shown:
            bio.push_notebook(self._handle)

    def show(self):
        self._handle = bpl.show(self.fig, notebook_handle=True)
        self.shown = True

    def _hover_tool(self):
        return HoverTool(tooltips=[('name', '@name'),
                                   ('t', '@x{%0.1f}'),
                                   ('y', '@y{%0.3f}')],
                         formatters={'x' : 'printf', 'y': 'printf'})



class ErrorFigure(Figure):

    def _hover_tool(self):
        return HoverTool(tooltips=[('name', '@name'),
                                   ('epoch', '@x{%d}'),
                                   ('error', '@y{%0.1f}')],
                         formatters={'x' : 'printf', 'y': 'printf'})

class LossFigure(Figure):

    def _hover_tool(self):
        return HoverTool(tooltips=[('name', '@name'),
                                   ('epoch', '@x{%d}'),
                                   ('loss', '@y{%0.1f}')],
                         formatters={'x' : 'printf', 'y': 'printf'})
