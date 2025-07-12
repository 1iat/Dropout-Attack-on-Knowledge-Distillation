import re
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import numpy as np
from os.path import exists

from misc import read_json

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 28})
global_ticksize = 8


def get_data_from_files(file_name, num_files, epochs, num_classes=10):
  '''
  Get the data from files (and average if necessary)

  Parameters:
  - file_name: file name of file to get model info of Note: File name should not include .json extension 
    --> this is so function can look at multiple files
  - num_files: the number of files each index of data_to_convert counts for. Many results are run several times, this
    parameter is for that.
  - epochs: the number of epochs the files 
  '''
  test_acc = 0
  test_rec_class = np.zeros(num_classes)
  test_prec_class = np.zeros(num_classes)
  train_loss = np.zeros(epochs)
  val_loss = np.zeros(epochs)
  train_acc = np.zeros(epochs)
  val_acc = np.zeros(epochs)
  minmax_acc = [1.1, 0]
  for i in range(1, num_files + 1):
    model_info = read_json(f'{file_name}-{i}.json')
    acc = model_info['accuracy']
    test_acc += acc
    test_rec_class += np.array(model_info['classRecall'])
    test_prec_class += np.array(model_info['classPrecision'])
    if minmax_acc[0] > acc:
      minmax_acc[0] = acc
    if minmax_acc[1] < acc:
      minmax_acc[1] = acc
    train_loss += np.array(model_info['metrics']['train_loss'])
    val_loss += np.array(model_info['metrics']['val_loss'])
    train_acc += np.array(model_info['metrics']['train_accuracy'])
    val_acc += np.array(model_info['metrics']['val_accuracy'])
  output_model = {
    'accuracy': test_acc / num_files,
    'classRecall': test_rec_class / num_files,
    'classPrecision': test_prec_class / num_files,
    'train_loss': train_loss / num_files,
    'val_loss': val_loss / num_files,
    'train_accuracy': train_acc / num_files,
    'val_accuracy': val_acc / num_files,
    'minAccuracy': minmax_acc[0],
    'maxAccuracy': minmax_acc[1],
  }
  return output_model

colors = ['tab:red', 'tab:blue', 'tab:orange', 'tab:purple',  'tab:pink', 'tab:brown']

def generate_linegraph(data, filepath, x_axis, y_axis, title, fig_size=None, xtickrotation=0, xscale='linear', error=None, baseline=None, ylim=None):
  '''
  Generates a line graph for the paper.
  
  Parameters:
  - data: input data for the graph. data is an array of dicts which should contain the following keys:
    - name: the group the dict belongs to
    - x: x-axis values
    - y: y-axis values
  - filepath: The path to the file
  - remaining parameters: used for the matplotlib graph generation
  '''
  ticksize = 14
  fig = plt.figure(figsize=fig_size)
  ax = fig.add_subplot(111)
  plt.subplots_adjust(top=0.98, right=0.99, left=0.09)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.spines['left'].set_linewidth(2)
  ax.spines['bottom'].set_linewidth(2)
  plt.title(title, loc='center', wrap=True)
  for i, d in enumerate(data):
    if d['name'] == 'model accuracy' and error is not None:
      ax.errorbar(d['x'], d['y'], yerr=error, fmt='-o',color=colors[i], label=d['name'], marker='.', markersize=20, linewidth=6)
    else:
      ax.plot(d['x'], d['y'], color=colors[i], label=d['name'], marker='.', markersize=20, linewidth=6)
  if baseline is not None:
    ax.axhline(baseline[0], linestyle='--', markersize=20, linewidth=6, label=baseline[1], color=colors[len(data)])
  ax.set_xlabel(x_axis)
  ax.set_xscale(xscale)
  ax.set_ylabel(y_axis)
  ax.tick_params(size=ticksize, which='major')
  ax.tick_params(size=global_ticksize, which='minor')
  if ylim is not None:
    plt.ylim(ylim)
  plt.setp(ax.get_xticklabels(), rotation=xtickrotation)
  ax.legend(frameon=False)
  plt.savefig(filepath)
  plt.clf()

def get_offset_array(width, size):
  '''
  Given a width of bars and x number of bars, return an array containing the offset necessary have x number of bars
  centered around some value
  '''
  if size % 2 == 1:
    return np.linspace(-(size // 2) * width, (size // 2) * width, size)
  else:
    return np.linspace(-(size / 2) * width + width / 2, (size / 2) * width - width / 2, size)

def generate_clusteredbargraph(data, filepath, x_axis, y_axis, title, xticks, bar_width, ncol, fig_size=None, ylim=None, legend_split_ind=None):
  '''
  Generates a clustered bar graph for the paper.

  Parameters: 
  - data: input data for the graph. data is an array of dicts which should contain the following keys:
    - name: the group the dict belongs to
    - values: the values of the bars in the group
  - filepath: The path to the file
  - remaining parameters: used for the matplotlib graph generation
  '''
  fig = plt.figure(figsize=fig_size)
  if title != '':
    plt.title(title, loc='center', wrap=True)
  ax = fig.add_subplot(111)
  plt.subplots_adjust(top=0.99, right=0.99, left=0.09)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.spines['left'].set_linewidth(2)
  ax.spines['bottom'].set_linewidth(2)
  xrange = range(0, len(data[0]['values']) * 2, 2)
  offset_array = get_offset_array(bar_width, len(data))
  patterns = [ "" , "//" , "\\\\" , "--" , "++" , "xx"]
  bars = []
  for i, d in enumerate(data):
    bars.append(ax.bar(xrange + offset_array[i], d['values'], bar_width, color=colors[i], label=d['name'], edgecolor='black', hatch=patterns[i]))
  ax.set_xlabel(x_axis)
  ax.set_ylabel(y_axis)
  plt.xticks(xrange, xticks)
  ax.tick_params(size=global_ticksize)
  if ylim is not None:
    plt.ylim(ylim)
  if legend_split_ind is None:
    ax.legend(frameon=False, ncol=ncol)
  else:
    # not very extendable code, maybe add bbox_to_anchor values if further graphs need to be modified
    kw = dict(frameon=False, borderaxespad=0)
    leg1 = plt.legend(handles=bars[:legend_split_ind], ncol=ncol, loc='lower right', bbox_to_anchor=(0.97, 0.85), **kw)
    plt.gca().add_artist(leg1)
    leg2 = plt.legend(handles=bars[legend_split_ind:], ncol=len(bars[legend_split_ind:]), loc='upper right', bbox_to_anchor=(0.9765, 0.85), **kw)
    # leg2.remove()
    # leg1._legend_box._children.append(leg2._legend_handle_box)
    # leg1._legend_box.stale = True
  plt.savefig(filepath)
  plt.clf()