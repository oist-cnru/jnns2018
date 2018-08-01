#!/usr/bin/env python

import sys
import os
import subprocess
import argparse


def run_ipynb(filepath):
    """Run a ipynb and produce a html output"""
    filename = os.path.basename(filepath)
    cmd = ('jupyter-nbconvert', '--to', 'html', '--execute',
           '--ClearOutputPreprocessor.enabled=True', filepath, '--output',
           filename)
    subprocess.check_call(cmd)


def clear_ipynb(filepath):
    """Strip a notebook from its output inplace"""
    filename = os.path.basename(filepath)
    cmd = ('jupyter-nbconvert', '--inplace', '--to', 'notebook',
           '--ClearOutputPreprocessor.enabled=True', filepath, '--output',
           filename)
    subprocess.check_call(cmd)

def is_ipynb(filepath):
    filename = os.path.basename(filepath)
    _, ext = os.path.splitext(filename)
    return ext == '.ipynb'

def find_ipynb(path):
    """Find notebooks in a directory and its subdirectories"""
    if os.path.isfile(path):
        return [path] if is_ipynb(path) else []

    filepaths = []
    for dirpath, dirnames, filenames in os.walk(path):
        if os.path.basename(dirpath) != '.ipynb_checkpoints':
            if dirpath == path:
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if is_ipynb(filepath):
                        filepaths.append(filepath)

            for dirname in dirnames:
                filepaths.extend(find_ipynb(os.path.join(dirpath, dirname)))
    return filepaths


if __name__ == '__main__':
    path = '.'
    if len(sys.argv) >= 2:
        path = sys.argv[1]

    ipynb_filepaths = find_ipynb(path)
    for filepath in ipynb_filepaths:
        clear_ipynb(filepath)
