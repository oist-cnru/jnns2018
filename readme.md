# Learning Timescales in MTRNNs

This is the code corresponding to the article "Learning Timescales in MTRNNs" by
Fabien C. Y. Benureau, and Jun Tani, submitted for review.
If the paper is published, this code will be released publicly.

## Run

Tested under Python 3.6.5.

First install PyTorch 0.4 (0.3 is not supported) by following the instructions
on http://pytorch.org/, in accordance to your configuration.

To install other dependencies:
```
pip install -r requirements.txt
```

Then run:
```
python run_training.py
```

Computation takes roughly one hour at ~500GFLOPS.

Finally, to produce the figures, launch jupyter:
```
jupyter notebook figures.ipynb
```

The figures probably won't be the same as the article. We produced those figures
using multithreaded, non-deterministic code. The current code should be
deterministic, and use only one thread per computation (the `run_training.py`
code launches seven processes in parallel).

## Documentation

Some generic documentation, generated from the code comments, is available. But
the best is probably to go into the code. To compile the documentation,
`sphinx` and a few dependencies need to be installed:

```
pip install sphinx sphinx-autobuild sphinx-rtd-theme sphinx_autodoc_typehints
cd docs/
make html
```

Then open the `./build/html/index.html` file.
