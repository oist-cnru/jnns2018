from .graph import Figure, ErrorFigure, LossFigure, OutputFigure
from .utils_bokeh import save_fig

# remove LAPACK/scipy harmless warning (see https://github.com/scipy/scipy/issues/5998)
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
