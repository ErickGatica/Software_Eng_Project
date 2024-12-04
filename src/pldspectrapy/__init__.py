from .constants import *
from .editmodule import *
from .igtools import *

from .linelist_conversions import *
from .pldhapi import *
from .td_support import *
from .model_creation import *
from .misc_tools import *
from .plotting_tools import *
from .config_handling import *
from .simulation_tools import *

# Use the packaged style file to set the plot style
import matplotlib.pyplot as plt

module_path = os.path.dirname(__file__)
# TODO: reorganize directory structure and move the style file to a more user-accessible location
style_path = os.path.join(module_path, "..", "resource", "pldl_style.mplstyle")
plt.style.use(style_path)
