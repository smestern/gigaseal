
import os
import sys

# Single source of truth for the package version (PEP 440).
# setuptools reads this statically via [tool.setuptools.dynamic] in pyproject.toml.
__version__ = "1.0.0b1"

# import subprocess

# # install ipfx without deps
# def install_ipfx():
#     subprocess.run([sys.executable, "-m", "pip", "install", "ipfx", "--no-deps"])

    
#only call if ipfx is not installed #this was an old issue and should
try:
    import ipfx
except ImportError:
    import warnings
    warnings.warn("ipfx is not installed. Some features may not be available.")

from . import patch_utils
from . import dataset
from . import featureExtractor
from . import ipfx_df
