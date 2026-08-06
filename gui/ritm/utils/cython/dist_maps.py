#!python
#cython: language_level=3

import pyximport

pyximport.install(pyximport=True, language_level=3, build_dir="C:\\tmp\\pyxbld")
# noinspection PyUnresolvedReferences
from ._get_dist_maps import get_dist_maps