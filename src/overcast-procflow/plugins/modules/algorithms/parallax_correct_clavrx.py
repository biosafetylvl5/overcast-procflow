"""OVERCAST Preprocessing Parallax Correction for CLAVR-X."""

import logging

from overcast_preprocessing import parallax_correction

LOG = logging.getLogger(__name__)

interface = "algorithms"
family = "xarray_to_xarray"
name = "parallax_correct_clavrx"


def call(xobj):
    return parallax_correction.parallax_correct_clavrx(xobj)
