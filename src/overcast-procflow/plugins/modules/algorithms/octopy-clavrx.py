"""Docstring."""

import logging

from geoips.interfaces import algorithms

CHANS = [
    "sensor_azimuth_angle",
    "sensor_zenith_angle",
    "surface_elevation",
    "cloud_phase",
    "cld_height_base",
    "cld_height_acha",
    "cld_cwp_dcomp",
    "freezing_altitude",
    "freezing_altitude_253",
    "freezing_altitude_268",
    "temp_10_4um_nom",
    "latitude",
    "longitude",
]

LOG = logging.getLogger(__name__)

interface = "algorithms"
family = "xarray_to_xarray"
name = "octopy-clavrx"


def call(xobj):
    """Docstring."""
    uv_oct = algorithms.get_plugin("uv_octopy")
    utvt_octopy = algorithms.get_plugin("utvt_octopy")

    xdict = xobj.copy()
    xdict["OCTOPY"] = uv_oct(xdict)
    output = utvt_octopy(xdict)
    output.cld_height_acha.assign_attrs(units=xobj["DATA"].cld_height_acha.units)
    output.cld_height_base.assign_attrs(units=xobj["DATA"].cld_height_base.units)
    return output
