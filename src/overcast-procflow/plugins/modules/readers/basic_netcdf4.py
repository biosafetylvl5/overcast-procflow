# # # This source code is protected under the license referenced at
# # # https://github.com/NRLMMD-GEOIPS.

"""CLAVR-x NetCDF4 Reader."""

import logging

import xarray as xr

LOG = logging.getLogger(__name__)

interface = "readers"
family = "standard"
name = "basic_netcdf4"


def call(fnames, metadata_only=False, chans=None, area_def=None, self_register=False):
    """Read basic NetCDF4 files."""
    if metadata_only or chans or area_def or self_register:
        LOG.warning(
            "This reader ONLY supports reading data sets, and does not"
            " implement metadata only reading, channel selection, "
            "area definitions, self registering. However, an arg was passed "
            "attempting to use one of those features. Ignoring.",
        )
    if len(fnames) == 1:
        return {"DATA": xr.open_dataset(fnames[0])}
    else:
        return {"DATA": [xr.open_dataset(fname) for fname in fnames]}
