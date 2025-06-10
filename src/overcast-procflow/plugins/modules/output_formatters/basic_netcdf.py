# # # This source code is protected under the license referenced at
# # # https://github.com/NRLMMD-GEOIPS.

"""Standard xarray-based NetCDF output format."""

import logging

LOG = logging.getLogger(__name__)

interface = "output_formatters"
family = "xarray_data"
name = "basic_netcdf"


def call(xarray_obj, product_names, output_fnames, clobber=False):
    """Write xarray-based NetCDF outputs to disk."""
    xarray_obj.to_netcdf(output_fnames[0])
    return output_fnames
