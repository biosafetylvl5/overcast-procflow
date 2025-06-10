"""OVERCAST Preprocessing Sanitization for CLAVR-X."""

import logging

LOG = logging.getLogger(__name__)

interface = "algorithms"
family = "xarray_to_xarray"
name = "crop_preprocessing_integration_data"


def call(xobj, x_range=range(3600, 3800), y_range=range(3600, 3800)):
    LOG.debug(f"Cropping xobj to x range {x_range} and y range {y_range}.")
    return xobj.isel(y=y_range, x=x_range)
