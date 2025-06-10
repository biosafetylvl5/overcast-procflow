"""OVERCAST Preprocessing Cloud Water Content Ingester.

This module-based plugin provides a wrapper to add Cloud Water Content (cwc) to
CLAVR-x data.

It uses functionality from `overcast_preprocessing.sanitization`.
"""

import logging

from overcast_preprocessing import sanitization

LOG = logging.getLogger(__name__)

# Specify internal data used by GeoIPS to create procflows
interface = "algorithms"
family = "xarray_to_xarray"
name = "ingest_cwc"


def call(xobj, cwc_path):
    """Add Cloud Water Content variable to dataset.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset of CLAVR-x variables to add CWC to.
    cwc_path: str or None
        Path to hdf5 file containing CWC information. If None,
        an array of NANs of the appropriate shape is created.

    Returns
    -------
    xarray.Dataset: The input dataset with one additional data variable containing
        cwc information, which is aligned with existing fields along latitude and
        longitude and which adds a "levels" dimension.

    """
    # Perform the 2D blend using the specified weighting function.
    # The actual blending logic is handled by the blend_2d_clavrx function.
    if not cwc_path:
        LOG.warning(
            f"Passed CWC filepath is '{cwc_path}', "
            "which isn't valid. Adding NANs instead of data.",
        )
    return sanitization.ingest_cwc(xobj, cwc_path=cwc_path)
