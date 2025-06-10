#!/usr/bin/env python3
"""Custom OVERCAST Preprocessing Procflow (demo).

This procflow reads, sanitizes, parallax-corrects and resamples CLAVRx
data using GeoIPS. The processed data are then resampled to a common grid
and blended into both 2D and 3D outputs.

Usage:
    python3 preprocessing.py
"""

import logging
from pathlib import Path

import xarray
from rich.logging import RichHandler

import geoips

# Configure logging to use RichHandler
FORMAT = "%(name)s - %(message)s"
logging.basicConfig(
    level="INFO",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)
logger = logging.getLogger("main")

# List of CLAVRx variables that will be read from the HDF4 files
CLAVRX_VARS = [
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
    "latitude",
    "longitude",
]


def get_data(filename):
    """
    Read the specified CLAVRx HDF4 file using the GeoIPS 'clavrx_hdf4' reader.

    Parameters
    ----------
    filename : str or Path
        The name of the CLAVRx file to read.

    Returns
    -------
    dict
        A dictionary containing the read data keyed by "DATA".
    """
    overcast_clavrx_dir = Path("/overcast-data")
    example_clavrx_file = Path(filename)

    # Retrieve the 'clavrx_hdf4' reader plugin from GeoIPS
    reader = geoips.interfaces.readers.get_plugin("clavrx_hdf4")

    # Construct the full path to the file
    file = str(overcast_clavrx_dir / example_clavrx_file)

    # Read in the data for the specified CLAVRx variables
    data = reader([file], chans=CLAVRX_VARS)
    return data


def process_file(filename, useGeoIPSInterp=True, logger=None):
    """Process a single CLAVRx file with preprocessing steps.

    Performs reading, sanitizing, parallax-correcting,
    and resampling to common OVERCAST grid.

    Parameters
    ----------
    filename : str or Path
        The path to the file to process.
    useGeoIPSInterp : bool, optional
        Whether to use the default GeoIPS interpolator or a custom one, by default True.
    logger : logging.Logger, optional
        Logger for status messages. If None, a new logger will be created.

    Returns
    -------
    xarray.Dataset
        The resampled dataset.
    """
    if not logger:
        logger = logging.getLogger(filename)

    logger.info("Reading file")
    data = get_data(filename)["DATA"]

    logger.info("Sanitizing file data")
    sanitize_plugin = geoips.interfaces.algorithms.get_plugin("sanitize_clavrx")
    sanitized = sanitize_plugin(data)

    logger.info("Ingesting CWC")
    cwc_plugin = geoips.interfaces.algorithms.get_plugin("ingest_cwc")
    cwc = cwc_plugin(
        sanitized,
        "clavrx_OR_ABI-L1b-RadF-M6C01_G16_s20233450600210.CWC.h5",
    )

    logger.info("Correcting file parallax")
    parallax_plugin = geoips.interfaces.algorithms.get_plugin("parallax_correct_clavrx")
    corrected = parallax_plugin(cwc)

    logger.info(
        "Resampling file data",
    )  # Need to update what variables are resampled when using the GeoIPS native code; nearest neighbor plugin
    if useGeoIPSInterp:
        # Use the standard nearest-neighbor interpolator from GeoIPS
        from overcast_preprocessing import resampling

        # Prepare an empty dataset for the interpolated output
        resampled = xarray.Dataset()

        # Define the geographic limits and create an area definition
        geolims = resampling.GeoLimits(lonmin=-180, lonmax=180, latmin=-70, latmax=70)
        area_def = resampling.generate_global_swath(geolims, 0.02)

        # Perform nearest-neighbor interpolation
        geoips.interfaces.interpolators.get_plugin("interp_nearest")(
            area_def,
            corrected,
            resampled,
            CLAVRX_VARS,
        )
    else:
        # Use a custom interpolator plugin
        custom_resampler = geoips.interfaces.interpolators.get_plugin(
            "overcast_preprocessing_resample",
        )
        resampled = custom_resampler(corrected)
        # If caching is desired, arguments could be passed like:
        # resampled = custom_resampler(corrected, cache_key=filename)

    logger.info("Done with file")
    return resampled


def run_procflow(test_data, sats, use_geoips_interp=False):
    """Demo procflow for OVERCAST Preprocessing.

    Run the processing workflow on multiple satellite CLAVRx files, blend them into
    2D and 3D composite datasets, and then output them to NetCDF files.

    Parameters
    ----------
    test_data : dict
        A dictionary of satellite names to file paths.
    sats : list of str
        A list of satellite keys to process (matching keys in `test_data`).
        Order of list controls the order of blending.
    use_geoips_interp : bool, optional
        Whether to use the built-in GeoIPS interpolator or external interpolation,
        Defaults to external interpolation.

    Notes
    -----
    Cloud water content has not been added as a plugin yet. Is still TODO.
    """
    # Process each file sequentially, storing the result in `datasets`
    datasets = [
        process_file(
            "/workspaces/geoips_overcast/geoips_overcast/" + test_data[sat],
            useGeoIPSInterp=use_geoips_interp,
        )
        for sat in sats
    ]

    logging.info("Read data, data now being blended")

    # Blend multiple 2D datasets using a plugin
    blend_2d_plugin = geoips.interfaces.algorithms.get_plugin("blend_2d_clavrx")
    blended_data_2d = blend_2d_plugin(datasets)

    # Output the 2D blended data to NetCDF
    nc_output_plugin = geoips.interfaces.output_formatters.get_plugin("netcdf_xarray")
    nc_output_plugin(
        blended_data_2d,
        "2d_blended",
        ["./2d_blended_test.nc4"],
        clobber=True,
    )
    print("Done blending 2D data!")

    # Construct 3D fields from the same input datasets
    print("Generating 3D fields")
    construct_3d_plugin = geoips.interfaces.algorithms.get_plugin(
        "construct_3d_cloud_field",
    )
    blended_data_3d = construct_3d_plugin(datasets)

    # Output the 3D blended data to NetCDF
    nc_output_plugin(
        blended_data_3d,
        "3d_blended",
        ["./3d_blended_test.nc4"],
        clobber=True,
    )
    print("Done blending 3D data!")


if __name__ == "__main__":
    # Example usage with just the two GOES full disks
    test_data = {
        "GOES-18": "clavrx_OR_ABI-L1b-RadF-M6C01_G18_s20242980350217.level2.hdf",
        "GOES-16": "clavrx_OR_ABI-L1b-RadF-M6C01_G16_s20242980350207.level2.hdf",
    }
    sats = ["GOES-16", "GOES-18"]

    logger.info("Running procflow on GOES18/16 full disk imagery.")
    run_procflow(test_data, sats, use_geoips_interp=False)

    # More extensive set of full disk imagery
    test_data = {
        "GOES-18": "clavrx_OR_ABI-L1b-RadF-M6C01_G18_s20242980350217.level2.hdf",
        "Meteosat-9": "clavrx_met9_2024_298_0345.level2.hdf",
        "Meteosat-10": "clavrx_met10_2024_298_0345.level2.hdf",
        "HIM9": "clavrx_H09_20241024_0350_B01_FLDK_DK_R10_S0110.DAT.level2.hdf",
        "GOES-16": "clavrx_OR_ABI-L1b-RadF-M6C01_G16_s20242980350207.level2.hdf",
    }
    sats = ["GOES-16", "Meteosat-10", "Meteosat-9", "HIM9", "GOES-18"]

    logger.info("Running procflow on five full disk datasets")
    run_procflow(test_data, sats, use_geoips_interp=False)
