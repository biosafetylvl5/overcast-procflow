"""TODO: Docstring.

TODO.
"""

import gc
import logging
import os
import pickle
from pathlib import Path

from overcast_preprocessing import (
    blending,
    construct_3d_field,
    parallax_correction,
    resampling,
    sanitization,
)
from rich.logging import RichHandler

from geoips.interfaces import readers as geoips_readers

FORMAT = "%(name)s - %(message)s"
logging.basicConfig(
    level="INFO",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)


def get_data(filename):
    overcast_clavrx_dir = Path("/overcast-data")

    example_clavrx_file = Path(filename)
    reader = geoips_readers.get_plugin("clavrx_hdf4")
    file = str(overcast_clavrx_dir / example_clavrx_file)
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
    data = reader([file], chans=CLAVRX_VARS)
    return data


def sanitize(data):
    return sanitization.sanitize_clavrx(data)


def parralax_correct(data):
    return parallax_correction.parallax_correct_clavrx(data)


def resample(data, use_cache=True, cache_key=""):
    # Generate a destination grid with a grid spacing of 0.02 degrees
    N_PROCESSORS = 20
    geolims = resampling.GeoLimits(lonmin=-180, lonmax=180, latmin=-70, latmax=70)
    destination_grid = resampling.generate_global_swath(geolims, 0.02)

    if use_cache:
        NN_N_NEIGHBORS = 1
        GAUSS_N_NEIGHBORS = 4
        NN_CACHE_FILENAME = f"{cache_key}-nn_resample_cache.pkl"
        GAUSS_CACHE_FILENAME = f"{cache_key}-gauss_resample_cache.pkl"
        if not all(map(os.path.exists, [GAUSS_CACHE_FILENAME, NN_CACHE_FILENAME])):
            # Precompute and save nearest neighbor resampling
            resampling.save_resampling_info(
                data,
                NN_N_NEIGHBORS,
                NN_CACHE_FILENAME,
                global_grid=destination_grid,
                nprocs=N_PROCESSORS,
            )
            # Precompute and save Gaussian resampling
            resampling.save_resampling_info(
                data,
                GAUSS_N_NEIGHBORS,
                GAUSS_CACHE_FILENAME,
                global_grid=destination_grid,
                nprocs=N_PROCESSORS,
            )
        return resampling.resample_clavrx(
            data,
            global_grid=destination_grid,
            nprocs=N_PROCESSORS,
            precomp_nn_resample_info=NN_CACHE_FILENAME,
            precomp_gauss_resample_info=GAUSS_CACHE_FILENAME,
        )

    else:
        return resampling.resample_clavrx(
            data,
            global_grid=destination_grid,
            nprocs=N_PROCESSORS,
        )


# output_formatter = geoips_output_formatters.get_plugin("netcdf_geoips")

# data = reader("./../tests/data/overcast_cropped_data/cropped_raw_input.nc")

test_data = {
    "GOES-18": "clavrx_OR_ABI-L1b-RadF-M6C01_G18_s20242980350217.level2.hdf",
    "Meteosat-9": "clavrx_met9_2024_298_0345.level2.hdf",
    "Meteosat-10": "clavrx_met10_2024_298_0345.level2.hdf",
    "HIM9": "clavrx_H09_20241024_0350_B01_FLDK_DK_R10_S0110.DAT.level2.hdf",
    "GOES-16": "clavrx_OR_ABI-L1b-RadF-M6C01_G16_s20242980350207.level2.hdf",
}


def process_file(filename, logger=None):
    if not logger:
        logger = logging.getLogger(filename)
    logger.info("Reading file")
    data = get_data(filename)["DATA"]
    logger.info("Sanitizing file data")
    sanitized = sanitize(data)
    logger.info("Correcting file parralax")
    corrected = parralax_correct(sanitized)
    logger.info("Resampling file data")
    resampled = resample(corrected, cache_key=filename)
    logger.info("Done with file")
    return resampled


if __name__ == "__main__":
    sats = ["GOES-16", "Meteosat-10", "Meteosat-9", "HIM9", "GOES-18"]
    cache = "merged_data.pkl"
    with open(cache, "rb") as p:
        datasets = pickle.load(p)
    print("Read data, data now being blended")
    gc.collect()
    blended_data = blending.blend_2d_clavrx(
        datasets,
        weight_function=blending.hard_cutoff_weights,
    )
    del datasets
    print("Done blending!")
    with open("blended_data.pkl", "wb") as p:
        pickle.dump(blended_data, p, protocol=pickle.HIGHEST_PROTOCOL)
    print("generating 3d fields")
    blended_data = construct_3d_field.compute_3d_clavrx_cloud_phase(
        blended_data,
        number_of_levels=41,
        max_height=20,
    )
    print("Done!")
