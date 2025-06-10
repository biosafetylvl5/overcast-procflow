"""
OCTOPY-CLAVRX GeoIPS Plugin for Temporal Interpolation.

This plugin integrates OCTOPY temporal interpolation capabilities with CLAVR-x
satellite data products. It performs motion vector calculation and temporal
interpolation of multiple satellite channels using optical flow techniques.

The plugin processes multi-temporal satellite imagery to create temporally
interpolated products on the OVERCAST grid, efficiently handling multiple
channels with optimized motion vector calculations.
"""

import logging
from typing import Any

import xarray as xr

from geoips.interfaces import algorithms

# Required channels for OCTOPY-CLAVRX processing
REQUIRED_CHANNELS = [
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
]

# Constants
MIN_TIME_STEPS = 2
PLUGIN_NAME = "neo_utvt_octopy"

LOG = logging.getLogger(__name__)

# GeoIPS plugin metadata
interface = "algorithms"
family = "xarray_to_xarray"
name = "neo_octopy_clavrx"


class OctopyClavrxError(Exception):
    """Base exception for OCTOPY-CLAVRX processing errors."""

    pass


class InvalidInputError(OctopyClavrxError):
    """Exception raised for invalid input data."""

    pass


class PluginLoadError(OctopyClavrxError):
    """Exception raised when required plugins cannot be loaded."""

    pass


class ProcessingError(OctopyClavrxError):
    """Exception raised during OCTOPY processing steps."""

    pass


class ValidationError(OctopyClavrxError):
    """Exception raised for validation failures."""

    pass


def validate_input_data(xobj: xr.Dataset) -> None:
    """
    Validate that required channels are present in the input dataset.

    Parameters
    ----------
    xobj : xr.Dataset
        Input xarray Dataset to validate

    Raises
    ------
    InvalidInputError
        If input is not valid format or missing required channels
    ValidationError
        If required channels are missing from the input dataset
    """
    if not isinstance(xobj, (xr.Dataset, dict)):
        raise InvalidInputError(
            f"Input must be xarray Dataset or dict, got {type(xobj)}",
        )

    # Extract data variables based on input structure
    if isinstance(xobj, dict):
        if "DATA" not in xobj:
            raise InvalidInputError("Input dict must contain 'DATA' key")
        if not isinstance(xobj["DATA"], xr.Dataset):
            raise InvalidInputError("'DATA' key must contain xarray Dataset")
        data_vars = list(xobj["DATA"].data_vars.keys())
    else:
        data_vars = list(xobj.data_vars.keys())

    # Check for required channels
    missing_channels = [ch for ch in REQUIRED_CHANNELS if ch not in data_vars]

    if missing_channels:
        error_msg = (
            f"Missing required channels: {missing_channels}. "
            f"Required: {REQUIRED_CHANNELS}. "
            f"Available: {data_vars}"
        )
        LOG.error(error_msg)
        raise ValidationError(error_msg)

    LOG.info(
        f"Input validation passed - all {len(REQUIRED_CHANNELS)} required channels present",
    )


def validate_temporal_metadata(xdict: dict[str, Any]) -> None:
    """
    Validate that required temporal metadata is present.

    Parameters
    ----------
    xdict : Dict[str, Any]
        Input dictionary with METADATA key

    Raises
    ------
    ValidationError
        If required temporal metadata is missing or invalid
    """
    if "METADATA" not in xdict:
        raise ValidationError("METADATA key missing from input dictionary")

    metadata = xdict["METADATA"]

    # Check for temporal information
    if not hasattr(metadata, "source_file_datetimes"):
        raise ValidationError(
            "Missing required temporal metadata: source_file_datetimes",
        )

    # Validate datetime structure
    try:
        datetimes = metadata.source_file_datetimes
        if len(datetimes) < MIN_TIME_STEPS:
            raise ValidationError(
                f"Need at least {MIN_TIME_STEPS} time steps, got {len(datetimes)}",
            )
    except (AttributeError, TypeError) as e:
        raise ValidationError(f"Invalid datetime structure: {e}") from e

    LOG.debug("Temporal metadata validation passed")


def prepare_octopy_input(xobj: xr.Dataset) -> dict[str, Any]:
    """
    Prepare input data structure for OCTOPY processing.

    Parameters
    ----------
    xobj : xr.Dataset
        Input xarray Dataset with satellite data

    Returns
    -------
    Dict[str, Any]
        Dictionary structured for OCTOPY algorithm input

    Raises
    ------
    InvalidInputError
        If input data cannot be properly structured
    ValidationError
        If temporal metadata validation fails
    """
    LOG.debug("Preparing input data structure for OCTOPY processing")

    try:
        if isinstance(xobj, dict) and "DATA" in xobj and "METADATA" in xobj:
            # Input already has expected structure
            xdict = xobj.copy()
            LOG.debug("Input data already in expected OCTOPY format")
        elif isinstance(xobj, xr.Dataset):
            # Restructure direct dataset input
            xdict = {
                "DATA": xobj.copy(),
                "METADATA": xr.Dataset(attrs=xobj.attrs),
            }
            LOG.debug("Restructured input data for OCTOPY format")
        else:
            raise InvalidInputError(
                f"Cannot prepare OCTOPY input from type {type(xobj)}",
            )

        # Validate the prepared structure
        validate_temporal_metadata(xdict)
        return xdict

    except (ValidationError, InvalidInputError):
        raise
    except Exception as e:
        raise InvalidInputError(f"Failed to prepare OCTOPY input structure: {e}") from e


def load_octopy_plugins() -> tuple[Any, Any]:
    """
    Load required OCTOPY algorithm plugins.

    Returns
    -------
    tuple[Any, Any]
        Tuple containing (uv_octopy_plugin, multi_variable_octopy_plugin)

    Raises
    ------
    PluginLoadError
        If any required plugins cannot be loaded
    """
    LOG.info("Loading OCTOPY algorithm plugins")

    required_plugins = ["uv_octopy", PLUGIN_NAME]
    loaded_plugins = {}

    for plugin_name in required_plugins:
        try:
            plugin = algorithms.get_plugin(plugin_name)
            if plugin is None:
                raise PluginLoadError(
                    f"Plugin '{plugin_name}' loaded but returned None",
                )
            loaded_plugins[plugin_name] = plugin
            LOG.debug(f"Successfully loaded plugin: {plugin_name}")
        except Exception as e:
            raise PluginLoadError(
                f"Failed to load required plugin '{plugin_name}': {e}",
            ) from e

    LOG.info("Successfully loaded all OCTOPY plugins")
    return loaded_plugins["uv_octopy"], loaded_plugins[PLUGIN_NAME]


def create_channel_overrides() -> dict[str, dict[str, Any]]:
    """
    Create channel-specific processing overrides.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary of channel-specific overrides
    """
    # Import the functions from the refactored module
    # This assumes the refactored code is available as a module
    try:
        from geoips.interfaces.algorithms.neo_multi_variable_octopy import (
            MIN_HEIGHT_THRESHOLD,
            cloud_height_finalize,
            cloud_height_sanitize,
        )

        return {
            "cld_height_acha": {
                "sanitize_func": cloud_height_sanitize,
                "finalize_func": cloud_height_finalize,
                "min_threshold": MIN_HEIGHT_THRESHOLD,
            },
            "cld_height_base": {
                "sanitize_func": cloud_height_sanitize,
                "finalize_func": cloud_height_finalize,
                "min_threshold": MIN_HEIGHT_THRESHOLD,
            },
        }
    except ImportError:
        LOG.warning("Could not import specialized functions, using defaults")
        return {}


def preserve_variable_attributes(
    output_dataset: xr.Dataset,
    original_dataset: xr.Dataset,
    variables: list[str],
) -> xr.Dataset:
    """
    Preserve original variable attributes in the output dataset.

    Parameters
    ----------
    output_dataset : xr.Dataset
        Output dataset from OCTOPY processing
    original_dataset : xr.Dataset
        Original input dataset with complete attributes
    variables : List[str]
        List of variable names to preserve attributes for

    Returns
    -------
    xr.Dataset
        Output dataset with preserved variable attributes

    Raises
    ------
    ProcessingError
        If attribute preservation fails
    """
    LOG.debug(f"Preserving attributes for {len(variables)} variables")

    try:
        # Get the data source
        if isinstance(original_dataset, dict) and "DATA" in original_dataset:
            source_data = original_dataset["DATA"]
        elif isinstance(original_dataset, xr.Dataset):
            source_data = original_dataset
        else:
            raise ProcessingError(
                f"Invalid original dataset type: {type(original_dataset)}",
            )

        # Preserve attributes for each specified variable
        preserved_count = 0
        for var_name in variables:
            if (
                var_name in output_dataset.data_vars
                and var_name in source_data.data_vars
            ):
                original_attrs = source_data[var_name].attrs
                output_dataset[var_name] = output_dataset[var_name].assign_attrs(
                    original_attrs,
                )
                preserved_count += 1
                LOG.debug(f"Preserved {len(original_attrs)} attributes for {var_name}")
            else:
                LOG.debug(
                    f"Skipping attribute preservation for {var_name} (not in both datasets)",
                )

        LOG.info(
            f"Preserved attributes for {preserved_count}/{len(variables)} variables",
        )
        return output_dataset

    except Exception as e:
        raise ProcessingError(f"Failed to preserve variable attributes: {e}") from e


def call(
    xobj: xr.Dataset,
    channels: list[str] | None = None,
    interp_method: str = "nn",
    use_occlusion_detection: bool = True,
    reference_mapping: dict[str, str] | None = None,
    default_reference: str = "temp_10_4um_nom",
) -> xr.Dataset:
    """
    Execute OCTOPY-CLAVRX temporal interpolation processing.

    This function orchestrates the complete OCTOPY-CLAVRX workflow using the
    enhanced multi-variable OCTOPY algorithm for efficient processing of
    multiple channels with optimized motion vector calculations.

    Parameters
    ----------
    xobj : xr.Dataset
        Input xarray Dataset containing multi-temporal CLAVR-x satellite data
    channels : Optional[List[str]], optional
        List of channels to process. If None, processes all required channels
    interp_method : str, optional
        Interpolation method ('nn' or 'bi'), by default "nn"
    use_occlusion_detection : bool, optional
        Whether to apply occlusion detection logic, by default True
    reference_mapping : Optional[Dict[str, str]], optional
        Custom mapping of channels to reference variables
    default_reference : str, optional
        Default reference variable for motion calculation, by default "temp_10_4um_nom"

    Returns
    -------
    xr.Dataset
        Output xarray Dataset containing temporally interpolated variables for
        all processed channels with preserved attributes and coordinate information

    Raises
    ------
    InvalidInputError
        If input data is invalid or improperly formatted
    ValidationError
        If required channels are missing or temporal metadata is invalid
    PluginLoadError
        If OCTOPY algorithm plugins cannot be loaded
    ProcessingError
        If any processing step fails

    Notes
    -----
    This plugin integrates two OCTOPY algorithms:
    - uv_octopy: Calculates motion vectors from brightness temperature fields
    - neo-multi-variable-octopy: Performs efficient multi-variable temporal interpolation

    The processing groups variables by reference variable to minimize motion vector
    calculations and maximize efficiency when processing multiple channels.

    Examples
    --------
    >>> # Process all required channels
    >>> result = algorithms.get_plugin("neo-octopy-clavrx")(clavrx_dataset)

    >>> # Process specific channels only
    >>> result = algorithms.get_plugin("neo-octopy-clavrx")(
    ...     clavrx_dataset,
    ...     channels=["cld_height_acha", "cld_height_base"]
    ... )

    >>> # Custom processing with different interpolation method
    >>> result = algorithms.get_plugin("neo-octopy-clavrx")(
    ...     clavrx_dataset,
    ...     interp_method="bi",
    ...     use_occlusion_detection=False
    ... )
    """
    LOG.info("Starting OCTOPY-CLAVRX temporal interpolation processing")

    # Use all required channels if none specified
    if channels is None:
        channels = REQUIRED_CHANNELS.copy()
        LOG.info(f"Processing all {len(channels)} required channels")
    else:
        LOG.info(f"Processing {len(channels)} specified channels: {channels}")

    # Validate input data - fail fast
    validate_input_data(xobj)

    # Prepare input data structure - fail fast
    xdict = prepare_octopy_input(xobj)

    # Load OCTOPY algorithm plugins - fail fast
    uv_octopy, multi_variable_octopy = load_octopy_plugins()

    # Calculate motion vectors
    LOG.info("Calculating motion vectors with UV OCTOPY")
    try:
        octopy_motion_vectors = uv_octopy(xdict)
        if octopy_motion_vectors is None:
            raise ProcessingError("Motion vector calculation returned None")
        xdict["OCTOPY"] = octopy_motion_vectors
        LOG.info("Motion vector calculation completed successfully")
    except Exception as e:
        if isinstance(e, ProcessingError):
            raise
        raise ProcessingError(f"Motion vector calculation failed: {e}") from e

    # Perform temporal interpolation for all channels
    LOG.info("Performing temporal interpolation with multi-variable OCTOPY")
    try:
        # Create channel overrides for specialized processing
        channel_overrides = create_channel_overrides()

        # Call the enhanced multi-variable OCTOPY algorithm
        interpolated_output = multi_variable_octopy(
            xdict,
            interp_method=interp_method,
            channels=channels,
            reference_mapping=reference_mapping,
            default_reference=default_reference,
            channel_overrides=channel_overrides,
            use_occlusion_detection=use_occlusion_detection,
        )

        if interpolated_output is None:
            raise ProcessingError("Temporal interpolation returned None")

        LOG.info("Temporal interpolation completed successfully for all channels")
    except Exception as e:
        if isinstance(e, ProcessingError):
            raise
        raise ProcessingError(f"Temporal interpolation failed: {e}") from e

    # Preserve original variable attributes
    try:
        final_output = preserve_variable_attributes(
            interpolated_output,
            xobj,
            channels,
        )
    except Exception as e:
        raise ProcessingError(f"Attribute preservation failed: {e}") from e

    # Validate output
    if final_output is None:
        raise ProcessingError("Final output is None after processing")

    # Verify expected variables are present
    missing_output_vars = [var for var in channels if var not in final_output.data_vars]
    if missing_output_vars:
        LOG.warning(f"Some expected output variables missing: {missing_output_vars}")

    # Log processing summary
    processed_vars = [var for var in channels if var in final_output.data_vars]
    LOG.info("OCTOPY-CLAVRX processing completed successfully")
    LOG.info(f"Output variables: {list(final_output.data_vars.keys())}")
    LOG.info(f"Output shape: {final_output.dims}")
    LOG.info(f"Processed {len(processed_vars)}/{len(channels)} requested channels")

    return final_output


def get_required_channels() -> list[str]:
    """
    Get the list of required channels for OCTOPY-CLAVRX processing.

    Returns
    -------
    List[str]
        List of required channel names
    """
    return REQUIRED_CHANNELS.copy()


def validate_octopy_plugins() -> bool:
    """
    Validate that required OCTOPY plugins are available.

    Returns
    -------
    bool
        True if all required plugins are available, False otherwise
    """
    required_plugins = ["uv_octopy", PLUGIN_NAME]

    try:
        for plugin_name in required_plugins:
            plugin = algorithms.get_plugin(plugin_name)
            if plugin is None:
                LOG.error(f"Plugin '{plugin_name}' loaded but returned None")
                return False
        LOG.info("All required OCTOPY plugins are available")
        return True
    except Exception as e:
        LOG.warning(f"OCTOPY plugin validation failed: {e}")
        return False


def validate_octopy_plugins_strict() -> None:
    """
    Validate that required OCTOPY plugins are available with strict error handling.

    Raises
    ------
    PluginLoadError
        If any required plugins are not available
    """
    if not validate_octopy_plugins():
        raise PluginLoadError("Required OCTOPY plugins are not available")


# Convenience functions for common processing patterns
def process_cloud_variables_only(xobj: xr.Dataset, **kwargs) -> xr.Dataset:
    """
    Process only cloud-related variables.

    Parameters
    ----------
    xobj : xr.Dataset
        Input dataset
    **kwargs
        Additional arguments passed to call()

    Returns
    -------
    xr.Dataset
        Dataset with interpolated cloud variables
    """
    cloud_channels = [
        "cloud_phase",
        "cld_height_base",
        "cld_height_acha",
        "cld_cwp_dcomp",
    ]
    return call(xobj, channels=cloud_channels, **kwargs)


def process_geometric_variables_only(xobj: xr.Dataset, **kwargs) -> xr.Dataset:
    """
    Process only geometric variables.

    Parameters
    ----------
    xobj : xr.Dataset
        Input dataset
    **kwargs
        Additional arguments passed to call()

    Returns
    -------
    xr.Dataset
        Dataset with interpolated geometric variables
    """
    geometric_channels = [
        "sensor_azimuth_angle",
        "sensor_zenith_angle",
        "surface_elevation",
    ]
    return call(xobj, channels=geometric_channels, **kwargs)


def process_atmospheric_variables_only(xobj: xr.Dataset, **kwargs) -> xr.Dataset:
    """
    Process only atmospheric variables.

    Parameters
    ----------
    xobj : xr.Dataset
        Input dataset
    **kwargs
        Additional arguments passed to call()

    Returns
    -------
    xr.Dataset
        Dataset with interpolated atmospheric variables
    """
    atmospheric_channels = [
        "freezing_altitude",
        "freezing_altitude_253",
        "freezing_altitude_268",
        "temp_10_4um_nom",
    ]
    return call(xobj, channels=atmospheric_channels, **kwargs)
