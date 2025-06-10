"""
Module to apply OCTOPY temporally interpolated u, v motion vectors.

Takes in data from two CLAVR-x images and u, v motion vectors produced from uv_octopy
algorithm, then uses this data to temporally interpolate specified variables
for a GeoStitched product on the OVERCAST grid.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from octopy.utils.modules.algorithms.interp_octane import jma_array_nn
from octopy.utils.modules.algorithms.octane_warp_optical_flow import (
    octane_warp_optical_flow,
)
from octopy.utils.modules.algorithms.octopy_utils import interpolate_from_ut_vt
from xarray import Dataset

LOG = logging.getLogger(__name__)

# Constants
INTERFACE = "algorithms"
FAMILY = "xarray_dict_to_xarray"
NAME = "multi_variable_octopy"
MIN_HEIGHT_THRESHOLD = 500
OCCLUSION_THRESHOLD = 500
FORMATION_THRESHOLD = 500
TIME_SPLIT_THRESHOLD = 0.5

interface = "algorithms"
family = "xarray_to_xarray"
name = "neo_utvt_octopy"


@dataclass(frozen=True)
class VariableSpec:
    """
    Specification for a single variable to be interpolated.

    Attributes
    ----------
    name : str
        Variable name in the dataset
    reference_var : str
        Reference variable for motion calculation
    sanitize_func : Optional[Callable]
        Function to sanitize input data
    finalize_func : Optional[Callable]
        Function to finalize output data
    min_threshold : Optional[float]
        Minimum threshold for valid data
    """

    name: str
    reference_var: str
    sanitize_func: Callable | None = None
    finalize_func: Callable | None = None
    min_threshold: float | None = None


@dataclass(frozen=True)
class ProcessingConfig:
    """
    Configuration for multi-variable OCTOPY processing.

    Attributes
    ----------
    variables : list[VariableSpec]
        list of variable specifications
    use_occlusion_detection : bool
        Whether to apply occlusion detection logic
    interp_method : str
        Interpolation method ('nn' or 'bi')
    """

    variables: list[VariableSpec]
    use_occlusion_detection: bool = True
    interp_method: str = "nn"


@dataclass(frozen=True)
class MotionVectors:
    """
    Immutable container for motion vector data.

    Attributes
    ----------
    u : np.ndarray
        Horizontal motion vectors
    v : np.ndarray
        Vertical motion vectors
    ut : np.ndarray
        Temporal horizontal motion vectors
    vt : np.ndarray
        Temporal vertical motion vectors
    """

    u: np.ndarray
    v: np.ndarray
    ut: np.ndarray
    vt: np.ndarray


@dataclass(frozen=True)
class TimeParameters:
    """
    Immutable container for time-related parameters.

    Attributes
    ----------
    norm_time_arr : np.ndarray
        Normalized time array for interpolation
    time_to_interp : Any
        Target time for interpolation
    shape : Tuple[int, ...]
        Shape of the data arrays
    """

    norm_time_arr: np.ndarray
    time_to_interp: Any
    shape: tuple[int, ...]


# Pure utility functions
def remove_time_attributes(attrs: dict[str, Any]) -> dict[str, Any]:
    """
    Remove any attributes containing 'time' from a dictionary.

    Parameters
    ----------
    attrs : dict[str, Any]
        Dictionary of attributes to filter

    Returns
    -------
    dict[str, Any]
        New dictionary with time-related attributes removed
    """
    LOG.debug(f"Removing time attributes from {len(attrs)} total attributes")
    filtered_attrs = {k: v for k, v in attrs.items() if "time" not in k.lower()}
    LOG.debug(f"Retained {len(filtered_attrs)} attributes after filtering")
    return filtered_attrs


def extract_variable_attributes(dataset: Dataset, var_name: str) -> dict[str, Any]:
    """
    Extract attributes for a variable from the dataset, excluding time-related ones.

    Parameters
    ----------
    dataset : Dataset
        xarray Dataset containing the variable
    var_name : str
        Name of the variable to extract attributes from

    Returns
    -------
    Dict[str, Any]
        Dictionary of filtered attributes for the variable
    """
    if var_name not in dataset.variables:
        LOG.warning(f"Variable '{var_name}' not found in dataset")
        return {}

    attrs = dataset.variables[var_name].attrs
    LOG.debug(
        f"Extracting attributes for variable '{var_name}': {len(attrs)} attributes found",
    )
    return remove_time_attributes(attrs)


def default_sanitize_data(
    data: np.ndarray,
    min_threshold: float | None = None,
) -> np.ndarray:
    """
    Replace NaN values with 0 and optionally filter values below threshold.

    Parameters
    ----------
    data : np.ndarray
        Input data array to sanitize
    min_threshold : Optional[float]
        Minimum valid threshold. Values below this are set to 0

    Returns
    -------
    np.ndarray
        Sanitized data array
    """
    nan_count = np.sum(np.isnan(data))
    if nan_count > 0:
        LOG.debug(f"Replacing {nan_count} NaN values with 0")

    sanitized = np.copy(data)
    sanitized[np.isnan(sanitized)] = 0

    if min_threshold is not None:
        below_threshold = np.sum(sanitized < min_threshold)
        if below_threshold > 0:
            LOG.debug(
                f"Setting {below_threshold} values below threshold {min_threshold} to 0",
            )
        sanitized[sanitized < min_threshold] = 0

    return sanitized


def cloud_height_sanitize(
    data: np.ndarray,
    min_threshold: float = MIN_HEIGHT_THRESHOLD,
) -> np.ndarray:
    """
    Sanitize height data by replacing NaN values with 0 and filtering low values.

    Parameters
    ----------
    data : np.ndarray
        Input height data array to sanitize
    min_threshold : float
        Minimum valid height threshold in meters

    Returns
    -------
    np.ndarray
        Sanitized height data array
    """
    LOG.debug(f"Sanitizing cloud height data with threshold {min_threshold}m")
    return default_sanitize_data(data, min_threshold)


def default_finalize_data(data: np.ndarray) -> np.ndarray:
    """
    Return unmodified data array (default finalization).

    Parameters
    ----------
    data : np.ndarray
        Input data array

    Returns
    -------
    np.ndarray
        Unmodified data array
    """
    LOG.debug("Applying default finalization (no modifications)")
    return data


def cloud_height_finalize(data: np.ndarray) -> np.ndarray:
    """
    Finalize cloud height data by setting invalid values to NaN.

    Parameters
    ----------
    data : np.ndarray
        Cloud height data array

    Returns
    -------
    np.ndarray
        Finalized cloud height with invalid values set to NaN
    """
    LOG.debug("Applying cloud height finalization")

    final_data = np.copy(data)
    invalid_count = np.sum(final_data <= 0)
    final_data[final_data <= 0] = np.nan

    LOG.debug(f"Set {invalid_count} invalid height pixels to NaN")
    return final_data


# Core processing functions
def extract_variable_data(
    dataset: Dataset,
    var_spec: VariableSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract and sanitize variable data for both time steps.

    Parameters
    ----------
    dataset : Dataset
        Input dataset containing variable data
    var_spec : VariableSpec
        Variable specification with sanitization settings

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Sanitized data arrays for time steps (t1, t2)
    """
    LOG.debug(f"Extracting data for variable: {var_spec.name}")

    variable = dataset.variables[var_spec.name]
    sanitize_func = var_spec.sanitize_func or default_sanitize_data

    # Create sanitization function with threshold if specified
    if var_spec.min_threshold is not None:
        sanitize_with_threshold = partial(
            sanitize_func,
            min_threshold=var_spec.min_threshold,
        )
    else:
        sanitize_with_threshold = sanitize_func

    # Extract and sanitize data (assuming time index 1=earlier, 0=later)
    var_t1 = sanitize_with_threshold(np.asarray(variable[1].data))
    var_t2 = sanitize_with_threshold(np.asarray(variable[0].data))

    return var_t1, var_t2


def extract_all_variable_data(
    dataset: Dataset,
    variable_specs: list[VariableSpec],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Extract data for all specified variables using functional approach.

    Parameters
    ----------
    dataset : Dataset
        Input dataset containing variable data
    variable_specs : List[VariableSpec]
        List of variable specifications

    Returns
    -------
    Dict[str, Tuple[np.ndarray, np.ndarray]]
        Dictionary mapping variable names to (t1, t2) data tuples
    """
    LOG.info(f"Extracting data for {len(variable_specs)} variables")

    extract_func = partial(extract_variable_data, dataset)
    variable_data = {spec.name: extract_func(spec) for spec in variable_specs}

    LOG.debug(
        f"Successfully extracted data for variables: {list(variable_data.keys())}",
    )
    return variable_data


def calculate_time_parameters(
    file_datetimes: Any,
    shape: tuple[int, ...],
) -> TimeParameters:
    """
    Calculate normalized time array and related parameters for temporal interpolation.

    Parameters
    ----------
    file_datetimes : Any
        Nested structure containing start and end times for each file
    shape : Tuple[int, ...]
        Shape of the data arrays to create time grid

    Returns
    -------
    TimeParameters
        Container with normalized time array and interpolation parameters
    """
    LOG.info("Calculating time parameters for temporal interpolation")

    st1 = file_datetimes[1][0]  # Start time of first file
    st2 = file_datetimes[0][0]  # Start time of second file
    et1 = file_datetimes[1][1]  # End time of first file

    tt1 = (et1 - st1).total_seconds()
    LOG.debug(f"Time span of first file: {tt1} seconds")

    scan_secs1 = np.linspace(0, tt1, num=np.prod(shape), endpoint=True)
    time_arr1 = scan_secs1.reshape(shape)

    time_to_interp = st1
    time_dt2 = (time_to_interp - st1).total_seconds()
    norm_time_arr = np.abs((time_dt2 - time_arr1) / (st2 - st1).total_seconds())

    LOG.debug(
        f"Normalized time array range: [{norm_time_arr.min():.3f}, {norm_time_arr.max():.3f}]",
    )

    return TimeParameters(
        norm_time_arr=norm_time_arr,
        time_to_interp=time_to_interp,
        shape=shape,
    )


def calculate_motion_vectors_for_reference(
    dataset: Dataset,
    reference_var: str,
    u: np.ndarray,
    v: np.ndarray,
    norm_time_arr: np.ndarray,
) -> MotionVectors:
    """
    Calculate temporal motion vectors for a specific reference variable.

    Parameters
    ----------
    dataset : Dataset
        Input dataset containing reference variable
    reference_var : str
        Name of reference variable for motion calculation
    u : np.ndarray
        Initial horizontal motion vectors
    v : np.ndarray
        Initial vertical motion vectors
    norm_time_arr : np.ndarray
        Normalized time array for interpolation

    Returns
    -------
    MotionVectors
        Container with original and temporal motion vectors
    """
    LOG.debug(f"Calculating motion vectors using reference variable: {reference_var}")

    reference_data = dataset.variables[reference_var]
    f1, f2 = np.asarray(reference_data[1].data), np.asarray(reference_data[0].data)

    LOG.debug(
        f"Input motion vector ranges - u: [{u.min():.2f}, {u.max():.2f}], v: [{v.min():.2f}, {v.max():.2f}]",
    )

    ut, vt = octane_warp_optical_flow(f1, f2, u, v, norm_time_arr, donowcast=False)

    LOG.debug(
        f"Temporal motion vector ranges - ut: [{ut.min():.2f}, {ut.max():.2f}], vt: [{vt.min():.2f}, {vt.max():.2f}]",
    )

    return MotionVectors(u=u, v=v, ut=ut, vt=vt)


def calculate_displacement_coordinates(
    shape: tuple[int, ...],
    motion_vectors: MotionVectors,
    norm_time_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate displacement coordinates for temporal interpolation.

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of the data arrays
    motion_vectors : MotionVectors
        Container with temporal motion vectors
    norm_time_arr : np.ndarray
        Normalized time array for interpolation weights

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Displacement coordinate arrays (utx0, vtx0, utx1, vtx1)
    """
    LOG.debug("Calculating displacement coordinates")

    yarr, xarr = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    utx0 = xarr - motion_vectors.vt * norm_time_arr
    vtx0 = yarr - motion_vectors.ut * norm_time_arr
    utx1 = xarr + motion_vectors.vt * (1 - norm_time_arr)
    vtx1 = yarr + motion_vectors.ut * (1 - norm_time_arr)

    LOG.debug(
        f"Displacement coordinate ranges - utx0: [{utx0.min():.1f}, {utx0.max():.1f}], vtx0: [{vtx0.min():.1f}, {vtx0.max():.1f}]",
    )

    return utx0, vtx0, utx1, vtx1


def interpolate_variable_at_coordinates(
    var_data_t1: np.ndarray,
    var_data_t2: np.ndarray,
    displacement_coords: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    interp_method: str = "nn",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate single variable data using displacement coordinates.

    Parameters
    ----------
    var_data_t1 : np.ndarray
        Variable data at time step 1
    var_data_t2 : np.ndarray
        Variable data at time step 2
    displacement_coords : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Displacement coordinate arrays
    interp_method : str
        Interpolation method

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Interpolated data at displacement coordinates (i0x0, i1x1)
    """
    utx0, vtx0, utx1, vtx1 = displacement_coords
    interpolate_func = partial(interpolate_from_ut_vt, interp_method=interp_method)

    var_i0x0 = interpolate_func(var_data_t1, utx0, vtx0)
    var_i1x1 = interpolate_func(var_data_t2, utx1, vtx1)

    return var_i0x0, var_i1x1


def interpolate_all_variables(
    variable_data: dict[str, tuple[np.ndarray, np.ndarray]],
    displacement_coords: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    interp_method: str = "nn",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Interpolate all variables using displacement coordinates.

    Parameters
    ----------
    variable_data : Dict[str, Tuple[np.ndarray, np.ndarray]]
        Dictionary of variable data at both time steps
    displacement_coords : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Displacement coordinate arrays
    interp_method : str
        Interpolation method

    Returns
    -------
    Dict[str, Tuple[np.ndarray, np.ndarray]]
        Dictionary of interpolated variable data
    """
    LOG.info(
        f"Interpolating {len(variable_data)} variables using {interp_method} method",
    )

    interpolate_func = partial(
        interpolate_variable_at_coordinates,
        displacement_coords=displacement_coords,
        interp_method=interp_method,
    )

    interpolated_data = {
        var_name: interpolate_func(var_t1, var_t2)
        for var_name, (var_t1, var_t2) in variable_data.items()
    }

    LOG.debug("Variable interpolation completed for all variables")
    return interpolated_data


def calculate_occlusion_conditions(
    reference_data: tuple[np.ndarray, np.ndarray],
    reference_interpolated: tuple[np.ndarray, np.ndarray],
    displacement_coords: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    norm_time_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate occlusion and formation conditions for advanced temporal interpolation.

    Parameters
    ----------
    reference_data : Tuple[np.ndarray, np.ndarray]
        Reference variable data at both time steps
    reference_interpolated : Tuple[np.ndarray, np.ndarray]
        Interpolated reference data at displacement coordinates
    displacement_coords : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Displacement coordinate arrays
    norm_time_arr : np.ndarray
        Normalized time array for interpolation weights

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Boolean condition arrays (cond1, cond2, cond3, cond4)
    """
    LOG.debug("Calculating occlusion and formation conditions")

    utx0, vtx0, utx1, vtx1 = displacement_coords
    ref_t1, ref_t2 = reference_data
    ref_i0x0, ref_i1x1 = reference_interpolated

    ref_dt = ref_t1 - ref_t2
    ref_dt2 = ref_i0x0 - ref_i1x1

    occlusion_reasoning_arr = jma_array_nn(ref_dt, utx0, vtx0)
    occlusion_reasoning_arr2 = jma_array_nn(ref_dt, utx1, vtx1)

    cond1 = occlusion_reasoning_arr > OCCLUSION_THRESHOLD
    cond2 = occlusion_reasoning_arr2 < -OCCLUSION_THRESHOLD
    cond3 = (ref_dt2 < -FORMATION_THRESHOLD) & (norm_time_arr < TIME_SPLIT_THRESHOLD)
    cond4 = (ref_dt2 > FORMATION_THRESHOLD) & (norm_time_arr >= TIME_SPLIT_THRESHOLD)

    condition_counts = [np.sum(cond) for cond in [cond1, cond2, cond3, cond4]]
    LOG.debug(
        f"Condition pixel counts - occlusion: {condition_counts[0]}, occlusion2: {condition_counts[1]}, formation1: {condition_counts[2]}, formation2: {condition_counts[3]}",
    )

    return cond1, cond2, cond3, cond4


def apply_temporal_interpolation_single_variable(
    interpolated_data: tuple[np.ndarray, np.ndarray],
    conditions: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    norm_time_arr: np.ndarray,
    use_occlusion_detection: bool = True,
) -> np.ndarray:
    """
    Apply temporal interpolation with optional conditions for a single variable.

    Parameters
    ----------
    interpolated_data : Tuple[np.ndarray, np.ndarray]
        Interpolated variable data at displacement coordinates
    conditions : Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        Boolean condition arrays for special cases
    norm_time_arr : np.ndarray
        Normalized time array for interpolation weights
    use_occlusion_detection : bool
        Whether to apply occlusion detection logic

    Returns
    -------
    np.ndarray
        Final interpolated data for the variable
    """
    var_i0x0, var_i1x1 = interpolated_data

    # Base temporal interpolation
    var_data = (1 - norm_time_arr) * var_i0x0 + norm_time_arr * var_i1x1

    # Apply conditions only if occlusion detection is enabled and conditions provided
    if use_occlusion_detection and conditions is not None:
        cond1, cond2, cond3, cond4 = conditions

        # Apply conditions that use t1 values
        t1_conditions = cond1 | cond3
        if np.any(t1_conditions):
            var_data[t1_conditions] = var_i0x0[t1_conditions]

        # Apply conditions that use t2 values
        t2_conditions = cond2 | cond4
        if np.any(t2_conditions):
            var_data[t2_conditions] = var_i1x1[t2_conditions]

    return var_data


def apply_temporal_interpolation_all_variables(
    interpolated_data: dict[str, tuple[np.ndarray, np.ndarray]],
    conditions: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    norm_time_arr: np.ndarray,
    use_occlusion_detection: bool = True,
) -> dict[str, np.ndarray]:
    """
    Apply temporal interpolation with conditions to all variables.

    Parameters
    ----------
    interpolated_data : Dict[str, Tuple[np.ndarray, np.ndarray]]
        Dictionary of interpolated variable data
    conditions : Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        Boolean condition arrays for special cases
    norm_time_arr : np.ndarray
        Normalized time array for interpolation weights
    use_occlusion_detection : bool
        Whether to apply occlusion detection logic

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of final interpolated data for all variables
    """
    LOG.info(
        f"Applying temporal interpolation to {len(interpolated_data)} variables (occlusion detection: {use_occlusion_detection})",
    )

    interpolate_func = partial(
        apply_temporal_interpolation_single_variable,
        conditions=conditions,
        norm_time_arr=norm_time_arr,
        use_occlusion_detection=use_occlusion_detection,
    )

    final_data = {
        var_name: interpolate_func(var_interpolated)
        for var_name, var_interpolated in interpolated_data.items()
    }

    if use_occlusion_detection and conditions is not None:
        cond1, cond2, cond3, cond4 = conditions
        total_modified = np.sum(cond1 | cond2 | cond3 | cond4)
        LOG.info(
            f"Modified {total_modified} pixels due to occlusion/formation conditions",
        )

    return final_data


def finalize_variable_data(
    variable_data: dict[str, np.ndarray],
    variable_specs: list[VariableSpec],
) -> dict[str, np.ndarray]:
    """
    Apply finalization functions to all variables.

    Parameters
    ----------
    variable_data : Dict[str, np.ndarray]
        Dictionary of interpolated variable data
    variable_specs : List[VariableSpec]
        List of variable specifications with finalization functions

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of finalized variable data
    """
    LOG.info(f"Finalizing {len(variable_data)} variables")

    # Create mapping of variable names to finalization functions
    finalize_funcs = {
        spec.name: spec.finalize_func or default_finalize_data
        for spec in variable_specs
    }

    finalized_data = {
        var_name: finalize_funcs[var_name](var_data)
        for var_name, var_data in variable_data.items()
    }

    LOG.debug("Variable finalization completed for all variables")
    return finalized_data


def build_output_dataset(
    xarray_dict: dict[str, Dataset],
    variable_data: dict[str, np.ndarray],
    variable_specs: list[VariableSpec],
) -> Dataset:
    """
    Build the output xarray Dataset with interpolated variable data and metadata.

    Parameters
    ----------
    xarray_dict : Dict[str, Dataset]
        Dictionary containing input datasets
    variable_data : Dict[str, np.ndarray]
        Dictionary of final interpolated variable data
    variable_specs : List[VariableSpec]
        List of variable specifications

    Returns
    -------
    Dataset
        Output xarray Dataset with interpolated variables and metadata
    """
    LOG.info(f"Building output dataset with {len(variable_data)} variables")

    data_dataset = xarray_dict["DATA"]

    # Build data variables
    data_vars = {
        var_name: (
            ["y", "x"],
            var_data,
            extract_variable_attributes(data_dataset, var_name),
        )
        for var_name, var_data in variable_data.items()
    }

    # Build coordinates
    coords = {
        "longitude": (
            ["y", "x"],
            data_dataset.variables["longitude"],
            extract_variable_attributes(data_dataset, "longitude"),
        ),
        "latitude": (
            ["y", "x"],
            data_dataset.variables["latitude"],
            extract_variable_attributes(data_dataset, "latitude"),
        ),
    }

    # Build metadata
    metadata_attrs = remove_time_attributes(xarray_dict["METADATA"].attrs)

    LOG.debug(
        f"Output dataset created with {len(data_vars)} variables and {len(coords)} coordinates",
    )

    return Dataset(data_vars=data_vars, coords=coords, attrs=metadata_attrs)


def validate_input_data(
    xarray_dict: dict[str, Dataset],
    config: ProcessingConfig,
) -> None:
    """
    Validate that all required variables are present in input data.

    Parameters
    ----------
    xarray_dict : Dict[str, Dataset]
        Dictionary of input datasets
    config : ProcessingConfig
        Processing configuration

    Raises
    ------
    KeyError
        If required dataset keys are missing
    ValueError
        If required variables are missing
    """
    # Check required dataset keys
    required_keys = ["DATA", "OCTOPY", "METADATA"]
    missing_keys = [key for key in required_keys if key not in xarray_dict]
    if missing_keys:
        msg = f"Missing required dataset keys: {missing_keys}"
        LOG.error(msg)
        raise KeyError(msg)

    data_dataset = xarray_dict["DATA"]

    # Check required variables
    all_required_vars = set()
    for spec in config.variables:
        all_required_vars.add(spec.name)
        all_required_vars.add(spec.reference_var)
    all_required_vars.update(["latitude", "longitude"])

    missing_vars = [
        var for var in all_required_vars if var not in data_dataset.variables
    ]
    if missing_vars:
        msg = f"Missing required variables: {missing_vars}"
        LOG.error(msg)
        raise ValueError(msg)

    LOG.info("All required variables found in input data")


def group_variables_by_reference(
    variable_specs: list[VariableSpec],
) -> dict[str, list[VariableSpec]]:
    """
    Group variable specifications by their reference variable.

    Parameters
    ----------
    variable_specs : List[VariableSpec]
        List of variable specifications

    Returns
    -------
    Dict[str, List[VariableSpec]]
        Dictionary mapping reference variables to lists of variable specs
    """
    groups = {}
    for spec in variable_specs:
        if spec.reference_var not in groups:
            groups[spec.reference_var] = []
        groups[spec.reference_var].append(spec)

    LOG.debug(
        f"Grouped {len(variable_specs)} variables into {len(groups)} reference groups",
    )
    return groups


def process_octopy_interpolation(
    xarray_dict: dict[str, Dataset],
    config: ProcessingConfig,
) -> Dataset:
    """
    Process OCTOPY temporal interpolation for multiple variables.

    Parameters
    ----------
    xarray_dict : Dict[str, Dataset]
        Dictionary of xarray Datasets containing input data
    config : ProcessingConfig
        Configuration specifying variables and processing options

    Returns
    -------
    Dataset
        xarray Dataset with temporally interpolated variable data

    Raises
    ------
    ValueError
        If required variables are missing from input data
    KeyError
        If required keys are missing from xarray_dict
    """
    LOG.info(f"Starting OCTOPY interpolation for {len(config.variables)} variables")

    # Validate input data
    validate_input_data(xarray_dict, config)

    # Extract datasets
    data_dataset = xarray_dict["DATA"]
    octopy_dataset = xarray_dict["OCTOPY"]
    metadata = xarray_dict["METADATA"]

    # Extract motion vectors
    u = octopy_dataset.variables["u"].data
    v = octopy_dataset.variables["v"].data
    LOG.debug(f"Motion vector shapes - u: {u.shape}, v: {v.shape}")

    # Calculate time parameters (using first variable's shape)
    first_var = data_dataset.variables[config.variables[0].name]
    time_params = calculate_time_parameters(
        metadata.source_file_datetimes,
        np.asarray(first_var[0].data).shape,
    )

    # Extract variable data
    variable_data = extract_all_variable_data(data_dataset, config.variables)

    # Group variables by reference variable for efficient processing
    reference_groups = group_variables_by_reference(config.variables)

    all_interpolated_data = {}
    all_conditions = {}

    # Process each reference group
    for reference_var, var_specs in reference_groups.items():
        LOG.info(
            f"Processing {len(var_specs)} variables with reference: {reference_var}",
        )

        # Calculate motion vectors for this reference
        motion_vectors = calculate_motion_vectors_for_reference(
            data_dataset,
            reference_var,
            u,
            v,
            time_params.norm_time_arr,
        )

        # Calculate displacement coordinates
        displacement_coords = calculate_displacement_coordinates(
            time_params.shape,
            motion_vectors,
            time_params.norm_time_arr,
        )

        # Extract variables for this group
        group_variable_data = {
            spec.name: variable_data[spec.name] for spec in var_specs
        }

        # Interpolate variables
        group_interpolated = interpolate_all_variables(
            group_variable_data,
            displacement_coords,
            config.interp_method,
        )
        all_interpolated_data.update(group_interpolated)

        # Calculate conditions (only if occlusion detection is enabled)
        if config.use_occlusion_detection:
            # Use first variable in group as reference for conditions
            ref_var_name = var_specs[0].name
            reference_data = variable_data[ref_var_name]
            reference_interpolated = group_interpolated[ref_var_name]

            conditions = calculate_occlusion_conditions(
                reference_data,
                reference_interpolated,
                displacement_coords,
                time_params.norm_time_arr,
            )

            # Use same conditions for all variables in this reference group
            for spec in var_specs:
                all_conditions[spec.name] = conditions

    # Apply temporal interpolation with conditions
    final_data = {}
    for spec in config.variables:
        var_conditions = (
            all_conditions.get(spec.name) if config.use_occlusion_detection else None
        )
        var_interpolated = all_interpolated_data[spec.name]

        final_var_data = apply_temporal_interpolation_single_variable(
            var_interpolated,
            var_conditions,
            time_params.norm_time_arr,
            config.use_occlusion_detection,
        )
        final_data[spec.name] = final_var_data

    # Finalize data
    finalized_data = finalize_variable_data(final_data, config.variables)

    # Build output dataset
    result = build_output_dataset(xarray_dict, finalized_data, config.variables)

    LOG.info("OCTOPY interpolation completed successfully")
    return result


# Configuration builders
def create_variable_spec(
    name: str,
    reference_var: str,
    sanitize_func: Callable | None = None,
    finalize_func: Callable | None = None,
    min_threshold: float | None = None,
) -> VariableSpec:
    """
    Create a variable specification.

    Parameters
    ----------
    name : str
        Variable name
    reference_var : str
        Reference variable for motion calculation
    sanitize_func : Optional[Callable]
        Function to sanitize input data
    finalize_func : Optional[Callable]
        Function to finalize output data
    min_threshold : Optional[float]
        Minimum threshold for valid data

    Returns
    -------
    VariableSpec
        Variable specification object
    """
    return VariableSpec(
        name=name,
        reference_var=reference_var,
        sanitize_func=sanitize_func,
        finalize_func=finalize_func,
        min_threshold=min_threshold,
    )


def create_cloud_height_specs() -> list[VariableSpec]:
    """
    Create variable specifications for cloud height processing.

    Returns
    -------
    List[VariableSpec]
        List of variable specifications for cloud height variables
    """
    LOG.debug("Creating cloud height variable specifications")
    return [
        create_variable_spec(
            "cld_height_acha",
            "temp_10_4um_nom",
            cloud_height_sanitize,
            cloud_height_finalize,
            MIN_HEIGHT_THRESHOLD,
        ),
        create_variable_spec(
            "cld_height_base",
            "temp_10_4um_nom",
            cloud_height_sanitize,
            cloud_height_finalize,
            MIN_HEIGHT_THRESHOLD,
        ),
    ]


def create_processing_config(
    variable_specs: list[VariableSpec],
    use_occlusion_detection: bool = True,
    interp_method: str = "nn",
) -> ProcessingConfig:
    """
    Create a processing configuration.

    Parameters
    ----------
    variable_specs : List[VariableSpec]
        List of variable specifications
    use_occlusion_detection : bool
        Whether to apply occlusion detection logic
    interp_method : str
        Interpolation method

    Returns
    -------
    ProcessingConfig
        Processing configuration object
    """
    LOG.debug(f"Creating processing configuration for {len(variable_specs)} variables")
    return ProcessingConfig(
        variables=variable_specs,
        use_occlusion_detection=use_occlusion_detection,
        interp_method=interp_method,
    )


# Convenience functions for common use cases
def process_single_variable(
    xarray_dict: dict[str, Dataset],
    variable_name: str,
    reference_var: str,
    **kwargs,
) -> Dataset:
    """
    Process a single variable with OCTOPY interpolation.

    Parameters
    ----------
    xarray_dict : Dict[str, Dataset]
        Dictionary of input datasets
    variable_name : str
        Name of variable to interpolate
    reference_var : str
        Reference variable for motion calculation
    **kwargs
        Additional configuration options

    Returns
    -------
    Dataset
        Dataset with interpolated variable
    """
    spec = create_variable_spec(variable_name, reference_var, **kwargs)
    config = create_processing_config([spec])
    return process_octopy_interpolation(xarray_dict, config)


def process_variable_list(
    xarray_dict: dict[str, Dataset],
    variables: list[str],
    reference_var: str,
    **kwargs,
) -> Dataset:
    """
    Process a list of variables with the same reference variable.

    Parameters
    ----------
    xarray_dict : Dict[str, Dataset]
        Dictionary of input datasets
    variables : List[str]
        List of variable names to interpolate
    reference_var : str
        Reference variable for motion calculation
    **kwargs
        Additional configuration options applied to all variables

    Returns
    -------
    Dataset
        Dataset with all interpolated variables
    """
    specs = [create_variable_spec(var, reference_var, **kwargs) for var in variables]
    config = create_processing_config(specs)
    return process_octopy_interpolation(xarray_dict, config)


# Add these functions to the refactored OCTOPY code (after the existing functions)


def create_default_reference_mapping() -> dict[str, str]:
    """
    Create default mapping of channels to reference variables.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping channel names to their default reference variables
    """
    return {
        # Cloud variables - use infrared temperature
        "cld_height_acha": "temp_10_4um_nom",
        "cld_height_base": "temp_10_4um_nom",
        "cloud_phase": "temp_10_4um_nom",
        "cld_cwp_dcomp": "temp_10_4um_nom",
        # Temperature variables - self-reference
        "temp_10_4um_nom": "temp_10_4um_nom",
        # Geometric variables - use infrared temperature for motion
        "sensor_azimuth_angle": "temp_10_4um_nom",
        "sensor_zenith_angle": "temp_10_4um_nom",
        # Surface and atmospheric variables - use infrared temperature
        "surface_elevation": "temp_10_4um_nom",
        "freezing_altitude": "temp_10_4um_nom",
        "freezing_altitude_253": "temp_10_4um_nom",
        "freezing_altitude_268": "temp_10_4um_nom",
    }


def create_channel_specs_from_list(
    channels: list[str],
    reference_mapping: dict[str, str] | None = None,
    default_reference: str = "temp_10_4um_nom",
    sanitize_func: Callable | None = None,
    finalize_func: Callable | None = None,
    min_threshold: float | None = None,
    channel_overrides: dict[str, dict[str, Any]] | None = None,
) -> list[VariableSpec]:
    """
    Create variable specifications from a list of channel names.

    Parameters
    ----------
    channels : List[str]
        List of channel names to create specs for
    reference_mapping : Optional[Dict[str, str]]
        Mapping of channels to reference variables. If None, uses default mapping
    default_reference : str
        Default reference variable for channels not in mapping
    sanitize_func : Optional[Callable]
        Default sanitization function for all channels
    finalize_func : Optional[Callable]
        Default finalization function for all channels
    min_threshold : Optional[float]
        Default minimum threshold for all channels
    channel_overrides : Optional[Dict[str, Dict[str, Any]]]
        Per-channel overrides for specific settings

    Returns
    -------
    List[VariableSpec]
        List of variable specifications for the channels
    """
    LOG.debug(f"Creating variable specs for {len(channels)} channels")

    # Use default mapping if none provided
    if reference_mapping is None:
        reference_mapping = create_default_reference_mapping()

    # Default overrides to empty dict
    if channel_overrides is None:
        channel_overrides = {}

    specs = []
    for channel in channels:
        # Get reference variable
        reference_var = reference_mapping.get(channel, default_reference)

        # Get channel-specific overrides
        overrides = channel_overrides.get(channel, {})

        # Create spec with defaults and overrides
        spec = VariableSpec(
            name=channel,
            reference_var=reference_var,
            sanitize_func=overrides.get("sanitize_func", sanitize_func),
            finalize_func=overrides.get("finalize_func", finalize_func),
            min_threshold=overrides.get("min_threshold", min_threshold),
        )
        specs.append(spec)

        LOG.debug(f"Created spec for {channel} with reference {reference_var}")

    return specs


def call(
    xarray_dict: dict[str, Dataset],
    interp_method: str = "nn",
    channels: list[str] | None = None,
    reference_mapping: dict[str, str] | None = None,
    default_reference: str = "temp_10_4um_nom",
    sanitize_func: Callable | None = None,
    finalize_func: Callable | None = None,
    min_threshold: float | None = None,
    channel_overrides: dict[str, dict[str, Any]] | None = None,
    use_occlusion_detection: bool = True,
) -> Dataset:
    """
    Enhanced call interface for multi-variable OCTOPY processing.

    Parameters
    ----------
    xarray_dict : Dict[str, Dataset]
        Dictionary of xarray Datasets containing input data
    interp_method : str, optional
        Interpolation method ('nn' or 'bi'), by default "nn"
    channels : Optional[List[str]], optional
        List of channels to process. If None, processes legacy cloud height variables
    reference_mapping : Optional[Dict[str, str]], optional
        Mapping of channels to reference variables. If None, uses defaults
    default_reference : str, optional
        Default reference variable, by default "temp_10_4um_nom"
    sanitize_func : Optional[Callable], optional
        Default sanitization function for all channels
    finalize_func : Optional[Callable], optional
        Default finalization function for all channels
    min_threshold : Optional[float], optional
        Default minimum threshold for all channels
    channel_overrides : Optional[Dict[str, Dict[str, Any]]], optional
        Per-channel overrides for specific settings
    use_occlusion_detection : bool, optional
        Whether to apply occlusion detection logic, by default True

    Returns
    -------
    Dataset
        Processed xarray Dataset with interpolated variable data
    """
    LOG.info("Using enhanced multi-variable OCTOPY interface")

    # Handle legacy case - default to cloud height processing
    if channels is None:
        LOG.info("No channels specified, using legacy cloud height configuration")
        cloud_height_specs = create_cloud_height_specs()
        config = create_processing_config(
            cloud_height_specs,
            use_occlusion_detection,
            interp_method,
        )
    else:
        LOG.info(f"Processing {len(channels)} specified channels")

        # Create specs for all channels
        channel_specs = create_channel_specs_from_list(
            channels=channels,
            reference_mapping=reference_mapping,
            default_reference=default_reference,
            sanitize_func=sanitize_func,
            finalize_func=finalize_func,
            min_threshold=min_threshold,
            channel_overrides=channel_overrides,
        )

        config = create_processing_config(
            channel_specs,
            use_occlusion_detection,
            interp_method,
        )

    return process_octopy_interpolation(xarray_dict, config)
