"""OVERCAST Preprocessing Sanitization for CLAVR-X."""

import logging
from collections.abc import Collection, Mapping
from itertools import product

import xarray as xr
from overcast_preprocessing import sanitization

LOG = logging.getLogger(__name__)

interface = "algorithms"
family = "xarray_to_xarray"
name = "sanitize_clavrx"


def permute_indexes(indexes: Mapping[str, Collection]) -> list[dict[str, list]]:
    """
    Generate all possible permutations of index values.

    Given a dictionary where each key is associated with an iterable of values, this
    function computes the Cartesian product of the provided values and returns a list
    of dictionaries. Each dictionary represents one permutation by mapping each key
    to one value from its corresponding iterable.

    Parameters
    ----------
    indexes : dict
        A dictionary where each key corresponds to a label and the associated value
        is an iterable (e.g., list, tuple) of possible values.

    Returns
    -------
    list of dict
        A list of dictionaries, each representing a unique permutation of index values.
        Each dictionary maps the keys from `indexes` to one value selected from the
        corresponding iterable in the input dictionary.

    Examples
    --------
    >>> from itertools import product
    >>> indexes = {'a': [1, 2], 'b': [3, 4]}
    >>> permute_indexes(indexes)
    [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]

    Notes
    -----
    This solution was inspired by this Stack Overflow answer:
    https://stackoverflow.com/a/61557885
    """
    value_product = product(*[list(indexes[index]) for index in indexes])
    return [dict(zip(indexes.keys(), values, strict=False)) for values in value_product]


def call(xobj: xr.Dataset) -> xr.Dataset:
    """Sanitize an xarray ``Dataset`` for CLAVR‑x processing.

    The function wraps ``overcast_preprocessing.sanitization.sanitize_clavrx``
    so that it operates on every combination of explicit index values found in
    *xobj*.  For each combination it

    1. selects the matching subset,
    2. removes redundant coordinate labels,
    3. sanitizes the subset, and
    4. restores the coordinate labels.

    Sanitized subsets are concatenated along the first explicit index
    dimension.  If *xobj* contains no indexes, the entire dataset is sanitized
    in one call. This is nessesary to handle categorical index values and/or non-number
    index values as the overcast_preprocessing library only handles numerical data.

    Parameters
    ----------
    xobj : xr.Dataset
        Dataset to be sanitized. Optionally has categorical index values.

    Returns
    -------
    xr.Dataset
        Sanitized dataset.
    """
    # no explicit indexes -> sanitize the whole dataset and return
    if not xobj.indexes:
        return sanitization.sanitize_clavrx(xobj)

    sanitized_combos: list[xr.Dataset] = []

    # Build every possible combination of explicit index values
    index_array = permute_indexes(xobj.indexes)

    for combo in index_array:
        # Select the subset corresponding to *combo* and drop coord labels
        selection = xobj.sel(**combo).reset_coords(names=combo.keys(), drop=True)
        # Sanitize the subset
        sanitized_subset = sanitization.sanitize_clavrx(selection)
        # Re‑attach the coordinate labels that identify the subset
        sanitized_subset = sanitized_subset.assign_coords(**combo)
        sanitized_combos.append(sanitized_subset)

    # Concatenate all sanitized subsets along first index (any index would do)
    concat_dim = next(iter(xobj.indexes.keys()))
    return xr.concat(sanitized_combos, concat_dim)
