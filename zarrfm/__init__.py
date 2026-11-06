from . import canonical
from .dataset import DatasetReader, DatasetWriter, assign_splits, validate_dataset
from .editor import ZarrFMEditor, append_events, repair, set_bad
from .errors import EditConflictError, ZarrFMIntegrityError, StaleMetadataWarning
from .reader import ZarrFMReader
from .sampler import WindowSampler, collate, torch_dataset
from .spec import FORMAT, VERSION
from .validate import assert_valid, validate
from .writer import ZarrFMWriter, from_array

__all__ = [
    "ZarrFMReader",
    "ZarrFMWriter",
    "ZarrFMEditor",
    "append_events",
    "set_bad",
    "repair",
    "ZarrFMIntegrityError",
    "EditConflictError",
    "StaleMetadataWarning",
    "from_edf",
    "DatasetReader",
    "DatasetWriter",
    "WindowSampler",
    "assign_splits",
    "canonical",
    "collate",
    "from_array",
    "from_mne",
    "to_mne",
    "torch_dataset",
    "validate",
    "validate_dataset",
    "assert_valid",
    "FORMAT",
    "VERSION",
]
__version__ = VERSION


def __getattr__(name):
    if name in ("from_mne", "to_mne"):
        from . import mne_io

        return getattr(mne_io, name)
    if name == "from_edf":
        from . import edf_io

        return edf_io.from_edf
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
