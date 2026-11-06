class ZarrFMIntegrityError(RuntimeError):
    pass


class EditConflictError(RuntimeError):
    pass


class StaleMetadataWarning(UserWarning):
    pass
