import warp as wp

try:
    from warp._src.types import _ArrayAnnotationBase

    def _array_annotation_or(self, other):
        return self

    def _array_annotation_ror(self, other):
        return self

    _ArrayAnnotationBase.__or__ = _array_annotation_or
    _ArrayAnnotationBase.__ror__ = _array_annotation_ror

except Exception:
    pass
