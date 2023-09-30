import legacy.frame_processor.mapper as _mapper
import frame_processor.stage as _stage
from .pipeline import create_pipeline


class mapper:
    process = _mapper.process
    unary = _mapper.unary
    past = _mapper.past
    future = _mapper.future
    to_double = _mapper.to_double
    to_uint = _mapper.to_uint


IMAGE = _mapper.Fields.IMAGE
POSITION = _mapper.Fields.POSITION


class stage:
    neighbour = _stage.neighbour
    each = _stage.each
    contiguous = _stage.contiguous
    store = _stage.store

    to_double = each(_mapper.to_double)
    to_uint = each(_mapper.to_uint)
