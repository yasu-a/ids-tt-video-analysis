DO_TYPE_CHECK = True  # TODO: turn off!


def check_dtype_and_shape(dtype, shape):
    if not DO_TYPE_CHECK:
        return lambda *_, **__: None

    def checker(a):
        if not isinstance(dtype, tuple):
            dtype_normalized = dtype,
        else:
            dtype_normalized = dtype

        if a.dtype not in dtype_normalized:
            raise TypeError(f'expected {dtype=}, provided {a.dtype=}')

        if a.ndim != len(shape):
            raise TypeError(f'expected ndim={len(shape)}, provided {a.ndim=}')

        for i in range(len(shape)):
            if shape[i] is None:
                continue
            if shape[i] != a.shape[i]:
                raise TypeError(
                    f'expected {shape=}, {shape[i]=} for dimension #{i}, '
                    f'provided array with {a.shape=}, whose size of dimension #{i} is {a.shape[i]}'
                )

    return checker


class OneWriteManyReadDescriptor:
    def __init__(self, default):
        self.__value = default
        self.__value_set = False

    def freeze(self):
        self.__value_set = True

    def __get__(self, obj, owner=None):
        if not self.__value_set:
            raise ValueError('attribute not set')
        return self.__value

    def __set__(self, obj, value):
        if self.__value_set:
            raise ValueError('attribute is read-only; only one write to attribute allowed')
        self.__value = value
        self.freeze()
