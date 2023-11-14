import sys

import numpy as np

np.set_printoptions(suppress=True)


def inspect(video_name, start=None, stop=None):
    assert start is None, 'providing start is forbidden'

    with dataset.VideoBaseFrameStorage(
            dataset.get_video_frame_dump_dir_path(video_name),
            mode='r'
    ) as vf_store:
        print(f'output={vf_store.count()}')
        total_shapes = {}
        for i in range(vf_store.count()):
            data_dct = vf_store.get(i)
            for k, v in data_dct.items():
                shape = v.shape
                total_shape = total_shapes.get(k)
                if total_shape is not None:
                    assert total_shape == shape, (total_shape, shape)
                total_shapes[k] = shape
        print({k: (vf_store.count(), *v) for k, v in total_shapes.items()})


def main(video_name):
    inspect(video_name)


if __name__ == '__main__':
    _, *args = sys.argv
    if args:
        vn = args[0]
    else:
        vn = None
    main(video_name=vn)
