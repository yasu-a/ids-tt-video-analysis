import concurrent.futures
import contextlib
import functools
import itertools
import os
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm

import npstorage_context as snp_context
import storage
import storage.npstorage as snp
from config import config

storage.context.forbid_writing = True


class FeatureDumpFailure(RuntimeError):
    pass


class DumpMotionFeature:
    def __init__(self, video_name, force=False):
        self._video_name = video_name
        self._force = force
        self._gt_path = os.path.join('./label_data/grand_truth', video_name + '.csv')
        self._out_path = os.path.join('features_out', video_name + '.csv')
        os.makedirs(os.path.dirname(self._out_path), exist_ok=True)

    @functools.cached_property
    def _gt_df(self):
        try:
            return pd.read_csv(self._gt_path)
        except FileNotFoundError:
            raise FeatureDumpFailure('grand truth not found')

    @functools.cached_property
    def _iter_frame_indexes(self):
        frame_index = self._gt_df.index
        return list(frame_index)

    def _iter_lk_motion(self):
        snp_lk_motion = storage.create_instance(
            domain='numpy_storage',
            entity=self._video_name,
            context='lk_motion',
            mode='r',
        )

        @contextlib.contextmanager
        def snp_lk_motion_enter():
            nonlocal snp_lk_motion
            try:
                with snp_lk_motion as snp_lk_motion:
                    yield snp_lk_motion
            except FileNotFoundError as e:
                raise FeatureDumpFailure(e)

        with snp_lk_motion_enter() as snp_lk_motion:
            assert isinstance(snp_lk_motion, snp.NumpyStorage)

            if np.any(snp_lk_motion.get_array('fi', fill_nan=-1) < 0):
                raise FeatureDumpFailure('lk_motion contains nan')

            for fi in tqdm(self._iter_frame_indexes):
                entry = snp_lk_motion.get_entry(fi)
                assert isinstance(entry, snp_context.SNPEntryLKMotion)
                assert entry.fi == fi

                yield entry

    NUM_GRIDS = np.array([16, 16])
    NUM_GRIDS_TOTAL = np.prod(NUM_GRIDS)

    def _iter_rows(self):
        def nanargmax(v):
            v = v[~np.isnan(v)]
            if len(v) == 0:
                return -1
            return np.argmax(v)

        for entry in self._iter_lk_motion():
            s, v = entry.start.astype(np.float16), entry.velocity.astype(np.float16)
            # assert np.all(np.logical_or(*np.isnan(s).T) == np.logical_or(*np.isnan(v).T))
            nonnull_index = ~np.isnan(s[:, 0])
            s, v = s[nonnull_index, :], v[nonnull_index, :]

            v_norm = np.linalg.norm(v, axis=1)

            grid_idx_xy = (s * self.NUM_GRIDS).astype(np.int32)
            grid_idx_serial = grid_idx_xy[:, 0] * self.NUM_GRIDS[1] + grid_idx_xy[:, 1]

            mat = np.full(shape=(*self.NUM_GRIDS, 2), dtype=np.float16, fill_value=np.nan)
            for x, y in itertools.product(range(self.NUM_GRIDS[0]), range(self.NUM_GRIDS[1])):
                max_idx = nanargmax(v_norm[grid_idx_serial == x * self.NUM_GRIDS[1] + y])
                if max_idx >= 0:  # nanargmax returns a valid index
                    mat[x, y, :] = v[max_idx].round(4)

            row = int(entry.fi), float(entry.timestamp), *mat.ravel()
            yield row

    @functools.cached_property
    def _gt_header(self):
        return self._gt_df.columns[1:]

    def _iter_row_with_gt(self):
        gt = self._gt_df
        for row in self._iter_rows():
            fi = row[0]
            row = *row, *gt.iloc[fi].values[1:]
            yield row

    def _cancel_if_exists(self):
        if self._force:
            return
        if os.path.exists(self._out_path):
            raise FeatureDumpFailure('dump already exists')

    def _dump(self):
        self._cancel_if_exists()

        header = [
            'fi',
            'ts',
            *(
                f'{axis}_{x}_{y}'
                for x in range(self.NUM_GRIDS[0])
                for y in range(self.NUM_GRIDS[1])
                for axis in 'xy'
            ),
            *self._gt_header
        ]
        rows = [row for row in self._iter_row_with_gt()]

        df = pd.DataFrame(rows, columns=header)
        df['ts'] = df['ts'].round(5)

        df.to_csv(self._out_path)

    @classmethod
    def dump(cls, *args, **kwargs):
        obj = cls(*args, **kwargs)
        obj._dump()
        del obj

    @classmethod
    def create_zip(cls):
        zip_path = './idsttva_dataset'
        src_dir_path = 'features_out'
        shutil.make_archive(zip_path, 'zip', root_dir=src_dir_path)


if __name__ == '__main__':
    def main(force=False):
        video_names = {sp.entity for sp in storage.StoragePath.list_storages()}

        futures = {}

        pool = concurrent.futures.ProcessPoolExecutor(max_workers=min(config.max_n_jobs, 4))
        with pool as pool:
            for video_name in video_names:
                f = pool.submit(DumpMotionFeature.dump, video_name, force=force)
                futures[f] = video_name

        results = []
        for future in concurrent.futures.as_completed(futures):
            video_name = futures[future]
            try:
                future.result()
                flag = 'SUCCESS'
            except FeatureDumpFailure as e:
                # import traceback
                # traceback.print_exc()
                flag = str(e)
            results.append(
                dict(
                    video_name=video_name,
                    flag=flag
                )
            )

        pd.set_option('display.max_columns', 500)
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.width', 190)
        pd.set_option('display.max_colwidth', 200)

        print(pd.DataFrame(results))

        DumpMotionFeature.create_zip()


    main(force=False)
