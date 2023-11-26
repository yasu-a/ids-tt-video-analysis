import argparse
import json
import os.path
import re

import cv2
import numpy as np
from tqdm import tqdm

import async_writer
import npstorage_context
import process
import storage
import storage.npstorage as snp
from config import config
from label_manager.frame_label.factory import VideoFrameLabelFactory
from label_manager.frame_label.sample_set import FrameAggregationResult


class ProcessStageLabeledFrameDump(process.ProcessStage):
    NAME = 'labeled-frame-dump'
    ALIASES = 'lfd',

    @classmethod
    def customize_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('video_names', type=str, nargs='+')
        parser.add_argument('--out', type=str, required=True)

    def __init__(self, video_names: list[str], out: str):
        raise NotImplementedError()

        self.__fac = VideoFrameLabelFactory.create_instance()

        real_video_names = []
        for maybe_video_name in video_names:
            if maybe_video_name in self.__fac.keys():
                real_video_names.append(maybe_video_name)
            else:
                pattern = re.compile(maybe_video_name)
                detected = False
                for video_name in self.__fac.keys():
                    if pattern.fullmatch(video_name):
                        real_video_names.append(video_name)
                        detected = True
                if not detected:
                    print(f'Warning: nothing extracted from {maybe_video_name!r}')
        real_video_names = list(set(real_video_names))

        self.__video_names = real_video_names
        self.__out_path = out

        os.makedirs(self.__out_path, exist_ok=True)

    def run(self):
        import app_logging
        app_logging.set_log_level(app_logging.INFO)

        print('Video names:')
        for vn in self.__video_names:
            print(f' {vn}')

        for video_name in self.__video_names:
            sample_set = self.__fac[video_name]
            agg: FrameAggregationResult = sample_set.aggregate_full()
            frame_groups = list(
                agg.extract_ordered_label_groups(
                    label_order=[
                        sample_set.label_array.index(ln)
                        for ln in ['Stay', 'Play', 'Ready']
                    ],
                    predicate=lambda entry: entry.reliability > 0.5
                )
            )

            video_path = os.path.join(config.video_location, video_name + '.mp4')
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            json_path = os.path.join(self.__out_path, video_name, 'frames.json')
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            json_data = {
                'frames': []
            }

            for i, group in enumerate(frame_groups):
                fi_start = int(
                    np.mean([
                        group.prev_frame.fi_center if group.prev_frame else 0,
                        group.frames[0].fi_center
                    ])
                )

                fi_end = int(
                    np.mean([
                        group.frames[-1].fi_center,
                        group.next_frame.fi_center if group.next_frame else frame_count - 1
                    ])
                )

                dump_file_name = f'{i:04d}.mp4'
                frame_json_data = dict(
                    i=i,
                    fi_start=fi_start,
                    fi_end=fi_end,
                    label_stay=group.frames[0].fi_center,
                    label_play=group.frames[1].fi_center,
                    label_ready=group.frames[2].fi_center,
                    dump_file_name=dump_file_name
                )
                frame_json_data = {
                    k: int(v) if isinstance(v, np.int32) else v
                    for k, v in frame_json_data.items()
                }
                json_data['frames'].append(frame_json_data)

            with open(json_path, 'w') as f:
                json.dump(json_data, f, sort_keys=True, ensure_ascii=True, indent=True)

            # FIXME: create dump from source video
            with storage.create_instance(
                    domain='numpy_storage',
                    entity=video_name,
                    context='frames',
                    mode='r',
            ) as snp_video_frame:
                assert isinstance(snp_video_frame, snp.NumpyStorage)

                tsd = np.diff(snp_video_frame.get_array('timestamp'))
                fps = 1 / tsd[tsd.mean() - tsd.std() <= tsd].mean()
                print(f'{fps=}')

                bar = tqdm(json_data['frames'])
                for entry in bar:
                    video_dump_path = os.path.join(
                        self.__out_path, video_name, 'dumps', entry['dump_file_name']
                    )
                    os.makedirs(os.path.dirname(video_dump_path), exist_ok=True)

                    with async_writer.AsyncVideoFrameWriter(video_dump_path, fps=fps) as w:
                        fis = np.arange(entry['fi_start'], entry['fi_end'] + 1)
                        for i, fi in enumerate(fis):
                            snp_entry: npstorage_context.SNPEntryVideoFrame \
                                = snp_video_frame.get_entry(fi)
                            w.write(snp_entry.original)
                            if i % 10 == 0:
                                bar.set_description(
                                    f'{video_name} {entry["dump_file_name"]} {i} / {len(fis)} '
                                    f'of frames[{fis[0]}:{fis[-1] + 1}]'
                                )

            cap.release()

        print('Done!')
