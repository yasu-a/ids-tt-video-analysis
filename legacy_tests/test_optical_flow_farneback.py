from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

import async_writer
import npstorage_context as snp_context
import storage
import storage.npstorage as snp
import train_input
from primitive_motion_detector import *

if __name__ == '__main__':
    def main():
        video_name = '20230205_04_Narumoto_Harimoto'
        start, stop = 200, 400

        rect = train_input.frame_rects.normalized(video_name)

        detector: Optional[PMDetector] = PMDetector(
            PMDetectorParameter(enable_motion_correction=True)
        )

        with storage.create_instance(
                domain='numpy_storage',
                entity=video_name,
                context='frames',
                mode='r',
        ) as snp_video_frame:
            assert isinstance(snp_video_frame, snp.NumpyStorage)

            start = start or 0
            stop = stop or snp_video_frame.count()

            tsd = np.diff(snp_video_frame.get_array('timestamp'))
            fps = 1 / tsd[tsd.mean() - tsd.std() <= tsd].mean()

            with async_writer.AsyncVideoFrameWriter(
                    path='../out.mp4',
                    fps=fps
            ) as vw:
                for i in tqdm(range(start, stop)):
                    snp_entry_prev = snp_video_frame.get_entry(i)
                    snp_entry_target = snp_video_frame.get_entry(i + 1)
                    assert isinstance(snp_entry_prev, snp_context.SNPEntryVideoFrame)
                    assert isinstance(snp_entry_target, snp_context.SNPEntryVideoFrame)

                    flow = cv2.calcOpticalFlowFarneback(
                        cv2.cvtColor(snp_entry_prev.original, cv2.COLOR_RGB2GRAY),
                        cv2.cvtColor(snp_entry_target.original, cv2.COLOR_RGB2GRAY),
                        None,
                        0.5,
                        3,
                        15,
                        3,
                        5,
                        1.2,
                        0
                    )

                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                    img = np.zeros_like(snp_entry_target.original)
                    img[..., 0] = ang * 180 / np.pi / 2
                    img[..., 1] = 255
                    img[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                    # img = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
                    # cv2.imshow('win', img)
                    # cv2.waitKey(20)
                    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
                    vw.write(img)


    main()
