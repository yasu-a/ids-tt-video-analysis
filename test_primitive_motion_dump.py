from typing import Optional

import cv2
import numpy as np
import skimage.util
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
                    path='out.mp4',
                    fps=fps
            ) as vw:
                for i in tqdm(range(start, stop)):
                    snp_entry_target = snp_video_frame.get_entry(i)
                    snp_entry_next = snp_video_frame.get_entry(i + 1)
                    assert isinstance(snp_entry_target, snp_context.SNPEntryVideoFrame)
                    assert isinstance(snp_entry_next, snp_context.SNPEntryVideoFrame)

                    detector: Optional[PMDetector] = PMDetector(
                        PMDetectorParameter(enable_motion_correction=True)
                    )
                    source = PMDetectorSource(
                        target_frame=PMDetectorSourceTimeSeriesEntry(
                            original_image=snp_entry_target.original,
                            diff_image=snp_entry_target.motion,
                            timestamp=float(snp_entry_target.timestamp)
                        ),
                        next_frame=PMDetectorSourceTimeSeriesEntry(
                            original_image=snp_entry_next.original,
                            diff_image=snp_entry_next.motion,
                            timestamp=float(snp_entry_next.timestamp)
                        ),
                        detection_rect_normalized=train_input.frame_rects.normalized(video_name)
                    )
                    result = detector.compute(source=source)

                    xs = result.local_centroid

                    img = skimage.util.img_as_ubyte(result.original_images_clipped[0].copy())
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    for p1, p2 in zip(xs[0], xs[1]):
                        cv2.arrowedLine(
                            img,
                            p1[::-1],
                            p2[::-1],
                            (255, 255, 0),
                        )
                    # img = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
                    # cv2.imshow('win', img)
                    # cv2.waitKey(20)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    vw.write(img)


    main()
