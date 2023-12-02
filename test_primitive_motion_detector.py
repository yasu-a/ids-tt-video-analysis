from typing import Optional

import npstorage_context as snp_context
import storage
import storage.npstorage as snp
import train_input
from primitive_motion_detector import *

if __name__ == '__main__':
    def main():
        video_name = '20230205_04_Narumoto_Harimoto'

        with storage.create_instance(
                domain='numpy_storage',
                entity=video_name,
                context='frames',
                mode='r',
        ) as snp_video_frame:
            assert isinstance(snp_video_frame, snp.NumpyStorage)

            i = 191

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

            tester = PMDetectorTester(source=source, result=result)
            tester.test_all()

            # TODO: normalize result


    main()
