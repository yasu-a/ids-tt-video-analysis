import os

import timing
from legacy.extract import VideoFrameReader
from async_writer import AsyncVideoFrameWriter


class Job(timing.Timer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vfr = None

    def setup(self):
        video_path = os.path.expanduser(
            r'~/Desktop/idsttvideos/singles\20230205_04_Narumoto_Harimoto.mp4'
        )
        self.vfr = VideoFrameReader(video_path)
        self.writer = AsyncVideoFrameWriter(
            path=os.path.expanduser('~/Desktop/idsttvideos/out/test.mp4'),
            fps=self.vfr.fps
        )

    def context(self):
        return self.writer

    def target(self):
        for i, (pos, image) in enumerate(self.vfr[0:100:1]):
            image = image / 256.
            image = image * image
            image = image * 256.
            self.writer.write(image)


class Job2(Job):
    def target(self):
        for i, (pos, image) in enumerate(self.vfr[0:100:1]):
            self.writer.write(image)


def main():
    Job.measure(_repeat=3)
    #  611.584ms ±   27.795ms (CI=95%)


if __name__ == '__main__':
    main()
