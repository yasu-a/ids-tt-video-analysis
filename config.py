import os
import platform

device_name = platform.node()

if device_name == 'unknown':
    VIDEO_DIR_PATH = os.path.expanduser(r'~/Desktop/idsttvideos/singles')
    FRAME_DUMP_DIR_PATH = r'D:\idstt\iDSTTVideoFrameDump\frames'
    MOTION_DUMP_DIR_PATH = r'D:\idstt\iDSTTVideoFrameDump\motions'
    DEFAULT_VIDEO_NAME = '20230205_04_Narumoto_Harimoto'
    FEATURE_CACHE_PATH = r'D:\idstt\iDSTTVideoFrameDump\cache'
elif device_name == 'DESKTOP-O6M276J':  # Desktop of yasu.a
    VIDEO_DIR_PATH = os.path.expanduser(r'H:/idsttvideos/singles')
    FRAME_DUMP_DIR_PATH = r'H:\idstt\iDSTTVideoFrameDump\frames'
    MOTION_DUMP_DIR_PATH = r'H:\idstt\iDSTTVideoFrameDump\motions'
    DEFAULT_VIDEO_NAME = '20230205_04_Narumoto_Harimoto'
    FEATURE_CACHE_PATH = r'H:\idstt\iDSTTVideoFrameDump\cache'
else:
    assert False, device_name
