import main_process_video_dump
from process import register_process_in_module, run

register_process_in_module(main_process_video_dump)

if __name__ == '__main__':
    run()
