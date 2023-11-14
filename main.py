import process_extract_rect_cli
import process_video_dump
from process import register_process_in_module, run

register_process_in_module(process_video_dump)
register_process_in_module(process_extract_rect_cli)

if __name__ == '__main__':
    run()
