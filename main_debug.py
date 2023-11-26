import main

import shutil

try:
    shutil.rmtree('./label_data/markers')
except FileNotFoundError:
    pass
# argv = ['mi', './label_data/__marker_source_json_v2/*.json']
argv = ['dump-video', '20230219_03_Narumoto_Ito']

main.run(argv)
