import os
import re

_dir_path = os.path.dirname(__file__)
process_stage_modules = []
for _name in os.listdir(_dir_path):
    _m = re.fullmatch(r'(process_.+)\.py', _name)
    if _m:
        _module_name = _m.group(1)
        exec(f'from . import {_module_name}')
        process_stage_modules.append(
            globals()[_module_name]
        )
