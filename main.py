import process_defs
from process import register_process_in_module, run

for module in process_defs.process_stage_modules:
    register_process_in_module(module)

if __name__ == '__main__':
    run()
