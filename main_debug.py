import main

run_name = 'primitive-motion-dump-debug'

if run_name == 'marker-full-restore':
    import shutil

    try:
        shutil.rmtree('./label_data/markers')
    except FileNotFoundError:
        pass

    argv = ['mi', './label_data/__marker_source_json_v2/*.json']

elif run_name == 'primitive-motion-dump-debug':
    argv = ['pmd', '20230219_03_Narumoto_Ito']

else:
    assert False, run_name

main.run(argv)
