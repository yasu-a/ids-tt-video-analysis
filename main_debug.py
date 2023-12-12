if __name__ == '__main__':
    import main

    run_name = 'ugt-debug'

    if run_name == 'marker-full-restore':
        import shutil

        try:
            shutil.rmtree('./label_data/markers')
        except FileNotFoundError:
            pass

        argv = ['mi', './label_data/__marker_source_json_v2/*.json']

    elif run_name == 'primitive-motion-dump-debug':
        argv = ['pmd', '20230219_03_Narumoto_Ito']

    elif run_name == 'primitive-motion-visualization-debug':
        argv = ['vpm', '20230219_03_Narumoto_Ito', '--start', '300', '--stop', '600']
    elif run_name == 'mu2-debug':
        argv = ['mu2']
    elif run_name == 'ugt-debug':
        argv = ['ugt']
    else:
        assert False, run_name

    main.run(argv)
