import os


def result_dataset_name(opts):
    benchmark = opts.benchmarks[0]
    if opts.trainset.lower() == "allweather" and benchmark.lower() in ["allweather", "all"]:
        return os.path.basename(os.path.normpath(opts.data_file_dir))
    return benchmark


def result_dir(opts):
    return os.path.join(os.getcwd(), "results", opts.checkpoint_id, result_dataset_name(opts))
