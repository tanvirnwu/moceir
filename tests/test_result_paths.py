import os
from types import SimpleNamespace

from utils.result_paths import result_dataset_name, result_dir


def test_allweather_flat_testset_uses_data_folder_name_for_results():
    opts = SimpleNamespace(
        checkpoint_id="2026_06_12_11_01_26",
        data_file_dir="/home/HDD/tanvir_HDD/datasets/ir/testsets/Test1/",
        trainset="allweather",
        benchmarks=["allweather"],
    )

    assert result_dataset_name(opts) == "Test1"
    assert result_dir(opts).endswith(os.path.join("results", "2026_06_12_11_01_26", "Test1"))


def test_existing_benchmark_result_folder_is_preserved():
    opts = SimpleNamespace(
        checkpoint_id="ckpt",
        data_file_dir="/datasets",
        trainset="standard",
        benchmarks=["derain"],
    )

    assert result_dataset_name(opts) == "derain"
