import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image


def install_dependency_stubs(monkeypatch):
    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.abs = lambda x: x
    torch.sum = lambda x, *args, **kwargs: x

    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")

    class Dataset:
        pass

    torch_utils_data.Dataset = Dataset
    torch_utils.data = torch_utils_data
    torch.utils = torch_utils

    torch_nn = types.ModuleType("torch.nn")

    class Module:
        pass

    torch_nn.Module = Module
    torch_nn.init = types.SimpleNamespace(
        uniform=lambda *args, **kwargs: None,
        constant=lambda *args, **kwargs: None,
        xavier_normal=lambda *args, **kwargs: None,
        kaiming_normal=lambda *args, **kwargs: None,
    )
    torch.nn = torch_nn

    torchvision = types.ModuleType("torchvision")
    torchvision_transforms = types.ModuleType("torchvision.transforms")

    class IdentityTransform:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, image):
            return image

    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, image):
            for transform in self.transforms:
                image = transform(image)
            return image

    class ToTensor:
        def __call__(self, image):
            return np.asarray(image)

    torchvision_transforms.ToPILImage = IdentityTransform
    torchvision_transforms.Compose = Compose
    torchvision_transforms.RandomCrop = IdentityTransform
    torchvision_transforms.ToTensor = ToTensor
    torchvision_transforms.Resize = IdentityTransform
    torchvision_transforms.Grayscale = IdentityTransform
    torchvision_transforms.InterpolationMode = types.SimpleNamespace(NEAREST="nearest")

    torchvision_utils = types.ModuleType("torchvision.utils")
    torchvision_utils.make_grid = lambda *args, **kwargs: None
    torchvision.transforms = torchvision_transforms
    torchvision.utils = torchvision_utils

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", torch_utils_data)
    monkeypatch.setitem(sys.modules, "torch.nn", torch_nn)
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.transforms", torchvision_transforms)
    monkeypatch.setitem(sys.modules, "torchvision.utils", torchvision_utils)
    monkeypatch.setitem(sys.modules, "cv2", types.ModuleType("cv2"))


@pytest.fixture()
def dataset_module(monkeypatch):
    install_dependency_stubs(monkeypatch)
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    monkeypatch.syspath_prepend(src_dir)

    for module_name in [
        "data.dataset_utils",
        "data.degradation_utils",
        "utils.image_utils",
    ]:
        sys.modules.pop(module_name, None)

    import data.dataset_utils as dataset_utils

    return dataset_utils


def save_image(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((32, 32, 3), value, dtype=np.uint8)
    Image.fromarray(image).save(path)


def test_allweather_train_dataset_pairs_flat_input_and_gt(dataset_module, tmp_path):
    root = tmp_path / "allweather"
    save_image(root / "input" / "rain_001.png", 10)
    save_image(root / "gt" / "rain_001.png", 200)
    save_image(root / "input" / "snow_001.jpg", 20)
    save_image(root / "gt" / "snow_001.jpg", 210)

    args = SimpleNamespace(
        data_file_dir=str(root),
        patch_size=16,
        de_type=["allweather"],
    )

    dataset = dataset_module.AllWeatherTrainDataset(args)

    assert len(dataset) == 2
    ([image_name, de_id], lr, hr) = dataset[0]
    assert Path(image_name).parent.name == "input"
    assert de_id == 0
    assert lr.shape == (16, 16, 3)
    assert hr.shape == (16, 16, 3)


def test_allweather_train_dataset_accepts_parent_dataset_root(dataset_module, tmp_path):
    root = tmp_path / "datasets"
    save_image(root / "allweather" / "input" / "sample.png", 10)
    save_image(root / "allweather" / "gt" / "sample.png", 200)

    args = SimpleNamespace(
        data_file_dir=str(root),
        patch_size=16,
        de_type=["allweather"],
    )

    dataset = dataset_module.AllWeatherTrainDataset(args)

    assert len(dataset) == 1


def test_allweather_train_dataset_rejects_unmatched_files(dataset_module, tmp_path):
    root = tmp_path / "allweather"
    save_image(root / "input" / "missing_gt.png", 10)
    (root / "gt").mkdir(parents=True)

    args = SimpleNamespace(
        data_file_dir=str(root),
        patch_size=16,
        de_type=["allweather"],
    )

    with pytest.raises(ValueError, match="Missing GT"):
        dataset_module.AllWeatherTrainDataset(args)


def test_allweather_test_dataset_reads_per_weather_folder(dataset_module, tmp_path):
    root = tmp_path / "allweather"
    save_image(root / "test" / "haze" / "input" / "haze_001.png", 10)
    save_image(root / "test" / "haze" / "gt" / "haze_001.png", 200)
    save_image(root / "test" / "rain" / "input" / "rain_001.png", 20)
    save_image(root / "test" / "rain" / "gt" / "rain_001.png", 210)

    args = SimpleNamespace(
        data_file_dir=str(root),
        benchmarks=["haze"],
        de_type=["allweather"],
    )

    dataset = dataset_module.AllWeatherTestDataset(args)

    assert len(dataset) == 1
    ([image_name, de_id], lr, hr) = dataset[0]
    assert Path(image_name).name == "haze_001.png"
    assert de_id == 0
    assert lr.shape == (32, 32, 3)
    assert hr.shape == (32, 32, 3)


def test_allweather_test_dataset_prefers_test_split_for_all_benchmark(dataset_module, tmp_path):
    root = tmp_path / "allweather"
    save_image(root / "input" / "train_sample.png", 1)
    save_image(root / "gt" / "train_sample.png", 2)
    save_image(root / "test" / "haze" / "input" / "haze_001.png", 10)
    save_image(root / "test" / "haze" / "gt" / "haze_001.png", 200)
    save_image(root / "test" / "rain" / "input" / "rain_001.png", 20)
    save_image(root / "test" / "rain" / "gt" / "rain_001.png", 210)

    args = SimpleNamespace(
        data_file_dir=str(root),
        benchmarks=["allweather"],
        de_type=["allweather"],
    )

    dataset = dataset_module.AllWeatherTestDataset(args)

    names = [Path(sample["input"]).name for sample in dataset.samples]
    assert names == ["haze_001.png", "rain_001.png"]
