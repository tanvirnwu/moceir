import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.profile_utils import count_parameters, format_number


class FakeParameter:
    def __init__(self, count, requires_grad):
        self.count = count
        self.requires_grad = requires_grad

    def numel(self):
        return self.count


class FakeModel:
    def parameters(self):
        return [
            FakeParameter(10, True),
            FakeParameter(5, False),
            FakeParameter(7, True),
        ]


def test_count_parameters_splits_trainable_and_non_trainable():
    counts = count_parameters(FakeModel())

    assert counts["trainable"] == 17
    assert counts["non_trainable"] == 5
    assert counts["total"] == 22


def test_format_number_adds_commas():
    assert format_number(1234567) == "1,234,567"
