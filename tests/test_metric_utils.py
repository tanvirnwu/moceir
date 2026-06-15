import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.metric_utils import format_restoration_report


def test_format_restoration_report_includes_fid():
    report = format_restoration_report(psnr=30.123456, ssim=0.912345, lpips=0.123456, fid=12.345678)

    assert report == "PSNR: 30.123456 SSIM: 0.912345 LPIPS: 0.123456 FID: 12.345678\n"
