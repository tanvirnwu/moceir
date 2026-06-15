def format_restoration_report(psnr, ssim, lpips, fid):
    return "PSNR: {:f} SSIM: {:f} LPIPS: {:f} FID: {:f}\n".format(
        psnr,
        ssim,
        lpips,
        fid,
    )
