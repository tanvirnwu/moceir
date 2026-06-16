def format_restoration_report(psnr, ssim, lpips, fid, brisque, niqe):
    return "PSNR: {:f} SSIM: {:f} LPIPS: {:f} FID: {:f} BRISQUE: {:f} NIQE: {:f}\n".format(
        psnr,
        ssim,
        lpips,
        fid,
        brisque,
        niqe,
    )
