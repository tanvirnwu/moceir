def format_restoration_report(psnr, ssim, lpips, fid, brisque, niqe, piqe):
    return "PSNR: {:f} SSIM: {:f} LPIPS: {:f} FID: {:f} BRISQUE: {:f} NIQE: {:f} PIQE: {:f}\n".format(
        psnr,
        ssim,
        lpips,
        fid,
        brisque,
        niqe,
        piqe,
    )
