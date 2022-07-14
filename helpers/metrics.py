import numpy as np
import torch
from skimage.metrics import structural_similarity
from helpers import ms_ssim


def get_mse(img1, img2):
    return np.mean(np.square(img1 - img2))


def get_rmse(img1, img2):
    mse = (img1 - img2) ** 2
    mse = np.mean(mse)
    return np.sqrt(mse)


def get_psnr(img1, img2):
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
        img2 = img2.detach().cpu().numpy()

    rmse = get_rmse(img1, img2)

    if rmse == 0:
        return 100
    return 20 * np.log10(1.0 / rmse)


def get_ssim(img1, img2):
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy().transpose(0, 2, 3, 1)
        img2 = img2.detach().cpu().numpy().transpose(0, 2, 3, 1)

        sum = 0
        for i in range(img1.shape[0]):
            i1 = img1[i]
            i2 = img2[i]

            sum += structural_similarity(i1, i2, multichannel=True, data_range=1.0, gaussian_weights=True,
                                         sigma=1.5, use_sample_covariance=False, win_size=11)
        return sum / img1.shape[0]

    res = structural_similarity(img1, img2, multichannel=True, data_range=1.0, gaussian_weights=True,
                                sigma=1.5, use_sample_covariance=False, win_size=11)
    return res


def get_ssim_pt(img1, img2):
    result, _ = ms_ssim.ssim(img1, img2, channel=3)
    return result


def get_psnr_pt(img1, img2):
    mse = torch.pow(img1 - img2, 2)
    mse = torch.mean(mse)
    rmse = torch.sqrt(mse)

    if rmse <= torch.finfo(img1.dtype).eps:
        return 100.0
    return 20 * torch.log10(1.0 / rmse)


if __name__ == '__main__':
    # torch.random.manual_seed(12345678)
    # assume BS 4

    img1 = torch.rand((4, 3, 256, 256))
    img2 = torch.rand((4, 3, 256, 256))

    print("psnr", get_psnr_pt(img1, img2), get_psnr(img1, img2))
    print("ssim", get_ssim_pt(img1, img2), get_ssim(img1, img2))
