import albumentations as A
import cv2

IGNORE_INDEX = 255


def zoom_in_out_transforms(out_size: int = 512, scale_range: tuple = (0.3, 1.5),
                           mask_fill: int = IGNORE_INDEX, image_fill: int = 0,
                           pad_position: str = "random") -> list:
    """Resize to out_size, rescale by s ~ U(scale_range), pad if s < 1, random-crop to out_size."""
    low, high = scale_range
    if not 0 < low <= high:
        raise ValueError(f"scale_range must satisfy 0 < low <= high, got {scale_range}")
    return [
        A.Resize(out_size, out_size, interpolation=cv2.INTER_AREA),
        A.RandomScale(scale_limit=(low - 1.0, high - 1.0), interpolation=cv2.INTER_LINEAR,
                      area_for_downscale="image", p=1.0),
        A.PadIfNeeded(min_height=out_size, min_width=out_size, position=pad_position,
                      border_mode=cv2.BORDER_CONSTANT, fill=image_fill, fill_mask=mask_fill),
        A.RandomCrop(height=out_size, width=out_size),
    ]


def zoom_in_only_transforms(out_size: int = 512, area_range: tuple = (0.3, 1.0)) -> list:
    """RandomResizedCrop: keeps a fraction of the frame area, so it can only zoom in."""
    low, high = area_range
    if not 0 < low <= high <= 1:
        raise ValueError(f"area_range must satisfy 0 < low <= high <= 1, got {area_range}")
    return [
        A.Resize(out_size, out_size, interpolation=cv2.INTER_AREA),
        A.RandomResizedCrop(size=(out_size, out_size), scale=(low, high), ratio=(1.0, 1.0),
                            interpolation=cv2.INTER_LINEAR, p=1.0),
    ]


def fixed_scale_transforms(scale: float, out_size: int = 512, mask_fill: int = IGNORE_INDEX,
                           image_fill: int = 0) -> list:
    """Centred, deterministic version of zoom_in_out_transforms at one scale."""
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    return [
        A.Resize(out_size, out_size, interpolation=cv2.INTER_AREA),
        A.RandomScale(scale_limit=(scale - 1.0, scale - 1.0), interpolation=cv2.INTER_LINEAR,
                      area_for_downscale="image", p=1.0),
        A.PadIfNeeded(min_height=out_size, min_width=out_size, position="center",
                      border_mode=cv2.BORDER_CONSTANT, fill=image_fill, fill_mask=mask_fill),
        A.CenterCrop(height=out_size, width=out_size),
    ]