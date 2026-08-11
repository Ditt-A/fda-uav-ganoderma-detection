from __future__ import annotations

import cv2
import numpy as np


def _low_frequency_bounds(height: int, width: int, beta: float) -> tuple[slice, slice]:
    if not 0.0 <= beta <= 0.5:
        raise ValueError("beta harus berada pada rentang [0, 0.5].")
    radius = max(1, int(np.floor(min(height, width) * beta)))
    cy, cx = height // 2, width // 2
    return slice(max(0, cy - radius), min(height, cy + radius + 1)), \
           slice(max(0, cx - radius), min(width, cx + radius + 1))


def fourier_amplitude_mix(
    content_bgr: np.ndarray,
    donor_bgr: np.ndarray,
    beta: float = 0.01,
    lambda_mix: float = 0.35,
    donor_resize: bool = True,
) -> np.ndarray:
    """
    Mempertahankan phase citra content, lalu mencampur low-frequency amplitude donor.

    A_mix = (1-lambda) * A_content + lambda * A_donor

    Bounding box content tetap valid karena tidak ada transformasi spasial.
    """
    if not 0.0 <= lambda_mix <= 1.0:
        raise ValueError("lambda_mix harus berada pada rentang [0, 1].")

    h, w = content_bgr.shape[:2]
    if donor_bgr.shape[:2] != (h, w):
        if not donor_resize:
            raise ValueError(
                "FDA donor/content dimensions differ while donor_resize=false: "
                f"content={(h, w)}, donor={donor_bgr.shape[:2]}"
            )
        interpolation = (
            cv2.INTER_AREA
            if donor_bgr.shape[0] >= h and donor_bgr.shape[1] >= w
            else cv2.INTER_LINEAR
        )
        donor_bgr = cv2.resize(donor_bgr, (w, h), interpolation=interpolation)

    content = content_bgr.astype(np.float32) / 255.0
    donor = donor_bgr.astype(np.float32) / 255.0

    fft_content = np.fft.fft2(content, axes=(0, 1))
    fft_donor = np.fft.fft2(donor, axes=(0, 1))

    amp_content = np.abs(fft_content)
    phase_content = np.angle(fft_content)
    amp_donor = np.abs(fft_donor)

    amp_content_shift = np.fft.fftshift(amp_content, axes=(0, 1))
    amp_donor_shift = np.fft.fftshift(amp_donor, axes=(0, 1))

    ys, xs = _low_frequency_bounds(h, w, beta)
    amp_content_shift[ys, xs, :] = (
        (1.0 - lambda_mix) * amp_content_shift[ys, xs, :]
        + lambda_mix * amp_donor_shift[ys, xs, :]
    )

    amp_mixed = np.fft.ifftshift(amp_content_shift, axes=(0, 1))
    mixed_fft = amp_mixed * np.exp(1j * phase_content)
    mixed = np.fft.ifft2(mixed_fft, axes=(0, 1)).real
    mixed = np.clip(mixed * 255.0, 0, 255).astype(np.uint8)
    return mixed
