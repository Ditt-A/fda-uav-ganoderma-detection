from __future__ import annotations

import cv2
import albumentations as A


def build_geometric_transform(
    cfg: dict,
) -> A.Compose:
    aug = cfg["augmentation"]

    transforms: list[
        A.BasicTransform
    ] = []

    horizontal_flip_p = float(
        aug.get(
            "horizontal_flip_p",
            0.5,
        )
    )

    vertical_flip_p = float(
        aug.get(
            "vertical_flip_p",
            0.5,
        )
    )

    rotate90_p = float(
        aug.get(
            "rotate90_p",
            0.0,
        )
    )

    affine_p = float(
        aug.get(
            "affine_p",
            0.0,
        )
    )

    if horizontal_flip_p > 0:
        transforms.append(
            A.HorizontalFlip(
                p=horizontal_flip_p
            )
        )

    if vertical_flip_p > 0:
        transforms.append(
            A.VerticalFlip(
                p=vertical_flip_p
            )
        )

    if rotate90_p > 0:
        transforms.append(
            A.RandomRotate90(
                p=rotate90_p
            )
        )

    # Affine tidak dimasukkan sama sekali ketika affine_p = 0.
    if affine_p > 0:
        scale_limit = float(
            aug.get(
                "scale_limit",
                0.0,
            )
        )

        translate_limit = float(
            aug.get(
                "translate_limit",
                0.0,
            )
        )

        rotate_limit = float(
            aug.get(
                "rotate_limit",
                0.0,
            )
        )

        transforms.append(
            A.Affine(
                scale=(
                    1.0 - scale_limit,
                    1.0 + scale_limit,
                ),
                translate_percent={
                    "x": (
                        -translate_limit,
                        translate_limit,
                    ),
                    "y": (
                        -translate_limit,
                        translate_limit,
                    ),
                },
                rotate=(
                    -rotate_limit,
                    rotate_limit,
                ),
                shear=0,
                fit_output=False,
                border_mode=(
                    cv2.BORDER_REFLECT_101
                ),
                p=affine_p,
            )
        )

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=[
                "class_labels"
            ],
            min_visibility=0.20,
            clip=True,
        ),
    )