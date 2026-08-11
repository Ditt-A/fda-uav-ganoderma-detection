from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from src.common import (
    find_data_yaml,
    find_split_dir,
    iter_images,
    load_yaml,
    matching_label,
    parse_names,
    read_image,
    safe_copy,
    save_json,
    save_yaml,
)


def filename_group_key(
    path: Path,
    delimiter: str,
) -> str:
    """
    Ambil base filename sebelum hash Roboflow.

    Contoh:
    ABC_PNG.rf.123456.jpg
    menjadi:
    ABC_PNG
    """
    if delimiter in path.name:
        return path.name.split(
            delimiter,
            1,
        )[0]

    return path.stem


def safe_stem(text: str) -> str:
    """
    Pastikan nama output aman dipakai sebagai filename.
    """
    cleaned = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        text,
    )

    return cleaned.strip("._") or "image"


# ============================================================
# Visual similarity
# Digunakan untuk memecah kelompok ukuran 6 menjadi 3 + 3.
# ============================================================

def dihedral_views(
    image: np.ndarray,
) -> list[np.ndarray]:
    """
    Delapan orientasi:
    - empat rotasi;
    - empat rotasi setelah horizontal flip.

    Ini membantu membandingkan citra yang berasal dari sumber
    sama tetapi sudah mendapat flip atau rotasi 90 derajat.
    """
    views = []

    for base_image in (
        image,
        cv2.flip(image, 1),
    ):
        for rotation_index in range(4):
            rotated = np.rot90(
                base_image,
                rotation_index,
            )

            views.append(
                np.ascontiguousarray(rotated)
            )

    return views


def perceptual_hash(
    image: np.ndarray,
) -> np.ndarray:
    """
    Perceptual hash berbasis DCT.
    """
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.resize(
        gray,
        (32, 32),
        interpolation=cv2.INTER_AREA,
    )

    dct = cv2.dct(
        np.float32(gray)
    )[:8, :8]

    median = np.median(
        dct.flatten()[1:]
    )

    return (
        dct > median
    ).flatten()


def orientation_hashes(
    image: np.ndarray,
) -> list[np.ndarray]:
    return [
        perceptual_hash(view)
        for view in dihedral_views(image)
    ]


def minimum_hash_distance(
    hashes_a: list[np.ndarray],
    hashes_b: list[np.ndarray],
) -> float:
    best_distance = 1.0

    for hash_a in hashes_a:
        for hash_b in hashes_b:
            distance = float(
                np.count_nonzero(
                    hash_a != hash_b
                )
                / hash_a.size
            )

            best_distance = min(
                best_distance,
                distance,
            )

    return best_distance


def color_histogram(
    image: np.ndarray,
) -> np.ndarray:
    """
    Histogram HSV membantu membandingkan tampilan warna global.
    """
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [16, 16],
        [0, 180, 0, 256],
    )

    cv2.normalize(
        histogram,
        histogram,
    )

    return histogram


def candidate_metadata(
    path: Path,
) -> dict:
    image = read_image(path)

    return {
        "path": path,
        "hashes": orientation_hashes(image),
        "histogram": color_histogram(image),
    }


def visual_distance(
    candidate_a: dict,
    candidate_b: dict,
) -> float:
    hash_distance = minimum_hash_distance(
        candidate_a["hashes"],
        candidate_b["hashes"],
    )

    histogram_distance = float(
        cv2.compareHist(
            candidate_a["histogram"],
            candidate_b["histogram"],
            cv2.HISTCMP_BHATTACHARYYA,
        )
    )

    return (
        0.80 * hash_distance
        + 0.20 * histogram_distance
    )


def mean_internal_distance(
    indices: tuple[int, ...],
    distance_matrix: np.ndarray,
) -> float:
    pairs = list(
        itertools.combinations(
            indices,
            2,
        )
    )

    if not pairs:
        return 0.0

    return float(
        np.mean(
            [
                distance_matrix[i, j]
                for i, j in pairs
            ]
        )
    )


def split_six_into_triplets(
    paths: list[Path],
) -> tuple[
    list[list[Path]],
    float,
    float,
]:
    """
    Kelompok berisi enam file dianggap berasal dari dua source image
    yang kebetulan memiliki base filename sama.

    Semua kemungkinan pembagian 3 + 3 diuji.
    Dipilih pembagian dengan jarak internal paling kecil.
    """
    if len(paths) != 6:
        raise ValueError(
            "split_six_into_triplets "
            "hanya menerima enam gambar."
        )

    paths = sorted(paths)

    metadata = [
        candidate_metadata(path)
        for path in paths
    ]

    distance_matrix = np.zeros(
        (6, 6),
        dtype=np.float32,
    )

    for i, j in itertools.combinations(
        range(6),
        2,
    ):
        distance = visual_distance(
            metadata[i],
            metadata[j],
        )

        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance

    partitions = []

    # Index 0 selalu dimasukkan ke cluster A.
    # Ini mencegah partisi sama dihitung dua kali.
    for pair in itertools.combinations(
        range(1, 6),
        2,
    ):
        cluster_a = tuple(
            sorted(
                (
                    0,
                    *pair,
                )
            )
        )

        cluster_b = tuple(
            index
            for index in range(6)
            if index not in cluster_a
        )

        score = (
            mean_internal_distance(
                cluster_a,
                distance_matrix,
            )
            + mean_internal_distance(
                cluster_b,
                distance_matrix,
            )
        )

        partitions.append(
            (
                score,
                cluster_a,
                cluster_b,
            )
        )

    partitions.sort(
        key=lambda item: item[0]
    )

    best_score, best_a, best_b = (
        partitions[0]
    )

    second_score = partitions[1][0]

    cluster_margin = float(
        second_score - best_score
    )

    cluster_a_paths = [
        paths[index]
        for index in best_a
    ]

    cluster_b_paths = [
        paths[index]
        for index in best_b
    ]

    return (
        [
            cluster_a_paths,
            cluster_b_paths,
        ],
        float(best_score),
        cluster_margin,
    )


# ============================================================
# Black-border detection dan pemilihan representatif
# ============================================================

def border_artifact_metrics(
    image: np.ndarray,
    dark_threshold: int,
    border_fraction: float = 0.12,
) -> dict[str, float]:
    """
    Mengukur artefak hitam yang terhubung ke tepi gambar.

    Shear dengan black-fill biasanya menghasilkan area gelap besar
    yang menyentuh salah satu sisi gambar.
    """
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    dark_mask = (
        gray <= dark_threshold
    ).astype(np.uint8)

    height, width = gray.shape

    border_height = max(
        1,
        int(height * border_fraction),
    )

    border_width = max(
        1,
        int(width * border_fraction),
    )

    border_mask = np.zeros_like(
        dark_mask,
        dtype=bool,
    )

    border_mask[:border_height, :] = True
    border_mask[-border_height:, :] = True
    border_mask[:, :border_width] = True
    border_mask[:, -border_width:] = True

    border_dark_ratio = float(
        dark_mask[border_mask].mean()
    )

    (
        component_count,
        _,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        dark_mask,
        connectivity=8,
    )

    connected_border_area = 0

    for component_id in range(
        1,
        component_count,
    ):
        x, y, w, h, area = (
            stats[component_id]
        )

        touches_border = (
            x == 0
            or y == 0
            or x + w >= width
            or y + h >= height
        )

        if touches_border:
            connected_border_area += int(area)

    connected_border_ratio = (
        connected_border_area
        / float(height * width)
    )

    global_black_ratio = float(
        dark_mask.mean()
    )

    artifact_score = (
        0.65 * connected_border_ratio
        + 0.25 * border_dark_ratio
        + 0.10 * global_black_ratio
    )

    return {
        "artifact_score": float(
            artifact_score
        ),
        "global_black_ratio": (
            global_black_ratio
        ),
        "border_dark_ratio": (
            border_dark_ratio
        ),
        "connected_border_ratio": float(
            connected_border_ratio
        ),
    }


def sharpness(
    image: np.ndarray,
) -> float:
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )


def choose_representative(
    paths: list[Path],
    dark_threshold: int,
) -> tuple[
    dict,
    list[dict],
    float,
]:
    """
    Pilih kandidat dengan:
    1. artifact score paling rendah;
    2. black ratio paling rendah;
    3. ketajaman paling tinggi sebagai tie-breaker.

    Ini adalah likely clean representative, bukan jaminan bahwa
    file tersebut selalu original.
    """
    candidates = []

    for path in sorted(paths):
        image = read_image(path)

        metrics = border_artifact_metrics(
            image,
            dark_threshold,
        )

        candidates.append(
            {
                "path": path,
                **metrics,
                "sharpness": sharpness(
                    image
                ),
            }
        )

    candidates.sort(
        key=lambda item: (
            item["artifact_score"],
            item["global_black_ratio"],
            -item["sharpness"],
        )
    )

    selection_margin = 1.0

    if len(candidates) > 1:
        selection_margin = float(
            candidates[1]["artifact_score"]
            - candidates[0]["artifact_score"]
        )

    return (
        candidates[0],
        candidates,
        selection_margin,
    )


# ============================================================
# File operations
# ============================================================

def copy_selected_pair(
    selected_image: Path,
    source_images: Path,
    source_labels: Path,
    output_images: Path,
    output_labels: Path,
    output_stem: str,
) -> None:
    output_images.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_labels.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_image = (
        output_images
        / f"{output_stem}"
        f"{selected_image.suffix.lower()}"
    )

    output_label = (
        output_labels
        / f"{output_stem}.txt"
    )

    safe_copy(
        selected_image,
        output_image,
    )

    source_label = matching_label(
        selected_image,
        source_images,
        source_labels,
    )

    if source_label.exists():
        safe_copy(
            source_label,
            output_label,
        )
    else:
        output_label.write_text(
            "",
            encoding="utf-8",
        )


def copy_split_unchanged(
    data_yaml_path: Path,
    split: str,
    output_root: Path,
) -> int:
    """
    Validation dan test disalin tanpa recovery karena augmentasi
    Roboflow diterapkan pada train set.
    """
    found = find_split_dir(
        data_yaml_path.parent,
        split,
    )

    if not found:
        return 0

    source_images, source_labels = found

    output_images = (
        output_root
        / split
        / "images"
    )

    output_labels = (
        output_root
        / split
        / "labels"
    )

    output_images.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_labels.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    for image_path in iter_images(
        source_images
    ):
        relative_path = image_path.relative_to(
            source_images
        )

        output_image = (
            output_images
            / relative_path
        )

        output_label = (
            output_labels
            / relative_path
        ).with_suffix(".txt")

        safe_copy(
            image_path,
            output_image,
        )

        source_label = matching_label(
            image_path,
            source_images,
            source_labels,
        )

        if source_label.exists():
            safe_copy(
                source_label,
                output_label,
            )
        else:
            output_label.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_label.write_text(
                "",
                encoding="utf-8",
            )

        count += 1

    return count


# ============================================================
# Review sheets
# ============================================================

def create_thumbnail(
    path: Path,
    title: str,
    selected: bool,
    size: int = 240,
) -> np.ndarray:
    image = read_image(path)

    thumbnail = cv2.resize(
        image,
        (size, size),
        interpolation=cv2.INTER_AREA,
    )

    border_color = (
        (0, 255, 0)
        if selected
        else (255, 255, 255)
    )

    cv2.rectangle(
        thumbnail,
        (1, 1),
        (size - 2, size - 2),
        border_color,
        5 if selected else 2,
    )

    cv2.rectangle(
        thumbnail,
        (0, 0),
        (size, 34),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        thumbnail,
        title[:32],
        (7, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        border_color,
        1,
        cv2.LINE_AA,
    )

    return thumbnail


def create_review_sheet(
    group_key: str,
    clusters: list[list[Path]],
    selected_paths: list[Path],
    output_path: Path,
) -> None:
    selected_names = {
        path.name
        for path in selected_paths
    }

    rows = []

    for cluster_index, paths in enumerate(
        clusters
    ):
        panels = []

        for path in paths:
            is_selected = (
                path.name in selected_names
            )

            title = (
                f"C{cluster_index + 1} "
                f"{'SELECTED' if is_selected else ''}"
            )

            panels.append(
                create_thumbnail(
                    path,
                    title,
                    is_selected,
                )
            )

        rows.append(
            cv2.hconcat(panels)
        )

    sheet = cv2.vconcat(rows)

    title_canvas = np.zeros(
        (
            45,
            sheet.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    cv2.putText(
        title_canvas,
        group_key[:90],
        (8, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    sheet = cv2.vconcat(
        [
            title_canvas,
            sheet,
        ]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        sheet,
    )


# ============================================================
# Main recovery
# ============================================================

def recover_source(
    source_name: str,
    source: dict,
) -> None:
    recovery = source.get(
        "recovery",
        {},
    )

    if not recovery.get(
        "enabled",
        False,
    ):
        print(
            f"[SKIP] {source_name}: "
            "recovery disabled"
        )
        return

    input_root = Path(
        recovery.get(
            "input",
            source.get(
                "download_output",
                source["output"],
            ),
        )
    )

    output_root = Path(
        recovery.get(
            "output",
            source["output"],
        )
    )

    report_root = Path(
        recovery.get(
            "report_dir",
            "reports/recovery",
        )
    )

    delimiter = str(
        recovery.get(
            "filename_delimiter",
            ".rf.",
        )
    )

    versions_per_source = int(
        recovery.get(
            "expected_versions_per_source",
            3,
        )
    )

    expected_train_sources = (
        recovery.get(
            "expected_train_sources"
        )
    )

    dark_threshold = int(
        recovery.get(
            "black_threshold",
            12,
        )
    )

    max_selected_black_ratio = float(
        recovery.get(
            "max_selected_black_ratio",
            0.03,
        )
    )

    selection_margin_review = float(
        recovery.get(
            "selection_margin_review",
            0.002,
        )
    )

    cluster_margin_review = float(
        recovery.get(
            "cluster_margin_review",
            0.01,
        )
    )

    overwrite = bool(
        recovery.get(
            "overwrite",
            True,
        )
    )

    if (
        input_root.resolve()
        == output_root.resolve()
    ):
        raise ValueError(
            f"{source_name}: recovery input "
            "dan output tidak boleh sama."
        )

    if not input_root.exists():
        raise FileNotFoundError(
            f"{source_name}: input recovery "
            f"tidak ditemukan: {input_root}"
        )

    if output_root.exists():
        if overwrite:
            shutil.rmtree(
                output_root
            )

        elif any(output_root.iterdir()):
            print(
                f"[SKIP] {source_name}: "
                f"output recovery sudah ada "
                f"di {output_root}"
            )
            return

    data_yaml_path = find_data_yaml(
        input_root
    )

    data_yaml = load_yaml(
        data_yaml_path
    )

    names = parse_names(
        data_yaml
    )

    train_found = find_split_dir(
        data_yaml_path.parent,
        "train",
    )

    if not train_found:
        raise FileNotFoundError(
            f"{source_name}: "
            "split train tidak ditemukan."
        )

    train_images, train_labels = (
        train_found
    )

    raw_groups = defaultdict(list)

    for image_path in iter_images(
        train_images
    ):
        group_key = filename_group_key(
            image_path,
            delimiter,
        )

        raw_groups[group_key].append(
            image_path
        )

    distribution = Counter(
        len(paths)
        for paths in raw_groups.values()
    )

    print(
        f"[{source_name}] "
        f"base-name groups: "
        f"{len(raw_groups)}"
    )

    print(
        f"[{source_name}] "
        f"group-size distribution: "
        f"{dict(sorted(distribution.items()))}"
    )

    source_clusters = []
    raw_group_clusters = {}

    for group_key, paths in sorted(
        raw_groups.items()
    ):
        paths = sorted(paths)

        if len(paths) == versions_per_source:
            clusters = [paths]

            cluster_score = 0.0
            cluster_margin = 1.0

        elif (
            versions_per_source == 3
            and len(paths) == 6
        ):
            (
                clusters,
                cluster_score,
                cluster_margin,
            ) = split_six_into_triplets(
                paths
            )

        else:
            raise RuntimeError(
                f"{source_name}: group "
                f"'{group_key}' memiliki "
                f"{len(paths)} file.\n"
                "Pipeline ini mengharapkan "
                "kelompok ukuran 3 atau 6."
            )

        raw_group_clusters[
            group_key
        ] = clusters

        for subgroup, cluster_paths in enumerate(
            clusters
        ):
            source_clusters.append(
                {
                    "group_key": group_key,
                    "subgroup": subgroup,
                    "paths": cluster_paths,
                    "raw_group_size": (
                        len(paths)
                    ),
                    "cluster_score": float(
                        cluster_score
                    ),
                    "cluster_margin": float(
                        cluster_margin
                    ),
                }
            )

    if (
        expected_train_sources is not None
        and len(source_clusters)
        != int(expected_train_sources)
    ):
        raise RuntimeError(
            f"{source_name}: recovered "
            f"train sources="
            f"{len(source_clusters)}, "
            f"expected="
            f"{int(expected_train_sources)}.\n"
            "Pipeline dihentikan agar "
            "tidak diam-diam menggunakan "
            "jumlah data yang salah."
        )

    report_rows = []

    selected_by_group = defaultdict(list)

    review_groups = set()

    selected_black_border_count = 0

    for cluster in source_clusters:
        (
            selected,
            candidates,
            selection_margin,
        ) = choose_representative(
            cluster["paths"],
            dark_threshold,
        )

        group_key = cluster[
            "group_key"
        ]

        subgroup = int(
            cluster["subgroup"]
        )

        output_stem = (
            f"{safe_stem(group_key)}"
            f"__SRC{subgroup:02d}"
        )

        copy_selected_pair(
            selected_image=selected["path"],
            source_images=train_images,
            source_labels=train_labels,
            output_images=(
                output_root
                / "train"
                / "images"
            ),
            output_labels=(
                output_root
                / "train"
                / "labels"
            ),
            output_stem=output_stem,
        )

        selected_by_group[
            group_key
        ].append(
            selected["path"]
        )

        selected_has_black_border = (
            selected[
                "global_black_ratio"
            ]
            > max_selected_black_ratio
        )

        if selected_has_black_border:
            selected_black_border_count += 1

        needs_review = (
            cluster["raw_group_size"] == 6
            or cluster["cluster_margin"]
            < cluster_margin_review
            or selection_margin
            < selection_margin_review
            or selected_has_black_border
        )

        if needs_review:
            review_groups.add(
                group_key
            )

        report_rows.append(
            {
                "source": source_name,
                "group_key": group_key,
                "subgroup": subgroup,
                "raw_group_size": (
                    cluster[
                        "raw_group_size"
                    ]
                ),
                "selected_file": (
                    selected[
                        "path"
                    ].name
                ),
                "output_stem": output_stem,
                "cluster_members": " | ".join(
                    path.name
                    for path in cluster[
                        "paths"
                    ]
                ),
                "cluster_score": (
                    cluster[
                        "cluster_score"
                    ]
                ),
                "cluster_margin": (
                    cluster[
                        "cluster_margin"
                    ]
                ),
                "selection_margin": (
                    selection_margin
                ),
                "artifact_score": (
                    selected[
                        "artifact_score"
                    ]
                ),
                "global_black_ratio": (
                    selected[
                        "global_black_ratio"
                    ]
                ),
                "border_dark_ratio": (
                    selected[
                        "border_dark_ratio"
                    ]
                ),
                "connected_border_ratio": (
                    selected[
                        "connected_border_ratio"
                    ]
                ),
                "sharpness": (
                    selected[
                        "sharpness"
                    ]
                ),
                "needs_review": (
                    needs_review
                ),
                "candidate_scores_json": (
                    json.dumps(
                        [
                            {
                                "file": (
                                    item[
                                        "path"
                                    ].name
                                ),
                                "artifact_score": (
                                    item[
                                        "artifact_score"
                                    ]
                                ),
                                "global_black_ratio": (
                                    item[
                                        "global_black_ratio"
                                    ]
                                ),
                                "sharpness": (
                                    item[
                                        "sharpness"
                                    ]
                                ),
                            }
                            for item in candidates
                        ],
                        ensure_ascii=False,
                    )
                ),
            }
        )

    valid_count = copy_split_unchanged(
        data_yaml_path,
        "valid",
        output_root,
    )

    test_count = copy_split_unchanged(
        data_yaml_path,
        "test",
        output_root,
    )

    output_yaml = {
        "path": str(
            output_root.resolve()
        ),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": names,
        "nc": len(names),
    }

    save_yaml(
        output_yaml,
        output_root / "data.yaml",
    )

    source_report_dir = (
        report_root
        / source_name
    )

    review_dir = (
        source_report_dir
        / "review_sheets"
    )

    source_report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for group_key in sorted(
        review_groups
    ):
        create_review_sheet(
            group_key=group_key,
            clusters=(
                raw_group_clusters[
                    group_key
                ]
            ),
            selected_paths=(
                selected_by_group[
                    group_key
                ]
            ),
            output_path=(
                review_dir
                / f"{safe_stem(group_key)}.jpg"
            ),
        )

    csv_path = (
        source_report_dir
        / "recovery_report.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                report_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            report_rows
        )

    summary = {
        "source": source_name,
        "input": str(input_root),
        "output": str(output_root),
        "filename_delimiter": (
            delimiter
        ),
        "expected_versions_per_source": (
            versions_per_source
        ),
        "base_name_groups": (
            len(raw_groups)
        ),
        "group_size_distribution": (
            dict(
                sorted(
                    distribution.items()
                )
            )
        ),
        "recovered_train_sources": (
            len(source_clusters)
        ),
        "valid_copied": valid_count,
        "test_copied": test_count,
        "review_group_count": (
            len(review_groups)
        ),
        "selected_black_border_count": (
            selected_black_border_count
        ),
        "method_note": (
            "Filename dipakai hanya untuk "
            "mengelompokkan turunan dengan "
            "base name yang sama. Hash setelah "
            ".rf. tidak mengidentifikasi "
            "original atau augmentation. "
            "Kelompok ukuran 6 dipecah menjadi "
            "dua triplet menggunakan kemiripan "
            "visual. Satu likely clean "
            "representative dipilih berdasarkan "
            "artefak border terendah."
        ),
    }

    save_json(
        summary,
        source_report_dir
        / "recovery_summary.json",
    )

    recovery_notes = f"""# Recovery notes: {source_name}

Input augmented dataset:

    {input_root}

Effective recovered dataset:

    {output_root}

Base-name groups:

    {len(raw_groups)}

Group-size distribution:

    {dict(sorted(distribution.items()))}

Recovered training sources:

    {len(source_clusters)}

Validation copied unchanged:

    {valid_count}

Test copied unchanged:

    {test_count}

Review groups:

    {len(review_groups)}

Selected images still above black-border threshold:

    {selected_black_border_count}

## Method limitation

The hash after `.rf.` does not indicate which file is the original.
The filename is used only to group variants. The pipeline then uses
visual similarity and black-border artifact scoring to choose one
likely clean representative. Review sheets are stored at:

    {review_dir}
"""

    (
        output_root
        / "RECOVERY_NOTES.md"
    ).write_text(
        recovery_notes,
        encoding="utf-8",
    )

    print()
    print(
        f"[OK] {source_name}: "
        f"recovered train="
        f"{len(source_clusters)}"
    )

    print(
        f"[OK] {source_name}: "
        f"valid={valid_count}, "
        f"test={test_count}"
    )

    print(
        f"[CHECK] Selected images with "
        f"black-border > "
        f"{max_selected_black_ratio:.1%}: "
        f"{selected_black_border_count}"
    )

    print(
        f"[OK] Effective output: "
        f"{output_root}"
    )

    print(
        f"[REVIEW] {review_dir}"
    )

    print(
        f"[REPORT] {csv_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/datasets.yaml",
    )

    args = parser.parse_args()

    cfg = load_yaml(
        args.config
    )

    for source_name, source in (
        cfg["sources"].items()
    ):
        recover_source(
            source_name,
            source,
        )


if __name__ == "__main__":
    main()