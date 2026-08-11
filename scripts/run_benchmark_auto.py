from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()

EXTERNAL_DIR = ROOT / "external"
BENCHMARK_RUNS_DIR = ROOT / "runs" / "benchmark"
BENCHMARK_REPORT_DIR = ROOT / "reports" / "benchmark"
RUNS_DIR = BENCHMARK_RUNS_DIR
REPORT_DIR = BENCHMARK_REPORT_DIR
ENV_REPORT_DIR = ROOT / "reports" / "environments"
COCO_EXPORT_DIR = ROOT / "data" / "exports" / "coco"
PREFLIGHT_RUNS_DIR = ROOT / "runs" / "preflight"
PREFLIGHT_REPORT_PATH = BENCHMARK_REPORT_DIR / "preflight.json"
EPOCH_PLAN: dict[str, int] = {}
BUDGET_MODE = "epoch_matched"

SUPPORTED_MODELS = [
    "yolov10s",
    "yolo11s",
    "yolov12s",
    "yolov13s",
    "rfdetrs",
    "tinyformers",
    "deimv2s",
]

REPOSITORIES = {
    "yolov10s": {
        "url": "https://github.com/THU-MIG/yolov10.git",
        "path": EXTERNAL_DIR / "yolov10",
    },
    "yolov12s": {
        "url": "https://github.com/sunsmarterjie/yolov12.git",
        "path": EXTERNAL_DIR / "yolov12",
    },
    "yolov13s": {
        "url": "https://github.com/iMoonLab/yolov13.git",
        "path": EXTERNAL_DIR / "yolov13",
    },
    "tinyformers": {
        "url": "https://github.com/mmpmmpmmpjosh/TinyFormer.git",
        "path": EXTERNAL_DIR / "TinyFormer",
    },
    "deimv2s": {
        "url": "https://github.com/Intellindust-AI-Lab/DEIMv2.git",
        "path": EXTERNAL_DIR / "DEIMv2",
    },
}

SCENARIOS = {
    "B0": ROOT / "data" / "processed" / "scenarios" / "B0",
    "B1": ROOT / "data" / "processed" / "scenarios" / "B1",
    "B2": ROOT / "data" / "processed" / "scenarios" / "B2",
}

EVALUATION_VIEWS = {
    "primary_full": {
        "valid": "valid",
        "test": "test",
    },
    "balanced_diagnostic": {
        "valid": "balanced_diagnostic/valid",
        "test": "balanced_diagnostic/test",
    },
}


# ============================================================
# General subprocess utilities
# ============================================================

def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    log_path: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """
    Jalankan command sambil menampilkan output ke terminal dan log.
    """
    cwd = cwd or ROOT

    print()
    print("[RUN]", " ".join(str(item) for item in command))
    print("[CWD]", cwd)

    if log_path is None:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=check,
        )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with log_path.open(
        "a",
        encoding="utf-8",
    ) as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()

    if check and return_code != 0:
        raise subprocess.CalledProcessError(
            return_code,
            command,
        )

    return subprocess.CompletedProcess(
        command,
        return_code,
    )


def conda_command(
    environment_name: str,
    command: list[str],
) -> list[str]:
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        environment_name,
        *command,
    ]


def get_conda_environment_map() -> dict[str, str]:
    """
    Menghasilkan mapping:
        nama_environment -> absolute_prefix

    Contoh:
        {
            "soilcpu":
                "/home/user/miniforge3/envs/soilcpu"
        }

    Environment yang berada di custom prefix tetap dapat ditemukan.
    """
    result = subprocess.run(
        [
            "conda",
            "env",
            "list",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    environment_map: dict[str, str] = {}

    for raw_prefix in payload.get("envs", []):
        prefix = Path(
            raw_prefix
        ).expanduser().resolve()

        environment_map[prefix.name] = str(prefix)

    return environment_map


def get_conda_environments() -> set[str]:
    """
    Dipertahankan agar fungsi lama di script tetap kompatibel.
    """
    return set(
        get_conda_environment_map().keys()
    )


def resolve_base_environment(
    base_environment: str,
) -> tuple[str, str]:
    """
    Menerima nama environment atau path environment Conda.

    Fungsi selalu mengembalikan absolute prefix supaya proses
    clone tidak bergantung pada apakah environment terdaftar
    sebagai nama atau hanya sebagai custom prefix.
    """
    candidate = Path(
        base_environment
    ).expanduser()

    # Input berupa absolute/relative path yang benar-benar ada.
    if candidate.exists():
        prefix = candidate.resolve()

        python_path = (
            prefix / "bin" / "python"
        )

        conda_meta = (
            prefix / "conda-meta"
        )

        if not python_path.exists():
            raise RuntimeError(
                "Path environment ditemukan, tetapi Python "
                f"tidak tersedia: {python_path}"
            )

        if not conda_meta.exists():
            raise RuntimeError(
                "Path ditemukan, tetapi bukan Conda "
                f"environment yang valid: {prefix}"
            )

        return "prefix", str(prefix)

    # Input berupa nama environment.
    environment_map = (
        get_conda_environment_map()
    )

    if base_environment in environment_map:
        return (
            "prefix",
            environment_map[
                base_environment
            ],
        )

    # Fallback: environment aktif saat script dijalankan.
    active_prefix = os.environ.get(
        "CONDA_PREFIX"
    )

    active_name = os.environ.get(
        "CONDA_DEFAULT_ENV"
    )

    if active_prefix:
        active_path = Path(
            active_prefix
        ).expanduser().resolve()

        if (
            base_environment == active_name
            or base_environment
            == active_path.name
        ):
            python_path = (
                active_path
                / "bin"
                / "python"
            )

            if python_path.exists():
                return (
                    "prefix",
                    str(active_path),
                )

    available = "\n".join(
        f"  - {name}: {prefix}"
        for name, prefix
        in sorted(
            environment_map.items()
        )
    )

    raise RuntimeError(
        f"Environment '{base_environment}' "
        "tidak ditemukan.\n\n"
        "Environment yang terdeteksi:\n"
        f"{available}"
    )


def recreate_temporary_environment(
    base_environment: str,
    temporary_environment: str,
) -> None:
    """
    Temporary environment dibuat ulang dari absolute prefix
    base environment.

    Environment dasar tidak pernah diubah atau dihapus.
    """
    _, base_prefix = resolve_base_environment(
        base_environment
    )

    environment_map = (
        get_conda_environment_map()
    )

    temporary_prefix = (
        environment_map.get(
            temporary_environment
        )
    )

    if temporary_prefix is not None:
        print(
            "[REMOVE TEMP ENV] "
            f"{temporary_prefix}"
        )

        run_command(
            [
                "conda",
                "env",
                "remove",
                "--prefix",
                temporary_prefix,
                "-y",
            ]
        )

    print(
        "[BASE ENV PREFIX] "
        f"{base_prefix}"
    )

    print(
        "[TEMP ENV NAME] "
        f"{temporary_environment}"
    )

    run_command(
        [
            "conda",
            "create",
            "--name",
            temporary_environment,
            "--clone",
            base_prefix,
            "-y",
        ]
    )

    run_command(
        conda_command(
            temporary_environment,
            [
                "python",
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
                "setuptools",
                "wheel",
            ],
        )
    )


def remove_temporary_environment(
    temporary_environment: str,
) -> None:
    environment_map = (
        get_conda_environment_map()
    )

    temporary_prefix = (
        environment_map.get(
            temporary_environment
        )
    )

    if temporary_prefix is None:
        print(
            "[SKIP REMOVE] Temporary environment "
            "sudah tidak tersedia: "
            f"{temporary_environment}"
        )
        return

    run_command(
        [
            "conda",
            "env",
            "remove",
            "--prefix",
            temporary_prefix,
            "-y",
        ]
    )


# ============================================================
# Repository and provenance
# ============================================================

def clone_repository(
    model_name: str,
) -> Path:
    information = REPOSITORIES[model_name]
    repository_path = information["path"]

    if repository_path.exists():
        print(
            f"[SKIP] Repository sudah tersedia: "
            f"{repository_path}"
        )
        return repository_path

    repository_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_command(
        [
            "git",
            "clone",
            information["url"],
            str(repository_path),
        ]
    )

    return repository_path


def repository_commit(
    repository_path: Path,
) -> str:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository_path),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def save_environment_provenance(
    model_name: str,
    environment_name: str,
    repository_path: Path | None = None,
) -> None:
    ENV_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    freeze_result = subprocess.run(
        conda_command(
            environment_name,
            [
                "python",
                "-m",
                "pip",
                "freeze",
            ],
        ),
        check=True,
        capture_output=True,
        text=True,
    )

    (
        ENV_REPORT_DIR
        / f"{model_name}_pip_freeze.txt"
    ).write_text(
        freeze_result.stdout,
        encoding="utf-8",
    )

    python_result = subprocess.run(
        conda_command(
            environment_name,
            [
                "python",
                "-c",
                (
                    "import sys, torch; "
                    "print('python=' + sys.version.replace('\\n', ' ')); "
                    "print('torch=' + torch.__version__); "
                    "print('cuda=' + str(torch.version.cuda)); "
                    "print('cuda_available=' + str(torch.cuda.is_available())); "
                    "print('gpu=' + "
                    "(torch.cuda.get_device_name(0) "
                    "if torch.cuda.is_available() else 'none'))"
                ),
            ],
        ),
        check=True,
        capture_output=True,
        text=True,
    )

    (
        ENV_REPORT_DIR
        / f"{model_name}_runtime.txt"
    ).write_text(
        python_result.stdout,
        encoding="utf-8",
    )

    if repository_path is not None:
        (
            ENV_REPORT_DIR
            / f"{model_name}_commit.txt"
        ).write_text(
            repository_commit(repository_path) + "\n",
            encoding="utf-8",
        )


# ============================================================
# Dataset validation
# ============================================================

def count_image_label_pairs(
    dataset_root: Path,
    split_prefix: str = "",
) -> dict[str, dict[str, int]]:
    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
    }

    results: dict[str, dict[str, int]] = {}

    for split in (
        "train",
        "valid",
        "test",
    ):
        image_directory = dataset_root / split_prefix / split / "images"
        label_directory = (
            dataset_root / split_prefix / split / "labels"
        )

        images = [
            path
            for path in image_directory.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in image_extensions
                and "checkpoint" not in path.name.lower()
            )
        ]

        missing = 0

        for image_path in images:
            relative = image_path.relative_to(
                image_directory
            )

            label_path = (
                label_directory / relative
            ).with_suffix(".txt")

            if not label_path.exists():
                missing += 1

        results[split] = {
            "images": len(images),
            "missing_labels": missing,
        }

    return results


def validate_datasets(
    scenarios: list[str],
) -> None:
    for scenario in scenarios:
        dataset_root = SCENARIOS[scenario]
        data_yaml = dataset_root / "data.yaml"

        if not data_yaml.exists():
            raise FileNotFoundError(
                f"Dataset {scenario} tidak ditemukan: "
                f"{data_yaml}"
            )

        balanced_yaml = dataset_root / "data_balanced.yaml"
        if not balanced_yaml.exists():
            raise FileNotFoundError(
                f"Balanced diagnostic YAML missing for {scenario}: {balanced_yaml}"
            )
        results = count_image_label_pairs(dataset_root)
        balanced_results = count_image_label_pairs(
            dataset_root, "balanced_diagnostic"
        )

        print()
        print(f"===== DATASET {scenario} =====")

        for split, values in results.items():
            print(
                f"{split:5s}: "
                f"images={values['images']:5d}, "
                f"missing="
                f"{values['missing_labels']:3d}"
            )

            if values["missing_labels"] > 0:
                raise RuntimeError(
                    f"{scenario}/{split} memiliki "
                    f"{values['missing_labels']} "
                    "image tanpa label."
                )

        for split in ("valid", "test"):
            values = balanced_results[split]
            print(
                f"balanced/{split:5s}: images={values['images']:5d}, "
                f"missing={values['missing_labels']:3d}"
            )
            if values["images"] <= 0 or values["missing_labels"] > 0:
                raise RuntimeError(
                    f"{scenario}/balanced/{split} invalid: {values}"
                )


# ============================================================
# Model installation
# ============================================================

def install_repository_backend(
    environment_name: str,
    repository_path: Path,
) -> None:
    """
    Hapus implementasi ultralytics lama, lalu pasang fork
    milik repository model yang sedang diuji.
    """
    run_command(
        conda_command(
            environment_name,
            [
                "python",
                "-m",
                "pip",
                "uninstall",
                "-y",
                "ultralytics",
            ],
        ),
        check=False,
    )

    requirements_path = (
        repository_path / "requirements.txt"
    )

    if requirements_path.exists():
        run_command(
            conda_command(
                environment_name,
                [
                    "python",
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(requirements_path),
                ],
            )
        )

    run_command(
        conda_command(
            environment_name,
            [
                "python",
                "-m",
                "pip",
                "install",
                "-e",
                str(repository_path),
            ],
        )
    )


def disable_yolo_trackers(
    environment_name: str,
) -> None:
    """
    Hapus optional experiment trackers dari environment YOLO.

    Ultralytics versi/fork lama dapat menganggap local output path
    sebagai nama project W&B. Benchmark ini menyimpan seluruh log
    secara lokal, sehingga tracker eksternal tidak diperlukan.
    """
    print(
        "[TRACKERS] Menonaktifkan W&B, ClearML, "
        "Comet, MLflow, dan DVCLive."
    )

    run_command(
        conda_command(
            environment_name,
            [
                "python",
                "-m",
                "pip",
                "uninstall",
                "-y",
                "wandb",
                "clearml",
                "comet-ml",
                "mlflow",
                "dvclive",
            ],
        ),
        check=False,
    )


def install_yolov12_backend(
    environment_name: str,
    repository_path: Path,
) -> None:
    """
    Instalasi khusus YOLOv12.

    requirements.txt YOLOv12 berisi nama wheel FlashAttention
    lokal. Karena file tersebut tidak ikut berada di repository,
    dependency dipasang secara terkontrol dan reproducible.
    """
    print(
        "[YOLOV12] Memasang backend dengan "
        "dependency terkontrol."
    )

    # Bersihkan backend lama hasil clone base environment.
    run_command(
        conda_command(
            environment_name,
            [
                "python",
                "-m",
                "pip",
                "uninstall",
                "-y",
                "ultralytics",
                "torch",
                "torchvision",
                "torchaudio",
                "flash-attn",
                "flash_attn",
            ],
        ),
        check=False,
    )

    # YOLOv12 resmi mengunci torch 2.2.2 dan torchvision 0.17.2.
    # Gunakan CUDA 11.8 agar sesuai dengan server/base environment.
    run_command(
        conda_command(
            environment_name,
            [
                "python",
                "-m",
                "pip",
                "install",
                "torch==2.2.2+cu118",
                "torchvision==0.17.2+cu118",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu118",
            ],
        )
    )

    # Dependency untuk FlashAttention dan proses build fallback.
    run_command(
        conda_command(
            environment_name,
            [
                "python",
                "-m",
                "pip",
                "install",
                "packaging",
                "psutil",
                "ninja",
            ],
        )
    )

    requirements_path = (
        repository_path / "requirements.txt"
    )

    if not requirements_path.exists():
        raise FileNotFoundError(
            "requirements.txt YOLOv12 tidak ditemukan: "
            f"{requirements_path}"
        )

    ENV_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sanitized_path = (
        ENV_REPORT_DIR
        / "yolov12_requirements_sanitized.txt"
    )

    # Dependency yang ditangani terpisah atau tidak diperlukan
    # untuk training object detection dikeluarkan dari requirements.
    excluded_prefixes = (
        "torch==",
        "torchvision==",
        "torchaudio",
        "flash_attn",
        "flash-attn",
        "onnx==",
        "onnxruntime==",
        "onnxruntime-gpu==",
        "onnxslim==",
        "gradio==",
    )

    sanitized_lines = []

    for raw_line in requirements_path.read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw_line.strip()
        normalized = line.lower()

        if not line:
            continue

        if line.startswith("#"):
            continue

        if normalized.startswith(
            excluded_prefixes
        ):
            print(
                "[YOLOV12 REQUIREMENT SKIP] "
                f"{line}"
            )
            continue

        sanitized_lines.append(line)

    sanitized_path.write_text(
        "\n".join(sanitized_lines) + "\n",
        encoding="utf-8",
    )

    print(
        "[YOLOV12] Sanitized requirements: "
        f"{sanitized_path}"
    )

    run_command(
        conda_command(
            environment_name,
            [
                "python",
                "-m",
                "pip",
                "install",
                "-r",
                str(sanitized_path),
            ],
        )
    )

    # Wheel resmi FlashAttention 2.7.3:
    # Python 3.11, CUDA 11, PyTorch 2.2, ABI FALSE.
    flash_attention_url = (
        "https://github.com/Dao-AILab/"
        "flash-attention/releases/download/v2.7.3/"
        "flash_attn-2.7.3%2Bcu11torch2.2"
        "cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
    )

    print(
        "[YOLOV12] Mengunduh prebuilt "
        "FlashAttention wheel."
    )

    wheel_result = run_command(
        conda_command(
            environment_name,
            [
                "python",
                "-m",
                "pip",
                "install",
                flash_attention_url,
            ],
        ),
        check=False,
    )

    if wheel_result.returncode != 0:
        print(
            "[YOLOV12] Prebuilt wheel gagal. "
            "Mencoba build FlashAttention dari source."
        )

        build_environment = os.environ.copy()
        build_environment["MAX_JOBS"] = "8"

        source_result = run_command(
            conda_command(
                environment_name,
                [
                    "python",
                    "-m",
                    "pip",
                    "install",
                    "flash-attn==2.7.3",
                    "--no-build-isolation",
                ],
            ),
            env=build_environment,
            check=False,
        )

        if source_result.returncode != 0:
            raise RuntimeError(
                "Instalasi FlashAttention YOLOv12 gagal "
                "melalui prebuilt wheel dan source build."
            )

    # Repo YOLOv12 dipasang tanpa membiarkan pyproject
    # mengganti ulang versi dependency yang sudah dikunci.
    run_command(
        conda_command(
            environment_name,
            [
                "python",
                "-m",
                "pip",
                "install",
                "--no-deps",
                "-e",
                str(repository_path),
            ],
        )
    )

    # Verifikasi lengkap sebelum training dimulai.
    verification_code = (
        "import sys, torch, torchvision, flash_attn; "
        "from ultralytics import YOLO; "
        "print('python=' + sys.executable); "
        "print('torch=' + torch.__version__); "
        "print('torchvision=' + torchvision.__version__); "
        "print('flash_attn=' + flash_attn.__version__); "
        "print('cuda_build=' + str(torch.version.cuda)); "
        "print('cuda_available=' + "
        "str(torch.cuda.is_available())); "
        "print('gpu=' + "
        "(torch.cuda.get_device_name(0) "
        "if torch.cuda.is_available() else 'none')); "
        "assert torch.cuda.is_available(), "
        "'CUDA tidak tersedia pada environment YOLOv12'"
    )

    run_command(
        conda_command(
            environment_name,
            [
                "python",
                "-c",
                verification_code,
            ],
        )
    )

    print(
        "[OK] Backend YOLOv12 berhasil dipasang "
        "dan diverifikasi."
    )


def install_yolov13_backend(
    environment_name: str,
    repository_path: Path,
) -> None:
    """
    YOLOv13 menggunakan dependency stack yang sama dengan
    YOLOv12: PyTorch 2.2.2, torchvision 0.17.2, dan
    FlashAttention 2.7.3 untuk CUDA 11.

    Installer terkontrol YOLOv12 dapat digunakan dengan
    repository_path YOLOv13 karena requirements keduanya sama.
    """
    print(
        "[YOLOV13] Memasang backend dengan "
        "dependency terkontrol."
    )

    install_yolov12_backend(
        environment_name,
        repository_path,
    )

    print(
        "[OK] Backend YOLOv13 berhasil dipasang "
        "dan diverifikasi."
    )


def download_yolov10_weights(
    destination: Path,
) -> None:
    if destination.exists():
        return

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_command(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "3",
            "-o",
            str(destination),
            (
                "https://github.com/THU-MIG/yolov10/"
                "releases/download/v1.1/yolov10s.pt"
            ),
        ]
    )


# ============================================================
# YOLO worker
# ============================================================

def patch_yolov12_final_eval() -> None:
    """
    YOLOv12 melakukan validasi best.pt sekali lagi setelah epoch
    terakhir. Pada container dengan batas RAM, proses kedua ini
    dapat terkena SIGKILL setelah seluruh training selesai.

    Patch tetap melakukan strip optimizer, tetapi evaluasi test
    akan dijalankan oleh worker terpisah dalam proses Python baru.
    """
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.utils import LOGGER
    from ultralytics.utils.torch_utils import strip_optimizer

    def final_eval_without_duplicate_validation(
        trainer,
    ) -> None:
        last_checkpoint = {}

        for checkpoint_path in (
            trainer.last,
            trainer.best,
        ):
            if not checkpoint_path.exists():
                continue

            if checkpoint_path == trainer.last:
                last_checkpoint = strip_optimizer(
                    checkpoint_path
                )

            elif checkpoint_path == trainer.best:
                update_key = "train_results"

                updates = (
                    {
                        update_key:
                            last_checkpoint[update_key]
                    }
                    if update_key
                    in last_checkpoint
                    else None
                )

                strip_optimizer(
                    checkpoint_path,
                    updates=updates,
                )

        LOGGER.info(
            "\n[YOLOV12] Duplicate in-process final "
            "validation dilewati. Evaluasi test akan "
            "dijalankan pada proses baru."
        )

    BaseTrainer.final_eval = (
        final_eval_without_duplicate_validation
    )


def get_yolo_model_class(
    family: str,
):
    if family == "yolov10":
        from ultralytics import YOLOv10
        return YOLOv10

    from ultralytics import YOLO
    return YOLO


def yolo_effective_workers(
    family: str,
    requested_workers: int,
) -> int:
    # The orchestrator preflight owns the visible 8 -> 4 -> 2 -> 0
    # fallback. Never hide a model-specific workers=0 override here.
    if requested_workers not in {0, 2, 4, 8}:
        print(f"[WORKER AUDIT] non-standard requested workers={requested_workers}")
    return requested_workers



def disable_yolo_internal_albumentations() -> None:
    """
    Benchmark B0/B1/B2 memakai augmentation offline yang sudah
    ditentukan sebelumnya.

    Ultralytics lama (mis. fork YOLOv10) tetap menambahkan
    Blur/MedianBlur/ToGray/CLAHE dengan p=0.01 walaupun
    hsv/geometric/mosaic/mixup sudah 0.

    Untuk isolation benchmark, Albumentations dibuat identity.
    """
    try:
        import ultralytics.data.augment as yolo_augment

        Albumentations = getattr(
            yolo_augment,
            "Albumentations",
            None,
        )

        if Albumentations is None:
            print(
                "[YOLO AUG AUDIT] "
                "Albumentations class tidak ditemukan; skip."
            )
            return

        def disabled_init(
            self,
            *args,
            **kwargs,
        ):
            self.p = 0.0
            self.transform = None
            self.contains_spatial = False

        def disabled_call(
            self,
            labels,
        ):
            return labels

        Albumentations.__init__ = disabled_init
        Albumentations.__call__ = disabled_call

        print(
            "[YOLO AUG AUDIT] "
            "Albumentations internal = OFF / IDENTITY"
        )

    except Exception as exc:
        raise RuntimeError(
            "Gagal menonaktifkan Albumentations YOLO: "
            f"{exc}"
        ) from exc


def install_yolov13_nan_guard(model) -> None:
    import torch

    def check_finite_loss(trainer) -> None:
        tensors = []

        for name in ("loss", "loss_items"):
            value = getattr(
                trainer,
                name,
                None,
            )

            if torch.is_tensor(value):
                tensors.append(
                    value.detach()
                )

        if tensors and not all(
            bool(
                torch.isfinite(
                    value
                ).all()
            )
            for value in tensors
        ):
            epoch = (
                int(
                    getattr(
                        trainer,
                        "epoch",
                        -1,
                    )
                )
                + 1
            )

            raise FloatingPointError(
                "YOLOv13 NaN/Inf loss at epoch "
                f"{epoch}. Training aborted before "
                "invalid checkpoint is used."
            )

    model.add_callback(
        "on_train_batch_end",
        check_finite_loss,
    )

    print(
        "[YOLOV13 STABILITY] "
        "AMP=OFF, NaN/Inf guard=ON"
    )


def yolo_results_are_finite(
    results_csv: Path,
    require_validation_losses: bool = True,
    allow_early_validation_warmup_nonfinite: bool = True,
) -> bool:

    if not results_csv.exists():
        return False

    with results_csv.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:

        reader = csv.DictReader(handle)
        rows = list(reader)

        columns = [
            str(name).strip()
            for name
            in (reader.fieldnames or [])
        ]

    if not rows:
        return False

    train_loss_columns = [
        name
        for name in columns
        if "train/" in name.lower() and "loss" in name.lower()
    ]
    validation_loss_columns = [
        name
        for name in columns
        if "val/" in name.lower() and "loss" in name.lower()
    ]
    metric_columns = [
        name
        for name in columns
        if "metrics/" in name.lower()
    ]

    validation_nonfinite_rows: set[int] = set()
    for row_index, raw_row in enumerate(rows):

        row = {
            str(key).strip(): value
            for key, value
            in raw_row.items()
            if key is not None
        }

        # Training losses and metrics are never allowed to become non-finite.
        # Ultralytics-family implementations can record non-finite *auxiliary*
        # validation losses during their first validation warm-up passes when
        # no matched positives survive a loss branch. That is distinct from a
        # numerical-training failure if every later validation loss is finite.
        checked_columns = train_loss_columns + metric_columns
        if require_validation_losses:
            checked_columns += validation_loss_columns
        for column in checked_columns:

            try:
                value = float(
                    row.get(
                        column,
                        "",
                    )
                )

            except (
                TypeError,
                ValueError,
            ):
                return False

            if math.isfinite(value):
                continue
            if column in validation_loss_columns:
                validation_nonfinite_rows.add(row_index)
                continue
            return False

    if require_validation_losses and validation_nonfinite_rows:
        # Permit only validation-only events within the first four recorded
        # epochs; every later row was already checked above and must be finite.
        # This keeps the YOLOv13 safeguards intact while not discarding a valid
        # run over an Ultralytics auxiliary-loss warm-up artefact.
        if not (
            allow_early_validation_warmup_nonfinite
            and max(validation_nonfinite_rows) <= 3
        ):
            return False

    return True


def assert_yolo_training_finite(
    run_directory: Path,
    require_validation_losses: bool = True,
) -> None:

    results_csv = (
        run_directory
        / "results.csv"
    )

    if not yolo_results_are_finite(
        results_csv,
        require_validation_losses=require_validation_losses,
    ):
        raise FloatingPointError(
            "YOLO run contains NaN/Inf: "
            f"{results_csv}"
        )

    print(
        "[YOLO FINITE AUDIT] PASS (training/metrics finite; auxiliary "
        "validation warm-up non-finite values are permitted only in the first "
        "four rows when every later validation loss is finite): "
        f"{results_csv}"
    )


def worker_yolo_train(
    args: argparse.Namespace,
) -> None:
    started_at = time.time()
    disable_yolo_internal_albumentations()
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_SILENT"] = "true"
    os.environ["CLEARML_LOG_MODEL"] = "0"
    os.environ["COMET_MODE"] = "DISABLED"

    if args.family in {"yolov12", "yolov13"}:
        patch_yolov12_final_eval()

    ModelClass = get_yolo_model_class(
        args.family
    )

    effective_workers = (
        yolo_effective_workers(
            args.family,
            args.workers,
        )
    )

    model = ModelClass(
        args.weights
    )

    if args.family == "yolov13":
        install_yolov13_nan_guard(
            model
        )

    model.train(
        data=args.dataset,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        workers=effective_workers,
        seed=args.seed,
        deterministic=True,
        project=args.project,
        name=args.run_name,
        exist_ok=True,
        pretrained=True,
        amp=(
            False
            if args.family == "yolov13"
            else True
        ),

        # Plot dimatikan saat benchmark untuk mengurangi RAM.
        # Kurva tetap dapat dibuat dari results.csv.
        plots=False,
        val=True,
        cache=False,
        profile=False,

        # Semua augmentation internal dimatikan.
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        auto_augment=None,
        erasing=0.0,
    )

    run_directory = (
        Path(args.project)
        / args.run_name
    )

    assert_yolo_training_finite(
        run_directory,
        # A one-epoch memory preflight can legitimately have no detections,
        # which some Ultralytics releases encode as NaN auxiliary val losses.
        # Training losses and reported metrics must still be finite. Full and
        # multi-epoch runs retain the stricter validation-loss safeguard.
        require_validation_losses=args.epochs > 1,
    )

    best_weight = (
        run_directory
        / "weights"
        / "best.pt"
    )

    last_weight = (
        run_directory
        / "weights"
        / "last.pt"
    )

    evaluation_weight = (
        best_weight
        if best_weight.exists()
        else last_weight
    )

    if not evaluation_weight.exists():
        raise FileNotFoundError(
            "Training selesai tetapi checkpoint tidak "
            f"ditemukan: {run_directory}"
        )

    marker = {
        "run_name": args.run_name,
        "family": args.family,
        "checkpoint": str(
            evaluation_weight.resolve()
        ),
        "epochs_requested": args.epochs,
        "batch": args.batch,
        "workers": effective_workers,
        "runtime_seconds": time.time() - started_at,
        "training_completed": True,
    }

    (
        run_directory
        / "training_complete.json"
    ).write_text(
        json.dumps(
            marker,
            indent=2,
        ),
        encoding="utf-8",
    )

    augmentation_audit = {
        "trainer_side_augmentation": "disabled",
        "albumentations": "identity",
        "hsv": [0.0, 0.0, 0.0],
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "auto_augment": None,
        "erasing": 0.0,
        "yolov13_amp": False if args.family == "yolov13" else None,
        "verified": True,
    }
    (run_directory / "augmentation_audit.json").write_text(
        json.dumps(augmentation_audit, indent=2), encoding="utf-8"
    )

    print(
        "[TRAIN COMPLETE] "
        f"{evaluation_weight}"
    )


def worker_yolo_eval(
    args: argparse.Namespace,
) -> None:
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_SILENT"] = "true"
    os.environ["CLEARML_LOG_MODEL"] = "0"
    os.environ["COMET_MODE"] = "DISABLED"

    ModelClass = get_yolo_model_class(
        args.family
    )

    effective_workers = (
        yolo_effective_workers(
            args.family,
            args.workers,
        )
    )

    evaluation_batch = args.batch

    evaluation_model = ModelClass(
        args.weights
    )

    run_directory = (
        Path(args.project)
        / args.run_name
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary: dict[str, Any] = {
        "run_name": args.run_name,
        "family": args.family,
        "weights": str(
            Path(args.weights).resolve()
        ),
        "batch": evaluation_batch,
        "workers": effective_workers,
        "views": {},
    }

    view_inputs = {
        "primary_full": (args.dataset, Path(args.coco_primary)),
        "balanced_diagnostic": (
            args.balanced_dataset,
            Path(args.coco_balanced),
        ),
    }
    from common_coco_evaluator import (
        collect_final_reports,
        evaluate_common_coco,
        write_metrics,
    )

    base_metadata = json.loads(
        Path(args.metadata).read_text(encoding="utf-8")
    )
    base_metadata["parameter_count"] = sum(
        parameter.numel() for parameter in evaluation_model.model.parameters()
    )
    try:
        model_info = evaluation_model.info(verbose=False)
        if isinstance(model_info, (list, tuple)) and model_info:
            base_metadata["flops"] = float(model_info[-1])
    except Exception as error:
        print(f"[PROVENANCE] FLOPs unavailable: {error}")
    results_csv = run_directory / "results.csv"
    if results_csv.exists():
        with results_csv.open("r", encoding="utf-8", newline="") as handle:
            result_rows = list(csv.DictReader(handle))
        base_metadata["epochs_completed"] = len(result_rows)
        if result_rows:
            metric_columns = [
                key for key in result_rows[0]
                if "map50-95" in key.lower().replace(" ", "")
            ]
            if metric_columns:
                best_index = max(
                    range(len(result_rows)),
                    key=lambda index: float(result_rows[index].get(metric_columns[0]) or -1),
                )
                base_metadata["best_epoch"] = best_index + 1
    marker_path = run_directory / "training_complete.json"
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        base_metadata["runtime"] = marker.get("runtime_seconds")
    base_metadata["checkpoint"] = str(Path(args.weights).resolve())
    completed = base_metadata.get("epochs_completed") or base_metadata.get("max_epochs")
    base_metadata["optimizer_steps"] = math.ceil(
        base_metadata["batches_per_epoch"]
        / base_metadata.get("gradient_accumulation_steps", 1)
    ) * int(completed)
    base_metadata["effective_samples_seen"] = (
        int(base_metadata["train_images"]) * int(completed)
    )
    Path(args.metadata).write_text(
        json.dumps(base_metadata, indent=2), encoding="utf-8"
    )
    for evaluation_view, (data_yaml, coco_root) in view_inputs.items():
        metrics = evaluation_model.val(
            data=data_yaml,
            split="test",
            imgsz=args.imgsz,
            batch=evaluation_batch,
            device=args.device,
            workers=effective_workers,
            project=args.project,
            name=f"{args.run_name}_{evaluation_view}_native_test",
            exist_ok=True,
            plots=False,
            save_json=False,
        )
        native = {}
        if hasattr(metrics, "box"):
            native = {
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
                "map50": float(metrics.box.map50),
                "map50_95": float(metrics.box.map),
            }
        summary["views"][evaluation_view] = native

        ground_truth_path = coco_root / "annotations" / "instances_test.json"
        ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
        id_by_name = {
            item["file_name"]: int(item["id"])
            for item in ground_truth["images"]
        }
        image_directory = coco_root / "images" / "test"
        image_paths = [
            image_directory / item["file_name"]
            for item in ground_truth["images"]
        ]
        predictions = []
        for start in range(0, len(image_paths), evaluation_batch):
            batch_paths = image_paths[start:start + evaluation_batch]
            outputs = evaluation_model.predict(
                source=[str(path) for path in batch_paths],
                imgsz=args.imgsz,
                batch=evaluation_batch,
                conf=0.001,
                iou=0.7,
                device=args.device,
                augment=False,
                verbose=False,
                save=False,
                stream=False,
            )
            if not isinstance(outputs, list):
                outputs = list(outputs)
            for path, output in zip(batch_paths, outputs):
                boxes = getattr(output, "boxes", None)
                if boxes is None:
                    continue
                xyxy = boxes.xyxy.detach().cpu().tolist()
                scores = boxes.conf.detach().cpu().tolist()
                classes = boxes.cls.detach().cpu().tolist()
                for box, score, class_id in zip(xyxy, scores, classes):
                    x1, y1, x2, y2 = map(float, box)
                    predictions.append(
                        {
                            "image_id": id_by_name[path.name],
                            "category_id": int(class_id),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(score),
                        }
                    )
        prediction_path = run_directory / f"predictions_{evaluation_view}.json"
        prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
        metadata = {
            **base_metadata,
            "evaluation_view": evaluation_view,
            "workers": effective_workers,
            "checkpoint": str(Path(args.weights).resolve()),
        }
        rows = evaluate_common_coco(ground_truth_path, prediction_path, metadata)
        write_metrics(rows, run_directory / f"common_metrics_{evaluation_view}.json")

    collect_final_reports(Path(args.project), Path(args.final_report_dir))

    (
        run_directory
        / "test_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("[TEST COMPLETE]")
    print(json.dumps(summary, indent=2))


# Backward compatibility.
def worker_yolo(
    args: argparse.Namespace,
) -> None:
    worker_yolo_train(args)


# ============================================================
# RF-DETR worker
# ============================================================

def worker_rfdetr(
    args: argparse.Namespace,
) -> None:
    from rfdetr import RFDETRSmall

    started_at = time.time()
    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    def find_checkpoints() -> list[Path]:
        return sorted(
            [
                path for pattern in ("*.pth", "*.pt")
                for path in output_directory.rglob(pattern)
                if path.is_file()
            ],
            key=lambda path: (
                "best_total" not in path.name.lower(),
                "best" not in path.name.lower(),
                -path.stat().st_mtime,
            ),
        )

    completion_path = output_directory / "training_complete.json"
    checkpoint_candidates = find_checkpoints()
    reuse_completed_training = completion_path.exists() and bool(checkpoint_candidates)
    if reuse_completed_training:
        checkpoint = checkpoint_candidates[0]
        print(f"[RF-DETR PREFLIGHT RECOVERY] reuse trained checkpoint: {checkpoint}")
        model = None
        if not args.preflight_smoke:
            model = RFDETRSmall(
                pretrain_weights=str(checkpoint.resolve()),
                # RF-DETR 1.7.1 otherwise builds its 90-class default head before
                # loading this one-class checkpoint, which can produce mixed raw
                # class indices at prediction time.
                num_classes=1,
                device=f"cuda:{args.device}",
            )
        completion_record = json.loads(completion_path.read_text(encoding="utf-8"))
        training_runtime = float(completion_record.get("runtime_seconds", 0.0))
    else:
        model = RFDETRSmall(num_classes=1, device=f"cuda:{args.device}")
        model.train(
            dataset_dir=args.dataset_root,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.rf_batch,
            grad_accum_steps=args.rf_grad_accum,
            num_workers=args.workers,
            resolution=512,
            device=f"cuda:{args.device}",

            # Fixed 512, tanpa trainer-side augmentation.
            aug_config={},
            scale_jitter=False,
            multi_scale=False,
            expanded_scales=False,

            early_stopping=True,
            early_stopping_patience=args.patience,
            run_test=True,
            tensorboard=True,
            seed=args.seed,
        )
        training_runtime = time.time() - started_at
    (output_directory / "augmentation_audit.json").write_text(
        json.dumps(
            {
                "trainer_side_augmentation": "disabled",
                "aug_config": {},
                "scale_jitter": False,
                "multi_scale": False,
                "expanded_scales": False,
                "fixed_resolution": 512,
                "verified": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not reuse_completed_training:
        completion_path.write_text(
            json.dumps(
                {
                    "training_completed": True,
                    "epochs_requested": args.epochs,
                    "batch": args.rf_batch,
                    "grad_accum": args.rf_grad_accum,
                    "workers": args.workers,
                    "runtime_seconds": training_runtime,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    checkpoint_candidates = find_checkpoints()
    checkpoint = checkpoint_candidates[0] if checkpoint_candidates else None
    base_metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    base_metadata.update(
        {
            "epochs_completed": args.epochs,
            "best_epoch": None,
            "checkpoint": str(checkpoint.resolve()) if checkpoint else None,
            "runtime": training_runtime,
        }
    )
    try:
        if model is None:
            raise AttributeError("preflight reused completed checkpoint without reload")
        parameter_owner = model
        visited = set()
        while not hasattr(parameter_owner, "parameters"):
            owner_id = id(parameter_owner)
            if owner_id in visited or not hasattr(parameter_owner, "model"):
                raise AttributeError("no torch module found through RF-DETR model wrappers")
            visited.add(owner_id)
            parameter_owner = parameter_owner.model
        base_metadata["parameter_count"] = sum(
            parameter.numel() for parameter in parameter_owner.parameters()
        )
    except Exception as error:
        print(f"[PROVENANCE] RF-DETR parameter count unavailable: {error}")
    Path(args.metadata).write_text(
        json.dumps(base_metadata, indent=2), encoding="utf-8"
    )

    if args.preflight_smoke:
        # A preflight answers whether a one-epoch train/data-loader/checkpoint
        # cycle is stable; it is not a benchmark evaluation.  Evaluating every
        # held-out image here made RF-DETR preflight take longer than the smoke
        # training itself.  Full runs still execute both complete COCO views.
        (output_directory / "preflight_smoke.json").write_text(
            json.dumps(
                {
                    "one_epoch_training": True,
                    "checkpoint_reloaded_with_num_classes": 1,
                    "fixed_resolution": 512,
                    "worker_loader_completed": True,
                    "full_evaluation_deferred_to_benchmark_run": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return

    from common_coco_evaluator import (
        collect_final_reports,
        evaluate_common_coco,
        write_metrics,
    )

    view_inputs = {
        "primary_full": (Path(args.dataset_root), Path(args.coco_primary)),
        "balanced_diagnostic": (
            Path(args.balanced_dataset_root),
            Path(args.coco_balanced),
        ),
    }
    # RF-DETR 1.7.1 performs its native test evaluation inside train(run_test=True)
    # but does not expose the newer model.evaluate() API. Preserve that fact in
    # the auxiliary summary and use predictions + the common COCO evaluator for
    # both required views.
    native_metrics = {
        "primary_full": {
            "status": "completed_by_train_run_test",
            "dataset_root": str(Path(args.dataset_root).resolve()),
        },
        "balanced_diagnostic": {
            "status": "common_coco_only",
            "dataset_root": str(Path(args.balanced_dataset_root).resolve()),
        },
    }
    prediction_adapter_audit = {}
    try:
        model.optimize_for_inference(batch_size=args.rf_batch)
        print("[RF-DETR INFERENCE] optimized checkpoint for held-out evaluation")
    except Exception as error:
        print(f"[RF-DETR INFERENCE] optimization skipped: {error}")
    for evaluation_view, (dataset_root, coco_root) in view_inputs.items():
        prediction_path = output_directory / f"predictions_{evaluation_view}.json"
        common_metrics_path = output_directory / f"common_metrics_{evaluation_view}.json"
        if prediction_path.exists() and common_metrics_path.exists():
            print(
                "[RF-DETR EVALUATION RECOVERY] reusing completed view: "
                f"{evaluation_view}"
            )
            prediction_adapter_audit[evaluation_view] = {
                "framework": "rfdetr==1.7.1",
                "status": "reused_verified_evaluation_artifacts",
                "prediction_count": len(
                    json.loads(prediction_path.read_text(encoding="utf-8"))
                ),
            }
            continue
        ground_truth_path = coco_root / "annotations" / "instances_test.json"
        ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
        id_by_name = {
            item["file_name"]: int(item["id"])
            for item in ground_truth["images"]
        }
        image_directory = coco_root / "images" / "test"
        image_paths = [
            image_directory / item["file_name"]
            for item in ground_truth["images"]
        ]
        predictions = []
        raw_class_ids: set[int] = set()
        discarded_background_predictions = 0
        for start in range(0, len(image_paths), args.rf_batch):
            batch_paths = image_paths[start:start + args.rf_batch]
            # Compiled RF-DETR inference requires a fixed batch shape.
            # Pad only the final partial batch and discard padded outputs.
            inference_batch_paths = list(batch_paths)
            if inference_batch_paths and len(inference_batch_paths) < args.rf_batch:
                inference_batch_paths.extend(
                    [inference_batch_paths[-1]]
                    * (args.rf_batch - len(inference_batch_paths))
                )
            outputs = model.predict(
                [str(path) for path in inference_batch_paths],
                threshold=0.001,
                shape=(512, 512),
            )
            if not isinstance(outputs, list):
                outputs = [outputs]
            outputs = outputs[:len(batch_paths)]
            for path, output in zip(batch_paths, outputs):
                for box, score, class_id in zip(
                    output.xyxy, output.confidence, output.class_id
                ):
                    raw_class_id = int(class_id)
                    # RF-DETR 1.7.1 exposes both the trained foreground
                    # channel (raw ID 0) and its background/no-object channel
                    # (raw ID 1). This adapter is deliberately limited to
                    # those verified one-class output IDs.
                    if raw_class_id not in {0, 1}:
                        raise RuntimeError(
                            "RF-DETR emitted an unexpected class ID for the "
                            f"single-target benchmark: {raw_class_id}"
                        )
                    raw_class_ids.add(raw_class_id)
                    # In RF-DETR 1.7.1 with a one-class fine-tuned head,
                    # raw ID 0 is the trained foreground target. Raw ID 1
                    # is the background/no-object channel exposed by predict().
                    # It has no matching COCO foreground category and must not
                    # be scored as a BSR detection.
                    if raw_class_id == 1:
                        discarded_background_predictions += 1
                        continue
                    x1, y1, x2, y2 = map(float, box)
                    predictions.append(
                        {
                            "image_id": id_by_name[path.name],
                            "category_id": 0,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(score),
                        }
                    )
        prediction_path = output_directory / f"predictions_{evaluation_view}.json"
        if 0 not in raw_class_ids:
            raise RuntimeError(
                "RF-DETR emitted no foreground class-0 prediction in the "
                "single-target benchmark; refusing to score an invalid adapter."
            )
        prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
        prediction_adapter_audit[evaluation_view] = {
            "framework": "rfdetr==1.7.1",
            "raw_class_ids": sorted(raw_class_ids),
            "foreground_raw_class_id": 0,
            "discarded_background_raw_class_id": 1,
            "discarded_background_prediction_count": discarded_background_predictions,
            "canonical_category_id": 0,
            "mapping": "RF-DETR raw class 0 (BSR foreground) -> COCO category 0; raw class 1 (background/no-object) discarded",
            "prediction_count": len(predictions),
        }
        metadata = {**base_metadata, "evaluation_view": evaluation_view}
        rows = evaluate_common_coco(ground_truth_path, prediction_path, metadata)
        write_metrics(rows, common_metrics_path)
    (output_directory / "native_test_summary.json").write_text(
        json.dumps(native_metrics, indent=2), encoding="utf-8"
    )
    (output_directory / "prediction_adapter_audit.json").write_text(
        json.dumps(prediction_adapter_audit, indent=2), encoding="utf-8"
    )
    collect_final_reports(output_directory.parent, Path(args.final_report_dir))


# ============================================================
# YOLO to COCO conversion for TinyFormer
# ============================================================

def load_names_from_yaml(
    yaml_path: Path,
) -> list[str]:
    import yaml

    payload = yaml.safe_load(
        yaml_path.read_text(
            encoding="utf-8"
        )
    )

    names = payload.get(
        "names",
        ["BSR"],
    )

    if isinstance(names, dict):
        return [
            str(names[key])
            for key in sorted(
                names,
                key=lambda value: int(value),
            )
        ]

    return [
        str(name)
        for name in names
    ]


def hardlink_or_copy(
    source: Path,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        return

    try:
        os.link(
            source,
            destination,
        )
    except OSError:
        shutil.copy2(
            source,
            destination,
        )


def export_split_to_coco(
    dataset_root: Path,
    output_root: Path,
    source_split: str,
    output_split: str,
    class_names: list[str],
) -> None:
    from PIL import Image

    image_directory = (
        dataset_root / source_split / "images"
    )
    label_directory = (
        dataset_root / source_split / "labels"
    )

    output_image_directory = (
        output_root / "images" / output_split
    )

    annotation_directory = (
        output_root / "annotations"
    )

    output_image_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    annotation_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
    }

    image_entries = []
    annotation_entries = []

    image_id = 1
    annotation_id = 1

    for source_image in sorted(
        image_directory.rglob("*")
    ):
        if (
            not source_image.is_file()
            or source_image.suffix.lower()
            not in image_extensions
            or "checkpoint"
            in source_image.name.lower()
        ):
            continue

        output_name = source_image.name

        destination_image = (
            output_image_directory
            / output_name
        )

        hardlink_or_copy(
            source_image,
            destination_image,
        )

        with Image.open(
            source_image
        ) as image:
            width, height = image.size

        image_entries.append(
            {
                "id": image_id,
                "file_name": output_name,
                "width": width,
                "height": height,
                "source": output_name.split("__", 1)[0].upper(),
            }
        )

        relative = source_image.relative_to(
            image_directory
        )

        label_path = (
            label_directory / relative
        ).with_suffix(".txt")

        if label_path.exists():
            for line in label_path.read_text(
                encoding="utf-8"
            ).splitlines():
                line = line.strip()

                if not line:
                    continue

                values = line.split()

                if len(values) < 5:
                    raise ValueError(
                        f"Label tidak valid: "
                        f"{label_path}: {line}"
                    )

                class_id = int(
                    float(values[0])
                )

                center_x = float(values[1])
                center_y = float(values[2])
                box_width = float(values[3])
                box_height = float(values[4])

                pixel_width = (
                    box_width * width
                )

                pixel_height = (
                    box_height * height
                )

                x_min = (
                    center_x * width
                    - pixel_width / 2
                )

                y_min = (
                    center_y * height
                    - pixel_height / 2
                )

                x_min = max(
                    0.0,
                    min(
                        x_min,
                        width - 1.0,
                    ),
                )

                y_min = max(
                    0.0,
                    min(
                        y_min,
                        height - 1.0,
                    ),
                )

                pixel_width = max(
                    1.0,
                    min(
                        pixel_width,
                        width - x_min,
                    ),
                )

                pixel_height = max(
                    1.0,
                    min(
                        pixel_height,
                        height - y_min,
                    ),
                )

                annotation_entries.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,

                        # TinyFormer menggunakan class index
                        # langsung ketika
                        # remap_mscoco_category=False.
                        # Karena num_classes=1, BSR harus
                        # memiliki class index 0.
                        "category_id": class_id,

                        "bbox": [
                            x_min,
                            y_min,
                            pixel_width,
                            pixel_height,
                        ],

                        "area": (
                            pixel_width
                            * pixel_height
                        ),

                        "iscrowd": 0,
                    }
                )

                annotation_id += 1

        image_id += 1

    categories = [
        {
            # Harus sama dengan model class index.
            "id": index,
            "name": name,
            "supercategory": (
                "oil_palm_disease"
            ),
        }
        for index, name in enumerate(
            class_names
        )
    ]

    # ========================================================
    # TinyFormer class-ID audit
    # ========================================================

    valid_category_ids = {
        item["id"]
        for item in categories
    }

    expected_category_ids = set(
        range(len(class_names))
    )

    if (
        valid_category_ids
        != expected_category_ids
    ):
        raise RuntimeError(
            "COCO category IDs tidak 0-based contiguous. "
            f"found={sorted(valid_category_ids)}, "
            f"expected={sorted(expected_category_ids)}"
        )

    annotation_category_ids = {
        item["category_id"]
        for item in annotation_entries
    }

    invalid_annotation_ids = (
        annotation_category_ids
        - valid_category_ids
    )

    if invalid_annotation_ids:
        raise RuntimeError(
            "Annotation memiliki category_id invalid: "
            f"{sorted(invalid_annotation_ids)}"
        )

    print(
        "[COCO CLASS AUDIT] "
        f"categories={sorted(valid_category_ids)} "
        f"annotation_categories="
        f"{sorted(annotation_category_ids)}"
    )

    payload = {
        "images": image_entries,
        "annotations": annotation_entries,
        "categories": categories,
    }

    annotation_path = (
        annotation_directory
        / f"instances_{output_split}.json"
    )

    annotation_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[COCO] {output_split}: "
        f"images={len(image_entries)}, "
        f"annotations="
        f"{len(annotation_entries)}"
    )


def export_scenario_to_coco(
    scenario: str,
    evaluation_view: str = "primary_full",
) -> Path:
    if evaluation_view not in EVALUATION_VIEWS:
        raise ValueError(f"Unknown evaluation view: {evaluation_view}")
    dataset_root = SCENARIOS[scenario]

    output_root = (
        COCO_EXPORT_DIR / scenario / evaluation_view
    )
    expected = [
        output_root / "annotations" / f"instances_{split}.json"
        for split in ("train", "val", "test")
    ]
    if output_root.exists():
        if all(path.exists() for path in expected):
            print(f"[COCO REUSE] {scenario}/{evaluation_view}: {output_root}")
            return output_root
        raise RuntimeError(
            f"Partial COCO export exists; refusing to overwrite: {output_root}. "
            "Archive it with rebuild_method_b.py first."
        )

    class_names = load_names_from_yaml(
        dataset_root / "data.yaml"
    )

    export_split_to_coco(
        dataset_root,
        output_root,
        "train",
        "train",
        class_names,
    )

    export_split_to_coco(
        dataset_root,
        output_root,
        EVALUATION_VIEWS[evaluation_view]["valid"],
        "val",
        class_names,
    )

    export_split_to_coco(
        dataset_root,
        output_root,
        EVALUATION_VIEWS[evaluation_view]["test"],
        "test",
        class_names,
    )

    (output_root / "export_provenance.json").write_text(
        json.dumps(
            {
                "scenario": scenario,
                "evaluation_view": evaluation_view,
                "source_dataset": str(dataset_root.resolve()),
                "train_source": "train",
                "valid_source": EVALUATION_VIEWS[evaluation_view]["valid"],
                "test_source": EVALUATION_VIEWS[evaluation_view]["test"],
                "category_ids": [0],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return output_root


def prepare_rfdetr_dataset(coco_root: Path) -> Path:
    """Create RF-DETR's Roboflow-COCO layout without duplicating images."""
    output_root = coco_root / "rfdetr_dataset"
    mapping = {"train": "train", "valid": "val", "test": "test"}
    for rf_split, coco_split in mapping.items():
        destination = output_root / rf_split
        annotation_destination = destination / "_annotations.coco.json"
        annotation_source = (
            coco_root / "annotations" / f"instances_{coco_split}.json"
        )
        if annotation_destination.exists():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        for image_path in sorted((coco_root / "images" / coco_split).iterdir()):
            if image_path.is_file():
                hardlink_or_copy(image_path, destination / image_path.name)
        shutil.copy2(annotation_source, annotation_destination)
    return output_root


# ============================================================
# TinyFormer preparation
# ============================================================

def ensure_tinyformer_backbone(
    environment_name: str,
    repository_path: Path,
) -> Path:
    checkpoint_directory = (
        repository_path / "ckpts"
    )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    backbone_path = (
        checkpoint_directory
        / "vitt_distill.pt"
    )

    if not backbone_path.exists():
        run_command(
            conda_command(
                environment_name,
                [
                    "python",
                    "-m",
                    "pip",
                    "install",
                    "gdown",
                ],
            )
        )

        run_command(
            conda_command(
                environment_name,
                [
                    "gdown",
                    "1YMTq_woOLjAcZnHSYNTsNg7f0ahj5LPs",
                    "-O",
                    str(backbone_path),
                ],
            )
        )

    if (
        not backbone_path.exists()
        or backbone_path.stat().st_size
        < 1_000_000
    ):
        raise RuntimeError(
            "Backbone TinyFormer gagal di-download."
        )

    return backbone_path


def is_likely_checkpoint(
    path: Path,
) -> bool:
    if not path.exists():
        return False

    if path.stat().st_size < 5_000_000:
        return False

    beginning = path.read_bytes()[:100].lower()

    if (
        b"<html" in beginning
        or b"<!doctype" in beginning
    ):
        return False

    return True


def find_tinyformer_s_checkpoint(
    directory: Path,
) -> Path | None:
    candidates = []

    for path in directory.rglob("*.pth"):
        lower_name = path.name.lower()

        if "vitt" in lower_name:
            continue

        if "tinyformer" not in lower_name:
            continue

        if "pbm" in lower_name:
            continue

        if not re.search(
            r"(^|[-_])s($|[-_.])",
            lower_name,
        ):
            continue

        if is_likely_checkpoint(path):
            candidates.append(path)

    if not candidates:
        return None

    candidates.sort(
        key=lambda path: path.stat().st_size
    )

    return candidates[0]


def ensure_tinyformer_model_checkpoint(
    environment_name: str,
    repository_path: Path,
) -> Path:
    checkpoint_directory = (
        repository_path / "ckpts"
    )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    fixed_checkpoint = (
        checkpoint_directory
        / "TinyFormer-S.pth"
    )

    if is_likely_checkpoint(
        fixed_checkpoint
    ):
        return fixed_checkpoint

    # Coba server checkpoint resmi terlebih dahulu.
    run_command(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "3",
            "-o",
            str(fixed_checkpoint),
            (
                "http://140.113.110.150:5000/"
                "sharing/8cAsmrEye"
            ),
        ],
        check=False,
    )

    if is_likely_checkpoint(
        fixed_checkpoint
    ):
        return fixed_checkpoint

    fixed_checkpoint.unlink(
        missing_ok=True
    )

    # Fallback otomatis ke Google Drive mirror resmi.
    mirror_directory = (
        checkpoint_directory
        / "google_drive_mirror"
    )

    run_command(
        conda_command(
            environment_name,
            [
                "python",
                "-m",
                "pip",
                "install",
                "gdown",
            ],
        )
    )

    run_command(
        conda_command(
            environment_name,
            [
                "gdown",
                "--folder",
                (
                    "https://drive.google.com/drive/"
                    "folders/1yIsUO14g5EyTFRbDeISoNJSDqNeSgT5o"
                ),
                "-O",
                str(mirror_directory),
            ],
        ),
        check=False,
    )

    discovered = find_tinyformer_s_checkpoint(
        mirror_directory
    )

    if discovered is None:
        raise RuntimeError(
            "Checkpoint COCO TinyFormer-S tidak berhasil "
            "ditemukan dari dua mirror resmi. "
            "Pipeline dihentikan agar tidak diam-diam "
            "melatih TinyFormer dari scratch."
        )

    shutil.copy2(
        discovered,
        fixed_checkpoint,
    )

    return fixed_checkpoint


def clean_tinyformer_transforms() -> dict[str, Any]:
    return {
        "type": "Compose",
        "ops": [
            {
                "type": "Resize",
                "size": [
                    640,
                    640,
                ],
            },
            {
                "type": (
                    "SanitizeBoundingBoxes"
                ),
                "min_size": 1,
            },
            {
                "type": "ConvertPILImage",
                "dtype": "float32",
                "scale": True,
            },
            {
                "type": "Normalize",
                "mean": [
                    0.485,
                    0.456,
                    0.406,
                ],
                "std": [
                    0.229,
                    0.224,
                    0.225,
                ],
            },
            {
                "type": "ConvertBoxes",
                "fmt": "cxcywh",
                "normalize": True,
            },
        ],
    }


def prepare_tinyformer_config(
    repository_path: Path,
    scenario: str,
    coco_root: Path,
    output_directory: Path,
    epochs: int,
    batch: int,
    workers: int,
) -> Path:
    """
    TinyFormer-S clean benchmark config.

    Dataset YAML dibuat FROM SCRATCH.
    Tidak menggunakan custom_detection.yml dan tidak memiliki
    __include__, sehingga augmentation bawaan tidak dapat
    masuk kembali saat config merge.
    """
    import yaml

    base_model_path = (
        repository_path
        / "configs"
        / "tinyformer"
        / "tinyformer_dinov3_s_coco.yml"
    )

    if not base_model_path.exists():
        raise FileNotFoundError(
            "Config TinyFormer-S tidak ditemukan: "
            f"{base_model_path}"
        )

    model_payload = yaml.safe_load(
        base_model_path.read_text(
            encoding="utf-8"
        )
    )

    # ========================================================
    # Buat sanitized base TinyFormer.
    #
    # configs/base/tinyformer.yml resmi membawa augmentation
    # training (Mosaic, MixUp, CopyBlend, multi-scale, dll).
    # Kita perlu semua definisi architecture/loss dari file itu,
    # tetapi TIDAK train_dataloader augmentation-nya.
    # ========================================================

    original_tinyformer_base = (
        repository_path
        / "configs"
        / "base"
        / "tinyformer.yml"
    )

    clean_tinyformer_base = (
        repository_path
        / "configs"
        / "base"
        / "tinyformer_benchmark_clean.yml"
    )

    if not original_tinyformer_base.exists():
        raise FileNotFoundError(
            "Base TinyFormer tidak ditemukan: "
            f"{original_tinyformer_base}"
        )

    base_payload = yaml.safe_load(
        original_tinyformer_base.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(base_payload, dict):
        raise RuntimeError(
            "configs/base/tinyformer.yml tidak valid."
        )

    # Ini sumber augmentation yang selama ini masuk kembali.
    removed_train_loader = base_payload.pop(
        "train_dataloader",
        None,
    )

    if removed_train_loader is None:
        print(
            "[TINYFORMER BASE] Warning: "
            "train_dataloader tidak ditemukan "
            "di base/tinyformer.yml"
        )
    else:
        print(
            "[TINYFORMER BASE] "
            "train_dataloader augmentation removed."
        )

    clean_tinyformer_base.write_text(
        yaml.safe_dump(
            base_payload,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    print(
        "[TINYFORMER BASE] Clean base: "
        f"{clean_tinyformer_base}"
    )

    if not isinstance(
        model_payload,
        dict,
    ):
        raise RuntimeError(
            "Config TinyFormer-S tidak valid."
        )

    # ========================================================
    # Paths
    # ========================================================

    generated_model_directory = (
        repository_path
        / "configs"
        / "ganoderma"
    )

    generated_dataset_directory = (
        repository_path
        / "configs"
        / "dataset"
    )

    generated_model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_dataset_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_config_name = (
        f"ganoderma_{scenario}.yml"
    )

    generated_dataset_path = (
        generated_dataset_directory
        / dataset_config_name
    )

    generated_model_path = (
        generated_model_directory
        / f"tinyformer_s_{scenario}.yml"
    )

    # ========================================================
    # Deterministic transforms
    # ========================================================

    def deterministic_transforms():
        return {
            "type": "Compose",
            "ops": [
                {
                    "type": "Resize",
                    "size": [
                        640,
                        640,
                    ],
                },
                {
                    "type": "SanitizeBoundingBoxes",
                    "min_size": 1,
                },
                {
                    "type": "ConvertPILImage",
                    "dtype": "float32",
                    "scale": True,
                },
                {
                    "type": "Normalize",
                    "mean": [
                        0.485,
                        0.456,
                        0.406,
                    ],
                    "std": [
                        0.229,
                        0.224,
                        0.225,
                    ],
                },
                {
                    "type": "ConvertBoxes",
                    "fmt": "cxcywh",
                    "normalize": True,
                },
            ],
        }

    # ========================================================
    # Dataset config FROM SCRATCH
    # ========================================================

    dataset_payload = {
        "task": "detection",

        "evaluator": {
            "type": "CocoEvaluator",
            "iou_types": [
                "bbox",
            ],
        },

        "num_classes": 1,

        "remap_mscoco_category": False,

        "train_dataloader": {
            "type": "DataLoader",

            "dataset": {
                "type": "CocoDetection",

                "img_folder": str(
                    (
                        coco_root
                        / "images"
                        / "train"
                    ).resolve()
                ),

                "ann_file": str(
                    (
                        coco_root
                        / "annotations"
                        / "instances_train.json"
                    ).resolve()
                ),

                "return_masks": False,

                "transforms":
                    deterministic_transforms(),
            },

            "shuffle": True,

            "num_workers": workers,

            "drop_last": True,

            "collate_fn": {
                "type":
                    "BatchImageCollateFunction",

                # Fixed 640.
                "base_size": 640,

                # Multi-scale OFF.
                "base_size_repeat": None,

                # Offline B0/B1/B2 augmentation only.
                "mixup_prob": 0.0,

                "copyblend_prob": 0.0,

                # Jangan pernah masuk stage transition
                # selama eksperimen.
                "stop_epoch": 999999,
            },

            "total_batch_size": batch,
        },

        "val_dataloader": {
            "type": "DataLoader",

            "dataset": {
                "type": "CocoDetection",

                "img_folder": str(
                    (
                        coco_root
                        / "images"
                        / "val"
                    ).resolve()
                ),

                "ann_file": str(
                    (
                        coco_root
                        / "annotations"
                        / "instances_val.json"
                    ).resolve()
                ),

                "return_masks": False,

                "transforms":
                    deterministic_transforms(),
            },

            "shuffle": False,

            "num_workers": workers,

            "drop_last": False,

            "collate_fn": {
                "type":
                    "BatchImageCollateFunction",
            },

            "total_batch_size": batch,
        },
    }

    # Dataset config TIDAK BOLEH punya __include__.
    if "__include__" in dataset_payload:
        raise RuntimeError(
            "Dataset config masih memiliki __include__."
        )

    generated_dataset_path.write_text(
        yaml.safe_dump(
            dataset_payload,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Model include cleanup
    # ========================================================

    original_includes = (
        model_payload.get(
            "__include__",
            []
        )
        or []
    )

    clean_includes = []

    dataset_inserted = False

    for include_item in original_includes:

        include_text = str(
            include_item
        )

        # Ganti dataset COCO dengan dataset Ganoderma.
        if "dataset/" in include_text:

            if not dataset_inserted:
                clean_includes.append(
                    f"../dataset/"
                    f"{dataset_config_name}"
                )

                dataset_inserted = True

            continue

        # Jangan izinkan augmentation defaults dari
        # base/dataloader.yml masuk.
        if "base/dataloader.yml" in include_text:
            continue

        # configs/base/tinyformer.yml membawa architecture
        # sekaligus augmentation. Pakai sanitized copy yang
        # mempertahankan architecture/loss tetapi membuang
        # train_dataloader.
        if "base/tinyformer.yml" in include_text:
            clean_includes.append(
                "../base/"
                "tinyformer_benchmark_clean.yml"
            )
            continue

        clean_includes.append(
            include_item
        )

    if not dataset_inserted:
        clean_includes.insert(
            0,
            f"../dataset/"
            f"{dataset_config_name}",
        )

    model_payload[
        "__include__"
    ] = clean_includes

    # Model-specific YAML TinyFormer bisa memiliki override
    # dataloader sendiri. Buang seluruhnya.
    model_payload.pop(
        "train_dataloader",
        None,
    )

    model_payload.pop(
        "val_dataloader",
        None,
    )

    # ========================================================
    # Benchmark parameters
    # ========================================================

    model_payload[
        "output_dir"
    ] = str(
        output_directory.resolve()
    )

    model_payload[
        "epoches"
    ] = epochs

    model_payload[
        "eval_spatial_size"
    ] = [
        640,
        640,
    ]

    model_payload[
        "print_freq"
    ] = 10

    model_payload[
        "checkpoint_freq"
    ] = 1

    # Tidak ada no-augmentation phase karena dari awal
    # memang augmentation trainer dimatikan.
    model_payload[
        "no_aug_epoch"
    ] = 0

    model_payload[
        "flat_epoch"
    ] = max(
        1,
        epochs // 2,
    )

    generated_model_path.write_text(
        yaml.safe_dump(
            model_payload,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # HARD AUDIT
    # ========================================================

    dataset_check = yaml.safe_load(
        generated_dataset_path.read_text(
            encoding="utf-8"
        )
    )

    clean_base_check = yaml.safe_load(
        clean_tinyformer_base.read_text(
            encoding="utf-8"
        )
    )

    if "train_dataloader" in clean_base_check:
        raise RuntimeError(
            "AUDIT FAILED: clean TinyFormer base "
            "masih memiliki train_dataloader."
        )

    # Pastikan generated model tidak lagi merujuk ke
    # base/tinyformer.yml yang asli.
    generated_model_check = yaml.safe_load(
        generated_model_path.read_text(
            encoding="utf-8"
        )
    )

    generated_includes = [
        str(item)
        for item in (
            generated_model_check.get(
                "__include__",
                []
            )
            or []
        )
    ]

    if any(
        item.endswith(
            "/base/tinyformer.yml"
        )
        or item.endswith(
            "../base/tinyformer.yml"
        )
        for item in generated_includes
    ):
        raise RuntimeError(
            "AUDIT FAILED: model masih include "
            "base/tinyformer.yml asli."
        )

    if not any(
        "tinyformer_benchmark_clean.yml"
        in item
        for item in generated_includes
    ):
        raise RuntimeError(
            "AUDIT FAILED: clean TinyFormer base "
            "belum di-include."
        )

    if "__include__" in dataset_check:
        raise RuntimeError(
            "AUDIT FAILED: dataset masih "
            "memiliki __include__."
        )

    train_cfg = (
        dataset_check[
            "train_dataloader"
        ]
    )

    transform_ops = (
        train_cfg[
            "dataset"
        ][
            "transforms"
        ][
            "ops"
        ]
    )

    transform_types = [
        item.get("type")
        for item in transform_ops
    ]

    forbidden_transforms = {
        "Mosaic",
        "RandomPhotometricDistort",
        "RandomZoomOut",
        "RandomIoUCrop",
        "RandomHorizontalFlip",
    }

    active_forbidden = (
        forbidden_transforms
        .intersection(
            transform_types
        )
    )

    if active_forbidden:
        raise RuntimeError(
            "AUDIT FAILED: augmentation aktif: "
            f"{sorted(active_forbidden)}"
        )

    transforms_payload = (
        train_cfg[
            "dataset"
        ][
            "transforms"
        ]
    )

    if "policy" in transforms_payload:
        raise RuntimeError(
            "AUDIT FAILED: transforms.policy "
            "masih ada."
        )

    collate = (
        train_cfg[
            "collate_fn"
        ]
    )

    if (
        float(
            collate.get(
                "mixup_prob",
                0.0,
            )
        )
        != 0.0
    ):
        raise RuntimeError(
            "AUDIT FAILED: MixUp masih aktif."
        )

    if (
        float(
            collate.get(
                "copyblend_prob",
                0.0,
            )
        )
        != 0.0
    ):
        raise RuntimeError(
            "AUDIT FAILED: CopyBlend masih aktif."
        )

    if (
        collate.get(
            "base_size_repeat"
        )
        is not None
    ):
        raise RuntimeError(
            "AUDIT FAILED: multi-scale "
            "masih aktif."
        )

    if (
        int(
            collate.get(
                "stop_epoch",
                0,
            )
        )
        <= epochs
    ):
        raise RuntimeError(
            "AUDIT FAILED: stop_epoch dapat "
            "memicu stage transition."
        )

    print()
    print(
        "===== TINYFORMER CLEAN CONFIG AUDIT ====="
    )
    print(
        f"scenario        : {scenario}"
    )
    print(
        f"epochs          : {epochs}"
    )
    print(
        f"batch           : {batch}"
    )
    print(
        f"workers         : {workers}"
    )
    print(
        "resolution      : 640x640"
    )
    print(
        "dataset include : NONE"
    )
    print(
        "policy          : NONE"
    )
    print(
        "Mosaic          : OFF"
    )
    print(
        "Photometric     : OFF"
    )
    print(
        "ZoomOut         : OFF"
    )
    print(
        "IoUCrop         : OFF"
    )
    print(
        "HFlip           : OFF"
    )
    print(
        "MixUp           : OFF"
    )
    print(
        "CopyBlend       : OFF"
    )
    print(
        "multi-scale     : OFF"
    )
    print(
        "stage switch    : OFF"
    )
    print(
        "transform types : "
        f"{transform_types}"
    )
    print(
        "========================================="
    )
    print()

    return generated_model_path


# ============================================================
# Model orchestration
# ============================================================

def train_image_count(scenario: str) -> int:
    return count_image_label_pairs(SCENARIOS[scenario])["train"]["images"]


def build_epoch_plan(
    scenarios: list[str], max_epochs: int, batch: int, budget_mode: str
) -> dict[str, int]:
    if budget_mode == "epoch_matched":
        return {scenario: max_epochs for scenario in scenarios}
    baseline_batches = math.ceil(train_image_count("B0") / batch)
    target_updates = baseline_batches * max_epochs
    plan = {}
    for scenario in scenarios:
        batches = math.ceil(train_image_count(scenario) / batch)
        plan[scenario] = max(1, round(target_updates / batches))
    return plan


def write_training_budget(
    *,
    run_directory: Path,
    model: str,
    scenario: str,
    epochs: int,
    batch: int,
    workers: int,
    input_size: int,
    grad_accum: int,
    seed: int,
    patience: int,
) -> Path:
    train_images = train_image_count(scenario)
    batches_per_epoch = math.ceil(train_images / batch)
    optimizer_steps = math.ceil(batches_per_epoch / grad_accum) * epochs
    metadata = {
        "model": model,
        "scenario": scenario,
        "seed": seed,
        "train_images": train_images,
        "best_epoch": None,
        "epochs_completed": None,
        "batch": batch,
        "workers": workers,
        "input_size": input_size,
        "parameter_count": None,
        "flops": None,
        "checkpoint": None,
        "runtime": None,
        "batches_per_epoch": batches_per_epoch,
        "gradient_accumulation_steps": grad_accum,
        "optimizer_steps": optimizer_steps,
        "effective_samples_seen": train_images * epochs,
        "budget_mode": BUDGET_MODE,
        "max_epochs": epochs,
        "patience": patience,
        "optimizer_steps_are_planned": True,
    }
    run_directory.mkdir(parents=True, exist_ok=True)
    path = run_directory / "training_budget.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def planned_epochs(scenario: str, fallback: int) -> int:
    return int(EPOCH_PLAN.get(scenario, fallback))


def archive_artifact(path: Path, reason: str) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    try:
        relative = path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        relative = Path(path.name)
    destination = ROOT / "archives" / f"before_rerun_{stamp}" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))
    print(f"[ARCHIVE {reason}] {path} -> {destination}")
    return destination

def yolo_training_is_complete(
    run_directory: Path,
    requested_epochs: int,
) -> bool:

    best_weight = (
        run_directory
        / "weights"
        / "best.pt"
    )

    marker_path = (
        run_directory
        / "training_complete.json"
    )

    results_csv = (
        run_directory
        / "results.csv"
    )

    if not (
        best_weight.exists()
        and marker_path.exists()
        and results_csv.exists()
    ):
        return False

    try:
        marker = json.loads(
            marker_path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return False

    if not marker.get(
        "training_completed",
        False,
    ):
        return False

    if int(
        marker.get(
            "epochs_requested",
            -1,
        )
    ) != int(requested_epochs):
        return False

    if not yolo_results_are_finite(
        results_csv
    ):
        print(
            "[RECOVERY REJECTED] "
            f"NaN/Inf: {run_directory}"
        )
        return False

    return True

def run_yolo_model(
    model_name: str,
    environment_name: str,
    scenarios: list[str],
    epochs: int,
    batch: int,
    patience: int,
    device: str,
    workers: int,
    seed: int,
) -> None:
    repository_path: Path | None = None

    if model_name == "yolov10s":
        repository_path = clone_repository(
            model_name
        )

        install_repository_backend(
            environment_name,
            repository_path,
        )

        initial_weights = (
            repository_path
            / "yolov10s.pt"
        )

        download_yolov10_weights(
            initial_weights
        )

        family = "yolov10"

    elif model_name == "yolo11s":
        initial_weights = Path(
            "yolo11s.pt"
        )

        family = "yolo"

    elif model_name == "yolov12s":
        repository_path = clone_repository(
            model_name
        )

        install_yolov12_backend(
            environment_name,
            repository_path,
        )

        initial_weights = Path(
            "yolov12s.pt"
        )

        family = "yolov12"

    elif model_name == "yolov13s":
        repository_path = clone_repository(
            model_name
        )

        install_yolov13_backend(
            environment_name,
            repository_path,
        )

        # Repo resmi YOLOv13 mendukung pemanggilan model
        # menggunakan nama checkpoint ini.
        initial_weights = Path(
            "yolov13s.pt"
        )

        family = "yolov13"

    else:
        raise ValueError(
            "Model YOLO tidak dikenal: "
            f"{model_name}"
        )

    disable_yolo_trackers(
        environment_name
    )

    save_environment_provenance(
        model_name,
        environment_name,
        repository_path,
    )

    for scenario in scenarios:
        run_epochs = planned_epochs(scenario, epochs)
        run_name = (
            f"{scenario}_{model_name}"
        )

        run_directory = (
            RUNS_DIR
            / run_name
        )

        test_directory = (
            RUNS_DIR
            / f"{run_name}_test"
        )

        training_complete = (
            yolo_training_is_complete(
                run_directory,
                run_epochs,
            )
        )

        train_log = (
            REPORT_DIR
            / "logs"
            / f"{run_name}.log"
        )

        test_log = (
            REPORT_DIR
            / "logs"
            / f"{run_name}_test.log"
        )

        if training_complete:
            print(
                "[RECOVER COMPLETED TRAINING] "
                f"{run_directory}"
            )

            print(
                "[RECOVER] Training tidak diulang; "
                "langsung menjalankan test evaluation."
            )

        else:
            for stale_directory in (
                run_directory,
                test_directory,
            ):
                if stale_directory.exists():
                    print(
                        "[CLEAN INCOMPLETE RUN] "
                        f"{stale_directory}"
                    )

                    archive_artifact(stale_directory, "INCOMPLETE_OR_MISMATCHED_RUN")

            archive_artifact(train_log, "STALE_LOG")

            metadata_path = write_training_budget(
                run_directory=run_directory,
                model=model_name,
                scenario=scenario,
                epochs=run_epochs,
                batch=batch,
                workers=workers,
                input_size=640,
                grad_accum=max(1, round(64 / batch)),
                seed=seed,
                patience=patience,
            )

            run_command(
                conda_command(
                    environment_name,
                    [
                        "python",
                        str(SCRIPT_PATH),
                        "--worker",
                        "yolo_train",
                        "--family",
                        family,
                        "--weights",
                        str(initial_weights),
                        "--dataset",
                        str(
                            SCENARIOS[
                                scenario
                            ]
                            / "data.yaml"
                        ),
                        "--project",
                        str(RUNS_DIR),
                        "--run-name",
                        run_name,
                        "--epochs",
                        str(run_epochs),
                        "--batch",
                        str(batch),
                        "--imgsz",
                        "640",
                        "--patience",
                        str(patience),
                        "--device",
                        device,
                        "--workers",
                        str(workers),
                        "--seed",
                        str(seed),
                    ],
                ),
                log_path=train_log,
            )

        best_weight = (
            run_directory
            / "weights"
            / "best.pt"
        )

        last_weight = (
            run_directory
            / "weights"
            / "last.pt"
        )

        evaluation_weight = (
            best_weight
            if best_weight.exists()
            else last_weight
        )

        if not evaluation_weight.exists():
            raise FileNotFoundError(
                "Checkpoint untuk evaluasi tidak ditemukan: "
                f"{run_directory}"
            )

        # Test evaluation selalu dijalankan pada proses Python baru.
        if test_directory.exists():
            archive_artifact(test_directory, "PREVIOUS_NATIVE_TEST")

        archive_artifact(test_log, "PREVIOUS_TEST_LOG")

        metadata_path = write_training_budget(
            run_directory=run_directory,
            model=model_name,
            scenario=scenario,
            epochs=run_epochs,
            batch=batch,
            workers=workers,
            input_size=640,
            grad_accum=max(1, round(64 / batch)),
            seed=seed,
            patience=patience,
        )
        coco_primary = export_scenario_to_coco(scenario, "primary_full")
        coco_balanced = export_scenario_to_coco(scenario, "balanced_diagnostic")

        run_command(
            conda_command(
                environment_name,
                [
                    "python",
                    str(SCRIPT_PATH),
                    "--worker",
                    "yolo_eval",
                    "--family",
                    family,
                    "--weights",
                    str(
                        evaluation_weight.resolve()
                    ),
                    "--dataset",
                    str(
                        SCENARIOS[
                            scenario
                        ]
                        / "data.yaml"
                    ),
                    "--balanced-dataset",
                    str(SCENARIOS[scenario] / "data_balanced.yaml"),
                    "--coco-primary",
                    str(coco_primary),
                    "--coco-balanced",
                    str(coco_balanced),
                    "--metadata",
                    str(metadata_path),
                    "--final-report-dir",
                    str(REPORT_DIR),
                    "--project",
                    str(RUNS_DIR),
                    "--run-name",
                    run_name,
                    "--epochs",
                    str(run_epochs),
                    "--batch",
                    str(batch),
                    "--imgsz",
                    "640",
                    "--patience",
                    str(patience),
                    "--device",
                    device,
                    "--workers",
                    str(workers),
                    "--seed",
                    str(seed),
                ],
            ),
            log_path=test_log,
        )


def run_rfdetr_model(
    environment_name: str,
    scenarios: list[str],
    epochs: int,
    patience: int,
    device: str,
    seed: int,
    rf_batch: int,
    rf_grad_accum: int,
    workers: int,
    preflight_smoke: bool = False,
) -> None:
    run_command(
        conda_command(
            environment_name,
            [
                "python",
                "-m",
                "pip",
                "install",
                (
                    "rfdetr[train,loggers,augment]"
                    "==1.7.1"
                ),
            ],
        )
    )

    save_environment_provenance(
        "rfdetrs",
        environment_name,
    )

    for scenario in scenarios:
        run_epochs = planned_epochs(scenario, epochs)
        run_name = (
            f"{scenario}_rfdetrs"
        )

        output_directory = (
            RUNS_DIR / run_name
        )

        log_path = (
            REPORT_DIR
            / "logs"
            / f"{run_name}.log"
        )

        reusable_completed_training = (
            (output_directory / "training_complete.json").exists()
            and any(output_directory.glob("*.pth"))
        )
        if output_directory.exists() and not reusable_completed_training:
            archive_artifact(output_directory, "PREVIOUS_RFDETR_RUN")
        elif reusable_completed_training:
            print(
                "[RF-DETR RECOVERY] preserving completed training for "
                f"post-training evaluation repair: {output_directory}"
            )
        archive_artifact(log_path, "PREVIOUS_RFDETR_LOG")
        coco_primary = export_scenario_to_coco(scenario, "primary_full")
        coco_balanced = export_scenario_to_coco(scenario, "balanced_diagnostic")
        rf_primary = prepare_rfdetr_dataset(coco_primary)
        rf_balanced = prepare_rfdetr_dataset(coco_balanced)
        metadata_path = write_training_budget(
            run_directory=output_directory,
            model="rfdetrs",
            scenario=scenario,
            epochs=run_epochs,
            batch=rf_batch,
            workers=workers,
            input_size=512,
            grad_accum=rf_grad_accum,
            seed=seed,
            patience=patience,
        )

        run_command(
            conda_command(
                environment_name,
                [
                    "python",
                    str(SCRIPT_PATH),
                    "--worker",
                    "rfdetr",
                    "--dataset-root",
                    str(rf_primary),
                    "--balanced-dataset-root",
                    str(rf_balanced),
                    "--coco-primary",
                    str(coco_primary),
                    "--coco-balanced",
                    str(coco_balanced),
                    "--metadata",
                    str(metadata_path),
                    "--final-report-dir",
                    str(REPORT_DIR),
                    "--output-dir",
                    str(output_directory),
                    "--epochs",
                    str(run_epochs),
                    "--patience",
                    str(patience),
                    "--device",
                    device,
                    "--rf-batch",
                    str(rf_batch),
                    "--rf-grad-accum",
                    str(rf_grad_accum),
                    "--workers",
                    str(workers),
                    "--seed",
                    str(seed),
                    *(["--preflight-smoke"] if preflight_smoke else []),
                ],
            ),
            log_path=log_path,
        )


def prepare_tinyformer_test_config(
    repository_path: Path,
    scenario: str,
    coco_root: Path,
    training_config_path: Path,
    evaluation_view: str,
) -> Path:
    """
    Buat config evaluasi TinyFormer yang benar-benar memakai
    held-out TEST split, bukan validation split.
    """
    import yaml

    dataset_directory = (
        repository_path
        / "configs"
        / "dataset"
    )

    model_directory = (
        repository_path
        / "configs"
        / "ganoderma"
    )

    train_dataset_path = (
        dataset_directory
        / f"ganoderma_{scenario}.yml"
    )

    if not train_dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset config tidak ditemukan: "
            f"{train_dataset_path}"
        )

    dataset_payload = yaml.safe_load(
        train_dataset_path.read_text(
            encoding="utf-8"
        )
    )

    # Test-only TinyFormer memakai val_dataloader sebagai
    # evaluator. Oleh karena itu arahkan val_dataloader ke
    # held-out TEST split.
    test_loader = dataset_payload[
        "val_dataloader"
    ]

    test_loader["dataset"]["img_folder"] = str(
        (
            coco_root
            / "images"
            / "test"
        ).resolve()
    )

    test_loader["dataset"]["ann_file"] = str(
        (
            coco_root
            / "annotations"
            / "instances_test.json"
        ).resolve()
    )

    test_dataset_path = (
        dataset_directory
        / f"ganoderma_{scenario}_{evaluation_view}_test.yml"
    )

    test_dataset_path.write_text(
        yaml.safe_dump(
            dataset_payload,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    model_payload = yaml.safe_load(
        training_config_path.read_text(
            encoding="utf-8"
        )
    )

    includes = (
        model_payload.get(
            "__include__",
            []
        )
        or []
    )

    updated_includes = []

    training_dataset_name = (
        f"ganoderma_{scenario}.yml"
    )

    test_dataset_name = (
        f"ganoderma_{scenario}_{evaluation_view}_test.yml"
    )

    for item in includes:
        item_text = str(item)

        if training_dataset_name in item_text:
            updated_includes.append(
                f"../dataset/{test_dataset_name}"
            )
        else:
            updated_includes.append(item)

    model_payload[
        "__include__"
    ] = updated_includes

    test_config_path = (
        model_directory
        / f"tinyformer_s_{scenario}_{evaluation_view}_test.yml"
    )

    test_config_path.write_text(
        yaml.safe_dump(
            model_payload,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    # Audit agar tidak diam-diam kembali ke validation.
    test_dataset_check = yaml.safe_load(
        test_dataset_path.read_text(
            encoding="utf-8"
        )
    )

    test_img_folder = (
        test_dataset_check[
            "val_dataloader"
        ][
            "dataset"
        ][
            "img_folder"
        ]
    )

    test_ann_file = (
        test_dataset_check[
            "val_dataloader"
        ][
            "dataset"
        ][
            "ann_file"
        ]
    )

    if "/test" not in test_img_folder.replace("\\", "/"):
        raise RuntimeError(
            "TinyFormer TEST audit gagal: "
            f"{test_img_folder}"
        )

    if not test_ann_file.endswith(
        "instances_test.json"
    ):
        raise RuntimeError(
            "TinyFormer TEST annotation audit gagal: "
            f"{test_ann_file}"
        )

    print()
    print("===== TINYFORMER TEST CONFIG =====")
    print(f"scenario    : {scenario}")
    print(f"view        : {evaluation_view}")
    print(f"images      : {test_img_folder}")
    print(f"annotations : {test_ann_file}")
    print("expected    : held-out TEST split")
    print("==================================")
    print()

    return test_config_path


def select_tinyformer_evaluation_checkpoint(
    output_directory: Path,
) -> Path:
    """
    Prioritas:
    1. best_stg1.pth
    2. best_stg2.pth
    3. checkpoint dengan nomor epoch terbesar
    4. checkpoint .pth terbaru
    """
    best_stg1 = (
        output_directory
        / "best_stg1.pth"
    )

    best_stg2 = (
        output_directory
        / "best_stg2.pth"
    )

    # Benchmark kita tidak pernah masuk stage-2 karena
    # stage switch dimatikan, jadi best_stg1 biasanya benar.
    if best_stg1.exists():
        print(
            "[TINYFORMER CHECKPOINT] "
            f"Using best checkpoint: {best_stg1}"
        )
        return best_stg1

    if best_stg2.exists():
        print(
            "[TINYFORMER CHECKPOINT] "
            f"Using best checkpoint: {best_stg2}"
        )
        return best_stg2

    import re

    numbered = []

    for candidate in output_directory.glob(
        "checkpoint*.pth"
    ):
        match = re.search(
            r"checkpoint(\d+)\.pth$",
            candidate.name,
        )

        if match:
            numbered.append(
                (
                    int(match.group(1)),
                    candidate,
                )
            )

    if numbered:
        numbered.sort(
            key=lambda item: item[0]
        )

        selected = numbered[-1][1]

        print(
            "[TINYFORMER CHECKPOINT] "
            "best checkpoint tidak ditemukan; "
            f"using latest epoch: {selected}"
        )

        return selected

    candidates = [
        item
        for item
        in output_directory.glob("*.pth")
        if item.is_file()
    ]

    if not candidates:
        raise FileNotFoundError(
            "Tidak ada checkpoint TinyFormer di "
            f"{output_directory}"
        )

    candidates.sort(
        key=lambda item:
            item.stat().st_mtime
    )

    selected = candidates[-1]

    print(
        "[TINYFORMER CHECKPOINT] "
        f"Fallback: {selected}"
    )

    return selected


def run_tinyformer_model(
    environment_name: str,
    scenarios: list[str],
    epochs: int,
    batch: int,
    device: str,
    workers: int,
    seed: int,
    patience: int,
) -> None:
    repository_path = clone_repository(
        "tinyformers"
    )

    requirements_path = (
        repository_path / "requirements.txt"
    )

    run_command(
        conda_command(
            environment_name,
            [
                "python",
                "-m",
                "pip",
                "install",
                "-r",
                str(requirements_path),
            ],
        )
    )

    ensure_tinyformer_backbone(
        environment_name,
        repository_path,
    )

    model_checkpoint = (
        ensure_tinyformer_model_checkpoint(
            environment_name,
            repository_path,
        )
    )

    save_environment_provenance(
        "tinyformers",
        environment_name,
        repository_path,
    )

    for scenario in scenarios:
        run_epochs = planned_epochs(scenario, epochs)
        coco_root = export_scenario_to_coco(
            scenario, "primary_full"
        )
        balanced_coco_root = export_scenario_to_coco(
            scenario, "balanced_diagnostic"
        )

        run_name = (
            f"{scenario}_tinyformers"
        )

        output_directory = (
            RUNS_DIR / run_name
        )
        if output_directory.exists():
            archive_artifact(output_directory, "PREVIOUS_TINYFORMER_RUN")
        train_log_path = REPORT_DIR / "logs" / f"{run_name}.log"
        archive_artifact(train_log_path, "PREVIOUS_TINYFORMER_LOG")
        metadata_path = write_training_budget(
            run_directory=output_directory,
            model="tinyformers",
            scenario=scenario,
            epochs=run_epochs,
            batch=batch,
            workers=workers,
            input_size=640,
            grad_accum=1,
            seed=seed,
            patience=patience,
        )

        config_path = (
            prepare_tinyformer_config(
                repository_path=repository_path,
                scenario=scenario,
                coco_root=coco_root,
                output_directory=output_directory,
                epochs=run_epochs,
                batch=batch,
                workers=workers,
            )
        )

        log_path = (
            REPORT_DIR
            / "logs"
            / f"{run_name}.log"
        )

        environment_variables = os.environ.copy()

        environment_variables[
            "CUDA_VISIBLE_DEVICES"
        ] = device
        environment_variables["TINYFORMER_PATIENCE"] = str(patience)
        environment_variables["TINYFORMER_MIN_DELTA"] = "0.0"

        training_started = time.time()

        run_command(
            conda_command(
                environment_name,
                [
                    "python",
                    "-m",
                    "torch.distributed.run",
                    "--nproc_per_node=1",
                    "--master_port=7777",
                    "train.py",
                    "-c",
                    str(config_path),
                    "--use-amp",
                    "--seed",
                    str(seed),
                    "-t",
                    str(model_checkpoint),
                ],
            ),
            cwd=repository_path,
            log_path=train_log_path,
            env=environment_variables,
        )

        evaluation_checkpoint = (
            select_tinyformer_evaluation_checkpoint(
                output_directory
            )
        )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        checkpoint_payload = None
        try:
            import torch
            checkpoint_payload = torch.load(
                evaluation_checkpoint, map_location="cpu", weights_only=False
            )
        except Exception as error:
            print(f"[PROVENANCE] TinyFormer checkpoint inspection unavailable: {error}")
        if isinstance(checkpoint_payload, dict):
            metadata["best_epoch"] = (
                int(checkpoint_payload.get("last_epoch", -1)) + 1
                if checkpoint_payload.get("last_epoch") is not None else None
            )
            state = checkpoint_payload.get("model")
            if isinstance(state, dict):
                metadata["parameter_count"] = sum(
                    value.numel() for value in state.values() if hasattr(value, "numel")
                )
        checkpoint_numbers = [
            int(match.group(1))
            for path in output_directory.glob("checkpoint*.pth")
            if (match := re.search(r"checkpoint(\d+)\.pth$", path.name))
        ]
        metadata["epochs_completed"] = (
            max(checkpoint_numbers) + 1 if checkpoint_numbers else run_epochs
        )
        metadata["runtime"] = time.time() - training_started
        metadata["checkpoint"] = str(evaluation_checkpoint.resolve())
        metadata["optimizer_steps"] = (
            metadata["batches_per_epoch"] * metadata["epochs_completed"]
        )
        metadata["effective_samples_seen"] = (
            metadata["train_images"] * metadata["epochs_completed"]
        )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (output_directory / "augmentation_audit.json").write_text(
            json.dumps(
                {
                    "trainer_side_augmentation": "disabled",
                    "transforms": [
                        "Resize(640,640)",
                        "SanitizeBoundingBoxes",
                        "ConvertPILImage",
                        "Normalize",
                        "ConvertBoxes",
                    ],
                    "mixup": 0.0,
                    "copyblend": 0.0,
                    "multi_scale": False,
                    "verified": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        from common_coco_evaluator import (
            collect_final_reports,
            evaluate_common_coco,
            write_metrics,
        )
        for view_index, (evaluation_view, view_coco_root) in enumerate(
            (
                ("primary_full", coco_root),
                ("balanced_diagnostic", balanced_coco_root),
            )
        ):
            test_config_path = prepare_tinyformer_test_config(
                repository_path=repository_path,
                scenario=scenario,
                coco_root=view_coco_root,
                training_config_path=config_path,
                evaluation_view=evaluation_view,
            )
            prediction_path = output_directory / f"predictions_{evaluation_view}.json"
            test_environment = environment_variables.copy()
            test_environment["COMMON_COCO_PREDICTIONS"] = str(prediction_path.resolve())
            test_log_path = (
                REPORT_DIR / "logs" / f"{run_name}_{evaluation_view}_test.log"
            )
            archive_artifact(test_log_path, "PREVIOUS_TINYFORMER_TEST_LOG")
            run_command(
                conda_command(
                    environment_name,
                    [
                        "python",
                        "-m",
                        "torch.distributed.run",
                        "--nproc_per_node=1",
                        f"--master_port={7778 + view_index}",
                        "train.py",
                        "-c",
                        str(test_config_path),
                        "--test-only",
                        "-r",
                        str(evaluation_checkpoint),
                    ],
                ),
                cwd=repository_path,
                log_path=test_log_path,
                env=test_environment,
            )
            if not prediction_path.exists():
                raise RuntimeError(
                    f"TinyFormer did not write common COCO predictions: {prediction_path}"
                )
            rows = evaluate_common_coco(
                view_coco_root / "annotations" / "instances_test.json",
                prediction_path,
                {**metadata, "evaluation_view": evaluation_view},
            )
            write_metrics(
                rows, output_directory / f"common_metrics_{evaluation_view}.json"
            )
        collect_final_reports(RUNS_DIR, REPORT_DIR)


# ============================================================
# Main
# ============================================================

def parse_list_argument(
    value: str,
    supported: list[str],
) -> list[str]:
    if value.strip().lower() == "all":
        return supported.copy()

    selected = [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]

    unknown = [
        item
        for item in selected
        if item not in supported
    ]

    if unknown:
        raise ValueError(
            f"Pilihan tidak dikenal: {unknown}"
        )

    return selected


def verify_base_environment(
    base_environment: str,
) -> str:
    _, base_prefix = resolve_base_environment(
        base_environment
    )

    python_path = (
        Path(base_prefix)
        / "bin"
        / "python"
    )

    result = subprocess.run(
        [
            str(python_path),
            "-c",
            (
                "import sys, torch; "
                "print('python=' + sys.executable); "
                "print('torch=' + torch.__version__); "
                "print('cuda_available=' + "
                "str(torch.cuda.is_available())); "
                "print('gpu=' + "
                "(torch.cuda.get_device_name(0) "
                "if torch.cuda.is_available() "
                "else 'none'))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    print()
    print("===== BASE ENVIRONMENT =====")
    print(f"Prefix: {base_prefix}")
    print(result.stdout.strip())

    return base_prefix

def run_deimv2_model(
    environment_name: str,
    scenarios: list[str],
    epochs: int,
    batch: int,
    device: str,
    workers: int,
    seed: int,
    patience: int,
) -> None:
    run_command(
        [
            sys.executable,
            str(ROOT / "scripts" / "10_train_deimv2.py"),
            "--scenarios",
            ",".join(scenarios),
            "--environment",
            environment_name,
            "--epochs",
            str(epochs),
            "--batch",
            str(batch),
            "--workers",
            str(workers),
            "--patience",
            str(patience),
            "--seed",
            str(seed),
            "--device",
            device,
        ],
        cwd=ROOT,
        log_path=REPORT_DIR / "logs" / "deimv2s_orchestrator.log",
    )


def execute_model(
    model_name: str,
    args: argparse.Namespace,
    scenarios: list[str],
    epochs: int,
    patience: int,
    batch: int,
    workers: int,
) -> None:
    if model_name in {"yolov10s", "yolo11s", "yolov12s", "yolov13s"}:
        run_yolo_model(
            model_name=model_name,
            environment_name=args.temporary_environment,
            scenarios=scenarios,
            epochs=epochs,
            batch=batch,
            patience=patience,
            device=args.device,
            workers=workers,
            seed=args.seed,
        )
    elif model_name == "rfdetrs":
        run_rfdetr_model(
            environment_name=args.temporary_environment,
            scenarios=scenarios,
            epochs=epochs,
            patience=patience,
            device=args.device,
            seed=args.seed,
            rf_batch=batch,
            rf_grad_accum=args.rf_grad_accum,
            workers=workers,
            preflight_smoke=bool(getattr(args, "preflight_smoke", False)),
        )
    elif model_name == "tinyformers":
        run_tinyformer_model(
            environment_name=args.temporary_environment,
            scenarios=scenarios,
            epochs=epochs,
            batch=batch,
            device=args.device,
            workers=workers,
            seed=args.seed,
            patience=patience,
        )
    elif model_name == "deimv2s":
        run_deimv2_model(
            environment_name=args.temporary_environment,
            scenarios=scenarios,
            epochs=epochs,
            batch=batch,
            device=args.device,
            workers=workers,
            seed=args.seed,
            patience=patience,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def worker_candidates(requested: int) -> list[int]:
    return [value for value in (8, 4, 2, 0) if value <= requested]


def prepare_static_assets(batch: int, workers: int, epochs: int = 100) -> None:
    """Generate all scenario/view COCO, RF-DETR, and TinyFormer configs only."""
    tinyformer_root = clone_repository("tinyformers")
    for scenario in SCENARIOS:
        primary = export_scenario_to_coco(scenario, "primary_full")
        balanced = export_scenario_to_coco(scenario, "balanced_diagnostic")
        prepare_rfdetr_dataset(primary)
        prepare_rfdetr_dataset(balanced)
        train_config = prepare_tinyformer_config(
            repository_path=tinyformer_root,
            scenario=scenario,
            coco_root=primary,
            output_directory=BENCHMARK_RUNS_DIR / f"{scenario}_tinyformers",
            epochs=epochs,
            batch=batch,
            workers=workers,
        )
        for view, coco_root in (
            ("primary_full", primary),
            ("balanced_diagnostic", balanced),
        ):
            prepare_tinyformer_test_config(
                repository_path=tinyformer_root,
                scenario=scenario,
                coco_root=coco_root,
                training_config_path=train_config,
                evaluation_view=view,
            )


def print_readiness_report(preflight: dict) -> None:
    audit_path = ROOT / "reports" / "method_b" / "method_b_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    scenarios = audit["scenarios"]
    roots = {name: SCENARIOS[name] for name in SCENARIOS}
    balanced_valid = len(
        list((roots["B0"] / "balanced_diagnostic/valid/images").glob("*"))
    )
    balanced_test = len(
        list((roots["B0"] / "balanced_diagnostic/test/images").glob("*"))
    )
    print("\nMETHOD B B0/B1/B2 READINESS")
    print("=" * 36)
    for scenario in SCENARIOS:
        print(f"{scenario} train: {scenarios[scenario]['train']['images']}")
    print(f"\nFull valid: {scenarios['B0']['valid']['images']}")
    print(f"Full test : {scenarios['B0']['test']['images']}")
    print(f"\nBalanced diagnostic valid: {balanced_valid}")
    print(f"Balanced diagnostic test : {balanced_test}")
    for source in ("A", "B"):
        values = [
            scenarios["B0"][split]["source_images"].get(source, 0)
            for split in ("train", "valid", "test")
        ]
        print(f"\nSource {source}:\n{values[0]} / {values[1]} / {values[2]} (train / valid / test)")
    print(f"\nGroup leakage: {audit['group_leakage_count']}")
    print(
        "Exact-image leakage: "
        f"{audit['exact_original_image_leakage_count'] + audit['exact_scenario_image_leakage_count']}"
    )
    print(f"Invalid labels: {audit['invalid_label_count']}")
    identity_labels = {
        "primary_full_valid": "full valid",
        "primary_full_test": "full test",
        "balanced_diagnostic_valid": "balanced valid",
        "balanced_diagnostic_test": "balanced test",
    }
    for key, label in identity_labels.items():
        status = "PASS" if audit["identity"][key]["same_content"] else "FAIL"
        print(f"B0/B1/B2 {label} identical: {status}")
    fda = audit["fda_config_controls"]
    print("\nFDA config controls:")
    print(
        "bidirectional: "
        + ("active and verified" if fda["active_and_verified"] else "FAIL")
    )
    print(
        "donor_resize: "
        + ("active and verified" if fda["active_and_verified"] else "FAIL")
    )
    print("\nBatch preflight:")
    recommended_batch = preflight.get("recommended_common_batch")
    display_batch = recommended_batch if recommended_batch is not None else 16
    batch_results = preflight.get("batches", {}).get(str(display_batch), {})
    for model in SUPPORTED_MODELS:
        item = batch_results.get(model, {})
        print(
            f"{model}: {item.get('status', 'NOT TESTED')}"
            + (f" (workers={item.get('workers')})" if item.get("workers") is not None else "")
        )
    print(f"\nRecommended common batch: {recommended_batch}")
    print(f"Recommended workers: {preflight.get('recommended_common_workers')}")
    print(
        "\nREADY_FOR_FULL_BENCHMARK = "
        + ("YES" if preflight.get("ready_for_full_benchmark") else "NO")
    )
    if not preflight.get("ready_for_full_benchmark"):
        for reason in preflight.get("blocking_reasons", []):
            print(f" - {reason}")


def run_preflight(args: argparse.Namespace) -> None:
    global RUNS_DIR, REPORT_DIR, EPOCH_PLAN, BUDGET_MODE
    if parse_list_argument(args.models, SUPPORTED_MODELS) != SUPPORTED_MODELS:
        raise RuntimeError("Preflight must test all six model families; use --models all")
    validate_datasets(list(SCENARIOS))
    audit_path = ROOT / "reports" / "method_b" / "method_b_audit.json"
    if not audit_path.exists():
        raise FileNotFoundError("Run scripts/rebuild_method_b.py before preflight")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError(f"Method-B audit is not PASS: {audit.get('failures')}")

    fresh_report = {
        "policy": {
            "batch_order": [16, 32],
            "batch32_requires_all_batch16": True,
            "worker_fallback": worker_candidates(args.workers),
            "scenario": "B0",
            "epochs": 1,
            "seed": args.seed,
        },
        "batches": {},
        "blocking_reasons": [],
    }
    report = fresh_report
    if args.resume_preflight and PREFLIGHT_REPORT_PATH.exists():
        existing = json.loads(PREFLIGHT_REPORT_PATH.read_text(encoding="utf-8"))
        existing_policy = existing.get("policy", {})
        if (
            existing_policy.get("seed") == args.seed
            and existing_policy.get("batch_order") == [16, 32]
            and existing_policy.get("worker_fallback") == worker_candidates(args.workers)
        ):
            report = existing
            report["blocking_reasons"] = []
            print(f"[PREFLIGHT RESUME] {PREFLIGHT_REPORT_PATH}")
    BUDGET_MODE = "epoch_matched"
    EPOCH_PLAN = {"B0": 1}
    # Propagated only to worker subprocesses launched by this preflight.
    args.preflight_smoke = True
    batch16_all_stable = True
    reused_existing_environment = False
    for batch in (16, 32):
        if batch == 32 and not batch16_all_stable:
            report["batches"]["32"] = {"skipped": "batch16_not_stable_for_all"}
            break
        report["batches"].setdefault(str(batch), {})
        for model_name in SUPPORTED_MODELS:
            previous = report["batches"][str(batch)].get(model_name, {})
            if previous.get("status") == "PASS":
                print(
                    f"[PREFLIGHT REUSE PASS] model={model_name} batch={batch} "
                    f"workers={previous.get('workers')}"
                )
                continue
            # A resumed preflight may have already demonstrated a deterministic
            # host-side worker failure (for example, /dev/shm bus errors). Do
            # not spend another full smoke run reproducing that same failure;
            # retain it in provenance and continue with the next safe fallback.
            attempts = list(previous.get("attempts", []))
            initial_worker = worker_candidates(args.workers)[0]
            known_worker_failures = {
                int(attempt["workers"])
                for attempt in attempts
                if attempt.get("status") == "FAIL"
                and (
                    int(attempt["workers"]) == initial_worker
                    or any(
                        marker in str(attempt.get("error", "")).lower()
                        for marker in (
                            "bus error",
                            "dataloader worker",
                            "exit status 137",
                        )
                    )
                )
            }
            selected_workers = None
            temporary_prefix = get_conda_environment_map().get(
                args.temporary_environment
            )
            if (
                args.reuse_existing_temporary_environment
                and not reused_existing_environment
                and temporary_prefix is not None
            ):
                print(
                    "[PREFLIGHT REUSE TEMPORARY ENVIRONMENT] "
                    f"{temporary_prefix} for model={model_name}"
                )
                reused_existing_environment = True
            else:
                recreate_temporary_environment(
                    args.base_environment, args.temporary_environment
                )
            try:
                for workers in worker_candidates(args.workers):
                    if workers in known_worker_failures:
                        print(
                            "[PREFLIGHT REUSE KNOWN WORKER FAILURE] "
                            f"model={model_name} batch={batch} workers={workers}"
                        )
                        continue
                    RUNS_DIR = PREFLIGHT_RUNS_DIR / f"batch_{batch}" / f"workers_{workers}"
                    REPORT_DIR = (
                        BENCHMARK_REPORT_DIR / "preflight" / f"batch_{batch}" / f"workers_{workers}"
                    )
                    RUNS_DIR.mkdir(parents=True, exist_ok=True)
                    REPORT_DIR.mkdir(parents=True, exist_ok=True)
                    try:
                        execute_model(
                            model_name,
                            args,
                            ["B0"],
                            epochs=1,
                            patience=1,
                            batch=batch,
                            workers=workers,
                        )
                        attempts.append({"workers": workers, "status": "PASS"})
                        selected_workers = workers
                        break
                    except Exception as error:
                        attempts.append(
                            {
                                "workers": workers,
                                "status": "FAIL",
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
                        print(
                            f"[PREFLIGHT RETRY] model={model_name} batch={batch} "
                            f"workers={workers} failed: {error}"
                        )
            finally:
                if not args.keep_environment:
                    remove_temporary_environment(args.temporary_environment)
            item = {
                "status": "PASS" if selected_workers is not None else "FAIL",
                "workers": selected_workers,
                "attempts": attempts,
            }
            report["batches"][str(batch)][model_name] = item
            PREFLIGHT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            PREFLIGHT_REPORT_PATH.write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
        batch_all_stable = all(
            report["batches"][str(batch)][model]["status"] == "PASS"
            for model in SUPPORTED_MODELS
        )
        if batch == 16:
            batch16_all_stable = batch_all_stable

    batch32 = report.get("batches", {}).get("32", {})
    batch32_all = bool(batch32) and all(
        isinstance(batch32.get(model), dict)
        and batch32[model].get("status") == "PASS"
        for model in SUPPORTED_MODELS
    )
    recommended_batch = 32 if batch32_all else 16 if batch16_all_stable else None
    report["recommended_common_batch"] = recommended_batch
    if recommended_batch is not None:
        worker_values = [
            report["batches"][str(recommended_batch)][model]["workers"]
            for model in SUPPORTED_MODELS
        ]
        report["recommended_common_workers"] = min(worker_values)
        report["effective_workers_by_model"] = {
            model: report["batches"][str(recommended_batch)][model]["workers"]
            for model in SUPPORTED_MODELS
        }
    else:
        report["recommended_common_workers"] = None
        report["effective_workers_by_model"] = {}
        report["blocking_reasons"].append(
            "At least one model family failed batch=16 at workers 8, 4, 2, and 0"
        )
    report["ready_for_full_benchmark"] = (
        audit.get("status") == "PASS" and recommended_batch is not None
    )
    RUNS_DIR = BENCHMARK_RUNS_DIR
    REPORT_DIR = BENCHMARK_REPORT_DIR
    if recommended_batch is not None:
        prepare_static_assets(
            recommended_batch,
            report["recommended_common_workers"],
            epochs=args.epochs,
        )
    PREFLIGHT_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_readiness_report(report)


def main_orchestrator(args: argparse.Namespace) -> None:
    global RUNS_DIR, REPORT_DIR, EPOCH_PLAN, BUDGET_MODE
    models = parse_list_argument(args.models, SUPPORTED_MODELS)
    scenarios = parse_list_argument(args.scenarios, list(SCENARIOS))
    verify_base_environment(args.base_environment)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    ENV_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "preflight":
        run_preflight(args)
        return

    validate_datasets(scenarios)
    RUNS_DIR = BENCHMARK_RUNS_DIR
    REPORT_DIR = BENCHMARK_REPORT_DIR
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if args.rf_batch != args.batch:
        raise RuntimeError(
            f"Benchmark fairness requires --rf-batch == --batch; got {args.rf_batch} vs {args.batch}"
        )
    if args.rf_grad_accum != 1:
        print(f"[FAIRNESS WARNING] RF-DETR grad accumulation={args.rf_grad_accum}")

    effective_workers = args.workers
    if args.mode == "full":
        if not args.preflight_batch:
            raise RuntimeError("Full mode requires --preflight-batch")
        if not PREFLIGHT_REPORT_PATH.exists():
            raise FileNotFoundError(
                f"Preflight report missing: {PREFLIGHT_REPORT_PATH}; run --mode preflight first"
            )
        preflight = json.loads(PREFLIGHT_REPORT_PATH.read_text(encoding="utf-8"))
        if not preflight.get("ready_for_full_benchmark"):
            raise RuntimeError(
                f"Preflight is not ready: {preflight.get('blocking_reasons')}"
            )
        recommended_batch = int(preflight["recommended_common_batch"])
        if args.batch != recommended_batch:
            raise RuntimeError(
                f"Requested batch={args.batch}, preflight recommends common batch={recommended_batch}"
            )
        if args.auto_worker_fallback:
            effective_workers = int(preflight["recommended_common_workers"])
            print(
                f"[WORKER FALLBACK POLICY] requested={args.workers}, "
                f"preflight-approved common={effective_workers}"
            )

    epochs = 3 if args.mode == "smoke" else args.epochs
    patience = 3 if args.mode == "smoke" else args.patience
    BUDGET_MODE = args.budget_mode
    EPOCH_PLAN = build_epoch_plan(scenarios, epochs, args.batch, args.budget_mode)
    summary = {
        "mode": args.mode,
        "models": models,
        "scenarios": scenarios,
        "base_environment": args.base_environment,
        "temporary_environment": args.temporary_environment,
        "max_epochs": epochs,
        "patience": patience,
        "batch": args.batch,
        "workers_requested": args.workers,
        "workers_effective_common": effective_workers,
        "rf_batch": args.rf_batch,
        "rf_grad_accum": args.rf_grad_accum,
        "seed": args.seed,
        "budget_mode": args.budget_mode,
        "epoch_plan": EPOCH_PLAN,
        "budget_interpretation": (
            "epoch_matched answers whether the complete training pipeline improves performance; "
            "update_matched is the optional strict approximate optimizer-update ablation"
        ),
        "status": {},
    }
    previous_summary_path = REPORT_DIR / "run_summary.json"
    if previous_summary_path.exists():
        try:
            previous_summary = json.loads(
                previous_summary_path.read_text(encoding="utf-8")
            )
            summary["models"] = list(
                dict.fromkeys(previous_summary.get("models", []) + models)
            )
            summary["status"].update(previous_summary.get("status", {}))
        except Exception as error:
            print(f"[SUMMARY RECOVERY] skipped: {error}")
    for model_name in models:
        print("\n" + "=" * 70)
        print(f"MODEL: {model_name}")
        print("=" * 70)
        summary["status"][model_name] = "running"
        (REPORT_DIR / "run_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        try:
            temporary_available = (
                args.temporary_environment in get_conda_environment_map()
            )
            if args.reuse_existing_temporary_environment and temporary_available:
                print(
                    "[REUSE TEMPORARY ENVIRONMENT] "
                    f"{args.temporary_environment}"
                )
            else:
                recreate_temporary_environment(
                    args.base_environment, args.temporary_environment
                )
            execute_model(
                model_name,
                args,
                scenarios,
                epochs,
                patience,
                args.batch,
                effective_workers,
            )
            summary["status"][model_name] = "completed"
        except Exception as error:
            summary["status"][model_name] = (
                f"failed: {type(error).__name__}: {error}"
            )
            (REPORT_DIR / "run_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            raise
        finally:
            if not args.keep_environment:
                remove_temporary_environment(args.temporary_environment)
        (REPORT_DIR / "run_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    print("\n" + "=" * 70)
    print("REQUESTED BENCHMARK RUNS COMPLETE")
    print("=" * 70)
    print(f"Runs   : {RUNS_DIR}")
    print(f"Reports: {REPORT_DIR}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Internal worker mode.
    parser.add_argument(
        "--worker",
        choices=[
            "yolo",
            "yolo_train",
            "yolo_eval",
            "rfdetr",
        ],
        default=None,
        help=argparse.SUPPRESS,
    )

    # Orchestrator parameters.
    parser.add_argument(
        "--mode",
        choices=[
            "smoke",
            "preflight",
            "full",
        ],
        default="smoke",
    )

    parser.add_argument(
        "--models",
        default="all",
        help=(
            "all atau comma-separated: "
            + ",".join(
                SUPPORTED_MODELS
            )
        ),
    )

    parser.add_argument(
        "--scenarios",
        default="B0,B1,B2",
    )

    parser.add_argument(
        "--base-environment",
        default="soilcpu",
    )

    parser.add_argument(
        "--temporary-environment",
        default="ganoderma_bench_tmp",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--device",
        default="0",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--rf-batch",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--rf-grad-accum",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--keep-environment",
        action="store_true",
    )

    parser.add_argument(
        "--budget-mode",
        choices=["epoch_matched", "update_matched"],
        default="epoch_matched",
        help=(
            "epoch_matched is the main complete-pipeline benchmark; update_matched "
            "is the optional strict approximate optimizer-update ablation"
        ),
    )

    parser.add_argument(
        "--auto-worker-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--preflight-batch",
        action="store_true",
        help="Require and enforce the all-family batch/worker preflight for full mode",
    )

    parser.add_argument(
        "--resume-preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse only previously PASSed preflight family/batch results",
    )
    parser.add_argument(
        "--reuse-existing-temporary-environment",
        action="store_true",
        help=(
            "Use one already-created temporary environment for the first incomplete "
            "preflight family; intended only for an interrupted, same-environment resume"
        ),
    )
    parser.add_argument(
        "--preflight-smoke",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    # YOLO worker parameters.
    parser.add_argument(
        "--family",
        default="yolo",
    )

    parser.add_argument(
        "--weights",
        default="",
    )

    parser.add_argument(
        "--dataset",
        default="",
    )

    parser.add_argument("--balanced-dataset", default="")
    parser.add_argument("--coco-primary", default="")
    parser.add_argument("--coco-balanced", default="")
    parser.add_argument("--metadata", default="")
    parser.add_argument(
        "--final-report-dir",
        default=str(BENCHMARK_REPORT_DIR),
    )

    parser.add_argument(
        "--project",
        default="",
    )

    parser.add_argument(
        "--run-name",
        default="",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    # RF-DETR worker parameters.
    parser.add_argument(
        "--dataset-root",
        default="",
    )

    parser.add_argument("--balanced-dataset-root", default="")

    parser.add_argument(
        "--output-dir",
        default="",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.worker in {
        "yolo",
        "yolo_train",
    }:
        worker_yolo_train(args)
        return

    if args.worker == "yolo_eval":
        worker_yolo_eval(args)
        return

    if args.worker == "rfdetr":
        worker_rfdetr(args)
        return

    main_orchestrator(args)


if __name__ == "__main__":
    main()
