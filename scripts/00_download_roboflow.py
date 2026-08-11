from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/datasets.yaml",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(
        Path(args.config).read_text(encoding="utf-8")
    ) or {}

    download_format = cfg.get(
        "download_format",
        "yolov8",
    )

    pending_downloads = []

    for source_name, source in cfg["sources"].items():
        # Raw Roboflow download disimpan di download_output.
        # Kalau tidak didefinisikan, gunakan output.
        output = Path(
            source.get(
                "download_output",
                source["output"],
            )
        )

        if (
            output.exists()
            and any(output.iterdir())
            and not args.overwrite
        ):
            print(
                f"[SKIP] {source_name}: "
                f"raw dataset sudah tersedia di {output}"
            )
            continue

        pending_downloads.append(
            (
                source_name,
                source,
                output,
            )
        )

    # Roboflow CLI tidak diperlukan kalau semua data sudah tersedia.
    if not pending_downloads:
        print("[OK] Tidak ada dataset yang perlu di-download.")
        return

    if shutil.which("roboflow") is None:
        raise SystemExit(
            "CLI Roboflow tidak ditemukan.\n"
            "Install dengan: pip install roboflow\n"
            "Kemudian login dengan: roboflow login"
        )

    for source_name, source, output in pending_downloads:
        if args.overwrite and output.exists():
            shutil.rmtree(output)

        output.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataset_url = (
            f'{source["workspace"]}/'
            f'{source["project"]}/'
            f'{int(source["version"])}'
        )

        command = [
            "roboflow",
            "download",
            "-f",
            download_format,
            "-l",
            str(output),
            dataset_url,
        ]

        print("[RUN]", " ".join(command))

        try:
            subprocess.run(
                command,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            print(
                f"\nDownload {source_name} gagal. "
                "Pastikan sudah login, versi benar, "
                "dan dataset dapat diakses.",
                file=sys.stderr,
            )
            raise SystemExit(
                exc.returncode
            ) from exc

        print(
            f"[OK] {source_name}: "
            f"downloaded to {output}"
        )


if __name__ == "__main__":
    main()