#!/usr/bin/env python3
"""
Packaging Script for Amazon ML Challenge 2026
Maps the clean flat project workspace into the official submission ZIP archive format:

    <team_name>_submission.zip
    ├── output/
    │   ├── matching_results.tsv
    │   └── candidate_pairs.tsv
    ├── code/
    │   └── business_entity_resolution/
    │       ├── src/
    │       ├── models/
    │       ├── README.md
    │       ├── requirements.txt
    │       └── run_pipeline.py
    └── Documentation_template.md

Usage:
    python package_submission.py [--team-name BrawlDevs]
"""

import os
import sys
import zipfile
import shutil
import argparse

EXCLUDE_EXTENSIONS = {'.pyc', '.pyo', '.tmp', '.DS_Store'}
EXCLUDE_DIRS = {'__pycache__', '.venv', '.idea', '.vscode'}


def make_submission_zip(team_name: str = "BrawlDevs"):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    zip_filename = os.path.join(root_dir, f"{team_name}_submission.zip")

    output_dir = os.path.join(root_dir, "output")
    matching_tsv = os.path.join(output_dir, "matching_results.tsv")
    candidate_tsv = os.path.join(output_dir, "candidate_pairs.tsv")

    if not os.path.exists(matching_tsv):
        print(f"Error: Required file missing: {matching_tsv}")
        sys.exit(1)

    print(f"Creating submission archive: {zip_filename}")
    file_count = 0
    total_bytes = 0

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # 1. Add output/ files
        for fname in ["matching_results.tsv", "candidate_pairs.tsv"]:
            fpath = os.path.join(output_dir, fname)
            if os.path.exists(fpath):
                arc = f"output/{fname}"
                print(f"  Adding: {arc} ({os.path.getsize(fpath):,} bytes)")
                zf.write(fpath, arcname=arc)
                file_count += 1
                total_bytes += os.path.getsize(fpath)
            else:
                print(f"  Warning: {fname} not found in output/")

        # 2. Add Documentation_template.md (mapped from README.md)
        readme_path = os.path.join(root_dir, "README.md")
        if os.path.exists(readme_path):
            arc = "Documentation_template.md"
            print(f"  Adding: {arc} (from README.md, {os.path.getsize(readme_path):,} bytes)")
            zf.write(readme_path, arcname=arc)
            file_count += 1
            total_bytes += os.path.getsize(readme_path)

        # 3. Add code/business_entity_resolution/ files
        code_prefix = "code/business_entity_resolution"
        
        # Root code files
        for fname in ["run_pipeline.py", "requirements.txt", "README.md"]:
            fpath = os.path.join(root_dir, fname)
            if os.path.exists(fpath):
                arc = f"{code_prefix}/{fname}"
                print(f"  Adding: {arc} ({os.path.getsize(fpath):,} bytes)")
                zf.write(fpath, arcname=arc)
                file_count += 1
                total_bytes += os.path.getsize(fpath)

        # src/ directory
        src_dir = os.path.join(root_dir, "src")
        if os.path.exists(src_dir):
            for root, dirs, files in os.walk(src_dir):
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                for file in files:
                    ext = os.path.splitext(file)[1]
                    if ext in EXCLUDE_EXTENSIONS or file.startswith("."):
                        continue
                    full_path = os.path.join(root, file)
                    rel_to_src = os.path.relpath(full_path, root_dir)
                    arc = f"{code_prefix}/{rel_to_src}".replace("\\", "/")
                    print(f"  Adding: {arc} ({os.path.getsize(full_path):,} bytes)")
                    zf.write(full_path, arcname=arc)
                    file_count += 1
                    total_bytes += os.path.getsize(full_path)

        # models/ directory
        models_dir = os.path.join(root_dir, "models")
        if os.path.exists(models_dir):
            for root, dirs, files in os.walk(models_dir):
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                for file in files:
                    full_path = os.path.join(root, file)
                    rel = os.path.relpath(full_path, root_dir)
                    arc = f"{code_prefix}/{rel}".replace("\\", "/")
                    print(f"  Adding: {arc} ({os.path.getsize(full_path):,} bytes)")
                    zf.write(full_path, arcname=arc)
                    file_count += 1
                    total_bytes += os.path.getsize(full_path)

    print(f"\nDone! Packaged {file_count} files ({total_bytes / (1024 * 1024):.1f} MB uncompressed) into {zip_filename}")
    print(f"Archive Size: {os.path.getsize(zip_filename) / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package flat workspace into official submission zip.")
    parser.add_argument("--team-name", type=str, default="BrawlDevs", help="Team name for zip prefix")
    args = parser.parse_args()

    make_submission_zip(team_name=args.team_name)
