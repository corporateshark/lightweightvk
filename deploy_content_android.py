#!/usr/bin/python3
# LightweightVK
#
# Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Packs the sample content into the archive the Android samples read, and puts it on the device.

The samples do not extract the archive, they read it in place (`TarFileReader` in `samples/VulkanApp.cpp`), and they look for
it in two places, in this order: the application's own OBB directory, then `$EXTERNAL_STORAGE/LVK/lvk_content.tar`. The
archive is the same for every sample, only its name differs when it is installed as an OBB.

    python deploy_content_android.py                     pack and push to $EXTERNAL_STORAGE/LVK/lvk_content.tar
    python deploy_content_android.py --obb <package>...  pack and install as the OBB of each installed package
    python deploy_content_android.py --pack-only         pack and stop, for a build that pushes later

`--out` overrides where the archive is written. Packing is skipped when the archive is already there, so delete it to
rebuild. `-DLVK_ANDROID_OBB_CONTENT=ON` makes the build run this with `--obb` for every generated Android sample.
"""

import argparse
import os
import subprocess
import sys
import tarfile

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ARCHIVE = os.path.join(ROOT, "third-party", "content", "archives", "lvk_content.tar")

paths = [
    (os.path.join(ROOT, "third-party", "content"), "content"),
    (os.path.join(ROOT, "third-party", "deps", "src", "3D-Graphics-Rendering-Cookbook", "data"), "deps/src/3D-Graphics-Rendering-Cookbook/data"),
    (os.path.join(ROOT, "third-party", "deps", "src", "ktx-software", "tests", "srcimages", "Iron_Bars"), "deps/src/ktx-software/tests/srcimages/Iron_Bars"),
]

exclude_abs = {
    os.path.join(ROOT, "third-party", "content", "archives"),
    os.path.join(ROOT, "third-party", "content", "patches"),
    os.path.join(ROOT, "third-party", "content", "src", "cloud"),
    os.path.join(ROOT, "third-party", "content", "src", "glTF-Sample-Models"),
    os.path.join(ROOT, "third-party", "content", "src", "CT_head"),
}

exclude_names = {".git"}


def pack(archive):
    if os.path.isfile(archive):
        print("{} already exists ({:.1f} MB), skipping creation".format(archive, os.path.getsize(archive) / (1024 * 1024)))
        return
    os.makedirs(os.path.dirname(archive), exist_ok=True)
    print("Creating {} ...".format(archive))
    total_files = 0
    with tarfile.open(archive, "w", format=tarfile.GNU_FORMAT) as tf:
        for desktop_path, archive_prefix in paths:
            if not os.path.isdir(desktop_path):
                print("  Warning: {} does not exist, skipping".format(desktop_path))
                continue
            print("  Adding {} ...".format(desktop_path))
            count = 0
            for root, dirs, files in os.walk(desktop_path):
                dirs[:] = [d for d in dirs if d not in exclude_names and os.path.abspath(os.path.join(root, d)) not in exclude_abs]
                for f in files:
                    full_path = os.path.join(root, f)
                    arcname = os.path.join(archive_prefix, os.path.relpath(full_path, desktop_path)).replace("\\", "/")
                    tf.add(full_path, arcname=arcname)
                    count += 1
            total_files += count
            print("    {} files".format(count))
    print("Created {} ({:.1f} MB, {} files)".format(archive, os.path.getsize(archive) / (1024 * 1024), total_files))


def adb(*args, capture=False):
    result = subprocess.run(["adb", *args], capture_output=capture, text=True)
    if result.returncode != 0:
        raise RuntimeError("adb {} failed with code {}".format(" ".join(args), result.returncode))
    return result.stdout if capture else ""


def installedPackages():
    out = adb("shell", "pm", "list", "packages", capture=True)
    return {line.strip()[len("package:"):] for line in out.splitlines() if line.startswith("package:")}


def externalStorage():
    return adb("shell", "echo", "$EXTERNAL_STORAGE", capture=True).strip() or None


def pushLoose(archive, storage):
    target = storage + "/LVK/lvk_content.tar"
    print("Uploading to {} ...".format(target))
    adb("push", archive, target)


def pushObb(archive, storage, package):
    target = "{}/Android/obb/{}/main.1.{}.obb".format(storage, package, package)
    adb("shell", "mkdir", "-p", "{}/Android/obb/{}".format(storage, package))
    print("Uploading to {} ...".format(target))
    adb("push", archive, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=DEFAULT_ARCHIVE, help="where to write the archive")
    parser.add_argument("--obb", nargs="+", metavar="PACKAGE", help="install as the OBB of these packages instead of pushing loose")
    parser.add_argument("--pack-only", action="store_true", help="do not touch the device")
    parser.add_argument("--all", action="store_true", help="with --obb, push to packages that are not installed too")
    args = parser.parse_args()

    pack(args.out)

    if args.pack_only:
        return 0

    try:
        storage = externalStorage()
        if not storage:
            print("External storage path is not found")
            return 1
        if not args.obb:
            pushLoose(args.out, storage)
            print("Completed")
            return 0
        wanted = args.obb
        if not args.all:
            installed = installedPackages()
            skipped = [p for p in wanted if p not in installed]
            wanted = [p for p in wanted if p in installed]
            if skipped:
                print("Not installed, skipping: {}".format(", ".join(skipped)))
            if not wanted:
                print("None of the packages are installed, nothing to do")
                return 0
        for package in wanted:
            pushObb(args.out, storage, package)
    except RuntimeError as e:
        print(e)
        return 1
    print("Completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
