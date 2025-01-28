#!/usr/bin/python3
# LightweightVK
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess

def get_external_storage():
    adb_command = ['adb', 'shell', 'echo', '$EXTERNAL_STORAGE']
    try:
        process = subprocess.Popen(adb_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        output = stdout.decode().strip()
        if stderr:
            print("Error executing adb shell command:", stderr.decode().strip())
        return output
    except Exception as e:
        print("An error occurred:", e)
        return None

external_storage_path = get_external_storage()
if external_storage_path is not None:
    paths = [(os.path.join("third-party", "content"),
              os.path.join(external_storage_path, "LVK", "content").replace("\\", "/")),
             (os.path.join("third-party", "deps", "src", "3D-Graphics-Rendering-Cookbook", "data"),
              os.path.join(external_storage_path, "LVK", "deps", "src", "3D-Graphics-Rendering-Cookbook", "data").replace("\\", "/"))]
    for (desktop_path, android_path) in paths:
        try:
            print('Copying {} to {} ...'.format(desktop_path, android_path))
            process = subprocess.Popen(['adb', 'push', desktop_path, android_path], shell=False)
            process.communicate()
            process.wait()
            print('Completed')
        except Exception as e:
            print("An error occurred:", e)
else:
    print("External storage path is not found")
