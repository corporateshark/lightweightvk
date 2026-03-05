#!/usr/bin/env python3

import os
import shutil
from pathlib import Path

base_dir=os.getcwd()
lib_dir=os.path.join(base_dir, "src", "openxr-sdk")

file = Path(os.path.join(lib_dir, "src", "CMakeLists.txt"))
file.write_text(file.read_text()
  .replace('find_package(Sanitizers)', 'find_package(Sanitizers QUIET)')
)
