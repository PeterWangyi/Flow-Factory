# Copyright 2026 The Flow-Factory Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import stat
from pathlib import Path

from PIL import Image

from flow_factory.logger.formatting import LogImage


def test_compressed_log_image_is_world_readable() -> None:
    image = LogImage(Image.new("RGB", (8, 8), color="red"))

    try:
        path = image.get_value()

        assert isinstance(path, str)
        assert stat.S_IMODE(Path(path).stat().st_mode) == 0o644
    finally:
        image.cleanup()
