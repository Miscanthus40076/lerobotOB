#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import platform
from typing import TYPE_CHECKING, cast

from lerobot.utils.import_utils import make_device_from_device_class

from .camera import Camera
from .configs import CameraConfig, Cv2Rotation

if TYPE_CHECKING:
    from .opencv.camera_opencv import OpenCVCamera
    from .opencv.configuration_opencv import OpenCVCameraConfig


def _resolve_opencv_copy_source(camera_config: "OpenCVCameraConfig") -> str | None:
    if camera_config.copy is False:
        return None

    if isinstance(camera_config.copy, int):
        return f"camera{camera_config.copy}"

    return camera_config.copy


def make_cameras_from_configs(camera_configs: dict[str, CameraConfig]) -> dict[str, Camera]:
    cameras: dict[str, Camera] = {}

    for key, cfg in camera_configs.items():
        # TODO(Steven): Consider just using the make_device_from_device_class for all types
        if cfg.type == "opencv":
            from .opencv import OpenCVCamera

            source_key = _resolve_opencv_copy_source(cfg)
            if source_key is None:
                cameras[key] = OpenCVCamera(cfg)
                continue

            if source_key not in cameras:
                raise ValueError(
                    f"OpenCV camera {key} copies from {source_key}, but {source_key} has not been declared earlier."
                )

            source_camera = cameras[source_key]
            if not isinstance(source_camera, OpenCVCamera):
                raise ValueError(f"OpenCV camera {key} copies from non-OpenCV camera {source_key}.")

            cameras[key] = OpenCVCamera(cfg, source_camera=source_camera, source_key=source_key)

        elif cfg.type == "intelrealsense":
            from .realsense.camera_realsense import RealSenseCamera

            cameras[key] = RealSenseCamera(cfg)

        elif cfg.type == "realsense_d435i_color":
            from .realsense.camera_rs_d435i import RealSenseD435iColorCamera

            cameras[key] = RealSenseD435iColorCamera(cfg)

        elif cfg.type == "realsense_d435i_depth":
            from .realsense.camera_rs_d435i import RealSenseD435iDepthCamera

            cameras[key] = RealSenseD435iDepthCamera(cfg)

        elif cfg.type == "realsense_d405_color":
            from .realsense.camera_rs_d405 import RealSenseD405ColorCamera

            cameras[key] = RealSenseD405ColorCamera(cfg)

        elif cfg.type == "realsense_d405_depth":
            from .realsense.camera_rs_d405 import RealSenseD405DepthCamera

            cameras[key] = RealSenseD405DepthCamera(cfg)

        elif cfg.type == "reachy2_camera":
            from .reachy2_camera.reachy2_camera import Reachy2Camera

            cameras[key] = Reachy2Camera(cfg)

        elif cfg.type == "orbbec_color":
            from .orbbec.camera_orbbec import OrbbecColorCamera

            cameras[key] = OrbbecColorCamera(cfg)

        elif cfg.type == "orbbec_depth":
            from .orbbec.camera_orbbec import OrbbecDepthCamera

            cameras[key] = OrbbecDepthCamera(cfg)

        else:
            try:
                cameras[key] = cast(Camera, make_device_from_device_class(cfg))
            except Exception as e:
                raise ValueError(f"Error creating camera {key} with config {cfg}: {e}") from e

    return cameras


def get_cv2_rotation(rotation: Cv2Rotation) -> int | None:
    import cv2  # type: ignore  # TODO: add type stubs for OpenCV

    if rotation == Cv2Rotation.ROTATE_90:
        return int(cv2.ROTATE_90_CLOCKWISE)
    elif rotation == Cv2Rotation.ROTATE_180:
        return int(cv2.ROTATE_180)
    elif rotation == Cv2Rotation.ROTATE_270:
        return int(cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return None


def get_cv2_backend() -> int:
    import cv2

    if platform.system() == "Windows":
        return int(cv2.CAP_MSMF)  # Use MSMF for Windows instead of AVFOUNDATION
    # elif platform.system() == "Darwin":  # macOS
    #     return cv2.CAP_AVFOUNDATION
    else:  # Linux and others
        return int(cv2.CAP_ANY)
