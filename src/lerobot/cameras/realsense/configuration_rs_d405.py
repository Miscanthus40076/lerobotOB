from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode, Cv2Rotation

_SUPPORTED_COLOR_STREAM_FORMATS = {
    "rgb8": "rgb8",
    "rs2_format_rgb8": "rgb8",
    "bgr8": "bgr8",
    "rs2_format_bgr8": "bgr8",
    "yuyv": "yuyv",
    "rs2_format_yuyv": "yuyv",
}


def _validate_rotation(rotation: Cv2Rotation) -> None:
    if rotation not in (
        Cv2Rotation.NO_ROTATION,
        Cv2Rotation.ROTATE_90,
        Cv2Rotation.ROTATE_180,
        Cv2Rotation.ROTATE_270,
    ):
        raise ValueError(
            f"`rotation` is expected to be in "
            f"{(Cv2Rotation.NO_ROTATION, Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_180, Cv2Rotation.ROTATE_270)}, "
            f"but {rotation} is provided."
        )


def _validate_stream_shape(fps: int | None, width: int | None, height: int | None) -> None:
    values = (fps, width, height)
    if any(v is not None for v in values) and any(v is None for v in values):
        raise ValueError("For `fps`, `width` and `height`, either all of them need to be set, or none of them.")


def _normalize_color_stream_format(color_stream_format: str) -> str:
    normalized = color_stream_format.strip().lower()
    if normalized not in _SUPPORTED_COLOR_STREAM_FORMATS:
        raise ValueError(
            f"`color_stream_format` is expected to be one of "
            f"{tuple(sorted(_SUPPORTED_COLOR_STREAM_FORMATS))}, but {color_stream_format} is provided."
        )
    return _SUPPORTED_COLOR_STREAM_FORMATS[normalized]


@CameraConfig.register_subclass("realsense_d405_color")
@dataclass
class RealSenseD405ColorCameraConfig(CameraConfig):
    serial_number_or_name: str
    color_mode: ColorMode = ColorMode.RGB
    color_stream_format: str = "rgb8"
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1

    def __post_init__(self) -> None:
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` is expected to be {ColorMode.RGB.value} or {ColorMode.BGR.value}, "
                f"but {self.color_mode} is provided."
            )

        _validate_rotation(self.rotation)
        _validate_stream_shape(self.fps, self.width, self.height)
        self.color_stream_format = _normalize_color_stream_format(self.color_stream_format)


@CameraConfig.register_subclass("realsense_d405_depth")
@dataclass
class RealSenseD405DepthCameraConfig(CameraConfig):
    serial_number_or_name: str
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1

    def __post_init__(self) -> None:
        _validate_rotation(self.rotation)
        _validate_stream_shape(self.fps, self.width, self.height)
