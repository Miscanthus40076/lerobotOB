from __future__ import annotations

import importlib
import queue
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.realsense.configuration_rs_d405 import (
    RealSenseD405ColorCameraConfig,
    RealSenseD405DepthCameraConfig,
)
from lerobot.cameras.utils import make_cameras_from_configs

d405_module = importlib.import_module("lerobot.cameras.realsense.camera_rs_d405")


@dataclass(frozen=True)
class FakeResolvedStream:
    width: int
    height: int
    fps: int


class FakeFrameProfile:
    def __init__(self, resolved_stream: FakeResolvedStream, frame_format: str):
        self._resolved_stream = resolved_stream
        self._frame_format = frame_format

    def as_video_stream_profile(self) -> "FakeFrameProfile":
        return self

    def width(self) -> int:
        return self._resolved_stream.width

    def height(self) -> int:
        return self._resolved_stream.height

    def fps(self) -> int:
        return self._resolved_stream.fps

    def format(self) -> str:
        return self._frame_format


class FakeFrame:
    def __init__(self, data: np.ndarray, resolved_stream: FakeResolvedStream, frame_format: str):
        self._data = data
        self.profile = FakeFrameProfile(resolved_stream, frame_format)

    def get_data(self) -> np.ndarray:
        return self._data


class FakeFrameSet:
    def __init__(
        self,
        color: np.ndarray | None = None,
        depth: np.ndarray | None = None,
        color_stream: FakeResolvedStream | None = None,
        depth_stream: FakeResolvedStream | None = None,
        color_format: str = "yuyv",
    ):
        self.keep_calls = 0
        self._color = (
            FakeFrame(color, color_stream, color_format) if color is not None and color_stream is not None else None
        )
        self._depth = (
            FakeFrame(depth, depth_stream, "z16") if depth is not None and depth_stream is not None else None
        )

    def get_color_frame(self) -> FakeFrame | None:
        return self._color

    def get_depth_frame(self) -> FakeFrame | None:
        return self._depth

    def keep(self) -> None:
        self.keep_calls += 1


class FakeVideoStreamProfile:
    def __init__(self, resolved_stream: FakeResolvedStream):
        self._resolved_stream = resolved_stream

    def as_video_stream_profile(self) -> "FakeVideoStreamProfile":
        return self

    def width(self) -> int:
        return self._resolved_stream.width

    def height(self) -> int:
        return self._resolved_stream.height

    def fps(self) -> int:
        return self._resolved_stream.fps


class FakePipelineProfile:
    def __init__(self, resolved_streams: dict[str, FakeResolvedStream]):
        self._resolved_streams = resolved_streams

    def get_stream(self, stream_name: str) -> FakeVideoStreamProfile:
        return FakeVideoStreamProfile(self._resolved_streams[stream_name])


class FakeDevice:
    def __init__(self, device_info: dict[str, str]):
        self._device_info = device_info

    def get_info(self, field: str) -> str:
        return self._device_info[field]


class FakeContext:
    def __init__(self, runtime: "FakeRealSenseRuntime"):
        self._runtime = runtime

    def query_devices(self) -> list[FakeDevice]:
        return [FakeDevice(device) for device in self._runtime.devices]


class FakePipelineWrapper:
    def __init__(self, pipeline: "FakePipeline"):
        self.pipeline = pipeline


class FakeAlign:
    def __init__(self, runtime: "FakeRealSenseRuntime", target_stream: str):
        self._runtime = runtime
        self.target_stream = target_stream

    def process(self, frames: FakeFrameSet) -> FakeFrameSet:
        self._runtime.align_process_calls += 1
        return frames


class FakeConfig:
    def __init__(self, runtime: "FakeRealSenseRuntime"):
        self._runtime = runtime
        self.serial_number: str | None = None
        self.requests: dict[str, dict[str, int | None | str]] = {}

    def enable_device(self, serial_number: str) -> None:
        self.serial_number = serial_number

    def enable_stream(self, stream_name: str, *args: Any) -> None:
        width = height = fps = None
        fmt = None
        if len(args) == 4:
            width, height, fmt, fps = args
        self.requests[stream_name] = {
            "width": None if width in (None, 0) else int(width),
            "height": None if height in (None, 0) else int(height),
            "fps": None if fps in (None, 0) else int(fps),
            "format": fmt,
        }

    def resolve(self, pipeline_wrapper: FakePipelineWrapper) -> FakePipelineProfile:
        del pipeline_wrapper
        return FakePipelineProfile(self._runtime.resolve_requests(self))


class FakePipeline:
    def __init__(self, runtime: "FakeRealSenseRuntime"):
        self._runtime = runtime
        self.serial_number: str | None = None
        self.started = False

    def start(self, config: FakeConfig) -> FakePipelineProfile:
        resolved_streams = self._runtime.start_pipeline(config)
        self.serial_number = config.serial_number
        self.started = True
        return FakePipelineProfile(resolved_streams)

    def try_wait_for_frames(self, timeout_ms: int) -> tuple[bool, FakeFrameSet | None]:
        if self.serial_number is None:
            raise RuntimeError("Pipeline was not started.")
        frame_queue = self._runtime.frames[self.serial_number]
        try:
            frames = frame_queue.get(timeout=max(timeout_ms / 1000.0, 0.01))
        except queue.Empty:
            return False, None
        return True, frames

    def stop(self) -> None:
        if self.started:
            self._runtime.stop_calls += 1
        self.started = False


class FakeRealSenseModule:
    def __init__(self, runtime: "FakeRealSenseRuntime"):
        self._runtime = runtime
        self.stream = SimpleNamespace(color="color", depth="depth")
        self.format = SimpleNamespace(yuyv="yuyv", bgr8="bgr8", rgb8="rgb8", z16="z16")
        self.camera_info = SimpleNamespace(
            name="name",
            serial_number="serial_number",
            firmware_version="firmware_version",
            usb_type_descriptor="usb_type_descriptor",
            product_line="product_line",
            product_id="product_id",
        )

    def context(self) -> FakeContext:
        return FakeContext(self._runtime)

    def config(self) -> FakeConfig:
        return FakeConfig(self._runtime)

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self._runtime)

    def pipeline_wrapper(self, pipeline: FakePipeline) -> FakePipelineWrapper:
        return FakePipelineWrapper(pipeline)

    def align(self, target_stream: str) -> FakeAlign:
        return FakeAlign(self._runtime, target_stream)


class FakeRealSenseRuntime:
    def __init__(self) -> None:
        self.devices = [
            {
                "name": "Intel RealSense D405",
                "serial_number": "405123",
                "firmware_version": "1.0.0",
                "usb_type_descriptor": "USB3",
                "product_line": "D400",
                "product_id": "0B5B",
            }
        ]
        self.supported_profiles = {
            "405123": {
                "color": [
                    FakeResolvedStream(width=2, height=2, fps=30),
                    FakeResolvedStream(width=640, height=480, fps=30),
                ],
                "depth": [
                    FakeResolvedStream(width=2, height=2, fps=30),
                    FakeResolvedStream(width=640, height=480, fps=30),
                ],
            }
        }
        self.default_profiles = {
            "405123": {
                "color": FakeResolvedStream(width=640, height=480, fps=30),
                "depth": FakeResolvedStream(width=640, height=480, fps=30),
            }
        }
        self.frames = {"405123": queue.Queue()}
        self.start_history: list[dict[str, Any]] = []
        self.stop_calls = 0
        self.align_process_calls = 0
        self.active_streams: dict[str, dict[str, FakeResolvedStream]] = {}
        self.module = FakeRealSenseModule(self)

    def resolve_requests(self, config: FakeConfig) -> dict[str, FakeResolvedStream]:
        if config.serial_number is None:
            raise RuntimeError("No serial number configured.")
        if config.serial_number not in self.supported_profiles:
            raise RuntimeError(f"Unknown serial {config.serial_number}.")

        resolved_streams: dict[str, FakeResolvedStream] = {}
        for stream_name, request in config.requests.items():
            resolved_streams[stream_name] = self._resolve_stream(
                serial_number=config.serial_number,
                stream_name=stream_name,
                width=request["width"],
                height=request["height"],
                fps=request["fps"],
            )
        return resolved_streams

    def _resolve_stream(
        self,
        serial_number: str,
        stream_name: str,
        width: int | None,
        height: int | None,
        fps: int | None,
    ) -> FakeResolvedStream:
        candidates = self.supported_profiles[serial_number][stream_name]
        if width is None or height is None:
            if fps is None:
                return self.default_profiles[serial_number][stream_name]
            for candidate in candidates:
                if candidate.fps == fps:
                    return candidate
            raise RuntimeError(f"No {stream_name} profile found for fps {fps}.")

        for candidate in candidates:
            if candidate.width == width and candidate.height == height and (fps is None or candidate.fps == fps):
                return candidate
        raise RuntimeError(f"No {stream_name} profile found for {width}x{height}@{fps}.")

    def start_pipeline(self, config: FakeConfig) -> dict[str, FakeResolvedStream]:
        resolved_streams = self.resolve_requests(config)
        if config.serial_number is not None:
            self.active_streams[config.serial_number] = resolved_streams
        self.start_history.append(
            {"serial_number": config.serial_number, "streams": resolved_streams, "requests": dict(config.requests)}
        )
        return resolved_streams

    def push_frame(
        self,
        serial_number: str,
        color: np.ndarray | None = None,
        depth: np.ndarray | None = None,
        color_format: str = "yuyv",
    ) -> FakeFrameSet:
        active_streams = self.active_streams.get(serial_number, self.default_profiles[serial_number])
        color_stream = active_streams.get("color", self.default_profiles[serial_number]["color"])
        depth_stream = active_streams.get("depth", self.default_profiles[serial_number]["depth"])
        frameset = FakeFrameSet(
            color=color,
            depth=depth,
            color_stream=color_stream,
            depth_stream=depth_stream,
            color_format=color_format,
        )
        self.frames[serial_number].put(frameset)
        return frameset


@pytest.fixture
def fake_realsense_runtime(monkeypatch: pytest.MonkeyPatch) -> FakeRealSenseRuntime:
    runtime = FakeRealSenseRuntime()
    d405_module.RealSenseD405Manager._registry.clear()
    monkeypatch.setattr(d405_module, "rs", runtime.module)
    yield runtime
    d405_module.RealSenseD405Manager._registry.clear()


def _wait_for_manager_frames(camera: Any, timeout_s: float = 1.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if camera.manager is not None:
            frames, seq = camera.manager.get_latest_frames_and_seq()
            if frames is not None and seq > 0:
                return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for the fake RealSense D405 manager to receive frames.")


def _wait_for_manager_seq(camera: Any, min_seq: int, timeout_s: float = 1.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if camera.manager is not None:
            _, seq = camera.manager.get_latest_frames_and_seq()
            if seq >= min_seq:
                return
        time.sleep(0.01)
    raise AssertionError(f"Timed out waiting for the fake RealSense D405 manager to reach seq>={min_seq}.")


def test_make_cameras_from_configs_supports_d405_types(fake_realsense_runtime: FakeRealSenseRuntime) -> None:
    cameras = make_cameras_from_configs(
        {
            "color": RealSenseD405ColorCameraConfig(
                serial_number_or_name="405123", width=640, height=480, fps=30
            ),
            "depth": RealSenseD405DepthCameraConfig(
                serial_number_or_name="405123", width=640, height=480, fps=30
            ),
        }
    )

    assert isinstance(cameras["color"], d405_module.RealSenseD405ColorCamera)
    assert isinstance(cameras["depth"], d405_module.RealSenseD405DepthCamera)


def test_connect_reuses_manager_and_restarts_when_depth_is_added(
    fake_realsense_runtime: FakeRealSenseRuntime,
) -> None:
    color = d405_module.RealSenseD405ColorCamera(
        RealSenseD405ColorCameraConfig(serial_number_or_name="405123", width=640, height=480, fps=30)
    )
    depth = d405_module.RealSenseD405DepthCamera(
        RealSenseD405DepthCameraConfig(serial_number_or_name="405123", width=640, height=480, fps=30)
    )

    color.connect(warmup=False)
    assert len(fake_realsense_runtime.start_history) == 1
    assert set(fake_realsense_runtime.start_history[0]["streams"]) == {"color"}
    assert fake_realsense_runtime.start_history[0]["requests"]["color"]["format"] == "rgb8"

    depth.connect(warmup=False)
    assert len(fake_realsense_runtime.start_history) == 2
    assert fake_realsense_runtime.stop_calls == 1
    assert set(fake_realsense_runtime.start_history[1]["streams"]) == {"color", "depth"}
    assert fake_realsense_runtime.start_history[1]["requests"]["color"]["format"] == "rgb8"
    assert color.manager is depth.manager

    color.disconnect()
    depth.disconnect()


def test_connect_allows_overriding_color_stream_format(
    fake_realsense_runtime: FakeRealSenseRuntime,
) -> None:
    color = d405_module.RealSenseD405ColorCamera(
        RealSenseD405ColorCameraConfig(
            serial_number_or_name="405123",
            width=640,
            height=480,
            fps=30,
            color_stream_format="RS2_FORMAT_YUYV",
        )
    )

    color.connect(warmup=False)

    assert fake_realsense_runtime.start_history[0]["requests"]["color"]["format"] == "yuyv"
    assert color.config.color_stream_format == "yuyv"

    color.disconnect()


def test_connect_rejects_mismatched_shared_fps(fake_realsense_runtime: FakeRealSenseRuntime) -> None:
    color = d405_module.RealSenseD405ColorCamera(
        RealSenseD405ColorCameraConfig(serial_number_or_name="405123", width=640, height=480, fps=30)
    )
    depth = d405_module.RealSenseD405DepthCamera(
        RealSenseD405DepthCameraConfig(serial_number_or_name="405123", width=640, height=480, fps=60)
    )

    color.connect(warmup=False)
    with pytest.raises(ValueError, match="same fps"):
        depth.connect(warmup=False)

    color.disconnect()


def test_connect_rejects_mismatched_aligned_resolution(
    fake_realsense_runtime: FakeRealSenseRuntime,
) -> None:
    color = d405_module.RealSenseD405ColorCamera(
        RealSenseD405ColorCameraConfig(serial_number_or_name="405123", width=640, height=480, fps=30)
    )
    depth = d405_module.RealSenseD405DepthCamera(
        RealSenseD405DepthCameraConfig(serial_number_or_name="405123", width=2, height=2, fps=30)
    )

    color.connect(warmup=False)
    with pytest.raises(ValueError, match="same color/depth resolution"):
        depth.connect(warmup=False)

    color.disconnect()


def test_connect_rejects_unsupported_device(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRealSenseRuntime()
    runtime.devices = [
        {
            "name": "Intel RealSense D435I",
            "serial_number": "123456",
            "firmware_version": "1.0.0",
            "usb_type_descriptor": "USB3",
            "product_line": "D400",
            "product_id": "0B3A",
        }
    ]
    runtime.supported_profiles = {
        "123456": {
            "color": [FakeResolvedStream(width=640, height=480, fps=30)],
            "depth": [FakeResolvedStream(width=640, height=480, fps=30)],
        }
    }
    runtime.default_profiles = {
        "123456": {
            "color": FakeResolvedStream(width=640, height=480, fps=30),
            "depth": FakeResolvedStream(width=640, height=480, fps=30),
        }
    }
    runtime.frames = {"123456": queue.Queue()}
    d405_module.RealSenseD405Manager._registry.clear()
    monkeypatch.setattr(d405_module, "rs", runtime.module)

    color = d405_module.RealSenseD405ColorCamera(
        RealSenseD405ColorCameraConfig(serial_number_or_name="123456", width=640, height=480, fps=30)
    )

    with pytest.raises(ValueError, match="only supports D405"):
        color.connect(warmup=False)


def test_async_read_times_out_without_frames(fake_realsense_runtime: FakeRealSenseRuntime) -> None:
    color = d405_module.RealSenseD405ColorCamera(
        RealSenseD405ColorCameraConfig(serial_number_or_name="405123", width=640, height=480, fps=30)
    )

    color.connect(warmup=False)
    with pytest.raises(TimeoutError):
        color.async_read(timeout_ms=50)
    color.disconnect()


def test_yuyv_color_decode_and_depth_colormap(fake_realsense_runtime: FakeRealSenseRuntime) -> None:
    color = d405_module.RealSenseD405ColorCamera(
        RealSenseD405ColorCameraConfig(
            serial_number_or_name="405123",
            width=2,
            height=2,
            fps=30,
            color_mode=ColorMode.BGR,
        )
    )
    depth = d405_module.RealSenseD405DepthCamera(
        RealSenseD405DepthCameraConfig(serial_number_or_name="405123", width=2, height=2, fps=30)
    )

    color.connect(warmup=False)
    depth.connect(warmup=False)

    yuyv_raw = np.array(
        [
            [[64, 128], [96, 128]],
            [[128, 128], [160, 128]],
        ],
        dtype=np.uint8,
    )
    depth_raw = np.array([[0, 100], [200, 300]], dtype=np.uint16)
    fake_realsense_runtime.push_frame("405123", color=yuyv_raw, depth=depth_raw, color_format="yuyv")

    _wait_for_manager_frames(color)

    color_frame = color.async_read(timeout_ms=500)
    depth_frame = depth.async_read(timeout_ms=500)

    expected_color = cv2.cvtColor(yuyv_raw, cv2.COLOR_YUV2BGR_YUYV)
    expected_depth = cv2.applyColorMap(cv2.convertScaleAbs(depth_raw, alpha=0.03), cv2.COLORMAP_JET)

    assert np.array_equal(color_frame, expected_color)
    assert np.array_equal(depth_frame, expected_depth)
    assert fake_realsense_runtime.align_process_calls > 0

    color.disconnect()
    depth.disconnect()


def test_depth_first_then_color_uses_aligned_output_spec(
    fake_realsense_runtime: FakeRealSenseRuntime,
) -> None:
    depth = d405_module.RealSenseD405DepthCamera(
        RealSenseD405DepthCameraConfig(serial_number_or_name="405123", width=2, height=2, fps=30)
    )
    color = d405_module.RealSenseD405ColorCamera(
        RealSenseD405ColorCameraConfig(
            serial_number_or_name="405123",
            width=2,
            height=2,
            fps=30,
            color_mode=ColorMode.BGR,
        )
    )

    depth.connect(warmup=False)
    color.connect(warmup=False)

    yuyv_raw = np.array(
        [
            [[80, 128], [96, 128]],
            [[112, 128], [128, 128]],
        ],
        dtype=np.uint8,
    )
    depth_raw = np.array([[50, 100], [150, 200]], dtype=np.uint16)
    fake_realsense_runtime.push_frame("405123", color=yuyv_raw, depth=depth_raw, color_format="yuyv")

    _wait_for_manager_frames(depth)

    depth_frame = depth.async_read(timeout_ms=500)
    expected_depth = cv2.applyColorMap(cv2.convertScaleAbs(depth_raw, alpha=0.03), cv2.COLORMAP_JET)

    assert np.array_equal(depth_frame, expected_depth)
    assert fake_realsense_runtime.align_process_calls > 0

    color.disconnect()
    depth.disconnect()


def test_manager_keeps_frames_and_tracks_client_drops(fake_realsense_runtime: FakeRealSenseRuntime) -> None:
    color = d405_module.RealSenseD405ColorCamera(
        RealSenseD405ColorCameraConfig(
            serial_number_or_name="405123",
            width=2,
            height=2,
            fps=30,
            color_mode=ColorMode.BGR,
        )
    )

    color.connect(warmup=False)

    first_frameset = fake_realsense_runtime.push_frame(
        "405123",
        color=np.full((2, 2, 2), 90, dtype=np.uint8),
        color_format="yuyv",
    )
    _wait_for_manager_seq(color, min_seq=1)
    color.async_read(timeout_ms=500)

    assert first_frameset.keep_calls == 1
    assert color._dropped_frame_count == 0

    for value in range(6):
        fake_realsense_runtime.push_frame(
            "405123",
            color=np.full((2, 2, 2), 100 + value, dtype=np.uint8),
            color_format="yuyv",
        )

    _wait_for_manager_seq(color, min_seq=7)
    color.async_read(timeout_ms=500)

    assert color._dropped_frame_count == 2
    assert color._max_seq_gap == 2
    assert color.manager is not None
    assert color.manager.get_status()["buffered_frames"] == 4

    color.disconnect()
