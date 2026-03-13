from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from threading import Condition, Event, Lock, Thread
from typing import Any, Literal

import cv2  # type: ignore  # TODO: add type stubs for OpenCV
import numpy as np  # type: ignore  # TODO: add type stubs for numpy
from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

try:
    import pyrealsense2 as rs  # type: ignore  # TODO: add type stubs for pyrealsense2
except Exception as e:
    rs = None
    logging.info(f"Could not import realsense: {e}")

from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_rs_d435i import (
    RealSenseD435iColorCameraConfig,
    RealSenseD435iDepthCameraConfig,
)

logger = logging.getLogger(__name__)

StreamKind = Literal["color", "depth"]
_SUPPORTED_D435I_NAMES = ("D435I",)


@dataclass(frozen=True)
class _StreamSpec:
    enabled: bool
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    format_name: str | None = None


@dataclass(frozen=True)
class _PipelineSpec:
    color: _StreamSpec
    depth: _StreamSpec

    def get(self, kind: StreamKind) -> _StreamSpec:
        return self.color if kind == "color" else self.depth

    @property
    def shared_fps(self) -> int | None:
        fps_values = {
            stream.fps for stream in (self.color, self.depth) if stream.enabled and stream.fps is not None
        }
        if len(fps_values) > 1:
            return None
        return next(iter(fps_values), None)

    def satisfies(self, required: "_PipelineSpec") -> bool:
        return _stream_satisfies(self.color, required.color) and _stream_satisfies(self.depth, required.depth)


@dataclass(frozen=True)
class _ClientRequest:
    kind: StreamKind
    fps: int | None
    width: int | None
    height: int | None
    color_stream_format: str | None = None


@dataclass(frozen=True)
class _FrameSnapshot:
    color: NDArray[Any] | None = None
    depth: NDArray[Any] | None = None


def _stream_satisfies(current: _StreamSpec, required: _StreamSpec) -> bool:
    if not required.enabled:
        return True
    if not current.enabled:
        return False
    if required.width is not None and current.width != required.width:
        return False
    if required.height is not None and current.height != required.height:
        return False
    if required.fps is not None and current.fps != required.fps:
        return False
    if required.format_name is not None and current.format_name != required.format_name:
        return False
    return True


def _require_realsense() -> Any:
    if rs is None:
        raise ImportError(
            "pyrealsense2 is required for RealSense D435i cameras. Install `lerobot[intelrealsense]`."
        )
    return rs


def _get_camera_info(device: Any, field: Any) -> str | None:
    try:
        value = device.get_info(field)
    except Exception:
        return None
    return str(value)


def _query_realsense_devices() -> list[dict[str, str]]:
    rs_module = _require_realsense()
    devices_info: list[dict[str, str]] = []
    context = rs_module.context()
    for device in context.query_devices():
        devices_info.append(
            {
                "name": _get_camera_info(device, rs_module.camera_info.name) or "",
                "serial_number": _get_camera_info(device, rs_module.camera_info.serial_number) or "",
                "firmware_version": _get_camera_info(device, rs_module.camera_info.firmware_version) or "",
                "usb_type_descriptor": _get_camera_info(device, rs_module.camera_info.usb_type_descriptor) or "",
                "product_line": _get_camera_info(device, rs_module.camera_info.product_line) or "",
                "product_id": _get_camera_info(device, rs_module.camera_info.product_id) or "",
            }
        )
    return devices_info


def _is_supported_d435i_device(device_info: dict[str, str]) -> bool:
    normalized_name = device_info["name"].upper().replace(" ", "")
    return any(token in normalized_name for token in _SUPPORTED_D435I_NAMES)


def _find_supported_device_info(serial_number_or_name: str) -> dict[str, str]:
    devices = _query_realsense_devices()
    matches = [device for device in devices if device["serial_number"] == serial_number_or_name]
    if not matches:
        matches = [device for device in devices if device["name"] == serial_number_or_name]

    if not matches:
        available = [f'{device["name"]} ({device["serial_number"]})' for device in devices]
        raise ValueError(
            f"No supported RealSense camera found for '{serial_number_or_name}'. Available devices: {available}"
        )

    if len(matches) > 1:
        serials = [device["serial_number"] for device in matches]
        raise ValueError(
            f"Multiple RealSense cameras found with name '{serial_number_or_name}'. "
            f"Please use the serial number instead. Found serials: {serials}"
        )

    device_info = matches[0]
    if not _is_supported_d435i_device(device_info):
        raise ValueError(
            f"RealSense D435i shared support only supports D435i devices, but got "
            f"'{device_info['name']}' ({device_info['serial_number']})."
        )

    return device_info


def _resolve_serial_number(serial_number_or_name: str) -> tuple[str, str | None]:
    if serial_number_or_name.isdigit():
        return serial_number_or_name, None

    device_info = _find_supported_device_info(serial_number_or_name)
    return device_info["serial_number"], device_info["name"]


def _get_rs_color_format(rs_module: Any, format_name: str | None) -> Any:
    normalized_format = format_name or "rgb8"
    try:
        return getattr(rs_module.format, normalized_format)
    except AttributeError as e:
        raise ValueError(f"Unsupported RealSense color stream format '{normalized_format}'.") from e


def _decode_color_frame_to_rgb(color_frame: Any) -> NDArray[Any]:
    rs_module = _require_realsense()
    color_profile = color_frame.profile.as_video_stream_profile()
    width = int(color_profile.width())
    height = int(color_profile.height())
    frame_format = color_frame.profile.format()
    color_raw = np.asanyarray(color_frame.get_data())

    if frame_format == rs_module.format.rgb8:
        return color_raw

    if frame_format == rs_module.format.bgr8:
        return cv2.cvtColor(color_raw, cv2.COLOR_BGR2RGB)

    if frame_format == rs_module.format.yuyv:
        color_raw = color_raw.view(np.uint8).reshape((height, width, 2))
        return cv2.cvtColor(color_raw, cv2.COLOR_YUV2RGB_YUYV)

    raise RuntimeError(f"Unsupported D435i color format: {frame_format}")


class RealSenseD435iManager:
    _registry: dict[str, "RealSenseD435iManager"] = {}
    _registry_lock = Lock()

    def __init__(self, serial_number: str, device_name: str | None = None):
        self.serial_number = serial_number
        self.device_name = device_name

        self._lock = Lock()
        self._clients: dict[int, _ClientRequest] = {}

        self.pipeline: Any = None
        self.profile: Any = None
        self.current_spec: _PipelineSpec | None = None

        self.frame_lock = Lock()
        self.frame_condition = Condition(self.frame_lock)
        self.latest_frames: _FrameSnapshot | None = None
        self.frames_seq = 0
        self.last_frames_ts: float | None = None

        self.thread: Thread | None = None
        self.stop_event = Event()

    @classmethod
    def get_or_create(cls, serial_number: str, device_name: str | None = None) -> "RealSenseD435iManager":
        with cls._registry_lock:
            manager = cls._registry.get(serial_number)
            if manager is None:
                manager = cls(serial_number=serial_number, device_name=device_name)
                cls._registry[serial_number] = manager
            elif device_name:
                manager.device_name = device_name
            return manager

    @classmethod
    def drop_if_unused(cls, serial_number: str, manager: "RealSenseD435iManager") -> None:
        with cls._registry_lock:
            if not manager.has_clients():
                current = cls._registry.get(serial_number)
                if current is manager:
                    cls._registry.pop(serial_number, None)

    def has_clients(self) -> bool:
        with self._lock:
            return bool(self._clients)

    def update_device_name(self, device_name: str | None) -> None:
        if device_name:
            with self._lock:
                self.device_name = device_name

    def connect_client(self, client_id: int, request: _ClientRequest) -> _PipelineSpec:
        with self._lock:
            if client_id in self._clients:
                if self.current_spec is None:
                    raise RuntimeError(f"Manager for {self.serial_number} lost its active pipeline state.")
                return self.current_spec

            desired_spec = self._build_desired_spec(extra_request=request)
            if self.current_spec is None or not self.current_spec.satisfies(desired_spec):
                self._restart_pipeline_locked(desired_spec, self.current_spec)

            self._clients[client_id] = request
            if self.current_spec is None:
                raise RuntimeError(f"Manager for {self.serial_number} failed to start a pipeline.")
            return self.current_spec

    def disconnect_client(self, client_id: int) -> None:
        should_drop = False
        with self._lock:
            self._clients.pop(client_id, None)
            if self._clients:
                return
            self._stop_pipeline_locked()
            should_drop = True

        if should_drop:
            self.drop_if_unused(self.serial_number, self)

    def get_latest_frames(self) -> _FrameSnapshot | None:
        with self.frame_lock:
            return self.latest_frames

    def get_latest_frames_and_seq(self) -> tuple[_FrameSnapshot | None, int]:
        with self.frame_lock:
            return self.latest_frames, self.frames_seq

    def wait_for_next_frames(self, last_seq: int, timeout_s: float) -> tuple[_FrameSnapshot | None, int]:
        deadline = time.monotonic() + max(timeout_s, 0.0)
        with self.frame_condition:
            while True:
                if self.latest_frames is not None and self.frames_seq != last_seq:
                    return self.latest_frames, self.frames_seq

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, last_seq

                self.frame_condition.wait(timeout=remaining)

    def get_output_stream_spec(self, kind: StreamKind) -> _StreamSpec:
        with self._lock:
            current_spec = self.current_spec

        if current_spec is None:
            raise RuntimeError(f"Manager for {self.serial_number} does not have an active pipeline.")

        return current_spec.get(kind)

    def get_status(self) -> dict[str, Any]:
        with self.frame_lock:
            last_frames_ts = self.last_frames_ts
            frames_seq = self.frames_seq

        with self._lock:
            current_spec = self.current_spec
            client_count = len(self._clients)
            device_name = self.device_name

        return {
            "serial_number": self.serial_number,
            "device_name": device_name,
            "has_pipeline": self.pipeline is not None,
            "clients": client_count,
            "frames_seq": frames_seq,
            "last_frames_age_s": None if last_frames_ts is None else (time.time() - last_frames_ts),
            "thread_alive": self.thread is not None and self.thread.is_alive(),
            "current_spec": current_spec,
        }

    def _build_desired_spec(self, extra_request: _ClientRequest | None = None) -> _PipelineSpec:
        requests = list(self._clients.values())
        if extra_request is not None:
            requests.append(extra_request)

        explicit_fps = {request.fps for request in requests if request.fps is not None}
        if len(explicit_fps) > 1:
            raise ValueError(
                f"RealSense D435i cameras sharing serial {self.serial_number} must use the same fps. "
                f"Requested fps values: {sorted(explicit_fps)}"
            )

        sticky_fps = self.current_spec.shared_fps if self.current_spec is not None else None
        fps = next(iter(explicit_fps), sticky_fps)

        color_enabled = any(request.kind == "color" for request in requests)
        depth_enabled = any(request.kind == "depth" for request in requests)
        if self.current_spec is not None:
            color_enabled = color_enabled or self.current_spec.color.enabled
            depth_enabled = depth_enabled or self.current_spec.depth.enabled

        color_spec = self._build_stream_spec(kind="color", enabled=color_enabled, requests=requests, fps=fps)
        depth_spec = self._build_stream_spec(kind="depth", enabled=depth_enabled, requests=requests, fps=fps)
        return _PipelineSpec(color=color_spec, depth=depth_spec)

    def _build_stream_spec(
        self, kind: StreamKind, enabled: bool, requests: list[_ClientRequest], fps: int | None
    ) -> _StreamSpec:
        if not enabled:
            return _StreamSpec(enabled=False)

        current_stream = self.current_spec.get(kind) if self.current_spec is not None else _StreamSpec(False)

        widths = {request.width for request in requests if request.kind == kind and request.width is not None}
        heights = {request.height for request in requests if request.kind == kind and request.height is not None}
        if len(widths) > 1 or len(heights) > 1:
            raise ValueError(
                f"RealSense D435i {kind} cameras sharing serial {self.serial_number} must use the same resolution."
            )

        width = next(iter(widths), current_stream.width if current_stream.enabled else None)
        height = next(iter(heights), current_stream.height if current_stream.enabled else None)
        format_name = None
        if kind == "color":
            formats = {
                request.color_stream_format
                for request in requests
                if request.kind == "color" and request.color_stream_format is not None
            }
            if len(formats) > 1:
                raise ValueError(
                    f"RealSense D435i color cameras sharing serial {self.serial_number} must use the same "
                    "color stream format."
                )
            format_name = next(iter(formats), current_stream.format_name if current_stream.enabled else "rgb8")

        return _StreamSpec(enabled=True, width=width, height=height, fps=fps, format_name=format_name)

    def _restart_pipeline_locked(
        self, desired_spec: _PipelineSpec, previous_spec: _PipelineSpec | None
    ) -> None:
        if previous_spec is not None:
            self._stop_pipeline_locked()
        self._start_pipeline_locked(desired_spec)

    def _start_pipeline_locked(self, desired_spec: _PipelineSpec) -> None:
        rs_module = _require_realsense()

        pipeline = rs_module.pipeline()
        rs_config = self._create_rs_config(desired_spec)
        profile = pipeline.start(rs_config)
        actual_spec = self._extract_actual_spec(profile, desired_spec)

        self.pipeline = pipeline
        self.profile = profile
        self.current_spec = actual_spec

        with self.frame_condition:
            self.latest_frames = None
            self.frames_seq = 0
            self.last_frames_ts = None

        self.stop_event = Event()
        self.thread = Thread(
            target=self._read_loop,
            name=f"RealSenseD435iManager[{self.serial_number}]_read_loop",
            daemon=True,
        )
        self.thread.start()

    def _create_rs_config(self, desired_spec: _PipelineSpec) -> Any:
        rs_module = _require_realsense()
        rs_config = rs_module.config()
        rs_config.enable_device(self.serial_number)
        self._enable_stream(rs_config, kind="color", spec=desired_spec.color)
        self._enable_stream(rs_config, kind="depth", spec=desired_spec.depth)
        return rs_config

    def _enable_stream(self, rs_config: Any, kind: StreamKind, spec: _StreamSpec) -> None:
        if not spec.enabled:
            return

        rs_module = _require_realsense()
        stream = rs_module.stream.color if kind == "color" else rs_module.stream.depth
        fmt = _get_rs_color_format(rs_module, spec.format_name) if kind == "color" else rs_module.format.z16

        if spec.width is None or spec.height is None:
            fps = spec.fps if spec.fps is not None else 0
            rs_config.enable_stream(stream, 0, 0, fmt, fps)
            return

        fps = spec.fps if spec.fps is not None else 0
        rs_config.enable_stream(stream, spec.width, spec.height, fmt, fps)

    def _extract_actual_spec(self, profile: Any, desired_spec: _PipelineSpec) -> _PipelineSpec:
        rs_module = _require_realsense()
        color_spec = _StreamSpec(enabled=False)
        depth_spec = _StreamSpec(enabled=False)

        if desired_spec.color.enabled:
            color_profile = profile.get_stream(rs_module.stream.color).as_video_stream_profile()
            color_spec = _StreamSpec(
                enabled=True,
                width=int(color_profile.width()),
                height=int(color_profile.height()),
                fps=int(color_profile.fps()),
                format_name=desired_spec.color.format_name,
            )

        if desired_spec.depth.enabled:
            depth_profile = profile.get_stream(rs_module.stream.depth).as_video_stream_profile()
            depth_spec = _StreamSpec(
                enabled=True,
                width=int(depth_profile.width()),
                height=int(depth_profile.height()),
                fps=int(depth_profile.fps()),
            )

        return _PipelineSpec(color=color_spec, depth=depth_spec)

    def _read_loop(self) -> None:
        pipeline = self.pipeline
        stop_event = self.stop_event
        if pipeline is None:
            return

        while not stop_event.is_set():
            try:
                ret, frames = pipeline.try_wait_for_frames(timeout_ms=100)
                if not ret or frames is None:
                    continue

                snapshot = self._build_frame_snapshot(frames)
                with self.frame_condition:
                    self.latest_frames = snapshot
                    self.frames_seq += 1
                    self.last_frames_ts = time.time()
                    self.frame_condition.notify_all()
            except Exception as e:
                if stop_event.is_set():
                    break
                logger.warning("RealSense D435i read loop error for %s: %s", self.serial_number, e)
                time.sleep(0.01)

    def _build_frame_snapshot(self, frames: Any) -> _FrameSnapshot:
        color_image = None
        depth_map = None

        color_frame = frames.get_color_frame()
        if color_frame is not None:
            color_image = np.ascontiguousarray(_decode_color_frame_to_rgb(color_frame))

        depth_frame = frames.get_depth_frame()
        if depth_frame is not None:
            depth_map = np.ascontiguousarray(np.asanyarray(depth_frame.get_data()).copy())

        return _FrameSnapshot(color=color_image, depth=depth_map)

    def _stop_pipeline_locked(self) -> None:
        thread = self.thread
        stop_event = self.stop_event
        pipeline = self.pipeline

        if stop_event is not None:
            stop_event.set()

        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass

        self.thread = None
        self.stop_event = Event()
        self.pipeline = None
        self.profile = None
        self.current_spec = None

        with self.frame_condition:
            self.latest_frames = None
            self.frames_seq = 0
            self.last_frames_ts = None
            self.frame_condition.notify_all()


class RealSenseD435iBaseCamera(Camera):
    KIND: StreamKind

    def __init__(self, config: RealSenseD435iColorCameraConfig | RealSenseD435iDepthCameraConfig):
        super().__init__(config)
        self.config = config
        self.serial_number, self.device_name = _resolve_serial_number(config.serial_number_or_name)
        self.manager: RealSenseD435iManager | None = None
        self.connected = False

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.new_frame_event = Event()
        self._last_frame_ts: float | None = None
        self._last_pipeline_seq = -1

        self.rotation = get_cv2_rotation(config.rotation)
        self.capture_width = self.width
        self.capture_height = self.height

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number})"

    @property
    def is_connected(self) -> bool:
        return self.connected

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return [
            {
                "name": device["name"],
                "type": "RealSense",
                "id": device["serial_number"],
                "firmware_version": device["firmware_version"],
                "usb_type_descriptor": device["usb_type_descriptor"],
                "product_line": device["product_line"],
                "product_id": device["product_id"],
            }
            for device in _query_realsense_devices()
            if _is_supported_d435i_device(device)
        ]

    def connect(self, warmup: bool = True) -> None:
        if self.connected:
            return

        device_info = _find_supported_device_info(self.serial_number)
        self.device_name = device_info["name"]
        self.manager = RealSenseD435iManager.get_or_create(self.serial_number, self.device_name)
        self.manager.update_device_name(self.device_name)

        try:
            spec = self.manager.connect_client(
                client_id=id(self),
                request=_ClientRequest(
                    kind=self.KIND,
                    fps=self.config.fps,
                    width=self.config.width,
                    height=self.config.height,
                    color_stream_format=getattr(self.config, "color_stream_format", None),
                ),
            )
        except Exception:
            RealSenseD435iManager.drop_if_unused(self.serial_number, self.manager)
            self.manager = None
            raise

        try:
            self._apply_stream_settings(spec.get(self.KIND))
        except Exception:
            if self.manager is not None:
                self.manager.disconnect_client(id(self))
            self.manager = None
            raise

        self.connected = True
        self.latest_frame = None
        self._last_frame_ts = None
        self._last_pipeline_seq = -1
        self.new_frame_event.clear()

        if warmup:
            warmup_s = float(getattr(self.config, "warmup_s", 0) or 0)
            if warmup_s > 0:
                try:
                    self.async_read(timeout_ms=warmup_s * 1000.0)
                except TimeoutError:
                    logger.warning("Warmup timed out for %s.", self)

    def _apply_stream_settings(self, stream_spec: _StreamSpec) -> None:
        if not stream_spec.enabled or stream_spec.width is None or stream_spec.height is None:
            raise RuntimeError(f"{self} failed to resolve active stream settings for {self.KIND}.")

        self.fps = stream_spec.fps
        self.capture_width = stream_spec.width
        self.capture_height = stream_spec.height

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            self.width = stream_spec.height
            self.height = stream_spec.width
        else:
            self.width = stream_spec.width
            self.height = stream_spec.height

    def _sync_stream_settings(self) -> None:
        if not self.connected or self.manager is None:
            return

        current_stream_spec = self.manager.get_output_stream_spec(self.KIND)
        if (
            current_stream_spec.enabled
            and (
                current_stream_spec.width != self.capture_width
                or current_stream_spec.height != self.capture_height
                or current_stream_spec.fps != self.fps
            )
        ):
            self._apply_stream_settings(current_stream_spec)

    def _read_from_frames(self, frames: _FrameSnapshot, color_mode: ColorMode | None = None) -> NDArray[Any] | None:
        raise NotImplementedError

    def _process_frame_from_frames(self, frames: _FrameSnapshot) -> NDArray[Any] | None:
        self._sync_stream_settings()
        return self._read_from_frames(frames)

    def _read_loop(self) -> None:
        if self.manager is None:
            return

        while not (self.stop_event and self.stop_event.is_set()):
            try:
                frames, seq = self.manager.wait_for_next_frames(self._last_pipeline_seq, timeout_s=0.5)
                if frames is None or seq == self._last_pipeline_seq:
                    continue

                self._last_pipeline_seq = seq
                frame = self._process_frame_from_frames(frames)
                if frame is None:
                    continue

                with self.frame_lock:
                    self.latest_frame = frame
                    self._last_frame_ts = time.time()
                    self.new_frame_event.set()
            except DeviceNotConnectedError:
                break
            except Exception as e:
                if self.stop_event and self.stop_event.is_set():
                    break
                logger.warning("RealSense D435i camera read loop error for %s: %s", self, e)
                time.sleep(0.01)

    def _start_read_thread(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name=f"{self.__class__.__name__}_read_loop", daemon=True)
        self.thread.start()

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 5000) -> NDArray[Any]:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.manager is None:
            raise RuntimeError(f"{self} does not have an active RealSense manager.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"Timed out waiting for frame from camera {self}.")

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Event set but no frame available for {self}.")

        return frame

    def disconnect(self) -> None:
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} already disconnected.")

        if self.thread is not None:
            self._stop_read_thread()

        if self.manager is not None:
            self.manager.disconnect_client(id(self))

        self.connected = False
        self.latest_frame = None
        self._last_pipeline_seq = -1
        self.new_frame_event.clear()
        self.manager = None

    def _require_frames(self) -> _FrameSnapshot | None:
        if not self.connected or self.manager is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._sync_stream_settings()
        return self.manager.get_latest_frames()


class RealSenseD435iColorCamera(RealSenseD435iBaseCamera):
    KIND = "color"

    def __init__(self, config: RealSenseD435iColorCameraConfig):
        super().__init__(config)
        self.color_mode = config.color_mode

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any] | None:
        frames = self._require_frames()
        if frames is None:
            return None
        return self._read_from_frames(frames, color_mode=color_mode)

    def _read_from_frames(self, frames: _FrameSnapshot, color_mode: ColorMode | None = None) -> NDArray[Any] | None:
        if frames.color is None:
            return None
        return self._postprocess_color_image(frames.color, color_mode=color_mode)

    def _postprocess_color_image(
        self, image: NDArray[Any], color_mode: ColorMode | None = None
    ) -> NDArray[Any]:
        if color_mode and color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid requested color mode '{color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(f"{self} frame is expected to be HxWx3, but got shape={image.shape}.")

        height, width, _ = image.shape
        if height != self.capture_height or width != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={width} or height={height} do not match configured width="
                f"{self.capture_width} or height={self.capture_height}."
            )

        requested_color_mode = color_mode or self.color_mode
        processed_image = image
        if requested_color_mode == ColorMode.BGR:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image


class RealSenseD435iDepthCamera(RealSenseD435iBaseCamera):
    KIND = "depth"

    def __init__(self, config: RealSenseD435iDepthCameraConfig):
        super().__init__(config)
        self.max_depth_m = float(config.max_depth_m)
        self.max_depth_raw = int(round(self.max_depth_m * 1000.0))

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any] | None:
        del color_mode
        frames = self._require_frames()
        if frames is None:
            return None
        return self._read_from_frames(frames)

    def _read_from_frames(self, frames: _FrameSnapshot, color_mode: ColorMode | None = None) -> NDArray[Any] | None:
        del color_mode
        if frames.depth is None:
            return None
        return self._postprocess_depth_image(frames.depth)

    def _postprocess_depth_image(self, depth_map: NDArray[Any]) -> NDArray[Any]:
        if depth_map.ndim != 2:
            raise RuntimeError(f"{self} depth frame is expected to be HxW, but got shape={depth_map.shape}.")

        height, width = depth_map.shape
        if height != self.capture_height or width != self.capture_width:
            raise RuntimeError(
                f"{self} depth frame width={width} or height={height} do not match configured width="
                f"{self.capture_width} or height={self.capture_height}."
            )

        depth_u16 = depth_map.astype(np.uint16, copy=False)
        valid_mask = (depth_u16 > 0) & (depth_u16 <= self.max_depth_raw)
        depth_8u = np.zeros(depth_u16.shape, dtype=np.uint8)
        if np.any(valid_mask):
            valid_depth = depth_u16[valid_mask].astype(np.float32, copy=False)
            depth_min = float(valid_depth.min())
            depth_max = float(valid_depth.max())
            if depth_max > depth_min:
                normalized_depth = (valid_depth - depth_min) * 255.0 / (depth_max - depth_min)
                depth_8u[valid_mask] = normalized_depth.astype(np.uint8)
        depth_rgb = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
        depth_rgb[~valid_mask] = 0

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            depth_rgb = cv2.rotate(depth_rgb, self.rotation)

        return depth_rgb
