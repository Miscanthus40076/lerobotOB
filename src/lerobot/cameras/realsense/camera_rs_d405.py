from __future__ import annotations

"""
RealSense D405 的共享式 color/depth 接入实现。

文件结构分为四层：

1. 数据结构与辅助函数
   - `_StreamSpec` / `_PipelineSpec` / `_ClientRequest`
   - 设备发现、serial 解析、颜色格式解码

2. `RealSenseD405Manager`
   - 一台物理 D405 对应一个 manager
   - manager 独占一个 RealSense pipeline 和一条后台采集线程
   - 负责合并多个逻辑相机实例的需求并统一协商 pipeline 参数

3. `RealSenseD405BaseCamera`
   - color/depth 逻辑相机的公共行为
   - connect/disconnect、async_read、掉帧统计、视图尺寸同步

4. `RealSenseD405ColorCamera` / `RealSenseD405DepthCamera`
   - 只负责从 manager 提供的 snapshot 中提取本视图并做最终后处理

这套实现的核心目标是：
  - 同一个物理 D405 只连接一次
  - color/depth 作为两个逻辑相机暴露给上层
  - 所有 SDK 读帧都集中在 manager 线程里完成，避免多线程直接抢设备
"""

from collections import deque
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
from .configuration_rs_d405 import RealSenseD405ColorCameraConfig, RealSenseD405DepthCameraConfig

logger = logging.getLogger(__name__)

StreamKind = Literal["color", "depth"]
_SUPPORTED_D405_NAMES = ("D405",)
_FRAME_HISTORY_SIZE = 4


@dataclass(frozen=True)
class _StreamSpec:
    """单一路 stream 的协商结果。

    对于 color/depth 各自记录：
      - 是否启用
      - 宽高
      - fps
      - color stream 的像素格式
    """

    enabled: bool
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    format_name: str | None = None


@dataclass(frozen=True)
class _PipelineSpec:
    """一次 pipeline 启动所需的完整 color/depth 规格。"""

    color: _StreamSpec
    depth: _StreamSpec

    def get(self, kind: StreamKind) -> _StreamSpec:
        return self.color if kind == "color" else self.depth

    @property
    def shared_fps(self) -> int | None:
        """返回 color/depth 的共享 fps。

        这里的语义是：
          - 如果启用的流里只有一个 fps 值，返回它
          - 如果出现多个不同 fps，返回 None，表示当前并不存在共享 fps
        """
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
    """单个逻辑相机实例向 manager 声明的需求。"""

    kind: StreamKind
    fps: int | None
    width: int | None
    height: int | None
    color_stream_format: str | None = None


@dataclass(frozen=True)
class _FrameSnapshot:
    """manager 线程产出的稳定帧快照。

    这里缓存的是 numpy 数据，而不是原始 `pyrealsense2 frameset`。
    这样做的原因是避免把底层 SDK 帧对象跨线程长期共享，降低生命周期和竞态问题。
    """

    color: NDArray[Any] | None = None
    depth: NDArray[Any] | None = None


def _stream_satisfies(current: _StreamSpec, required: _StreamSpec) -> bool:
    """判断当前已启动的 stream 是否满足新的要求。"""
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
    """确保 `pyrealsense2` 可用，否则给出清晰的安装提示。"""
    if rs is None:
        raise ImportError(
            "pyrealsense2 is required for RealSense D405 cameras. Install `lerobot[realsense]`."
        )
    return rs


def _get_camera_info(device: Any, field: Any) -> str | None:
    """安全读取 RealSense 设备元数据字段。"""
    try:
        value = device.get_info(field)
    except Exception:
        return None
    return str(value)


def _query_realsense_devices() -> list[dict[str, str]]:
    """枚举当前机器上的 RealSense 设备，并整理成可展示/可匹配的字典。"""
    rs_module = _require_realsense()
    devices_info: list[dict[str, str]] = []
    context = rs_module.context()
    for device in context.query_devices():
        info = {
            "name": _get_camera_info(device, rs_module.camera_info.name) or "",
            "serial_number": _get_camera_info(device, rs_module.camera_info.serial_number) or "",
            "firmware_version": _get_camera_info(device, rs_module.camera_info.firmware_version) or "",
            "usb_type_descriptor": _get_camera_info(device, rs_module.camera_info.usb_type_descriptor) or "",
            "product_line": _get_camera_info(device, rs_module.camera_info.product_line) or "",
            "product_id": _get_camera_info(device, rs_module.camera_info.product_id) or "",
        }
        devices_info.append(info)
    return devices_info


def _is_supported_d405_device(device_info: dict[str, str]) -> bool:
    """过滤出 D405 设备。"""
    normalized_name = device_info["name"].upper().replace(" ", "")
    return any(token in normalized_name for token in _SUPPORTED_D405_NAMES)


def _find_supported_device_info(serial_number_or_name: str) -> dict[str, str]:
    """按 serial 或设备名解析目标 D405，并做型号校验。"""
    devices = _query_realsense_devices()
    matches = [device for device in devices if device["serial_number"] == serial_number_or_name]
    if matches:
        device_info = matches[0]
    else:
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

    if not _is_supported_d405_device(device_info):
        raise ValueError(
            f"RealSense D405 shared support only supports D405 devices, but got "
            f"'{device_info['name']}' ({device_info['serial_number']})."
        )

    return device_info


def _resolve_serial_number(serial_number_or_name: str) -> tuple[str, str | None]:
    """把用户输入的 serial / device name 统一解析成 serial。"""
    if serial_number_or_name.isdigit():
        return serial_number_or_name, None

    device_info = _find_supported_device_info(serial_number_or_name)
    return device_info["serial_number"], device_info["name"]


def _decode_color_frame(color_frame: Any) -> NDArray[Any]:
    """把 D405 彩色帧统一解码成 BGR numpy 图像。"""
    rs_module = _require_realsense()
    color_profile = color_frame.profile.as_video_stream_profile()
    width = int(color_profile.width())
    height = int(color_profile.height())
    frame_format = color_frame.profile.format()
    color_raw = np.asanyarray(color_frame.get_data())

    if frame_format == rs_module.format.yuyv:
        color_raw = color_raw.view(np.uint8).reshape((height, width, 2))
        return cv2.cvtColor(color_raw, cv2.COLOR_YUV2BGR_YUYV)

    if frame_format == rs_module.format.bgr8:
        return color_raw

    if frame_format == rs_module.format.rgb8:
        return cv2.cvtColor(color_raw, cv2.COLOR_RGB2BGR)

    raise RuntimeError(f"Unsupported D405 color format: {frame_format}")


def _get_rs_color_format(rs_module: Any, format_name: str | None) -> Any:
    """把配置里的格式名字映射到 `pyrealsense2.format.*` 常量。"""
    normalized_format = format_name or "rgb8"
    try:
        return getattr(rs_module.format, normalized_format)
    except AttributeError as e:
        raise ValueError(f"Unsupported RealSense color stream format '{normalized_format}'.") from e


class RealSenseD405Manager:
    """一台物理 D405 的共享管理器。

    它是整个 split-camera 设计的核心：
      - 同一 serial 只会有一个 manager
      - 统一维护 pipeline/profile/align
      - 后台线程集中采集并写入 frame history
      - 多个逻辑相机实例通过它读取同一设备的 color/depth 数据
    """

    _registry: dict[str, "RealSenseD405Manager"] = {}
    _registry_lock = Lock()

    def __init__(
        self, serial_number: str, device_name: str | None = None, usb_type_descriptor: str | None = None
    ):
        self.serial_number = serial_number
        self.device_name = device_name
        self.usb_type_descriptor = usb_type_descriptor

        self._lock = Lock()
        self._clients: dict[int, _ClientRequest] = {}

        self.pipeline: Any = None
        self.profile: Any = None
        self.current_spec: _PipelineSpec | None = None
        self.align_processor: Any = None

        self.frame_lock = Lock()
        self.frame_condition = Condition(self.frame_lock)
        self.frame_history = deque(maxlen=_FRAME_HISTORY_SIZE)
        self.frames_seq = 0
        self.last_frames_ts: float | None = None

        self.total_wait_failures = 0
        self.total_wait_exceptions = 0
        self.last_wait_exception: str | None = None
        self._last_status_log_ts = 0.0
        self._last_empty_poll_log_ts = 0.0

        self.thread: Thread | None = None
        self.stop_event = Event()

    @classmethod
    def get_or_create(
        cls,
        serial_number: str,
        device_name: str | None = None,
        usb_type_descriptor: str | None = None,
    ) -> "RealSenseD405Manager":
        """按 serial 获取或创建共享 manager。"""
        with cls._registry_lock:
            manager = cls._registry.get(serial_number)
            if manager is None:
                manager = cls(
                    serial_number=serial_number,
                    device_name=device_name,
                    usb_type_descriptor=usb_type_descriptor,
                )
                cls._registry[serial_number] = manager
            else:
                if device_name:
                    manager.device_name = device_name
                if usb_type_descriptor:
                    manager.usb_type_descriptor = usb_type_descriptor
            return manager

    @classmethod
    def drop_if_unused(cls, serial_number: str, manager: "RealSenseD405Manager") -> None:
        """在没有逻辑客户端后，把 manager 从全局 registry 中移除。"""
        with cls._registry_lock:
            if not manager.has_clients():
                current = cls._registry.get(serial_number)
                if current is manager:
                    cls._registry.pop(serial_number, None)

    def has_clients(self) -> bool:
        """判断当前 manager 是否仍被逻辑相机使用。"""
        with self._lock:
            return bool(self._clients)

    def update_device_info(self, device_name: str | None, usb_type_descriptor: str | None) -> None:
        """更新发现阶段拿到的设备元信息。"""
        with self._lock:
            if device_name:
                self.device_name = device_name
            if usb_type_descriptor:
                self.usb_type_descriptor = usb_type_descriptor

    def connect_client(self, client_id: int, request: _ClientRequest) -> _PipelineSpec:
        """注册一个逻辑客户端，并在必要时启动/重启 pipeline。"""
        with self._lock:
            if client_id in self._clients:
                if self.current_spec is None:
                    raise RuntimeError(f"Manager for {self.serial_number} lost its active pipeline state.")
                return self.current_spec

            desired_spec = self._build_desired_spec(extra_request=request)
            if self.current_spec is None or not self.current_spec.satisfies(desired_spec):
                previous_spec = self.current_spec
                self._restart_pipeline_locked(desired_spec, previous_spec)

            self._clients[client_id] = request
            if self.current_spec is None:
                raise RuntimeError(f"Manager for {self.serial_number} failed to start a pipeline.")
            return self.current_spec

    def disconnect_client(self, client_id: int) -> None:
        """注销逻辑客户端；最后一个客户端断开时真正停止 pipeline。"""
        should_drop = False
        with self._lock:
            self._clients.pop(client_id, None)
            if self._clients:
                return
            self._stop_pipeline_locked()
            should_drop = True

        if should_drop:
            self.drop_if_unused(self.serial_number, self)

    def get_latest_frames(self) -> Any:
        """返回最近一帧 snapshot。"""
        with self.frame_lock:
            if not self.frame_history:
                return None
            return self.frame_history[-1][1]

    def wait_for_next_frames(self, last_seq: int, timeout_s: float) -> tuple[Any, int]:
        """等待比 `last_seq` 更新的一帧。

        这里依赖小型 ring buffer，而不是单槽 `latest_frames`，这样在消费端稍慢时，
        仍有机会拿到最近几帧中的下一帧，而不是永远只看到“当前最新值”。
        """
        deadline = time.monotonic() + max(timeout_s, 0.0)
        with self.frame_condition:
            while True:
                for seq, frames in self.frame_history:
                    if seq > last_seq:
                        return frames, seq

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, last_seq

                self.frame_condition.wait(timeout=remaining)

    def get_output_stream_spec(self, kind: StreamKind) -> _StreamSpec:
        """返回某一路逻辑相机实际应该看到的输出规格。"""
        with self._lock:
            current_spec = self.current_spec
            align_enabled = self.align_processor is not None

        if current_spec is None:
            raise RuntimeError(f"Manager for {self.serial_number} does not have an active pipeline.")

        if kind == "depth" and align_enabled and current_spec.color.enabled:
            return current_spec.color

        return current_spec.get(kind)

    def get_status(self) -> dict[str, Any]:
        """导出调试状态，便于 timeout / 掉帧日志定位问题。"""
        with self.frame_lock:
            last_frames_ts = self.last_frames_ts
            frames_seq = self.frames_seq
            buffered_frames = len(self.frame_history)
            oldest_buffer_seq = self.frame_history[0][0] if self.frame_history else None

        with self._lock:
            current_spec = self.current_spec
            client_count = len(self._clients)
            device_name = self.device_name
            usb_type_descriptor = self.usb_type_descriptor
            align_enabled = self.align_processor is not None

        return {
            "serial_number": self.serial_number,
            "device_name": device_name,
            "usb_type_descriptor": usb_type_descriptor,
            "has_pipeline": self.pipeline is not None,
            "clients": client_count,
            "frames_seq": frames_seq,
            "last_frames_age_s": None if last_frames_ts is None else (time.time() - last_frames_ts),
            "thread_alive": self.thread is not None and self.thread.is_alive(),
            "current_spec": current_spec,
            "align_enabled": align_enabled,
            "buffered_frames": buffered_frames,
            "oldest_buffer_seq": oldest_buffer_seq,
            "total_wait_failures": self.total_wait_failures,
            "total_wait_exceptions": self.total_wait_exceptions,
            "last_wait_exception": self.last_wait_exception,
        }

    def _record_wait_failure(self) -> None:
        self.total_wait_failures += 1
        now = time.time()
        if now - self._last_status_log_ts > 2.0:
            self._last_status_log_ts = now
            logger.warning(
                "Timed out waiting for RealSense D405 frames for %s "
                "(failures=%s, usb=%s).",
                self.serial_number,
                self.total_wait_failures,
                self.usb_type_descriptor,
            )

    def _record_empty_poll(self) -> None:
        self.total_wait_failures += 1
        now = time.time()

        # `poll_for_frames()` is non-blocking: an empty poll usually means "next frame
        # is not ready yet", not a real timeout. Only surface a warning if the pipeline
        # has produced nothing for an unusually long interval.
        with self.frame_lock:
            last_frames_ts = self.last_frames_ts

        if last_frames_ts is not None and (now - last_frames_ts) < 1.0:
            return

        if now - self._last_empty_poll_log_ts > 5.0:
            self._last_empty_poll_log_ts = now
            logger.warning(
                "RealSense D405 has not produced frames recently for %s "
                "(empty_polls=%s, usb=%s, last_frame_age_s=%s).",
                self.serial_number,
                self.total_wait_failures,
                self.usb_type_descriptor,
                None if last_frames_ts is None else (now - last_frames_ts),
            )

    def _record_wait_exception(self, e: Exception) -> None:
        self.total_wait_exceptions += 1
        self.last_wait_exception = repr(e)
        now = time.time()
        if now - self._last_status_log_ts > 2.0:
            self._last_status_log_ts = now
            logger.warning(
                "RealSense D405 read loop exception for %s (exceptions=%s, usb=%s): %s",
                self.serial_number,
                self.total_wait_exceptions,
                self.usb_type_descriptor,
                e,
            )

    def _build_desired_spec(self, extra_request: _ClientRequest | None = None) -> _PipelineSpec:
        """把所有逻辑客户端的需求合并成一次 pipeline 规格。"""
        requests = list(self._clients.values())
        if extra_request is not None:
            requests.append(extra_request)

        explicit_fps = {request.fps for request in requests if request.fps is not None}
        if len(explicit_fps) > 1:
            raise ValueError(
                f"RealSense D405 cameras sharing serial {self.serial_number} must use the same fps. "
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

        if (
            color_spec.enabled
            and depth_spec.enabled
            and color_spec.width is not None
            and color_spec.height is not None
            and depth_spec.width is not None
            and depth_spec.height is not None
            and (color_spec.width != depth_spec.width or color_spec.height != depth_spec.height)
        ):
            raise ValueError(
                f"RealSense D405 cameras sharing serial {self.serial_number} must use the same color/depth "
                f"resolution when both streams are enabled because depth is aligned to color."
            )

        return _PipelineSpec(color=color_spec, depth=depth_spec)

    def _build_stream_spec(
        self, kind: StreamKind, enabled: bool, requests: list[_ClientRequest], fps: int | None
    ) -> _StreamSpec:
        """构造单路 stream 规格，并在这里做所有共享约束校验。"""
        if not enabled:
            return _StreamSpec(enabled=False)

        current_stream = self.current_spec.get(kind) if self.current_spec is not None else _StreamSpec(False)

        widths = {request.width for request in requests if request.kind == kind and request.width is not None}
        heights = {request.height for request in requests if request.kind == kind and request.height is not None}
        if len(widths) > 1 or len(heights) > 1:
            raise ValueError(
                f"RealSense D405 {kind} cameras sharing serial {self.serial_number} must use the same resolution."
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
                    f"RealSense D405 color cameras sharing serial {self.serial_number} must use the same "
                    "color stream format."
                )
            format_name = next(iter(formats), current_stream.format_name if current_stream.enabled else "rgb8")

        return _StreamSpec(enabled=True, width=width, height=height, fps=fps, format_name=format_name)

    def _restart_pipeline_locked(
        self, desired_spec: _PipelineSpec, previous_spec: _PipelineSpec | None
    ) -> None:
        """在持锁状态下重启 pipeline。

        若新规格启动失败，并且存在旧规格，则尝试回滚恢复。
        """
        if previous_spec is not None:
            self._preflight_config_locked(desired_spec)

        if previous_spec is not None:
            self._stop_pipeline_locked()

        try:
            self._start_pipeline_locked(desired_spec)
        except Exception:
            if previous_spec is not None:
                try:
                    self._start_pipeline_locked(previous_spec)
                except Exception as restore_error:
                    logger.error(
                        "Failed to restore previous RealSense D405 pipeline for %s: %s",
                        self.serial_number,
                        restore_error,
                    )
            raise

    def _preflight_config_locked(self, desired_spec: _PipelineSpec) -> None:
        """在真正 stop/start 前，先用 `resolve()` 预检查配置是否合法。"""
        rs_module = _require_realsense()
        try:
            probe_pipeline = rs_module.pipeline()
            probe_config = self._create_rs_config(desired_spec)
            if hasattr(rs_module, "pipeline_wrapper") and hasattr(probe_config, "resolve"):
                probe_config.resolve(rs_module.pipeline_wrapper(probe_pipeline))
        except Exception as e:
            raise RuntimeError(
                f"Failed to resolve RealSense D405 pipeline config for {self.serial_number}: {e}"
            ) from e

    def _start_pipeline_locked(self, desired_spec: _PipelineSpec) -> None:
        """在持锁状态下启动 RealSense pipeline 和后台采集线程。"""
        rs_module = _require_realsense()
        pipeline = rs_module.pipeline()
        rs_config = self._create_rs_config(desired_spec)

        try:
            profile = pipeline.start(rs_config)
        except Exception as e:
            message = str(e)
            if "resource busy" in message.lower():
                raise RuntimeError(
                    "D405 is busy. Close other processes that may be using the camera "
                    "(for example another viewer, RealSense Viewer, ffmpeg, or a previous script)."
                ) from e
            raise ConnectionError(
                f"Failed to open RealSense D405 camera {self.serial_number} with requested streams. "
                f"Desired spec: {desired_spec}"
            ) from e

        try:
            actual_spec = self._extract_actual_spec(profile, desired_spec)
            self._validate_started_pipeline(desired_spec, actual_spec)
        except Exception:
            try:
                pipeline.stop()
            except Exception:
                logger.warning("Error cleaning up failed RealSense D405 pipeline start for %s.", self.serial_number)
            raise

        self.pipeline = pipeline
        self.profile = profile
        self.current_spec = actual_spec
        self.align_processor = None
        if desired_spec.color.enabled and desired_spec.depth.enabled:
            self.align_processor = rs_module.align(rs_module.stream.color)

        self.total_wait_failures = 0
        self.total_wait_exceptions = 0
        self.last_wait_exception = None
        self._last_status_log_ts = 0.0

        with self.frame_condition:
            self.frame_history.clear()
            self.frames_seq = 0
            self.last_frames_ts = None

        usb_type = self.usb_type_descriptor or ""
        if usb_type.startswith("2"):
            logger.warning(
                "D405 %s is currently connected as USB %s. Color+depth streaming may stall. "
                "A USB 3.x port/cable is strongly recommended.",
                self.serial_number,
                usb_type,
            )

        self.stop_event = Event()
        self.thread = Thread(
            target=self._read_loop,
            name=f"RealSenseD405Manager[{self.serial_number}]_read_loop",
            daemon=True,
        )
        self.thread.start()

    def _create_rs_config(self, desired_spec: _PipelineSpec) -> Any:
        """把内部 `_PipelineSpec` 转成 RealSense SDK `config`。"""
        rs_module = _require_realsense()
        rs_config = rs_module.config()
        rs_config.enable_device(self.serial_number)
        self._enable_stream(rs_config, kind="depth", spec=desired_spec.depth)
        self._enable_stream(rs_config, kind="color", spec=desired_spec.color)
        return rs_config

    def _enable_stream(self, rs_config: Any, kind: StreamKind, spec: _StreamSpec) -> None:
        """把某一路 stream 规格写入 SDK config。"""
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
        """从启动后的 profile 里提取实际生效的规格。"""
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

    def _validate_started_pipeline(self, desired_spec: _PipelineSpec, actual_spec: _PipelineSpec) -> None:
        """确认 SDK 真正启动出来的规格没有悄悄偏离请求。"""
        for kind in ("color", "depth"):
            desired_stream = desired_spec.get(kind)
            actual_stream = actual_spec.get(kind)
            if desired_stream.enabled and not actual_stream.enabled:
                raise RuntimeError(
                    f"RealSense D405 camera {self.serial_number} did not start the requested {kind} stream."
                )

            if desired_stream.width is not None and actual_stream.width != desired_stream.width:
                raise RuntimeError(
                    f"RealSense D405 camera {self.serial_number} started {kind} width "
                    f"{actual_stream.width}, expected {desired_stream.width}."
                )

            if desired_stream.height is not None and actual_stream.height != desired_stream.height:
                raise RuntimeError(
                    f"RealSense D405 camera {self.serial_number} started {kind} height "
                    f"{actual_stream.height}, expected {desired_stream.height}."
                )

            if desired_stream.fps is not None and actual_stream.fps != desired_stream.fps:
                raise RuntimeError(
                    f"RealSense D405 camera {self.serial_number} started {kind} fps "
                    f"{actual_stream.fps}, expected {desired_stream.fps}."
                )

        if actual_spec.color.enabled and actual_spec.depth.enabled and actual_spec.color.fps != actual_spec.depth.fps:
            raise RuntimeError(
                f"RealSense D405 camera {self.serial_number} started color fps "
                f"{actual_spec.color.fps} and depth fps {actual_spec.depth.fps}, but they must match."
            )

    def _read_loop(self) -> None:
        """manager 唯一的后台采集线程。

        这条线程是整个文件里唯一直接触碰 RealSense pipeline 的读帧路径。
        它负责：
          - 从 SDK 拉取新 frames
          - 必要时做 `align(color)`
          - 立刻解包成稳定的 numpy snapshot
          - 写入 ring buffer 并唤醒等待中的逻辑相机
        """
        pipeline = self.pipeline
        stop_event = self.stop_event
        align_processor = self.align_processor
        if pipeline is None:
            return

        use_poll_for_frames = hasattr(pipeline, "poll_for_frames")
        while not stop_event.is_set():
            try:
                if use_poll_for_frames:
                    frames = pipeline.poll_for_frames()
                    ret = bool(frames)
                    if not ret:
                        self._record_empty_poll()
                        stop_event.wait(0.002)
                        continue
                else:
                    ret, frames = pipeline.try_wait_for_frames(timeout_ms=100)

                if not ret or frames is None:
                    self._record_wait_failure()
                    continue

                if align_processor is not None:
                    frames = align_processor.process(frames)

                snapshot = self._build_frame_snapshot(frames)

                with self.frame_condition:
                    next_seq = self.frames_seq + 1
                    self.frame_history.append((next_seq, snapshot))
                    self.frames_seq = next_seq
                    self.last_frames_ts = time.time()
                    self.frame_condition.notify_all()

            except Exception as e:
                if stop_event.is_set():
                    break
                self._record_wait_exception(e)
                time.sleep(0.05)

    def _build_frame_snapshot(self, frames: Any) -> _FrameSnapshot:
        """把 SDK frameset 解包为线程间可安全共享的 numpy snapshot。"""
        color_image_bgr = None
        depth_map = None

        color_frame = frames.get_color_frame()
        if color_frame is not None:
            color_image_bgr = np.ascontiguousarray(_decode_color_frame(color_frame))

        depth_frame = frames.get_depth_frame()
        if depth_frame is not None:
            depth_map = np.ascontiguousarray(np.asanyarray(depth_frame.get_data()).copy())

        return _FrameSnapshot(color=color_image_bgr, depth=depth_map)

    def _stop_pipeline_locked(self) -> None:
        """停止后台线程和 pipeline，并清空运行时状态。"""
        thread = self.thread
        stop_event = self.stop_event
        pipeline = self.pipeline

        if stop_event is not None:
            stop_event.set()

        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning("RealSense D405 manager thread for %s did not stop cleanly.", self.serial_number)

        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception as e:
                logger.warning("Error stopping RealSense D405 pipeline for %s: %s", self.serial_number, e)

        self.thread = None
        self.stop_event = Event()

        self.pipeline = None
        self.profile = None
        self.current_spec = None
        self.align_processor = None

        with self.frame_condition:
            self.frame_history.clear()
            self.frames_seq = 0
            self.last_frames_ts = None
            self.frame_condition.notify_all()


class RealSenseD405BaseCamera(Camera):
    """D405 color/depth 逻辑相机的公共基类。"""

    KIND: StreamKind

    def __init__(self, config: RealSenseD405ColorCameraConfig | RealSenseD405DepthCameraConfig):
        super().__init__(config)
        self.config = config
        self.serial_number, self.device_name = _resolve_serial_number(config.serial_number_or_name)
        self.manager: RealSenseD405Manager | None = None
        self.connected = False

        self.frame_lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self._last_frame_ts: float | None = None
        self._none_frame_count = 0
        self._ok_frame_count = 0
        self._dropped_frame_count = 0
        self._max_seq_gap = 0
        self._last_none_log_ts = 0.0
        self._last_drop_log_ts = 0.0
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
        """返回当前机器上的 D405 设备列表，供 `find-cameras` 使用。"""
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
            if _is_supported_d405_device(device)
        ]

    def connect(self, warmup: bool = True) -> None:
        """把当前逻辑相机接到共享 manager 上，并在需要时做 warmup。"""
        if self.connected:
            return

        device_info = _find_supported_device_info(self.serial_number)
        self.device_name = device_info["name"]
        usb_type_descriptor = device_info["usb_type_descriptor"]
        self.manager = RealSenseD405Manager.get_or_create(
            self.serial_number,
            device_name=self.device_name,
            usb_type_descriptor=usb_type_descriptor,
        )
        self.manager.update_device_info(self.device_name, usb_type_descriptor)

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
            RealSenseD405Manager.drop_if_unused(self.serial_number, self.manager)
            self.manager = None
            raise

        try:
            if self.manager is None:
                raise RuntimeError(f"{self} manager is not available after connect.")
            self._apply_stream_settings(self.manager.get_output_stream_spec(self.KIND))
        except Exception:
            if self.manager is not None:
                self.manager.disconnect_client(id(self))
            self.manager = None
            raise

        self.connected = True
        self.latest_frame = None
        self._last_frame_ts = None
        self._none_frame_count = 0
        self._ok_frame_count = 0
        self._dropped_frame_count = 0
        self._max_seq_gap = 0
        self._last_none_log_ts = 0.0
        self._last_drop_log_ts = 0.0
        self._last_pipeline_seq = -1

        if warmup:
            warmup_s = float(getattr(self.config, "warmup_s", 0) or 0)
            deadline = time.time() + warmup_s
            got_frame = False
            last_error: Exception | None = None
            while time.time() < deadline:
                try:
                    frame = self.read()
                    if frame is not None:
                        got_frame = True
                        break
                except Exception as e:
                    last_error = e
                time.sleep(0.05)

            if not got_frame and warmup_s > 0:
                if last_error is not None:
                    logger.warning("Warmup timed out for %s. Last error: %s", self, last_error)
                else:
                    logger.warning("Warmup timed out for %s.", self)

    def _apply_stream_settings(self, stream_spec: _StreamSpec) -> None:
        """把 manager 协商出的实际规格同步到当前逻辑相机实例。"""
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
        """如果 manager 在重启后修改了规格，则把当前实例的缓存尺寸同步过来。"""
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

    def _read_from_frames(self, frames: Any, color_mode: ColorMode | None = None) -> NDArray[Any] | None:
        """由子类实现：从 snapshot 中提取本视图。"""
        raise NotImplementedError

    def _process_frame_from_frames(self, frames: Any) -> NDArray[Any] | None:
        """统一处理一次“从 snapshot 到可返回图像”的流程，并更新统计信息。"""
        self._sync_stream_settings()
        frame = self._read_from_frames(frames)

        if frame is None:
            self._none_frame_count += 1
            now = time.time()
            if now - self._last_none_log_ts > 2.0:
                self._last_none_log_ts = now
                logger.debug(
                    "%s read() returned None (kind=%s, none_count=%s, ok_count=%s, manager_status=%s)",
                    self,
                    self.KIND,
                    self._none_frame_count,
                    self._ok_frame_count,
                    self.manager.get_status() if self.manager is not None else None,
                )
            return None

        with self.frame_lock:
            self.latest_frame = frame

        self._ok_frame_count += 1
        self._last_frame_ts = time.time()
        return frame

    def _advance_pipeline_seq(self, seq: int) -> None:
        """更新客户端看到的 seq，并统计在消费端发生的跳帧。"""
        if self._last_pipeline_seq < 0:
            dropped = max(seq - 1, 0)
        else:
            dropped = max(seq - self._last_pipeline_seq - 1, 0)

        if dropped > 0:
            self._dropped_frame_count += dropped
            self._max_seq_gap = max(self._max_seq_gap, dropped)
            now = time.time()
            if now - self._last_drop_log_ts > 2.0:
                self._last_drop_log_ts = now
                logger.debug(
                    "%s dropped %s frame(s) before consuming seq=%s (total_dropped=%s, max_gap=%s, manager_status=%s)",
                    self,
                    dropped,
                    seq,
                    self._dropped_frame_count,
                    self._max_seq_gap,
                    self.manager.get_status() if self.manager is not None else None,
                )

        self._last_pipeline_seq = seq

    def async_read(self, timeout_ms: float = 5000) -> NDArray[Any]:
        """等待 manager 的下一帧并返回本逻辑相机视图。"""
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.manager is None:
            raise RuntimeError(f"{self} does not have an active RealSense manager.")

        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                break

            frames, seq = self.manager.wait_for_next_frames(self._last_pipeline_seq, timeout_s=remaining_s)
            if frames is None or seq == self._last_pipeline_seq:
                break

            self._advance_pipeline_seq(seq)
            frame = self._process_frame_from_frames(frames)
            if frame is not None:
                return frame

        last_age = None if self._last_frame_ts is None else (time.time() - self._last_frame_ts)
        manager_status = self.manager.get_status()
        logger.warning(
            "Timed out waiting for frame from %s (kind=%s) after %sms. manager_thread_alive=%s, "
            "last_frame_age_s=%s, none_count=%s, ok_count=%s, manager_status=%s",
            self,
            self.KIND,
            timeout_ms,
            self.manager.thread is not None and self.manager.thread.is_alive(),
            last_age,
            self._none_frame_count,
            self._ok_frame_count,
            {
                **manager_status,
                "client_dropped_frames": self._dropped_frame_count,
                "client_max_seq_gap": self._max_seq_gap,
            },
        )
        raise TimeoutError(f"Timed out waiting for frame from camera {self}.")

    def disconnect(self) -> None:
        """从 manager 注销当前逻辑相机。"""
        if not self.connected:
            raise DeviceNotConnectedError(f"{self} already disconnected.")

        if self.manager is not None:
            self.manager.disconnect_client(id(self))

        self.connected = False
        self.latest_frame = None
        self._last_pipeline_seq = -1
        self.manager = None

    def _require_frames(self) -> Any:
        """获取当前最新 snapshot；若设备未连接则抛错。"""
        if not self.connected or self.manager is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._sync_stream_settings()

        frames = self.manager.get_latest_frames()
        if frames is None:
            return None
        return frames


class RealSenseD405ColorCamera(RealSenseD405BaseCamera):
    """D405 的彩色逻辑相机。"""

    KIND = "color"

    def __init__(self, config: RealSenseD405ColorCameraConfig):
        super().__init__(config)
        self.color_mode = config.color_mode

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any] | None:
        """同步读取当前最新彩色图。"""
        frames = self._require_frames()
        if frames is None:
            return None

        return self._read_from_frames(frames, color_mode=color_mode)

    def _read_from_frames(self, frames: Any, color_mode: ColorMode | None = None) -> NDArray[Any] | None:
        """从 snapshot 中提取彩色图并做颜色/旋转后处理。"""
        if frames.color is None:
            return None

        return self._postprocess_color_image(frames.color, color_mode=color_mode)

    def _postprocess_color_image(
        self, image_bgr: NDArray[Any], color_mode: ColorMode | None = None
    ) -> NDArray[Any]:
        """把 manager 产出的 BGR 图转换成调用方要求的最终格式。"""
        if color_mode and color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid requested color mode '{color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise RuntimeError(f"{self} frame is expected to be HxWx3, but got shape={image_bgr.shape}.")

        height, width, _ = image_bgr.shape
        if height != self.capture_height or width != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={width} or height={height} do not match configured width="
                f"{self.capture_width} or height={self.capture_height}."
            )

        requested_color_mode = color_mode or self.color_mode
        processed_image = image_bgr
        if requested_color_mode == ColorMode.RGB:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image


class RealSenseD405DepthCamera(RealSenseD405BaseCamera):
    """D405 的深度逻辑相机。"""

    KIND = "depth"

    def __init__(self, config: RealSenseD405DepthCameraConfig):
        super().__init__(config)

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any] | None:
        """同步读取当前最新深度图。"""
        del color_mode
        frames = self._require_frames()
        if frames is None:
            return None

        return self._read_from_frames(frames)

    def _read_from_frames(self, frames: Any, color_mode: ColorMode | None = None) -> NDArray[Any] | None:
        """从 snapshot 中提取深度图并做深度后处理。"""
        del color_mode
        if frames.depth is None:
            return None

        return self._postprocess_depth_image(frames.depth)

    def _postprocess_depth_image(self, depth_map: NDArray[Any]) -> NDArray[Any]:
        """把原始深度图转换成当前对外暴露的深度图像格式。"""
        if depth_map.ndim != 2:
            raise RuntimeError(f"{self} depth frame is expected to be HxW, but got shape={depth_map.shape}.")

        height, width = depth_map.shape
        if height != self.capture_height or width != self.capture_width:
            raise RuntimeError(
                f"{self} depth frame width={width} or height={height} do not match configured width="
                f"{self.capture_width} or height={self.capture_height}."
            )

        depth_rgb = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_map.astype(np.uint16, copy=False), alpha=0.03),
            cv2.COLORMAP_JET,
        )

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            depth_rgb = cv2.rotate(depth_rgb, self.rotation)

        return depth_rgb
