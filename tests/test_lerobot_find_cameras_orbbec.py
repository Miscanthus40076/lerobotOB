from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np

find_cameras_module = importlib.import_module("lerobot.scripts.lerobot_find_cameras")


@dataclass
class FakeOrbbecColorCameraConfig:
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color_mode: object | None = None


@dataclass
class FakeOrbbecDepthCameraConfig:
    fps: int | None = None
    width: int | None = None
    height: int | None = None


class FakeOrbbecColorCamera:
    @staticmethod
    def find_cameras() -> list[dict[str, object]]:
        return [
            {
                "name": "Orbbec Gemini 335",
                "type": "Orbbec",
                "id": "ORB123",
                "serial_number": "ORB123",
                "default_stream_profile": {
                    "stream_type": "Color",
                    "format": "RGB",
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                },
                "default_color_stream_profile": {
                    "stream_type": "Color",
                    "format": "RGB",
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                },
            }
        ]

    def __init__(self, config: FakeOrbbecColorCameraConfig):
        self.config = config
        self.is_connected = False

    def connect(self, warmup: bool = True) -> None:
        del warmup
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False

    def read(self) -> np.ndarray:
        height = int(self.config.height or 1)
        width = int(self.config.width or 1)
        return np.zeros((height, width, 3), dtype=np.uint8)


class FakeOrbbecDepthCamera(FakeOrbbecColorCamera):
    def __init__(self, config: FakeOrbbecDepthCameraConfig):
        super().__init__(config)


def _patch_orbbec_support(monkeypatch) -> None:
    monkeypatch.setattr(
        find_cameras_module,
        "_load_orbbec_camera_support",
        lambda: (
            FakeOrbbecColorCamera,
            FakeOrbbecColorCameraConfig,
            FakeOrbbecDepthCamera,
            FakeOrbbecDepthCameraConfig,
        ),
    )


def test_find_all_orbbec_cameras_uses_orbbec_discovery(monkeypatch) -> None:
    _patch_orbbec_support(monkeypatch)

    cameras = find_cameras_module.find_all_orbbec_cameras()

    assert len(cameras) == 1
    assert cameras[0]["type"] == "Orbbec"
    assert cameras[0]["id"] == "ORB123"
    assert cameras[0]["default_color_stream_profile"]["width"] == 640


def test_create_camera_instance_builds_orbbec_color_camera(monkeypatch) -> None:
    _patch_orbbec_support(monkeypatch)
    meta = FakeOrbbecColorCamera.find_cameras()[0]

    camera_dict = find_cameras_module.create_camera_instance(meta)

    assert camera_dict is not None
    instance = camera_dict["instance"]
    assert isinstance(instance, FakeOrbbecColorCamera)
    assert instance.config.width == 640
    assert instance.config.height == 480
    assert instance.read().shape == (480, 640, 3)


def test_create_camera_instance_falls_back_to_orbbec_depth_camera(monkeypatch) -> None:
    _patch_orbbec_support(monkeypatch)
    meta = {
        "name": "Orbbec Gemini 335",
        "type": "Orbbec",
        "id": "ORB123",
        "default_stream_profile": {
            "stream_type": "Depth",
            "format": "Y16",
            "width": 640,
            "height": 400,
            "fps": 30,
        },
        "default_depth_stream_profile": {
            "stream_type": "Depth",
            "format": "Y16",
            "width": 640,
            "height": 400,
            "fps": 30,
        },
    }

    camera_dict = find_cameras_module.create_camera_instance(meta)

    assert camera_dict is not None
    instance = camera_dict["instance"]
    assert isinstance(instance, FakeOrbbecDepthCamera)
    assert instance.config.width == 640
    assert instance.config.height == 400


def test_save_images_from_all_cameras_skips_additional_orbbec_devices(monkeypatch, tmp_path) -> None:
    created_ids: list[str] = []

    def fake_find_and_print_cameras(camera_type_filter: str | None = None) -> list[dict[str, object]]:
        assert camera_type_filter == "orbbec"
        return [
            {"type": "Orbbec", "id": "ORB123"},
            {"type": "Orbbec", "id": "ORB456"},
        ]

    def fake_create_camera_instance(cam_meta: dict[str, object]) -> dict[str, object]:
        created_ids.append(str(cam_meta["id"]))
        return {
            "instance": FakeOrbbecColorCamera(FakeOrbbecColorCameraConfig(width=1, height=1)),
            "meta": cam_meta,
        }

    monkeypatch.setattr(find_cameras_module, "find_and_print_cameras", fake_find_and_print_cameras)
    monkeypatch.setattr(find_cameras_module, "create_camera_instance", fake_create_camera_instance)

    find_cameras_module.save_images_from_all_cameras(tmp_path, record_time_s=0, camera_type="orbbec")

    assert created_ids == ["ORB123"]
