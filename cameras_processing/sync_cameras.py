# sync_cameras_from_yaml.py
import os
import json
import yaml
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))      
PROJECT_DIR = os.path.dirname(CURRENT_DIR)                    

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from core.db_pg import upsert_video_source_from_yaml, mark_cameras_deleted_not_in_external_ids

BASE_DIR = PROJECT_DIR
CAMERAS_CONFIG_PATH = os.path.join(BASE_DIR, "cameras_processing", "cameras.yaml")

def sync_cameras_from_yaml(yaml_path: str | None = None):
    if yaml_path is None:
        yaml_path = CAMERAS_CONFIG_PATH

    if not os.path.exists(yaml_path):
        print(f"Файл {yaml_path} не найден")
        return

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    cameras = data.get("cameras", [])
    print(f"Найдено камер в YAML: {len(cameras)}")

    yaml_external_ids: list[str] = []

    for cam in cameras:
        external_id = cam["id"]
        yaml_external_ids.append(external_id)

        name = cam["name"]
        connection_type = cam["connection_type"]
        enabled = bool(cam.get("enabled", True))
        location = cam.get("location")

        if connection_type in ("rtsp", "http_mjpeg"):
            url = cam["url"]
        elif connection_type == "file":
            url = cam["path"]
        elif connection_type == "device":
            device_index = cam.get("device_index", 0)
            url = str(device_index)
        else:
            print(f"Пропуск камеры {external_id}: неизвестный connection_type={connection_type}")
            continue

        camera_params: dict = {}
        for key in ("detection_logic", "video_processing", "stability", "tracking", "stream"):
            if key in cam:
                camera_params[key] = cam[key]

        if connection_type == "device":
            camera_params.setdefault("device", {})["index"] = cam.get("device_index", 0)

        vp = camera_params.get("video_processing", {})
        target_fps = vp.get("target_fps")

        source_id = upsert_video_source_from_yaml(
            external_id=external_id,
            name=name,
            connection_type=connection_type,
            url=url,
            location=location,
            enabled=enabled,
            target_fps=target_fps,
            camera_params=camera_params,
        )

        print(f"Синхронизирована камера external_id={external_id}, video_source_id={source_id}")

    # Помечаем как удалённые те камеры, которых нет в YAML
    if yaml_external_ids:
        deleted_count = mark_cameras_deleted_not_in_external_ids(yaml_external_ids)
        if deleted_count:
            print(f"Помечено как удалённые (camera_deleted=TRUE) камер, отсутствующих в cameras.yaml: {deleted_count}")
    else:
        print("В cameras.yaml нет камер, пометку camera_deleted не выполняем")


if __name__ == "__main__":
    sync_cameras_from_yaml()