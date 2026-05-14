# stream_worker.py
import os
import time
import sys

from ultralytics import YOLO

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))      
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from core.config_loader import load_config, build_config_for_source
from core.video_processing import analyze_camera_stream
from core.db_pg import (
    get_active_stream_runs,
    get_active_cameras,
    create_stream_run_if_not_exists,
    get_video_source,
    mark_run_as_taken,
    finish_processing_run,
    reset_stuck_stream_runs,
    reset_stream_run_to_processing
)

BASE_DIR = PROJECT_DIR
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

def main():
    config = load_config(CONFIG_PATH)
    person_model = YOLO(config.person_detector.path)
    ppe_model = YOLO(config.ppe_detector.path)

    current_run_id = None

    print("[stream_worker] Запущен...")

    while True:
        try:
            # 1. Для каждой активной камеры пытаемся создать stream-run (атомарно)
            active_cameras = get_active_cameras()  # только type='camera', is_active=TRUE, camera_deleted=FALSE
            for cam in active_cameras:
                print(f"[stream_worker] Проверка камеры id={cam.id}, is_active={cam.is_active}, deleted={getattr(cam, 'camera_deleted', None)}")
                run_id = create_stream_run_if_not_exists(
                    video_source_id=cam.id,
                    config=build_config_for_source(global_config=config, source=cam),
                )
                if run_id is not None:
                    print(f"[stream_worker] Создан stream-run run_id={run_id} для камеры video_source_id={cam.id}")

            # 2. Берём только run'ы со status='processing'
            runs = get_active_stream_runs()
            if not runs:
                time.sleep(5)
                continue

            for run in runs:
                # 3. Пытаемся "захватить" run. Если не получилось — кто-то уже взял.
                if not mark_run_as_taken(run.id):
                    continue

                print(f"[stream_worker] Обработка run_id={run.id}, video_source_id={run.video_source_id}")

                source = get_video_source(run.video_source_id)
                if not source or source.type != "camera" or not source.is_active:
                    print(f"[stream_worker] Не найден video_source_id={run.video_source_id} для run_id={run.id}")
                    finish_processing_run(
                        run_id=run.id,
                        duration_sec=0.0,
                        video_fps=0.0,
                        total_frames=0,
                        processed_frames=0,
                        result_video_path=None,
                        error_message=f"video_source_id={run.video_source_id} не найден",
                    )
                    continue

                if source.type != "camera":
                    print(f"[stream_worker] video_source_id={source.id} не является камерой (type={source.type})")
                    finish_processing_run(
                        run_id=run.id,
                        duration_sec=0.0,
                        video_fps=0.0,
                        total_frames=0,
                        processed_frames=0,
                        result_video_path=None,
                        error_message=f"Источник id={source.id} не является камерой",
                    )
                    continue

                if not source.is_active:
                    print(f"[stream_worker] Камера video_source_id={source.id} неактивна (is_active=false)")
                    finish_processing_run(
                        run_id=run.id,
                        duration_sec=0.0,
                        video_fps=0.0,
                        total_frames=0,
                        processed_frames=0,
                        result_video_path=None,
                        error_message=f"Камера id={source.id} неактивна",
                    )
                    continue

                # Подбор camera_url по connection_type
                if source.connection_type in ("rtsp", "http_mjpeg"):
                    camera_url = source.camera_url
                elif source.connection_type == "file":
                    camera_url = source.camera_url
                elif source.connection_type == "device":
                    try:
                        device_index = int(source.camera_url) if source.camera_url is not None else 0
                    except ValueError:
                        device_index = 0
                    camera_url = device_index
                else:
                    print(f"[stream_worker] Неизвестный connection_type={source.connection_type} для camera id={source.id}")
                    finish_processing_run(
                        run_id=run.id,
                        duration_sec=0.0,
                        video_fps=0.0,
                        total_frames=0,
                        processed_frames=0,
                        result_video_path=None,
                        error_message=f"Неизвестный connection_type={source.connection_type}",
                    )
                    continue

                # Собираем конфиг
                config_local = build_config_for_source(global_config=config, source=source)

                # Ставим текущий run_id
                current_run_id = run.id

                try:
                    result = analyze_camera_stream(
                        camera_url=camera_url,
                        config=config_local,
                        person_model=person_model,
                        ppe_model=ppe_model,
                        run_id=run.id,
                        video_source_id=source.id,
                    )
                    print(f"[stream_worker] Завершён run_id={run.id}")
                except Exception as e:
                    print(f"[stream_worker] Ошибка при обработке run_id={run.id}: {e}")

            time.sleep(2)

        except KeyboardInterrupt:
            print("[stream_worker] Остановлен пользователем (KeyboardInterrupt).")
            if current_run_id is not None:
                try:
                    restored = reset_stream_run_to_processing(current_run_id)
                    if restored:
                        print(f"[stream_worker] run {current_run_id} возвращён в статус 'processing' для повторной обработки.")
                    else:
                        print(f"[stream_worker] run {current_run_id} не был в статусе 'processing_in_worker' или не найден.")
                except Exception as fe:
                    print(f"[stream_worker] Ошибка при сбросе статуса run_id={current_run_id} после KeyboardInterrupt: {fe}")
            break

        except Exception as e:
            print(f"[stream_worker] Необработанное исключение в основном цикле: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()