# tasks_processing.py

import os
from typing import Any, Dict, List
from dataclasses import dataclass
import cv2
import numpy as np

from core.processing import analyze_frame, draw_annotations
from core.video_processing import analyze_video_file
from core.video_overlay import generate_annotated_video
from core.db_pg import finish_processing_run, insert_violation, get_stable_violations_for_run, get_run_status
from core.config_loader import AppConfig, DetectionLogicConfig


def analyze_video_run_background(
    run_id: int,
    original_path: str,
    config_local: AppConfig,
    person_model,
    ppe_model,
    original_filename: str,
) -> None:
    """
    Фоновая задача: читает сохранённый файл и запускает analyze_video_file.
    Обновление run и запись нарушений происходят внутри analyze_video_file
    и/или через finish_processing_run.
    """
    try:
        # Защита от повторного запуска для уже обработанного run'а
        status = get_run_status(run_id)
        if status is None:
            print(f"[tasks_processing] run_id={run_id} не найден, analyze_video_run_background пропущен.")
            return
        if status != "processing":
            print(f"[tasks_processing] run_id={run_id} в статусе '{status}', повторный анализ не выполняется.")
            return

        with open(original_path, "rb") as f:
            video_bytes = f.read()

        result: Dict[str, Any] = analyze_video_file(
            video_bytes=video_bytes,
            config=config_local,
            person_model=person_model,
            ppe_model=ppe_model,
            original_filename=original_filename,
            existing_run_id=run_id,
        )

    except Exception as e:
        print(f"[tasks_processing] Ошибка analyze_video_run_background для run_id={run_id}: {e}")
        finish_processing_run(
            run_id=run_id,
            duration_sec=0,
            video_fps=0,
            total_frames=0,
            processed_frames=0,
            result_video_path=None,
            error_message=str(e),
        )


def analyze_video_overlay_run_background(
    run_id: int,
    original_path: str,
    config_local: AppConfig,
    person_model,
    ppe_model,
) -> None:
    """
    Фоновая задача: строит размеченное видео и завершает run в БД.
    """
    try:
        status = get_run_status(run_id)
        if status is None:
            print(f"[tasks_processing] run_id={run_id} не найден, overlay-просчёт пропущен.")
            return
        if status != "processing":
            print(f"[tasks_processing] run_id={run_id} в статусе '{status}', повторный overlay не выполняется.")
            return

        with open(original_path, "rb") as f:
            video_bytes = f.read()

        result: Dict[str, Any] = generate_annotated_video(
            video_bytes=video_bytes,
            config=config_local,
            person_model=person_model,
            ppe_model=ppe_model,
        )

        if "error" in result:
            raise RuntimeError(result["error"])

        finish_processing_run(
            run_id=run_id,
            duration_sec=float(result.get("duration_sec") or 0),
            video_fps=float(result.get("video_fps") or 0),
            total_frames=int(result.get("total_frames") or 0),
            processed_frames=int(result.get("processed_frames") or 0),
            result_video_path=result.get("out_path"),
            error_message=None,
        )
    except Exception as e:
        print(f"[tasks_processing] Ошибка analyze_video_overlay_run_background для run_id={run_id}: {e}")
        finish_processing_run(
            run_id=run_id,
            duration_sec=0,
            video_fps=0,
            total_frames=0,
            processed_frames=0,
            result_video_path=None,
            error_message=str(e),
        )


@dataclass
class ImageAnalysisResult:
    filename: str
    analysis: Dict[str, Any]
    processing_time_sec: float


def analyze_image_run_background(
    run_id: int,
    original_path: str,
    config_local: AppConfig,
    person_model,
    ppe_model,
    original_filename: str,
    violations_dir: str
) -> None:
    """
    Фоновая задача: анализирует одно изображение и сохраняет результат.
    """
    try:
        status = get_run_status(run_id)
        if status is None:
            print(f"[tasks_processing] run_id={run_id} не найден, overlay-просчёт пропущен.")
            return
        if status != "processing":
            print(f"[tasks_processing] run_id={run_id} в статусе '{status}', повторный overlay не выполняется.")
            return
        with open(original_path, "rb") as f:
            image_bytes = f.read()

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            finish_processing_run(
                run_id=run_id,
                duration_sec=0,
                video_fps=0,
                total_frames=0,
                processed_frames=0,
                result_video_path=None,
                error_message=f"Не удалось прочитать изображение: {original_filename}",
            )
            return

        # Анализ кадра
        analysis = analyze_frame(
            frame_bgr=frame_bgr,
            person_model=person_model,
            ppe_model=ppe_model,
            detection_logic=config_local.detection_logic,
        )

        # Рисуем аннотации
        annotated_bgr = draw_annotations(frame_bgr, analysis)

        # Время обработки
        import time
        processing_time_sec = time.perf_counter() - getattr(analyze_image_run_background, '_start', time.perf_counter())

        # Сохраняем изображение с разметкой в violations_frames
        os.makedirs(violations_dir, exist_ok=True)
        annotated_filename = f"annotated_{os.path.splitext(original_filename)[0]}.png"
        annotated_path = os.path.join(violations_dir, annotated_filename)
        cv2.imwrite(annotated_path, annotated_bgr)

        # Время обработки (переменная)
        processing_time_sec = 0.0

        result = ImageAnalysisResult(
            filename=original_filename,
            analysis=analysis,
            processing_time_sec=processing_time_sec,
        )

        # Сохраняем результат
        import json
        results_dir = os.path.join(violations_dir, "image_results")
        os.makedirs(results_dir, exist_ok=True)
        result_path = os.path.join(results_dir, f"run_{run_id}_result.json")
        with open(result_path, "w") as f:
            json.dump({
                "filename": result.filename,
                "analysis": result.analysis,
                "processing_time_sec": result.processing_time_sec,
                "annotated_image_path": annotated_path,
            }, f)

        finish_processing_run(
            run_id=run_id,
            duration_sec=processing_time_sec,
            video_fps=0,
            total_frames=0,
            processed_frames=1,
            result_video_path=annotated_path,
            error_message=None,
        )

    except Exception as e:
        print(f"[tasks_processing] Ошибка analyze_image_run_background для run_id={run_id}: {e}")
        finish_processing_run(
            run_id=run_id,
            duration_sec=0,
            video_fps=0,
            total_frames=0,
            processed_frames=0,
            result_video_path=None,
            error_message=str(e),
        )
