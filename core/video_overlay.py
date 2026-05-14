# core/video_overlay.py
from typing import Dict, Any
import cv2
import numpy as np
import os
import tempfile
import shutil
import subprocess

from .processing import analyze_frame, draw_annotations
from .config_loader import AppConfig


def generate_annotated_video(
    video_bytes: bytes,
    config: AppConfig,
    person_model,
    ppe_model
) -> Dict[str, Any]:
    import time

    # 1. Сохраняем входное видео во временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        cap.release()
        os.remove(tmp_path)
        return {"error": "Не удалось открыть видеофайл"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    # ---------- ВАЖНО: понижаем частоту обработки ----------
    # Нормализуем параметры, чтобы избежать None/NaN/нулевых значений.
    try:
        target_fps = float(config.video_processing.target_fps)
    except Exception:
        target_fps = 2.0
    if target_fps <= 0:
        target_fps = 2.0

    frame_step = max(1, int(round(fps / max(target_fps, 0.01))))
    # Например, fps=25, target_fps=2 => frame_step=12 (примерно каждый 12-й кадр)

    # 2. Берём один кадр, чтобы определить размер после ресайза
    ret, frame_bgr = cap.read()
    if not ret:
        cap.release()
        os.remove(tmp_path)
        return {"error": "Пустой видеофайл"}

    try:
        max_width = int(config.video_processing.max_width)
    except Exception:
        max_width = 0
    h0, w0 = frame_bgr.shape[:2]
    if max_width > 0 and w0 > max_width:
        new_h = int(max_width * h0 / w0)
        out_size = (max_width, new_h)
    else:
        out_size = (w0, h0)

    # Переходим в начало
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 3. Готовим выходной файл
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    output_dir = os.path.join(project_root, "output_videos")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Пишем промежуточный файл максимально совместимым с OpenCV способом.
    # Финальный браузерный формат сделаем ниже через ffmpeg (H.264/yuv420p).
    codec_candidates = [
        ("mp4v", ".mp4"),
        ("avc1", ".mp4"),
        ("MJPG", ".avi"),
    ]
    out = None
    out_path = None
    out_filename = None
    used_codec = None

    for codec_name, ext in codec_candidates:
        candidate_filename = f"annotated_{timestamp}_{codec_name}{ext}"
        candidate_path = os.path.join(output_dir, candidate_filename)
        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        candidate_out = cv2.VideoWriter(candidate_path, fourcc, target_fps, out_size)
        if candidate_out.isOpened():
            out = candidate_out
            out_path = candidate_path
            out_filename = candidate_filename
            used_codec = codec_name
            break
        candidate_out.release()

    if out is None:
        cap.release()
        os.remove(tmp_path)
        return {"error": "Не удалось открыть VideoWriter ни с одним кодеком"}

    processed_frames = 0
    frame_index = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_index += 1

        # пропускаем кадры, чтобы обрабатывать только с шагом frame_step
        if frame_index % frame_step != 0:
            continue

        # Масштабирование
        h, w = frame_bgr.shape[:2]
        if max_width > 0 and w > max_width:
            new_h = int(max_width * h / w)
            frame_bgr = cv2.resize(frame_bgr, (max_width, new_h), interpolation=cv2.INTER_AREA)

        # Анализ кадра
        analysis = analyze_frame(
            frame_bgr=frame_bgr,
            person_model=person_model,
            ppe_model=ppe_model,
            detection_logic=config.detection_logic,
        )

        # Рисуем аннотации
        annotated_bgr = draw_annotations(frame_bgr, analysis)

        out.write(annotated_bgr)
        processed_frames += 1

    cap.release()
    out.release()
    os.remove(tmp_path)

    # Если writer открылся, но ни одного кадра не записалось или файл пустой,
    # возвращаем информативную ошибку, чтобы фронтенд показал причину.
    if processed_frames == 0:
        try:
            if out_path and os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass
        return {"error": "Не удалось сформировать видео: нет обработанных кадров"}

    if not out_path or not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return {"error": "Не удалось сохранить выходной видеофайл"}

    # Всегда пытаемся получить браузерный H.264 MP4.
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        h264_filename = f"annotated_{timestamp}_h264.mp4"
        h264_path = os.path.join(output_dir, h264_filename)
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", out_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            h264_path,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0 and os.path.exists(h264_path) and os.path.getsize(h264_path) > 0:
                try:
                    os.remove(out_path)
                except Exception:
                    pass
                out_path = h264_path
                out_filename = h264_filename
                used_codec = "h264_ffmpeg"
            else:
                return {
                    "error": "ffmpeg не смог перекодировать видео в H.264",
                    "ffmpeg_path": ffmpeg_bin,
                    "ffmpeg_stderr": (proc.stderr or "")[-1000:],
                }
        except Exception as e:
            return {
                "error": f"Ошибка запуска ffmpeg: {e}",
                "ffmpeg_path": ffmpeg_bin,
            }
    else:
        return {
            "error": "ffmpeg не найден: невозможно подготовить браузерный H.264 MP4",
        }

    return {
        "out_path": out_path,
        "duration_sec": duration_sec,
        "duration_str": _sec_to_hms(duration_sec),
        "video_fps": fps,              # исходный fps
        "target_fps": target_fps,      # фактический fps выхода
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "frame_step": frame_step,
        "output_filename": out_filename,
        "writer_codec": used_codec,
    }


def _sec_to_hms(sec: float) -> str:
    import time
    return time.strftime("%H:%M:%S", time.gmtime(sec))