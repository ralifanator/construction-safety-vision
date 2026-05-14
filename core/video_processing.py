# core/video_processing.py
from typing import Dict, Any, List
from collections import deque
import cv2
import numpy as np
import os
import base64
import time
from typing import Dict, Any, Optional

from .processing import analyze_frame, draw_annotations
import base64
from .config_loader import AppConfig
from .db_pg import (
    create_video_source_for_file,
    create_processing_run,
    finish_processing_run,
    insert_violation,
    update_processing_run_progress
)
from .processing import analyze_frame, draw_annotations
from .config_loader import AppConfig


def _dominant_violation_from_history(history) -> tuple[str, str]:
    """
    По окну статусов кадров (как в analyze_frame: OK / NO HELMET / …)
    возвращает (violation_type для БД, представительный статус для подписи).
    Нужен, потому что устойчивое нарушение может сработать на кадре, где уже OK,
    а доля нарушений набрана предыдущими кадрами окна.
    """
    viols = [s for s in history if s != "OK"]
    if not viols:
        return "unknown", "OK"
    if "NO HELMET & NO VEST" in viols:
        return "no_helmet_and_vest", "NO HELMET & NO VEST"
    nh = sum(1 for s in viols if s == "NO HELMET")
    nv = sum(1 for s in viols if s == "NO VEST")
    if nh > 0 and nv > 0:
        return "no_helmet_and_vest", "NO HELMET & NO VEST"
    if nh > nv:
        return "no_helmet", "NO HELMET"
    if nv > nh:
        return "no_vest", "NO VEST"
    if nh > 0:
        return "no_helmet", "NO HELMET"
    if nv > 0:
        return "no_vest", "NO VEST"
    return "unknown", viols[-1]


def analyze_video_file(
    video_bytes: bytes,
    config: AppConfig,
    person_model,
    ppe_model,
    original_filename: str,
    existing_run_id: int | None = None,
) -> Dict[str, Any]:
    """
    Анализ видеофайла по кадрам с пониженной частотой с трекингом людей по IoU.
    По каждому человеку (track_id) ведём окно последних N кадров,
    считаем долю кадров с нарушением; если доля >= T, фиксируем устойчивое нарушение
    и сохраняем кадр с разметкой (PNG в base64).
    """

    import tempfile
    import os

    start_time = time.time()
    
    target_fps = config.video_processing.target_fps
    max_width = config.video_processing.max_width

    window_size = config.stability.window_size_n
    ratio_t = config.stability.violation_ratio_t
    min_window = config.stability.min_window_for_violation

    iou_threshold = config.tracking.iou_threshold
    max_track_lost_frames = config.tracking.max_track_lost_frames

    # 1. Временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        cap.release()
        os.remove(tmp_path)
        return {
            "error": "Не удалось открыть видеофайл"
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    # 2. Используем существующий run (для фоновой обработки) или создаём новый
    if existing_run_id is not None:
        run_id = existing_run_id
    else:
        video_source_id = create_video_source_for_file(original_filename, None)
        run_id = create_processing_run(video_source_id, "file_analyze", config)

    frame_step = max(1, int(round(fps / max(target_fps, 0.01))))

    processed_frames = 0
    # Сколько кадров реально попадёт в анализ (шаг frame_step, см. цикл ниже).
    frames_to_process = (
        (total_frames + frame_step - 1) // frame_step if total_frames > 0 else 0
    )

    # Инициализируем прогресс в БД сразу после старта.
    # Это убирает NULL в /api/video/status до завершения обработки.
    update_processing_run_progress(
        run_id=run_id,
        processed_frames=0,
        total_frames=total_frames,
        frames_to_process=frames_to_process,
    )

    # --- треки ---
    tracks: Dict[int, Dict[str, Any]] = {}
    next_track_id = 1

    # Устойчивые нарушения по людям (и кадрам)
    stable_violations_by_person: List[Dict[str, Any]] = []

    frame_index = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_index % frame_step != 0:
            frame_index += 1
            continue

        frame_index += 1
        processed_frames += 1

        # Обновляем прогресс не на каждом кадре, чтобы не перегружать БД.
        if processed_frames % 10 == 0:
            update_processing_run_progress(
                run_id=run_id,
                processed_frames=processed_frames,
                total_frames=total_frames,
                frames_to_process=frames_to_process,
            )

        print(f"[FRAME] frame_index={frame_index}, existing_tracks={len(tracks)}")

        # Масштабирование
        if max_width > 0:
            h, w = frame_bgr.shape[:2]
            if w > max_width:
                new_h = int(max_width * h / w)
                frame_bgr = cv2.resize(frame_bgr, (max_width, new_h), interpolation=cv2.INTER_AREA)

        # Анализ кадра
        analysis = analyze_frame(
            frame_bgr=frame_bgr,
            person_model=person_model,
            ppe_model=ppe_model,
            detection_logic=config.detection_logic,
        )

        persons = analysis.get("persons", [])

        # --- сопоставление людей с треками по IoU ---
        assigned_tracks = set()

        for person in persons:
            bbox = person["bbox"]  # [x1, y1, x2, y2]
            status = person.get("status", "OK")

            # 1) находим трек с максимальным IoU
            best_track_id = None
            best_iou = 0.0

            for tid, tinfo in tracks.items():
                if tid in assigned_tracks:
                    continue
                iou = _bbox_iou(bbox, tinfo["last_bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = tid

            # 2) если IoU достаточен -> продолжаем трек, иначе создаём новый
            if best_track_id is not None and best_iou >= iou_threshold:
                track_id = best_track_id
                assigned_tracks.add(track_id)
                track = tracks[track_id]
                created_new = False
            else:
                track_id = next_track_id
                next_track_id += 1
                tracks[track_id] = {
                    "last_bbox": bbox,
                    "last_frame_index": frame_index,
                    "history": deque(maxlen=window_size),
                    "active_violation": False
                }
                assigned_tracks.add(track_id)
                track = tracks[track_id]
                created_new = True

            # Обновляем данные трека
            track["last_bbox"] = bbox
            track["last_frame_index"] = frame_index

            # В окне храним реальные статусы кадров, чтобы тип нарушения не терялся,
            # когда на текущем кадре уже OK, а порог доли набран по предыдущим кадрам.
            track["history"].append(status)

            total_in_window = len(track["history"])
            if total_in_window > 0:
                num_violations = sum(1 for s in track["history"] if s != "OK")
                ratio = num_violations / total_in_window
            else:
                ratio = 0.0

            print(
                f"  track_id={track_id} "
                f"({'NEW' if created_new else 'OLD'}) "
                f"frame={frame_index} "
                f"status={status} "
                f"history_len={total_in_window} "
                f"violations={num_violations} "
                f"ratio={ratio:.2f}"
            )

            # Если уже зафиксировано устойчивое нарушение для этого трека, пропускаем
            if track["active_violation"]:
                continue
            
            # Если доля >= порога -> фиксируем устойчивое нарушение по этому человеку
            if total_in_window >= min_window and ratio >= ratio_t and num_violations > 0:
                time_sec = (frame_index - 1) / fps if fps > 0 else 0.0

                # Рисуем аннотации на текущем кадре
                annotated_bgr = draw_annotations(frame_bgr, analysis)

                # Сохраняем кадр нарушения в файл
                violations_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "violations_frames")
                os.makedirs(violations_dir, exist_ok=True)
                frame_filename = f"run_{run_id}_track_{track_id}_frame_{frame_index - 1}.png"
                frame_path = os.path.join(violations_dir, frame_filename)
                frame_url = f"/violations_frames/{frame_filename}"
                cv2.imwrite(frame_path, annotated_bgr)

                violation_type, dominant_status = _dominant_violation_from_history(
                    track["history"]
                )
                if status == "OK" and dominant_status != "OK":
                    status_note = f"{dominant_status} (устойчиво по окну; текущий кадр OK)"
                else:
                    status_note = status

                # Запись в БД
                insert_violation(
                    run_id=run_id,
                    track_index=track_id,
                    violation_type=violation_type,
                    frame_index=frame_index - 1,
                    time_sec=time_sec,
                    time_str=_sec_to_hms(time_sec),
                    window_size=total_in_window,
                    violations_in_window=num_violations,
                    violation_ratio=ratio,
                    status_note=status_note,
                    frame_image_path=frame_path,
                    frame_image_url=frame_url
                )

                annotated_bgr = draw_annotations(frame_bgr, analysis)

                violations_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "violations_frames")
                os.makedirs(violations_dir, exist_ok=True)

                frame_filename = f"run_{run_id}_track_{track_id}_frame_{frame_index - 1}.png"
                frame_path = os.path.join(violations_dir, frame_filename)
                cv2.imwrite(frame_path, annotated_bgr)

                # относительный URL
                frame_rel_url = f"/violations_frames/{frame_filename}"

                stable_violations_by_person.append({
                    "track_id": track_id,
                    "frame_index": frame_index - 1,
                    "time_sec": time_sec,
                    "time_str": _sec_to_hms(time_sec),
                    "window_size": total_in_window,
                    "violations_in_window": num_violations,
                    "violation_ratio": ratio,
                    "violation_type": violation_type,
                    "status_note": status_note,
                    "frame_image_url": frame_rel_url,
                })

                track["active_violation"] = True

        # --- удаляем "забытые" треки ---
        to_delete = []
        for tid, tinfo in tracks.items():
            if frame_index - tinfo["last_frame_index"] > max_track_lost_frames:
                to_delete.append(tid)
        for tid in to_delete:
            del tracks[tid]

    cap.release()
    os.remove(tmp_path)

    # Финальное обновление прогресса перед завершением run.
    update_processing_run_progress(
        run_id=run_id,
        processed_frames=processed_frames,
        total_frames=total_frames,
        frames_to_process=frames_to_process,
    )

    finish_processing_run(
    run_id=run_id,
    duration_sec=duration_sec,
    video_fps=fps,
    total_frames=total_frames,
    processed_frames=processed_frames,
    result_video_path=None,
    error_message=None
    )

    result: Dict[str, Any] = {
        "duration_sec": duration_sec,
        "duration_str": _sec_to_hms(duration_sec),
        "video_fps": fps,
        "target_fps": target_fps,
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "stability_window_size": window_size,
        "stability_ratio_t": ratio_t,
        "tracking_iou_threshold": iou_threshold,
        "tracking_max_track_lost_frames": max_track_lost_frames,
        "stable_violations_by_person": stable_violations_by_person
    }

    return result


def analyze_camera_stream(
    camera_url: str | int,
    config: AppConfig,
    person_model,
    ppe_model,
    run_id: int,
    video_source_id: int,
) -> Dict[str, Any]:
    """
    Анализ видеопотока с камеры.

    Логика аналогична analyze_video_file:
      - трекинг людей по IoU,
      - по каждому треку фиксируется окно из последних N статусов,
      - при достижении порога доли нарушений — записывается устойчивое нарушение и кадр,
      - треки чистятся по max_track_lost_frames.

    Отличия:
      - источник кадров: VideoCapture(camera_url),
      - нет total_frames (стрим), только processed_frames,
      - run ограничен по времени (сессия), max_run_duration_sec,
      - сохраняется "последний кадр" камеры в last_frames/camera_{video_source_id}_last.jpg,
      - track_id сбрасывается при достижении лимита TRACK_ID_MAX.
    """

    target_fps = config.video_processing.target_fps
    max_width = config.video_processing.max_width

    window_size = config.stability.window_size_n
    ratio_t = config.stability.violation_ratio_t
    min_window = config.stability.min_window_for_violation

    iou_threshold = config.tracking.iou_threshold
    max_track_lost_frames = config.tracking.max_track_lost_frames

    # Ограничение длительности одной сессии (run).
    max_run_duration_sec = config.stream.max_run_duration_sec

    print(f"[analyze_camera_stream] Открываем источник: {camera_url} (type={type(camera_url)})")
    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        finish_processing_run(
            run_id=run_id,
            duration_sec=0.0,
            video_fps=0.0,
            total_frames=0,
            processed_frames=0,
            result_video_path=None,
            error_message=f"Не удалось открыть поток: {camera_url}",
        )
        return {"error": "cannot_open_stream"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    # Рассчитываем шаг кадров по target_fps
    if not target_fps or target_fps <= 0:
        target_fps = fps
    frame_step = max(1, int(round(fps / max(target_fps, 0.01))))

    processed_frames = 0
    frame_index = 0

    # Для стрима total_frames и frames_to_process нам не известны
    total_frames = None
    frames_to_process = None

    # Инициализируем прогресс
    update_processing_run_progress(
        run_id=run_id,
        processed_frames=0,
        total_frames=total_frames,
        frames_to_process=frames_to_process,
    )

    tracks: Dict[int, Dict[str, Any]] = {}
    next_track_id = 1
    TRACK_ID_MAX = 1_000_000

    stable_violations_by_person: List[Dict[str, Any]] = []

    start_time = time.time()

    base_dir = os.path.dirname(os.path.dirname(__file__))
    violations_dir = os.path.join(base_dir, "violations_frames")
    os.makedirs(violations_dir, exist_ok=True)

    last_frames_dir = os.path.join(base_dir, "last_frames")
    os.makedirs(last_frames_dir, exist_ok=True)

    try:
        while True:
            # Ограничение по длительности сессии
            if max_run_duration_sec and (time.time() - start_time) >= max_run_duration_sec:
                break

            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_index % frame_step != 0:
                frame_index += 1
                continue

            frame_index += 1
            processed_frames += 1

            # Масштабирование
            if max_width > 0:
                h, w = frame_bgr.shape[:2]
                if w > max_width:
                    new_h = int(max_width * h / w)
                    frame_bgr = cv2.resize(frame_bgr, (max_width, new_h), interpolation=cv2.INTER_AREA)

            # Анализ кадра
            analysis = analyze_frame(
                frame_bgr=frame_bgr,
                person_model=person_model,
                ppe_model=ppe_model,
                detection_logic=config.detection_logic,
            )

            persons = analysis.get("persons", [])

            # Рисуем аннотации (и используем один и тот же annotated_bgr для "последнего кадра" и для нарушений)
            annotated_bgr = draw_annotations(frame_bgr, analysis)

            # Сохраняем "последний кадр" камеры
            last_frame_filename = f"camera_{video_source_id}_last.jpg"
            last_frame_path = os.path.join(last_frames_dir, last_frame_filename)
            cv2.imwrite(last_frame_path, annotated_bgr)

            # --- трекинг и устойчивые нарушения ---
            assigned_tracks = set()

            for person in persons:
                bbox = person["bbox"]  # [x1, y1, x2, y2]
                status = person.get("status", "OK")

                # 1) находим трек с максимальным IoU
                best_track_id = None
                best_iou = 0.0

                for tid, tinfo in tracks.items():
                    if tid in assigned_tracks:
                        continue
                    iou = _bbox_iou(bbox, tinfo["last_bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_track_id = tid

                # 2) если IoU достаточен -> продолжаем трек, иначе создаём новый
                if best_track_id is not None and best_iou >= iou_threshold:
                    track_id = best_track_id
                    assigned_tracks.add(track_id)
                    track = tracks[track_id]
                    created_new = False
                else:
                    # Проверка лимита на track_id
                    if next_track_id > TRACK_ID_MAX:
                        tracks.clear()
                        next_track_id = 1

                    track_id = next_track_id
                    next_track_id += 1

                    tracks[track_id] = {
                        "last_bbox": bbox,
                        "last_frame_index": frame_index,
                        "history": deque(maxlen=window_size),
                        "active_violation": False
                    }
                    assigned_tracks.add(track_id)
                    track = tracks[track_id]
                    created_new = True

                # Обновляем трек
                track["last_bbox"] = bbox
                track["last_frame_index"] = frame_index
                track["history"].append(status)

                total_in_window = len(track["history"])
                num_violations = sum(1 for s in track["history"] if s != "OK")
                ratio = (num_violations / total_in_window) if total_in_window > 0 else 0.0

                # Если уже зафиксировано устойчивое нарушение — пропускаем
                if track["active_violation"]:
                    continue

                # Условие устойчивого нарушения
                if total_in_window >= min_window and ratio >= ratio_t and num_violations > 0:
                    time_sec = (frame_index - 1) / fps if fps > 0 else 0.0

                    frame_filename = f"run_{run_id}_track_{track_id}_frame_{frame_index - 1}.png"
                    frame_path = os.path.join(violations_dir, frame_filename)
                    frame_url = f"/violations_frames/{frame_filename}"
                    cv2.imwrite(frame_path, annotated_bgr)

                    violation_type, dominant_status = _dominant_violation_from_history(
                        track["history"]
                    )
                    if status == "OK" and dominant_status != "OK":
                        status_note = f"{dominant_status} (устойчиво по окну; текущий кадр OK)"
                    else:
                        status_note = status

                    insert_violation(
                        run_id=run_id,
                        track_index=track_id,
                        violation_type=violation_type,
                        frame_index=frame_index - 1,
                        time_sec=time_sec,
                        time_str=_sec_to_hms(time_sec),
                        window_size=total_in_window,
                        violations_in_window=num_violations,
                        violation_ratio=ratio,
                        status_note=status_note,
                        frame_image_path=frame_path,
                        frame_image_url=frame_url
                    )

                    frame_rel_url = f"/violations_frames/{frame_filename}"
                    stable_violations_by_person.append({
                        "track_id": track_id,
                        "frame_index": frame_index - 1,
                        "time_sec": time_sec,
                        "time_str": _sec_to_hms(time_sec),
                        "window_size": total_in_window,
                        "violations_in_window": num_violations,
                        "violation_ratio": ratio,
                        "violation_type": violation_type,
                        "status_note": status_note,
                        "frame_image_url": frame_rel_url,
                    })

                    track["active_violation"] = True

            # Удаляем "забытые" треки
            to_delete = []
            for tid, tinfo in tracks.items():
                if frame_index - tinfo["last_frame_index"] > max_track_lost_frames:
                    to_delete.append(tid)
            for tid in to_delete:
                del tracks[tid]

            # Обновление прогресса раз в 30 обработанных кадров
            if processed_frames % 30 == 0:
                update_processing_run_progress(
                    run_id=run_id,
                    processed_frames=processed_frames,
                    total_frames=total_frames,
                    frames_to_process=frames_to_process,
                )

    except KeyboardInterrupt:
        raise
    
    except Exception as e:
        cap.release()
        duration_sec = time.time() - start_time
        finish_processing_run(
            run_id=run_id,
            duration_sec=duration_sec,
            video_fps=fps,
            total_frames=0,
            processed_frames=processed_frames,
            result_video_path=None,
            error_message=str(e),
        )
        return {"error": str(e)}

    cap.release()
    duration_sec = time.time() - start_time

    # Финальное обновление прогресса
    update_processing_run_progress(
        run_id=run_id,
        processed_frames=processed_frames,
        total_frames=total_frames,
        frames_to_process=frames_to_process,
    )

    finish_processing_run(
        run_id=run_id,
        duration_sec=duration_sec,
        video_fps=fps,
        total_frames=0,
        processed_frames=processed_frames,
        result_video_path=None,
        error_message=None,
    )

    return {
        "duration_sec": duration_sec,
        "duration_str": _sec_to_hms(duration_sec),
        "video_fps": fps,
        "target_fps": target_fps,
        "total_frames": None,
        "processed_frames": processed_frames,
        "stability_window_size": window_size,
        "stability_ratio_t": ratio_t,
        "tracking_iou_threshold": iou_threshold,
        "tracking_max_track_lost_frames": max_track_lost_frames,
        "stable_violations_by_person": stable_violations_by_person,
    }


def _sec_to_hms(sec: float) -> str:
    import time
    return time.strftime("%H:%M:%S", time.gmtime(sec))


def _bbox_iou(b1, b2) -> float:
    x1, y1, x2, y2 = b1
    x1g, y1g, x2g, y2g = b2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    area1 = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = max(0, x2g - x1g) * max(0, y2g - y1g)
    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union