# main_server.py
import os
from fastapi.staticfiles import StaticFiles

from typing import List
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, APIRouter, Query
from fastapi.responses import JSONResponse, Response
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import time
import json
from copy import deepcopy
from datetime import datetime

from core.config_loader import load_config, DetectionLogicConfig, VideoProcessingConfig, StabilityConfig
from core.processing import analyze_frame, draw_annotations
from core.video_processing import analyze_video_file
from core.video_overlay import generate_annotated_video
from core.tasks_processing import analyze_video_run_background, analyze_video_overlay_run_background, analyze_image_run_background
from core.db_pg import (
    create_video_source_for_file,
    create_processing_run,
    finish_processing_run,
    get_processing_run,
    get_video_run_result,
    get_stable_violations_for_run,
    get_all_cameras,
    get_camera_processing_states,
    get_camera_violations_for_period
)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

OUTPUT_VIDEOS_DIR = os.path.join(BASE_DIR, "output_videos")
os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)

VIOLATIONS_DIR = os.path.join(BASE_DIR, "violations_frames")
os.makedirs(VIOLATIONS_DIR, exist_ok=True)

LAST_FRAMES_DIR = os.path.join(BASE_DIR, "last_frames")
os.makedirs(LAST_FRAMES_DIR, exist_ok=True)

# фронт
app.mount("/app", StaticFiles(directory=STATIC_DIR, html=True), name="static")

# Раздаём output_videos как статические файлы
app.mount("/output_videos", StaticFiles(directory=OUTPUT_VIDEOS_DIR), name="output_videos")

app.mount("/violations_frames", StaticFiles(directory=VIOLATIONS_DIR), name="violations_frames")

app.mount("/last_frames", StaticFiles(directory=LAST_FRAMES_DIR), name="last_frames")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Путь к config.yaml относительно этого файла ---
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

config = load_config(CONFIG_PATH)

# Модели YOLO
person_model = YOLO(config.person_detector.path)
ppe_model = YOLO(config.ppe_detector.path)

@app.get("/api/config_defaults")
def get_config_defaults():
    """
    Возвращает значения по умолчанию для настроек обработки
    из config.yaml, чтобы фронтенд мог подставить их в поля ввода.
    """
    return {
        "image": {
            "helmet_confidence_threshold": config.detection_logic.helmet_confidence_threshold,
            "vest_confidence_threshold": config.detection_logic.vest_confidence_threshold,
            "person_confidence_threshold": config.detection_logic.person_confidence_threshold,
            "person_min_area": config.detection_logic.person_min_area,
        },
        "video": {
            "target_fps": config.video_processing.target_fps,
            "helmet_confidence_threshold": config.detection_logic.helmet_confidence_threshold,
            "vest_confidence_threshold": config.detection_logic.vest_confidence_threshold,
            "person_confidence_threshold": config.detection_logic.person_confidence_threshold,
            "window_size_n": config.stability.window_size_n,
            "violation_ratio_t": config.stability.violation_ratio_t,
            "min_window_for_violation": config.stability.min_window_for_violation,
            "max_width": config.video_processing.max_width,
        }
    }

@app.post("/api/image/start_analyze")
async def start_analyze_image_endpoint(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    params: str = Form(None),
):
    """
    Старт фоновой обработки изображений.
    HTTP-ответ возвращает run_id сразу, не дожидаясь конца анализа.
    """
    if not files:
        return JSONResponse(
            status_code=400,
            content={"error": "Не переданы файлы"}
        )

    # Базовые конфиги
    base_det_cfg = config.detection_logic
    det_cfg_local = base_det_cfg

    if params:
        try:
            p = json.loads(params)
            det_cfg_local = DetectionLogicConfig(
                helmet_vest_match_iou_threshold=base_det_cfg.helmet_vest_match_iou_threshold,
                person_min_area=base_det_cfg.person_min_area,
                helmet_confidence_threshold=p.get("helmet_confidence_threshold", base_det_cfg.helmet_confidence_threshold),
                vest_confidence_threshold=p.get("vest_confidence_threshold", base_det_cfg.vest_confidence_threshold),
                person_confidence_threshold=p.get("person_confidence_threshold", base_det_cfg.person_confidence_threshold),
            )
        except Exception as e:
            print("Ошибка разбора params для изображений:", e)
            det_cfg_local = base_det_cfg

    config_local = deepcopy(config)
    config_local.detection_logic = det_cfg_local

    results = []

    for file in files:
        try:
            image_bytes = await file.read()
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": f"Не удалось прочитать файл: {e}"
            })
            continue

        input_images_dir = os.path.join(BASE_DIR, "input_images")
        os.makedirs(input_images_dir, exist_ok=True)
        original_path = os.path.join(input_images_dir, file.filename)
        try:
            with open(original_path, "wb") as f:
                f.write(image_bytes)
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": f"Не удалось сохранить изображение: {e}"
            })
            continue

        video_source_id = create_video_source_for_file(file.filename, original_path)
        run_id = create_processing_run(video_source_id, "image_analyze", config_local)

        background_tasks.add_task(
            analyze_image_run_background,
            run_id=run_id,
            original_path=original_path,
            config_local=config_local,
            person_model=person_model,
            ppe_model=ppe_model,
            original_filename=file.filename,
            violations_dir=VIOLATIONS_DIR
        )

        results.append({"run_id": run_id, "filename": file.filename, "status": "queued"})

    return JSONResponse(content={"results": results})


@app.get("/api/image/status/{run_id}")
def get_image_status(run_id: int):
    """
    Возвращает статус фоновой обработки изображения.
    """
    run = get_processing_run(run_id)
    if not run:
        return JSONResponse(status_code=404, content={"error": "run not found"})

    return {
        "run_id": run.id,
        "status": run.status,
        "error_message": run.error_message,
        "duration_sec": run.duration_sec,
    }


@app.get("/api/image/result/{run_id}")
def get_image_result(run_id: int):
    """
    Возвращает результат фоновой обработки изображения.
    """
    run = get_video_run_result(run_id)
    if not run:
        return JSONResponse(status_code=404, content={"error": "run not found"})

    if run.error_message:
        return JSONResponse(status_code=500, content={"error": run.error_message})

    # Пытаемся прочитать JSON-результат
    result_path = None
    # Проверяем сначала в violations_frames/image_results/
    results_dir = os.path.join(BASE_DIR, "violations_frames", "image_results")
    if os.path.exists(results_dir):
        result_file = os.path.join(results_dir, f"run_{run_id}_result.json")
        if os.path.exists(result_file):
            result_path = result_file
    if not result_path:
        results_dir_old = os.path.join(BASE_DIR, "image_results")
        if os.path.exists(results_dir_old):
            result_file = os.path.join(results_dir_old, f"run_{run_id}_result.json")
            if os.path.exists(result_file):
                result_path = result_file

    if result_path:
        with open(result_path, "r") as f:
            result_data = json.load(f)
        # Формируем URL для размеченного изображения
        annotated_path = result_data.get("annotated_image_path")
        if annotated_path and os.path.exists(annotated_path):
            result_data["annotated_image_url"] = "/violations_frames/" + os.path.basename(annotated_path)
        return JSONResponse(content={"result": result_data})
    else:
        # Фаллбэк: возвращаем данные из run
        return JSONResponse(content={
            "result": {
                "runid": run.id,
                "filename": run.result_video_path or f"run{run.id}",
                "status": run.status,
                "error": run.error_message,
                "annotated_image_url": run.result_video_path and f"/violations_frames/{os.path.basename(run.result_video_path)}" or None,
            }
        })


@app.post("/api/image/analyze_with_overlay")
async def analyze_image_with_overlay_endpoint(
    files: List[UploadFile] = File(...),
    params: str = Form(None),
):
    """
    Принимает один или несколько файлов изображений (ключ 'files'),
    возвращает список результатов:
      - filename
      - analysis
      - image_base64 (PNG)
      - processing_time_sec (время обработки одного изображения в секундах)
    """
    if not files:
        return JSONResponse(
            status_code=400,
            content={"error": "Не переданы файлы"}
        )

    base_det_cfg = config.detection_logic
    det_cfg_local = base_det_cfg

    if params:
        try:
            p = json.loads(params)
            det_cfg_local = DetectionLogicConfig(
                helmet_vest_match_iou_threshold=base_det_cfg.helmet_vest_match_iou_threshold,
                person_min_area=p.get("person_min_area", base_det_cfg.person_min_area),
                helmet_confidence_threshold=p.get("helmet_confidence_threshold", base_det_cfg.helmet_confidence_threshold),
                vest_confidence_threshold=p.get("vest_confidence_threshold", base_det_cfg.vest_confidence_threshold),
                person_confidence_threshold=p.get("person_confidence_threshold", base_det_cfg.person_confidence_threshold),
            )
        except Exception as e:
            print("Ошибка разбора params для изображений:", e)
            det_cfg_local = base_det_cfg

    results = []

    for file in files:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            results.append({
                "filename": file.filename,
                "error": "Не удалось прочитать изображение"
            })
            continue

        # --- замеряем время обработки одного изображения ---
        t_start = time.perf_counter()

        # 1. Анализ
        analysis = analyze_frame(
            frame_bgr=frame_bgr,
            person_model=person_model,
            ppe_model=ppe_model,
            detection_logic=det_cfg_local,
        )

        # 2. Рисуем аннотации
        annotated_bgr = draw_annotations(frame_bgr, analysis)

        # --- конец замера ---
        t_end = time.perf_counter()
        processing_time_sec = t_end - t_start

        # Кодируем в PNG и потом в base64
        success, encoded_image = cv2.imencode(".png", annotated_bgr)
        if not success:
            results.append({
                "filename": file.filename,
                "error": "Не удалось закодировать изображение"
            })
            continue

        b64_bytes = base64.b64encode(encoded_image.tobytes())
        b64_str = b64_bytes.decode("utf-8")

        results.append({
            "filename": file.filename,
            "analysis": analysis,
            "image_base64": b64_str,
            "image_mime_type": "image/png",
            "processing_time_sec": processing_time_sec
        })

    return JSONResponse(content={"results": results})


@app.post("/api/video/analyze")
async def analyze_video_endpoint(
    files: List[UploadFile] = File(...),
    params: str = Form(None),
):
    """
    Принимает один или несколько видеофайлов (ключ 'files'),
    для каждого выполняет анализ по настройкам из config.yaml
    и возвращает список результатов.
    """
    if not files:
        return JSONResponse(
            status_code=400,
            content={"error": "Не переданы файлы"}
        )

    print(">>> /api/video/analyze params raw:", params)

    base_det_cfg = config.detection_logic
    base_vproc_cfg = config.video_processing
    base_stab_cfg = config.stability

    det_cfg_local = base_det_cfg
    vproc_cfg_local = base_vproc_cfg
    stab_cfg_local = base_stab_cfg

    if params:
        try:
            p = json.loads(params)

            # detection_logic
            d = p.get("detection_logic", {})
            det_cfg_local = DetectionLogicConfig(
                helmet_vest_match_iou_threshold=base_det_cfg.helmet_vest_match_iou_threshold,
                person_min_area=base_det_cfg.person_min_area,
                helmet_confidence_threshold=d.get("helmet_confidence_threshold", base_det_cfg.helmet_confidence_threshold),
                vest_confidence_threshold=d.get("vest_confidence_threshold", base_det_cfg.vest_confidence_threshold),
                person_confidence_threshold=d.get("person_confidence_threshold", base_det_cfg.person_confidence_threshold),
            )

            # video_processing
            target_fps = p.get("target_fps", base_vproc_cfg.target_fps)
            max_width = p.get("max_width", base_vproc_cfg.max_width)
            vproc_cfg_local = VideoProcessingConfig(
                target_fps=target_fps,
                max_width=max_width,
            )

            # stability
            s = p.get("stability", {})
            stab_cfg_local = StabilityConfig(
                window_size_n=s.get("window_size_n", base_stab_cfg.window_size_n),
                violation_ratio_t=s.get("violation_ratio_t", base_stab_cfg.violation_ratio_t),
                min_window_for_violation=s.get("min_window_for_violation", base_stab_cfg.min_window_for_violation),
            )

        except Exception as e:
            print("Ошибка разбора params для видео:", e)
            det_cfg_local = base_det_cfg
            vproc_cfg_local = base_vproc_cfg
            stab_cfg_local = base_stab_cfg

    config_local = deepcopy(config)
    config_local.detection_logic = det_cfg_local
    config_local.video_processing = vproc_cfg_local
    config_local.stability = stab_cfg_local

    results = []

    for file in files:
        video_bytes = await file.read()

        result = analyze_video_file(
            video_bytes=video_bytes,
            config=config_local,
            person_model=person_model,
            ppe_model=ppe_model,
            original_filename=file.filename,
        )

        result["filename"] = file.filename
        results.append(result)

    return JSONResponse(content={"results": results})


@app.post("/api/video/analyze_with_overlay")
async def analyze_video_with_overlay_endpoint(
    file: UploadFile = File(...),
    params: str = Form(None),
):
    if not file:
        return JSONResponse(
            status_code=400,
            content={"error": "Файл не передан"}
        )

    try:
        video_bytes = await file.read()
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Не удалось прочитать файл: {e}"}
        )

    # Базовые конфиги
    base_det_cfg = config.detection_logic
    base_vproc_cfg = config.video_processing

    det_cfg_local = base_det_cfg
    vproc_cfg_local = base_vproc_cfg

    # Пытаемся применить параметры из фронтенда
    if params:
        try:
            p = json.loads(params)

            # detection_logic
            d = p.get("detection_logic", {})
            det_cfg_local = DetectionLogicConfig(
                helmet_vest_match_iou_threshold=base_det_cfg.helmet_vest_match_iou_threshold,
                person_min_area=base_det_cfg.person_min_area,
                helmet_confidence_threshold=d.get("helmet_confidence_threshold", base_det_cfg.helmet_confidence_threshold),
                vest_confidence_threshold=d.get("vest_confidence_threshold", base_det_cfg.vest_confidence_threshold),
                person_confidence_threshold=d.get("person_confidence_threshold", base_det_cfg.person_confidence_threshold),
            )

            # video_processing: target_fps и max_width
            target_fps = p.get("target_fps", base_vproc_cfg.target_fps)
            max_width = p.get("max_width", base_vproc_cfg.max_width)
            vproc_cfg_local = VideoProcessingConfig(
                target_fps=target_fps,
                max_width=max_width,
            )

        except Exception as e:
            print("Ошибка разбора params для видео overlay:", e)
            det_cfg_local = base_det_cfg
            vproc_cfg_local = base_vproc_cfg

    # Собираем локальный config, чтобы не менять глобальный
    config_local = deepcopy(config)
    config_local.detection_logic = det_cfg_local
    config_local.video_processing = vproc_cfg_local

    # Генерация размеченного видео
    result = generate_annotated_video(
        video_bytes=video_bytes,
        config=config_local,
        person_model=person_model,
        ppe_model=ppe_model,
    )

    if "error" in result:
        return JSONResponse(status_code=500, content=result)

    annotated_rel_url = "/output_videos/" + result["output_filename"]

    return JSONResponse(content={
        "filename": file.filename,
        "annotated_video_url": annotated_rel_url,
        "duration_sec": result.get("duration_sec"),
        "duration_str": result.get("duration_str"),
        "video_fps": result.get("video_fps"),
        "target_fps": config_local.video_processing.target_fps,
    })


@app.post("/api/video/start_analyze_with_overlay")
async def start_analyze_video_with_overlay_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    params: str = Form(None),
):
    """
    Старт фоновой генерации размеченного видео.
    Возвращает run_id сразу; клиент опрашивает /api/video/status/{run_id}
    и затем читает /api/video/result/{run_id}.
    """
    try:
        video_bytes = await file.read()
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Не удалось прочитать файл: {e}"}
        )

    input_videos_dir = os.path.join(BASE_DIR, "input_videos")
    os.makedirs(input_videos_dir, exist_ok=True)
    original_path = os.path.join(input_videos_dir, file.filename)
    try:
        with open(original_path, "wb") as f:
            f.write(video_bytes)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Не удалось сохранить видео: {e}"}
        )

    # Базовые конфиги
    base_det_cfg = config.detection_logic
    base_vproc_cfg = config.video_processing

    det_cfg_local = base_det_cfg
    vproc_cfg_local = base_vproc_cfg

    # Пытаемся применить параметры из фронтенда
    if params:
        try:
            p = json.loads(params)

            d = p.get("detection_logic", {})
            det_cfg_local = DetectionLogicConfig(
                helmet_vest_match_iou_threshold=base_det_cfg.helmet_vest_match_iou_threshold,
                person_min_area=base_det_cfg.person_min_area,
                helmet_confidence_threshold=d.get("helmet_confidence_threshold", base_det_cfg.helmet_confidence_threshold),
                vest_confidence_threshold=d.get("vest_confidence_threshold", base_det_cfg.vest_confidence_threshold),
                person_confidence_threshold=d.get("person_confidence_threshold", base_det_cfg.person_confidence_threshold),
            )

            target_fps = p.get("target_fps", base_vproc_cfg.target_fps)
            max_width = p.get("max_width", base_vproc_cfg.max_width)
            vproc_cfg_local = VideoProcessingConfig(
                target_fps=target_fps,
                max_width=max_width,
            )
        except Exception as e:
            print("Ошибка разбора params в /api/video/start_analyze_with_overlay:", e)
            det_cfg_local = base_det_cfg
            vproc_cfg_local = base_vproc_cfg

    config_local = deepcopy(config)
    config_local.detection_logic = det_cfg_local
    config_local.video_processing = vproc_cfg_local

    video_source_id = create_video_source_for_file(file.filename, original_path)
    run_id = create_processing_run(video_source_id, "file_overlay", config_local)

    background_tasks.add_task(
        analyze_video_overlay_run_background,
        run_id=run_id,
        original_path=original_path,
        config_local=config_local,
        person_model=person_model,
        ppe_model=ppe_model,
    )

    return {"run_id": run_id, "status": "queued"}


@app.post("/api/video/start_analyze")
async def start_analyze_video_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    params: str = Form(None),
):
    """
    Старт фоновой обработки одного видеофайла.
    HTTP-ответ возвращает run_id сразу, не дожидаясь конца анализа.
    """
    try:
        video_bytes = await file.read()
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Не удалось прочитать файл: {e}"}
        )
    input_videos_dir = os.path.join(BASE_DIR, "input_videos")
    os.makedirs(input_videos_dir, exist_ok=True)
    original_path = os.path.join(input_videos_dir, file.filename)
    try:
        with open(original_path, "wb") as f:
            f.write(video_bytes)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Не удалось сохранить видео: {e}"}
        )

    # 2. Регистрируем источник видео и run в БД
    video_source_id = create_video_source_for_file(file.filename, original_path)
    run_id = create_processing_run(video_source_id, "file_analyze", config)

    # 3. Формируем локальный config_local из params
    base_det_cfg = config.detection_logic
    base_vproc_cfg = config.video_processing
    base_stab_cfg = config.stability

    det_cfg_local = base_det_cfg
    vproc_cfg_local = base_vproc_cfg
    stab_cfg_local = base_stab_cfg

    if params:
        try:
            p = json.loads(params)

            # detection_logic
            d = p.get("detection_logic", {})
            det_cfg_local = DetectionLogicConfig(
                helmet_vest_match_iou_threshold=base_det_cfg.helmet_vest_match_iou_threshold,
                person_min_area=base_det_cfg.person_min_area,
                helmet_confidence_threshold=d.get("helmet_confidence_threshold", base_det_cfg.helmet_confidence_threshold),
                vest_confidence_threshold=d.get("vest_confidence_threshold", base_det_cfg.vest_confidence_threshold),
                person_confidence_threshold=d.get("person_confidence_threshold", base_det_cfg.person_confidence_threshold),
            )

            # video_processing
            target_fps = p.get("target_fps", base_vproc_cfg.target_fps)
            max_width = p.get("max_width", base_vproc_cfg.max_width)
            vproc_cfg_local = VideoProcessingConfig(
                target_fps=target_fps,
                max_width=max_width,
            )

            # stability
            s = p.get("stability", {})
            stab_cfg_local = StabilityConfig(
                window_size_n=s.get("window_size_n", base_stab_cfg.window_size_n),
                violation_ratio_t=s.get("violation_ratio_t", base_stab_cfg.violation_ratio_t),
                min_window_for_violation=s.get("min_window_for_violation", base_stab_cfg.min_window_for_violation),
            )
        except Exception as e:
            print("Ошибка разбора params в /api/video/start_analyze:", e)
            det_cfg_local = base_det_cfg
            vproc_cfg_local = base_vproc_cfg
            stab_cfg_local = base_stab_cfg

    config_local = deepcopy(config)
    config_local.detection_logic = det_cfg_local
    config_local.video_processing = vproc_cfg_local
    config_local.stability = stab_cfg_local

    # 4. Ставим задачу в фон.
    #    В tasks_processing.analyze_video_run_background передаём всё необходимое.
    background_tasks.add_task(
        analyze_video_run_background,
        run_id=run_id,
        original_path=original_path,
        config_local=config_local,
        person_model=person_model,
        ppe_model=ppe_model,
        original_filename=file.filename,
    )

    # 5. Сразу отвечаем клиенту
    return {"run_id": run_id, "status": "queued"}


# Эндпоинт: статус запуска
@app.get("/api/video/status/{run_id}")
def get_video_status(run_id: int):
    run = get_processing_run(run_id)
    if not run:
        return JSONResponse(status_code=404, content={"error": "run not found"})

    return {
        "run_id": run.id,
        "status": run.status,
        "error_message": run.error_message,
        "duration_sec": run.duration_sec,
        "target_fps": run.target_fps,
        "video_fps": run.video_fps,
        "total_frames": run.total_frames,
        "processed_frames": run.processed_frames,
        "frames_to_process": run.frames_to_process,
    }


@app.get("/api/video/result/{run_id}")
def get_video_result(run_id: int):
    """
    Возвращает детальный результат анализа видео по run_id,
    в формате, близком к /api/video/analyze (один объект result).
    """
    run = get_video_run_result(run_id)
    if not run:
        return JSONResponse(status_code=404, content={"error": "run not found"})

    # Вспомогательная функция перевода секунд в HH:MM:SS
    def _sec_to_hms(sec: float) -> str:
        try:
            total = int(sec)
        except (TypeError, ValueError):
            return "?"
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    duration_str = _sec_to_hms(run.duration_sec or 0.0)

    # Устойчивые нарушения
    viols = get_stable_violations_for_run(run_id)
    stable_violations = []

    for v in viols:
        frame_image_url = None
        if v.frame_image_path:
            filename = os.path.basename(v.frame_image_path)
            frame_image_url = "/violations_frames/" + filename

        stable_violations.append({
            "trackid": v.track_index,
            "frameindex": v.frame_index,
            "timesec": float(v.time_sec) if v.time_sec is not None else None,
            "timestr": v.time_str or _sec_to_hms(v.time_sec or 0.0),
            "windowsize": v.window_size,
            "violationsinwindow": v.violations_in_window,
            "violationratio": float(v.violation_ratio) if v.violation_ratio is not None else None,
            "statusnote": v.status_note,
            "violationtype": v.violation_type,
            "frameimageurl": frame_image_url,
        })

    res = {
        "runid": run.id,
        "filename": run.result_video_path or f"run{run.id}",
        "status": run.status,
        "processingtype": run.processing_type,
        "error": run.error_message,
        "durationsec": float(run.duration_sec) if run.duration_sec is not None else None,
        "durationstr": duration_str,
        "targetfps": float(run.target_fps) if run.target_fps is not None else None,
        "videofps": float(run.video_fps) if run.video_fps is not None else None,
        "totalframes": run.total_frames,
        "processedframes": run.processed_frames,
        "maxwidth": run.max_width,
        "stableviolationsbyperson": stable_violations,
        "processingparams": run.processing_params,
    }

    if run.result_video_path:
        res["annotatedvideourl"] = "/output_videos/" + os.path.basename(run.result_video_path)
    else:
        res["annotatedvideourl"] = None

    if run.processing_params and isinstance(run.processing_params, dict):
        ftp = run.processing_params.get("frames_to_process")
        if ftp is not None:
            res["framestoprocess"] = int(ftp)

    return {"result": res}

@app.get("/api/cameras")
def api_get_cameras():
    """
    Возвращает список актуальных камер (type='camera', camera_deleted=FALSE)
    с флагом enabled и состоянием is_processing_now (обрабатывается ли сейчас воркером).
    """
    try:
        cameras = get_all_cameras()  # камеры из video_sources, camera_deleted=FALSE
        processing_states = get_camera_processing_states()  # video_source_id -> bool

        items = []
        for cam in cameras:
            is_processing = bool(processing_states.get(cam.id, False))
            items.append({
                "id": cam.id,
                "external_id": cam.external_id,
                "name": cam.name,
                "location": cam.location,
                "type": cam.type,
                "connection_type": cam.connection_type,
                "enabled": bool(cam.is_active),
                "is_processing_now": is_processing,
                "target_fps": cam.target_fps,
                "camera_params": cam.camera_params or {},
            })
        return {"cameras": items}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Ошибка чтения списка камер: {e}"},
        )

@app.get("/api/cameras/{camera_id}/violations")
def api_get_camera_violations(
    camera_id: int,
    from_ts: str = Query(..., alias="from"),
    to_ts: str = Query(..., alias="to"),
):
    """
    Отчёт по нарушениям с камеры за период [from, to].

    Параметры:
      camera_id — id камеры (video_source_id)
      from, to — ISO-строки дат/времени, например:
                 2026-05-12T00:00:00Z или 2026-05-12T00:00
    """
    try:
        # Приводим ISO-строки к datetime, поддерживаем варианты с 'Z'
        def parse_iso(s: str) -> datetime:
            s = s.strip()
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            return datetime.fromisoformat(s)

        from_dt = parse_iso(from_ts)
        to_dt = parse_iso(to_ts)

        violations = get_camera_violations_for_period(camera_id, from_dt, to_dt)

        return {"violations": violations}
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": "Некорректный формат дат. Используйте ISO-строки, например 2026-05-12T00:00:00Z или 2026-05-12T00:00"},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Ошибка при формировании отчёта по нарушениям: {e}"},
        )

if __name__ == "__main__":
    uvicorn.run("main_server:app", host="0.0.0.0", port=8000, workers=4)