# core/db_pg.py
import psycopg2
from psycopg2.extras import Json
from typing import Optional, List, Dict, Any
from . import db_config
import time
from datetime import datetime, timezone

class ProcessingRun:
    def __init__(
        self,
        id: int,
        status: str,
        error_message: Optional[str],
        duration_sec: Optional[float],
        target_fps: Optional[float],
        video_fps: Optional[float],
        total_frames: Optional[int],
        processed_frames: Optional[int],
        frames_to_process: Optional[int] = None,
    ):
        self.id = id
        self.status = status
        self.error_message = error_message
        self.duration_sec = duration_sec
        self.target_fps = target_fps
        self.video_fps = video_fps
        self.total_frames = total_frames
        self.processed_frames = processed_frames
        self.frames_to_process = frames_to_process

class VideoRunResult:
    def __init__(
        self,
        id: int,
        video_source_id: int,
        status: str,
        processing_type: str,
        error_message: Optional[str],
        duration_sec: Optional[float],
        target_fps: Optional[float],
        video_fps: Optional[float],
        total_frames: Optional[int],
        processed_frames: Optional[int],
        max_width: Optional[int],
        processing_params: Optional[dict],
        result_video_path: Optional[str],
    ):
        self.id = id
        self.video_source_id = video_source_id
        self.status = status
        self.processing_type = processing_type
        self.error_message = error_message
        self.duration_sec = duration_sec
        self.target_fps = target_fps
        self.video_fps = video_fps
        self.total_frames = total_frames
        self.processed_frames = processed_frames
        self.max_width = max_width
        self.processing_params = processing_params
        self.result_video_path = result_video_path


class StableViolationRecord:
    def __init__(
        self,
        id: int,
        track_index: Optional[int],
        violation_type: str,
        frame_index: int,
        time_sec: float,
        time_str: str,
        window_size: int,
        violations_in_window: int,
        violation_ratio: float,
        status_note: Optional[str],
        frame_image_path: Optional[str],
    ):
        self.id = id
        self.track_index = track_index
        self.violation_type = violation_type
        self.frame_index = frame_index
        self.time_sec = time_sec
        self.time_str = time_str
        self.window_size = window_size
        self.violations_in_window = violations_in_window
        self.violation_ratio = violation_ratio
        self.status_note = status_note
        self.frame_image_path = frame_image_path

class VideoSource:
    def __init__(
        self,
        id: int,
        type: str,
        name: str,
        file_path: Optional[str],
        camera_url: Optional[str],
        location: Optional[str],
        connection_type: Optional[str],
        external_id: Optional[str],
        is_active: bool,
        target_fps: Optional[float],
        camera_params: Optional[dict],
    ):
        self.id = id
        self.type = type
        self.name = name
        self.file_path = file_path
        self.camera_url = camera_url
        self.location = location
        self.connection_type = connection_type
        self.external_id = external_id
        self.is_active = is_active
        self.target_fps = target_fps
        self.camera_params = camera_params

class StreamRun:
    def __init__(
        self,
        id: int,
        video_source_id: int,
        status: str,
        processing_type: str,
        processing_params: Optional[dict],
        target_fps: Optional[float],
        max_width: Optional[int],
        error_message: Optional[str],
    ):
        self.id = id
        self.video_source_id = video_source_id
        self.status = status
        self.processing_type = processing_type
        self.processing_params = processing_params
        self.target_fps = target_fps
        self.max_width = max_width
        self.error_message = error_message

def get_conn():
    return psycopg2.connect(
        host=db_config.DB_HOST,
        port=db_config.DB_PORT,
        dbname=db_config.DB_NAME,
        user=db_config.DB_USER,
        password=db_config.DB_PASSWORD
    )

def get_processing_run(run_id: int) -> Optional[ProcessingRun]:
    """
    Возвращает информацию о запуске анализа видео из таблицы video_processing_runs.
    Если запись не найдена — возвращает None.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id,
                       status,
                       error_message,
                       duration_sec,
                       target_fps,
                       video_fps,
                       total_frames,
                       processed_frames,
                       (processing_params->>'frames_to_process')::int AS frames_to_process
                FROM video_processing_runs
                WHERE id = %s
                """,
                (run_id,),
            )
            row = cur.fetchone()
            if not row:
                return None

            return ProcessingRun(
                id=row[0],
                status=row[1],
                error_message=row[2],
                duration_sec=row[3],
                target_fps=row[4],
                video_fps=row[5],
                total_frames=row[6],
                processed_frames=row[7],
                frames_to_process=row[8],
            )
    finally:
        conn.close()

def create_video_source_for_file(filename: str, file_path: Optional[str]) -> int:
    """
    Создаёт запись об источнике типа 'file'.
    file_path может быть None, если исходное видео не сохраняется на диск.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO video_sources (type, name, file_path)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                ("file", filename, file_path)
            )
            video_source_id = cur.fetchone()[0]
        conn.commit()
        return video_source_id
    finally:
        conn.close()


def create_processing_run(
    video_source_id: int,
    processing_type: str,
    config
) -> int:
    """
    Создаёт запись в video_processing_runs со статусом 'processing'
    и сохраняет ключевые параметры запуска в JSONB processing_params.
    """
    params = {
        "helmet_thr": config.detection_logic.helmet_confidence_threshold,
        "vest_thr": config.detection_logic.vest_confidence_threshold,
        "person_thr": config.detection_logic.person_confidence_threshold,
        "window_size_n": config.stability.window_size_n,
        "violation_ratio_t": config.stability.violation_ratio_t,
        "min_window_for_violation": config.stability.min_window_for_violation,
        "tracking_iou_thr": config.tracking.iou_threshold,
        "max_track_lost_frames": config.tracking.max_track_lost_frames,
    }

    # Гарантируем, что в target_fps и max_width пойдут числа, а не сложные объекты
    try:
        target_fps = float(config.video_processing.target_fps)
    except Exception:
        target_fps = None

    try:
        max_width = int(config.video_processing.max_width)
    except Exception:
        max_width = None

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO video_processing_runs (
                    video_source_id,
                    status,
                    processing_type,
                    target_fps,
                    max_width,
                    processing_params
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    video_source_id,
                    "processing",
                    processing_type,
                    target_fps,
                    max_width,
                    Json(params),   # JSONB
                )
            )
            run_id = cur.fetchone()[0]
        conn.commit()
        return run_id
    finally:
        conn.close()


def finish_processing_run(
    run_id: int,
    duration_sec: float,
    video_fps: float,
    total_frames: int,
    processed_frames: int,
    result_video_path: str | None,
    error_message: str | None,
):
    """
    Завершает запуск: проставляет finished_at, статус ('done' или 'error')
    и итоговую статистику в video_processing_runs.
    """
    status = "error" if error_message else "done"

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE video_processing_runs
                SET
                    finished_at      = NOW(),
                    status           = %s,
                    duration_sec     = %s,
                    video_fps        = %s,
                    total_frames     = %s,
                    processed_frames = %s,
                    result_video_path = %s,
                    error_message    = %s
                WHERE id = %s
                """,
                (
                    status,
                    duration_sec,
                    video_fps,
                    total_frames,
                    processed_frames,
                    result_video_path,
                    error_message,
                    run_id,
                )
            )
        conn.commit()
    finally:
        conn.close()

def update_processing_run_progress(
    run_id: int,
    processed_frames: int,
    total_frames: int | None = None,
    frames_to_process: int | None = None,
):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            merge = {}
            if frames_to_process is not None:
                merge["frames_to_process"] = frames_to_process
            cur.execute(
                """
                UPDATE video_processing_runs
                SET
                    processed_frames = %s,
                    total_frames     = COALESCE(total_frames, %s),
                    processing_params = COALESCE(processing_params, '{}'::jsonb) || %s::jsonb
                WHERE id = %s
                """,
                (processed_frames, total_frames, Json(merge), run_id)
            )
        conn.commit()
    finally:
        conn.close()


def insert_violation(
    run_id: int,
    track_index: int,
    violation_type: str,
    frame_index: int,
    time_sec: float,
    time_str: str,
    window_size: int,
    violations_in_window: int,
    violation_ratio: float,
    status_note: str,
    frame_image_path: str,
    frame_image_url: str
):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO violations (
                    video_processing_run_id,
                    track_index,
                    violation_type,
                    frame_index,
                    time_sec,
                    time_str,
                    window_size,
                    violations_in_window,
                    violation_ratio,
                    status_note,
                    frame_image_path,
                    frame_image_url
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    run_id,
                    track_index,
                    violation_type,
                    frame_index,
                    time_sec,
                    time_str,
                    window_size,
                    violations_in_window,
                    violation_ratio,
                    status_note,
                    frame_image_path,
                    frame_image_url
                )
            )
        conn.commit()
    finally:
        conn.close()

def get_video_run_result(run_id: int) -> Optional[VideoRunResult]:
    """
    Читает одну запись из video_processing_runs по id.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id,
                       video_source_id,
                       status,
                       processing_type,
                       error_message,
                       duration_sec,
                       target_fps,
                       video_fps,
                       total_frames,
                       processed_frames,
                       max_width,
                       processing_params,
                       result_video_path
                FROM video_processing_runs
                WHERE id = %s
                """,
                (run_id,),
            )
            row = cur.fetchone()
            if not row:
                return None

            return VideoRunResult(
                id=row[0],
                video_source_id=row[1],
                status=row[2],
                processing_type=row[3],
                error_message=row[4],
                duration_sec=row[5],
                target_fps=row[6],
                video_fps=row[7],
                total_frames=row[8],
                processed_frames=row[9],
                max_width=row[10],
                processing_params=row[11],
                result_video_path=row[12],
                )
    finally:
        conn.close()

def get_stable_violations_for_run(run_id: int) -> List[StableViolationRecord]:
    """
    Возвращает список устойчивых нарушений из таблицы violations
    для конкретного video_processing_run_id.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id,
                       track_index,
                       violation_type,
                       frame_index,
                       time_sec,
                       time_str,
                       window_size,
                       violations_in_window,
                       violation_ratio,
                       status_note,
                       frame_image_path
                FROM violations
                WHERE video_processing_run_id = %s
                ORDER BY time_sec
                """,
                (run_id,),
            )
            rows = cur.fetchall() or []
            items: List[StableViolationRecord] = []
            for row in rows:
                items.append(
                    StableViolationRecord(
                        id=row[0],
                        track_index=row[1],
                        violation_type=row[2],
                        frame_index=row[3],
                        time_sec=row[4],
                        time_str=row[5],
                        window_size=row[6],
                        violations_in_window=row[7],
                        violation_ratio=row[8],
                        status_note=row[9],
                        frame_image_path=row[10],
                    )
                )
            return items
    finally:
        conn.close()

def upsert_video_source_from_yaml(
    external_id: str,
    name: str,
    connection_type: str,
    url: str,
    location: Optional[str],
    enabled: bool,
    target_fps: Optional[float],
    camera_params: Optional[dict],
) -> int:
    """
    Создаёт или обновляет запись в video_sources по external_id (из cameras.yaml).

    При обновлении:
      - camera_deleted = FALSE
      - is_active = enabled

    При вставке:
      - camera_deleted = FALSE
      - is_active = enabled
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM video_sources WHERE external_id = %s",
                (external_id,)
            )
            row = cur.fetchone()

            params_json = Json(camera_params) if camera_params is not None else None

            if row:
                video_source_id = row[0]
                cur.execute(
                    """
                    UPDATE video_sources
                    SET
                        name             = %s,
                        type             = 'camera',
                        camera_url       = %s,
                        location         = %s,
                        connection_type  = %s,
                        is_active        = %s,
                        target_fps       = %s,
                        camera_params    = %s,
                        camera_deleted   = FALSE
                    WHERE id = %s
                    """,
                    (
                        name,
                        url,
                        location,
                        connection_type,
                        enabled,
                        target_fps,
                        params_json,
                        video_source_id,
                    )
                )
            else:
                cur.execute(
                    """
                    INSERT INTO video_sources (
                        external_id,
                        type,
                        name,
                        camera_url,
                        location,
                        connection_type,
                        is_active,
                        target_fps,
                        camera_params,
                        camera_deleted
                    )
                    VALUES (
                        %s,            -- external_id
                        'camera',      -- type
                        %s,            -- name
                        %s,            -- camera_url
                        %s,            -- location
                        %s,            -- connection_type
                        %s,            -- is_active
                        %s,            -- target_fps
                        %s,            -- camera_params
                        FALSE          -- camera_deleted
                    )
                    RETURNING id
                    """,
                    (
                        external_id,
                        name,
                        url,
                        location,
                        connection_type,
                        enabled,
                        target_fps,
                        params_json,
                    )
                )
                video_source_id = cur.fetchone()[0]

        conn.commit()
        return video_source_id
    finally:
        conn.close()


def get_video_source(video_source_id: int) -> Optional[VideoSource]:
    """
    Возвращает объект VideoSource по id из таблицы video_sources.
    Если запись не найдена — возвращает None.

    Ожидается, что в video_sources есть поля:
      id, type, name, file_path, camera_url, location,
      connection_type, external_id, is_active, target_fps, camera_params
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    type,
                    name,
                    file_path,
                    camera_url,
                    location,
                    connection_type,
                    external_id,
                    is_active,
                    target_fps,
                    camera_params
                FROM video_sources
                WHERE id = %s
                """,
                (video_source_id,)
            )
            row = cur.fetchone()
            if not row:
                return None

            # row[10] = camera_params (JSONB) -> psycopg2 вернёт dict или None
            return VideoSource(
                id=row[0],
                type=row[1],
                name=row[2],
                file_path=row[3],
                camera_url=row[4],
                location=row[5],
                connection_type=row[6],
                external_id=row[7],
                is_active=row[8],
                target_fps=row[9],
                camera_params=row[10],
            )
    finally:
        conn.close()

def get_active_stream_runs() -> List[StreamRun]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    video_source_id,
                    status,
                    processing_type,
                    processing_params,
                    target_fps,
                    max_width,
                    error_message
                FROM video_processing_runs
                WHERE processing_type = 'stream'
                  AND status = 'processing'
                ORDER BY id
                """
            )
            rows = cur.fetchall() or []
            runs: List[StreamRun] = []
            for row in rows:
                runs.append(
                    StreamRun(
                        id=row[0],
                        video_source_id=row[1],
                        status=row[2],
                        processing_type=row[3],
                        processing_params=row[4],
                        target_fps=row[5],
                        max_width=row[6],
                        error_message=row[7],
                    )
                )
            return runs
    finally:
        conn.close()


def mark_run_as_taken(run_id: int) -> bool:
    """
    Пытается "захватить" run для обработки воркером.
    Меняет статус с 'processing' на 'processing_in_worker'.
    Возвращает True, если захват успешен (строка обновлена).
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE video_processing_runs
                SET status = 'processing_in_worker'
                WHERE id = %s
                  AND status = 'processing'
                """,
                (run_id,)
            )
            updated = cur.rowcount
        conn.commit()
        return updated == 1
    finally:
        conn.close()

def get_active_cameras() -> List[VideoSource]:
    """
    Возвращает все активные камеры из video_sources (type='camera', is_active=true).
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    type,
                    name,
                    file_path,
                    camera_url,
                    location,
                    connection_type,
                    external_id,
                    is_active,
                    target_fps,
                    camera_params
                FROM video_sources
                WHERE type = 'camera'
                  AND is_active = TRUE
                ORDER BY id
                """
            )
            rows = cur.fetchall() or []
            result: List[VideoSource] = []
            for row in rows:
                result.append(
                    VideoSource(
                        id=row[0],
                        type=row[1],
                        name=row[2],
                        file_path=row[3],
                        camera_url=row[4],
                        location=row[5],
                        connection_type=row[6],
                        external_id=row[7],
                        is_active=row[8],
                        target_fps=row[9],
                        camera_params=row[10],
                    )
                )
            return result
    finally:
        conn.close()

def has_active_stream_run(video_source_id: int) -> bool:
    """
    Проверяет, есть ли для данной камеры (video_source_id)
    активный stream-запуск:
      processing_type = 'stream'
      статус в ('processing', 'processing_in_worker')
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM video_processing_runs
                WHERE video_source_id = %s
                  AND processing_type = 'stream'
                  AND status IN ('processing', 'processing_in_worker')
                LIMIT 1
                """,
                (video_source_id,)
            )
            row = cur.fetchone()
            return row is not None
    finally:
        conn.close()

def create_stream_run_if_not_exists(
    video_source_id: int,
    config
) -> Optional[int]:
    """
    Атомарно создаёт stream-запуск для камеры, если ещё нет активного stream-run'а.
    Активный: processing_type='stream', status IN ('processing', 'processing_in_worker').

    Возвращает run_id, если run был создан, либо None, если активный run уже существует.
    """
    
    ERROR_RETRY_INTERVAL_SEC = 30

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # 0) Смотрим последний stream-run для камеры
            cur.execute(
                """
                SELECT id, status, started_at, finished_at
                FROM video_processing_runs
                WHERE video_source_id = %s
                  AND processing_type = 'stream'
                ORDER BY id DESC
                LIMIT 1
                """,
                (video_source_id,)
            )
            last = cur.fetchone()
            if last:
                last_id, last_status, last_created_at, last_finished_at = last

                # Если последний run завершился ошибкой — проверяем, не слишком ли свежая ошибка
                if last_status == 'error':
                    # Берём время окончания, если есть, иначе созидание
                    t_ref = last_finished_at or last_created_at
                    if t_ref is not None:
                        # если t_ref timezone-aware:
                        if getattr(t_ref, "tzinfo", None) is not None:
                            last_err_ts = t_ref.timestamp()
                        else:
                            # считаем, что это UTC naive
                            last_err_ts = t_ref.replace(tzinfo=timezone.utc).timestamp()
                        now_ts = time.time()
                        if now_ts - last_err_ts < ERROR_RETRY_INTERVAL_SEC:
                            # Слишком рано пытаться снова -> не создаём новый run
                            conn.commit()
                            return None

            # 1) Проверяем наличие активного stream-run'а (processing или уже в воркере)
            cur.execute(
                """
                SELECT id
                FROM video_processing_runs
                WHERE video_source_id = %s
                  AND processing_type = 'stream'
                  AND status IN ('processing', 'processing_in_worker')
                LIMIT 1
                """,
                (video_source_id,)
            )
            row = cur.fetchone()
            if row:
                conn.commit()
                return None

            # 2) Активного нет -> создаём новый
            params = {
                "helmet_thr": config.detection_logic.helmet_confidence_threshold,
                "vest_thr": config.detection_logic.vest_confidence_threshold,
                "person_thr": config.detection_logic.person_confidence_threshold,
                "window_size_n": config.stability.window_size_n,
                "violation_ratio_t": config.stability.violation_ratio_t,
                "min_window_for_violation": config.stability.min_window_for_violation,
                "tracking_iou_thr": config.tracking.iou_threshold,
                "max_track_lost_frames": config.tracking.max_track_lost_frames,
            }

            try:
                target_fps = float(config.video_processing.target_fps)
            except Exception:
                target_fps = None

            try:
                max_width = int(config.video_processing.max_width)
            except Exception:
                max_width = None

            cur.execute(
                """
                INSERT INTO video_processing_runs (
                    video_source_id,
                    status,
                    processing_type,
                    target_fps,
                    max_width,
                    processing_params
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    video_source_id,
                    "processing",
                    "stream",
                    target_fps,
                    max_width,
                    Json(params),
                )
            )
            run_id = cur.fetchone()[0]

        conn.commit()
        return run_id
    finally:
        conn.close()

def get_all_cameras() -> List[VideoSource]:
    """
    Возвращает все камеры, которые сейчас есть в cameras.yaml:
      type = 'camera'
      camera_deleted = FALSE
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    type,
                    name,
                    file_path,
                    camera_url,
                    location,
                    connection_type,
                    external_id,
                    is_active,
                    target_fps,
                    camera_params,
                    camera_deleted
                FROM video_sources
                WHERE type = 'camera'
                  AND camera_deleted = FALSE
                ORDER BY id
                """
            )
            rows = cur.fetchall() or []
            result: List[VideoSource] = []
            for row in rows:
                result.append(
                    VideoSource(
                        id=row[0],
                        type=row[1],
                        name=row[2],
                        file_path=row[3],
                        camera_url=row[4],
                        location=row[5],
                        connection_type=row[6],
                        external_id=row[7],
                        is_active=row[8],
                        target_fps=row[9],
                        camera_params=row[10],
                    )
                )
            return result
    finally:
        conn.close()

def get_camera_processing_states() -> Dict[int, bool]:
    """
    Возвращает словарь video_source_id -> is_processing_now (bool) для камер.

    Камера считается "обрабатываемой сейчас", если есть хотя бы один
    stream-run со статусом 'processing_in_worker' для её video_source_id.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT video_source_id
                FROM video_processing_runs
                WHERE processing_type = 'stream'
                  AND status = 'processing_in_worker'
                """
            )
            rows = cur.fetchall() or []
            states: Dict[int, bool] = {}
            for row in rows:
                vs_id = row[0]
                states[vs_id] = True
            return states
    finally:
        conn.close()

def reset_stuck_stream_runs() -> int:
    """
    Переводит stream-run'ы, которые зависли в статусе 'processing_in_worker',
    обратно в 'processing', чтобы их мог подхватить воркер при следующем запуске.

    Возвращает количество обновлённых строк.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE video_processing_runs
                SET status = 'processing'
                WHERE processing_type = 'stream'
                  AND status = 'processing_in_worker'
                """
            )
            updated = cur.rowcount
        conn.commit()
        return updated
    finally:
        conn.close()

def deactivate_cameras_not_in_external_ids(active_external_ids: List[str]) -> int:
    """
    Помечает как неактивные (is_active = false) все камеры (type='camera'),
    у которых external_id отсутствует в переданном списке active_external_ids.

    Возвращает количество обновлённых строк.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE video_sources
                SET is_active = FALSE
                WHERE type = 'camera'
                  AND (external_id IS NULL OR external_id <> ALL(%s))
                """,
                (active_external_ids or ["__none__"],)  # если список пуст, чтобы не упасть
            )
            updated = cur.rowcount
        conn.commit()
        return updated
    finally:
        conn.close()

def delete_cameras_not_in_external_ids(active_external_ids: List[str]) -> int:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM video_sources
                WHERE type = 'camera'
                  AND (external_id IS NULL OR external_id <> ALL(%s))
                """,
                (active_external_ids or ["__none__"],)
            )
            deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()

def mark_cameras_deleted_not_in_external_ids(active_external_ids: List[str]) -> int:
    """
    Помечает камеры (type='camera') как удалённые из YAML,
    если их external_id нет в списке active_external_ids.

    Устанавливает:
      camera_deleted = TRUE
      is_active = FALSE

    Возвращает количество обновлённых строк.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE video_sources
                SET
                    camera_deleted = TRUE,
                    is_active = FALSE
                WHERE type = 'camera'
                  AND (external_id IS NULL OR external_id <> ALL(%s))
                """,
                (active_external_ids or ["__none__"],)
            )
            updated = cur.rowcount
        conn.commit()
        return updated
    finally:
        conn.close()

def reset_stream_run_to_processing(run_id: int) -> bool:
    """
    Переводит stream-run из 'processing_in_worker' обратно в 'processing'
    (используется при нештатной остановке воркера — KeyboardInterrupt).

    Возвращает True, если статус был изменён.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE video_processing_runs
                SET status = 'processing'
                WHERE id = %s
                  AND processing_type = 'stream'
                  AND status = 'processing_in_worker'
                """,
                (run_id,)
            )
            updated = cur.rowcount
        conn.commit()
        return updated == 1
    finally:
        conn.close()

def get_camera_violations_for_period(
    video_source_id: int,
    from_dt: datetime,
    to_dt: datetime,
) -> List[Dict[str, Any]]:
    """
    Возвращает список нарушений по камере за период [from_dt, to_dt]
    по полю violations.created_at.

    video_source_id — это id из video_sources.id.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    v.id,
                    r.video_source_id,
                    v.video_processing_run_id,
                    v.violation_type,
                    v.frame_index,
                    v.time_sec,
                    v.track_index,
                    v.frame_image_url,
                    v.created_at
                FROM violations v
                JOIN video_processing_runs r
                  ON v.video_processing_run_id = r.id
                WHERE r.video_source_id = %s
                  AND v.created_at >= %s
                  AND v.created_at <= %s
                  AND r.processing_type = 'stream'
                ORDER BY v.created_at DESC, v.id
                """,
                (video_source_id, from_dt, to_dt)
            )
            rows = cur.fetchall() or []

            result: List[Dict[str, Any]] = []
            for row in rows:
                result.append({
                    "id": row[0],
                    "video_source_id": row[1],
                    "run_id": row[2],
                    "violation_type": row[3],
                    "frame_index": row[4],
                    "time_sec": row[5],
                    "track_index": row[6],
                    "frame_image_url": row[7],
                    "created_at": row[8].isoformat() if row[8] else None,
                })
            return result
    finally:
        conn.close()

def get_run_status(run_id: int) -> Optional[str]:
    """
    Возвращает статус run'а по id или None, если не найден.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status FROM video_processing_runs WHERE id = %s",
                (run_id,)
            )
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()

def get_active_cameras_count() -> int:
    """
    Возвращает количество активных камер:
      type = 'camera'
      is_active = TRUE
      camera_deleted = FALSE
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM video_sources
                WHERE type = 'camera'
                  AND is_active = TRUE
                  AND camera_deleted = FALSE
                """
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0
    finally:
        conn.close()