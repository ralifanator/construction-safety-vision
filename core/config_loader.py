# core/config_loader.py
import yaml
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ModelConfig:
    path: str
    confidence_threshold: float
    nms_iou_threshold: float


@dataclass
class StreamConfig:
    max_run_duration_sec: float


@dataclass
class DetectionLogicConfig:
    helmet_vest_match_iou_threshold: float
    person_min_area: int
    helmet_confidence_threshold: float
    vest_confidence_threshold: float
    person_confidence_threshold: float


@dataclass
class VideoProcessingConfig:
    target_fps: float
    max_width: int


@dataclass
class StabilityConfig:
    window_size_n: int
    violation_ratio_t: float
    min_window_for_violation: int


@dataclass
class TrackingConfig:
    iou_threshold: float
    max_track_lost_frames: int


@dataclass
class AppConfig:
    person_detector: ModelConfig
    ppe_detector: ModelConfig
    detection_logic: DetectionLogicConfig
    video_processing: VideoProcessingConfig
    stability: StabilityConfig
    tracking: TrackingConfig
    stream: StreamConfig


def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    m_person = raw["models"]["person_detector"]
    m_ppe = raw["models"]["ppe_detector"]
    d_logic = raw["detection_logic"]
    v_proc = raw.get("video_processing", {})
    stab = raw.get("stability", {})
    track = raw.get("tracking", {})
    stream_raw = raw.get("stream", {})
    stream_cfg = StreamConfig(
        max_run_duration_sec=stream_raw.get("max_run_duration_sec", 3600.0),
    )

    person_cfg = ModelConfig(
        path=m_person["path"],
        confidence_threshold=None,
        nms_iou_threshold=m_person.get("nms_iou_threshold", 0.45),
    )

    ppe_cfg = ModelConfig(
        path=m_ppe["path"],
        confidence_threshold=None,
        nms_iou_threshold=m_ppe.get("nms_iou_threshold", 0.45),
    )

    det_logic_cfg = DetectionLogicConfig(
        helmet_vest_match_iou_threshold=d_logic.get("helmet_vest_match_iou_threshold", 0.3),
        person_min_area=d_logic.get("person_min_area", 500),
        helmet_confidence_threshold=d_logic.get("helmet_confidence_threshold", 0.5),
        vest_confidence_threshold=d_logic.get("vest_confidence_threshold", 0.5),
        person_confidence_threshold=d_logic.get("person_confidence_threshold", 0.4),
    )

    video_proc_cfg = VideoProcessingConfig(
        target_fps=v_proc.get("target_fps", 1.0),
        max_width=v_proc.get("max_width", 960),
    )

    stability_cfg = StabilityConfig(
    window_size_n=stab.get("window_size_n", 10),
    violation_ratio_t=stab.get("violation_ratio_t", 0.6),
    min_window_for_violation=stab.get("min_window_for_violation", 3),
    )

    tracking_cfg = TrackingConfig(
        iou_threshold=track.get("iou_threshold", 0.4),
        max_track_lost_frames=track.get("max_track_lost_frames", 20),
    )

    return AppConfig(
        person_detector=person_cfg,
        ppe_detector=ppe_cfg,
        detection_logic=det_logic_cfg,
        video_processing=video_proc_cfg,
        stability=stability_cfg,
        tracking=tracking_cfg,
        stream=stream_cfg,
    )


def build_config_for_source(global_config: AppConfig, source: Any) -> AppConfig:
    """
    Формирует AppConfig для конкретного источника (камеры) на основе:
      - глобального config.yaml (global_config),
      - пер-камерных настроек source.camera_params (dict или None),
      - поля source.target_fps (если задано).

    Ожидается, что у source есть атрибуты:
      - camera_params: dict | None
      - target_fps: float | None
    """
    from copy import deepcopy

    config_local = deepcopy(global_config)

    # camera_params в БД храним как JSONB -> в Python это dict или None
    params: Optional[dict] = getattr(source, "camera_params", None) or {}

    # --- detection_logic ---
    base_dl = global_config.detection_logic
    dl = params.get("detection_logic", {})

    config_local.detection_logic = DetectionLogicConfig(
        helmet_vest_match_iou_threshold=dl.get(
            "helmet_vest_match_iou_threshold",
            base_dl.helmet_vest_match_iou_threshold
        ),
        person_min_area=dl.get("person_min_area", base_dl.person_min_area),
        helmet_confidence_threshold=dl.get(
            "helmet_confidence_threshold",
            base_dl.helmet_confidence_threshold
        ),
        vest_confidence_threshold=dl.get(
            "vest_confidence_threshold",
            base_dl.vest_confidence_threshold
        ),
        person_confidence_threshold=dl.get(
            "person_confidence_threshold",
            base_dl.person_confidence_threshold
        ),
    )

    # --- video_processing ---
    base_vp = global_config.video_processing
    vp = params.get("video_processing", {})

    # приоритет target_fps:
    #   1) camera_params.video_processing.target_fps
    #   2) source.target_fps (отдельная колонка)
    #   3) глобальный config.yaml
    vp_target_fps = vp.get("target_fps")
    src_target_fps = getattr(source, "target_fps", None)

    if vp_target_fps is not None:
        target_fps = vp_target_fps
    elif src_target_fps is not None:
        target_fps = src_target_fps
    else:
        target_fps = base_vp.target_fps

    max_width = vp.get("max_width", base_vp.max_width)

    config_local.video_processing = VideoProcessingConfig(
        target_fps=target_fps,
        max_width=max_width,
    )

    # --- stability ---
    base_st = global_config.stability
    st = params.get("stability", {})

    config_local.stability = StabilityConfig(
        window_size_n=st.get("window_size_n", base_st.window_size_n),
        violation_ratio_t=st.get("violation_ratio_t", base_st.violation_ratio_t),
        min_window_for_violation=st.get("min_window_for_violation", base_st.min_window_for_violation),
    )

    # --- tracking ---
    base_tr = global_config.tracking
    tr = params.get("tracking", {})

    config_local.tracking = TrackingConfig(
        iou_threshold=tr.get("iou_threshold", base_tr.iou_threshold),
        max_track_lost_frames=tr.get(
            "max_track_lost_frames",
            base_tr.max_track_lost_frames
        ),
    )

    base_stream = global_config.stream
    stream_p = params.get("stream", {})

    config_local.stream = StreamConfig(
        max_run_duration_sec=stream_p.get(
            "max_run_duration_sec",
            base_stream.max_run_duration_sec
        )
    )

    return config_local