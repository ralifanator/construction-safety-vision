# core/processing.py
from typing import Dict, Any, List
import cv2
import numpy as np

from .config_loader import DetectionLogicConfig

# Классы модели СИЗ
CLASS_HELMET = 0
CLASS_VEST = 1
CLASS_HEAD = 2
CLASS_PERSON = 3


def analyze_frame(
    frame_bgr: np.ndarray,
    person_model,
    ppe_model,
    detection_logic: DetectionLogicConfig
) -> Dict[str, Any]:
    """
    Анализ одного кадра (изображения) без GUI.
    Логика полностью повторяет process_frame из старой программы,
    но:
      - не рисует на кадре,
      - возвращает JSON-подобную структуру.

    :param frame_bgr: изображение в формате BGR (как из OpenCV)
    :param person_model: YOLO-модель детекции людей (yolov8s.pt)
    :param ppe_model: YOLO-модель детекции касок/жилетов/голов (best.pt)
    :param detection_logic: настройки логики (пока используем person_min_area)
    :return: словарь с результатами анализа
    """
    height, width = frame_bgr.shape[:2]

    person_min_area = detection_logic.person_min_area
    helmet_conf_thr = detection_logic.helmet_confidence_threshold
    vest_conf_thr = detection_logic.vest_confidence_threshold
    person_conf_thr = detection_logic.person_confidence_threshold

    # ---------- 1. Детектируем людей (person_model, COCO, class 0 = person) ----------
    res_person = person_model(frame_bgr, verbose=False)
    persons: List[Dict[str, Any]] = []
    rp = res_person[0]

    if rp.boxes is not None:
        for box in rp.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id != 0:  # в COCO класс 0 = person
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # фильтрация слишком маленьких людей по площади bbox
            area = max(0, (x2 - x1)) * max(0, (y2 - y1))
            if area < person_min_area:
                continue

            persons.append({
                "bbox": (x1, y1, x2, y2),
                "conf": conf
            })

    # ---------- 2. Детектируем helmet / head / vest (ppe_model) ----------
    res_ppe = ppe_model(frame_bgr, verbose=False)
    helmets: List[Dict[str, Any]] = []
    heads: List[Dict[str, Any]] = []
    vests: List[Dict[str, Any]] = []
    re = res_ppe[0]

    if re.boxes is not None:
        for box in re.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Фильтрация по уверенности в зависимости от класса
            if cls_id == CLASS_HELMET and conf < helmet_conf_thr:
                continue
            if cls_id == CLASS_VEST and conf < vest_conf_thr:
                continue
            if cls_id == CLASS_HEAD and conf < person_conf_thr:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            det = {
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
                "cls": cls_id
            }

            if cls_id == CLASS_HELMET:
                helmets.append(det)
            elif cls_id == CLASS_HEAD:
                heads.append(det)
            elif cls_id == CLASS_VEST:
                vests.append(det)

    # Имена классов ppe‑модели (для удобства вывода/визуализации)
    class_names = {
        CLASS_HELMET: "helmet",
        CLASS_VEST: "vest",
        CLASS_HEAD: "head"
    }

    # ---------- 3. Логика по каждому человеку ----------
    persons_out: List[Dict[str, Any]] = []

    for person in persons:
        x1, y1, x2, y2 = person["bbox"]
        ph = y2 - y1
        person_conf = person["conf"]

        # head в верхней части bbox человека
        person_heads = []
        for hbox in heads:
            hx1, hy1, hx2, hy2 = hbox["bbox"]
            cx = (hx1 + hx2) / 2
            cy = (hy1 + hy2) / 2
            if cx < x1 or cx > x2:
                continue
            if cy < y1 or cy > y1 + ph * 0.5:
                continue
            person_heads.append(hbox)

        # helmet над/на голове
        person_helmets = []
        for hb in helmets:
            hx1, hy1, hx2, hy2 = hb["bbox"]
            cx = (hx1 + hx2) / 2
            cy = (hy1 + hy2) / 2
            if cx < x1 or cx > x2:
                continue
            if cy < y1 - ph * 0.2 or cy > y1 + ph * 0.5:
                continue
            person_helmets.append(hb)

        # vest в средней части человека
        person_vests = []
        for vb in vests:
            vx1, vy1, vx2, vy2 = vb["bbox"]
            cx = (vx1 + vx2) / 2
            cy = (vy1 + vy2) / 2
            if cx < x1 or cx > x2:
                continue
            if cy < y1 + ph * 0.3 or cy > y1 + ph * 0.8:
                continue
            person_vests.append(vb)

        has_helmet = len(person_helmets) > 0
        has_head = len(person_heads) > 0
        has_vest = len(person_vests) > 0

        # логика каски по твоему правилу:
        # - если helmet есть -> каска есть
        # - если head и нет helmet -> каски нет
        # - если нет ни head, ни helmet -> считаем, что каски нет
        if has_helmet:
            helmet_ok = True
        elif has_head and not has_helmet:
            helmet_ok = False
        else:
            helmet_ok = False

        vest_ok = has_vest

        if not helmet_ok and not vest_ok:
            status = "NO HELMET & NO VEST"
        elif not helmet_ok:
            status = "NO HELMET"
        elif not vest_ok:
            status = "NO VEST"
        else:
            status = "OK"

        persons_out.append({
            "bbox": [x1, y1, x2, y2],
            "conf": person_conf,
            "helmet_ok": helmet_ok,
            "vest_ok": vest_ok,
            "status": status
        })

    # Сформируем также списки helmet/head/vest
    helmets_out = []
    for h in helmets:
        x1, y1, x2, y2 = h["bbox"]
        helmets_out.append({
            "bbox": [x1, y1, x2, y2],
            "conf": h["conf"],
            "cls": h["cls"],
            "name": class_names.get(h["cls"], str(h["cls"]))
        })

    heads_out = []
    for h in heads:
        x1, y1, x2, y2 = h["bbox"]
        heads_out.append({
            "bbox": [x1, y1, x2, y2],
            "conf": h["conf"],
            "cls": h["cls"],
            "name": class_names.get(h["cls"], str(h["cls"]))
        })

    vests_out = []
    for v in vests:
        x1, y1, x2, y2 = v["bbox"]
        vests_out.append({
            "bbox": [x1, y1, x2, y2],
            "conf": v["conf"],
            "cls": v["cls"],
            "name": class_names.get(v["cls"], str(v["cls"]))
        })

    result: Dict[str, Any] = {
        "image_width": width,
        "image_height": height,
        "persons": persons_out,
        "helmets": helmets_out,
        "heads": heads_out,
        "vests": vests_out,
    }

    return result


def draw_annotations(frame_bgr: np.ndarray, analysis: Dict[str, Any]) -> np.ndarray:
    """
    Рисует на кадре те же прямоугольники и подписи:
      - цветные рамки вокруг людей с финальным статусом,
      - тонкие рамки вокруг helmet/head/vest.
    Использует структуру, возвращаемую analyze_frame.
    """
    frame = frame_bgr.copy()

    # ---------- 1. Рисуем людей с итоговым статусом ----------
    for p in analysis.get("persons", []):
        x1, y1, x2, y2 = p["bbox"]
        status = p["status"]
        conf = p["conf"]

        if status == "OK":
            color = (0, 255, 0)          # зелёный
        elif status == "NO VEST":
            color = (0, 165, 255)        # оранжевый
        else:
            # "NO HELMET" или "NO HELMET & NO VEST"
            color = (0, 0, 255)          # красный

        text = f"person {conf:.2f} | {status}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text_y = y1 + 20
        cv2.putText(frame, text, (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ---------- 2. Рисуем helmet/head/vest тонкими рамками ----------
    def draw_dets(dets: List[Dict[str, Any]], color):
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            conf = d["conf"]
            name = d.get("name", str(d.get("cls", "?")))
            text = f"{name} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            text_y = y1 + 20
            cv2.putText(frame, text, (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    draw_dets(analysis.get("helmets", []), (255, 0, 0))    # шлемы – красный
    draw_dets(analysis.get("heads", []),   (0, 0, 255))    # головы – синий
    draw_dets(analysis.get("vests", []),   (0, 255, 255))  # жилеты – жёлтый

    return frame