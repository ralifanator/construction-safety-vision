import os
import sys
import time
import subprocess

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))      
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from core.db_pg import get_active_cameras_count, reset_stuck_stream_runs 

def main():
    # 1. Определяем, сколько воркеров запускать
    num_workers = get_active_cameras_count()

    if num_workers <= 0:
        print("[manager] Активных камер нет, воркеры запускать не будем.")
        return

    print(f"[manager] Активных камер: {num_workers}. Запускаем {num_workers} экземпляров stream_worker.py")

    processes = []

    for i in range(num_workers):
        # Запускаем отдельный процесс Python с тем же интерпретатором
        p = subprocess.Popen(
            [sys.executable, "cameras_processing/stream_worker.py"],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        processes.append(p)
        print(f"[manager] Запущен stream_worker #{i+1} с PID={p.pid}")
        # Чуть разнесём по времени старт, чтобы не биться сильно за одни и те же run'ы
        time.sleep(1)

    try:
        # Основной цикл: ждём, пока воркеры работают
        while True:
            alive = [p for p in processes if p.poll() is None]
            if not alive:
                print("[manager] Все stream_worker завершились.")
                break
            time.sleep(5)

    except KeyboardInterrupt:
        # Аналогилично stream_worker: аккуратно завершаем "текущую обработку"
        print("[manager] Остановлен пользователем (KeyboardInterrupt). Останавливаем всех воркеров...")

        # 1) Останавливаем все дочерние процессы
        for p in processes:
            if p.poll() is None:
                p.terminate()
        time.sleep(2)
        for p in processes:
            if p.poll() is None:
                p.kill()
        print("[manager] Все воркеры остановлены.")

        # 2) Сбрасываем все stream-run'ы из 'processing_in_worker' в 'processing'
        try:
            restored = reset_stuck_stream_runs()
            if restored:
                print(f"[manager] Сброшено stream-run'ов из 'processing_in_worker' в 'processing': {restored}")
            else:
                print("[manager] Нет stream-run'ов в статусе 'processing_in_worker' для сброса.")
        except Exception as e:
            print(f"[manager] Ошибка при сбросе статусов stream-run'ов: {e}")

    except Exception as e:
        print(f"[manager] Необработанное исключение в менеджере: {e}")
        # В случае ошибки менеджера тоже корректно останавливаем воркеры
        for p in processes:
            if p.poll() is None:
                p.terminate()
        time.sleep(2)
        for p in processes:
            if p.poll() is None:
                p.kill()
        print("[manager] Все воркеры остановлены после ошибки.")
        # И сбрасываем зависшие run'ы
        try:
            restored = reset_stuck_stream_runs()
            if restored:
                print(f"[manager] Сброшено stream-run'ов из 'processing_in_worker' в 'processing': {restored}")
            else:
                print("[manager] Нет stream-run'ов в статусе 'processing_in_worker' для сброса.")
        except Exception as e2:
            print(f"[manager] Ошибка при сбросе статусов stream-run'ов после исключения: {e2}")

    else:
        # Нормальное завершение без исключений (например, все воркеры сами отработали)
        # На всякий случай тоже попробуем сбросить зависшие run'ы
        try:
            restored = reset_stuck_stream_runs()
            if restored:
                print(f"[manager] (normal exit) Сброшено stream-run'ов из 'processing_in_worker' в 'processing': {restored}")
        except Exception as e:
            print(f"[manager] (normal exit) Ошибка при сбросе статусов stream-run'ов: {e}")


if __name__ == "__main__":
    main()