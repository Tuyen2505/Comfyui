import os
import time
import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading

class ImageWatcher(FileSystemEventHandler):
    latest_image = None
    last_queued_time = 0
    queue_interval = 5  # giây

    def on_created(self, event):
        # Kiểm tra nếu là file ảnh
        if not event.is_directory and event.src_path.lower().endswith(('.jpg', '.png')):
            ImageWatcher.latest_image = event.src_path
            current_time = time.time()
            # Chỉ queue nếu đã qua khoảng thời gian tối thiểu
            if current_time - ImageWatcher.last_queued_time > ImageWatcher.queue_interval:
                self.queue_prompt()
                ImageWatcher.last_queued_time = current_time

    def queue_prompt(self):
        # Gửi yêu cầu queue prompt qua API
        try:
            url = "http://localhost:8188/prompt"
            payload = {
                "prompt": {
                    # Thay thế bằng prompt/workflow thực tế của bạn
                    "workflow": """
                                    {
                                        "2": {
                                            "inputs": {},
                                            "class_type": "LoadLatestImage",
                                            "_meta": {
                                            "title": "Load Latest Image"
                                            }
                                        },
                                        "3": {
                                            "inputs": {
                                            "images": [
                                                "2",
                                                0
                                            ]
                                            },
                                            "class_type": "PreviewImage",
                                            "_meta": {
                                            "title": "Preview Image"
                                            }
                                        }
                                    }
                                """
                }
            }
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                print("Đã queue prompt thành công")
            else:
                print(f"Queue thất bại: {response.status_code}")
        except Exception as e:
            print(f"Lỗi khi queue prompt: {e}")

class LoadLatestImage:
    WATCHED_DIR = r"D:/Received_Images"  # Thay bằng đường dẫn thư mục của bạn

    def __init__(self):
        self.start_watching()

    def start_watching(self):
        # Khởi động bộ theo dõi thư mục
        event_handler = ImageWatcher()
        observer = Observer()
        observer.schedule(event_handler, self.WATCHED_DIR, recursive=False)
        observer_thread = threading.Thread(target=observer.start, daemon=True)
        observer_thread.start()

    def load_latest_image(self):
        # Logic load ảnh (giả định đã có)
        image_path = ImageWatcher.latest_image
        if not image_path:
            return None
        print(f"Đã load ảnh: {image_path}")
        return image_path

# Sử dụng node
NODE_CLASS_MAPPINGS = {"LoadLatestImage": LoadLatestImage}
NODE_DISPLAY_NAME_MAPPINGS = {"LoadLatestImage": "Load Latest Image"}