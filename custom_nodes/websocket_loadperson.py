import os
import torch
import hashlib
import numpy as np
from PIL import Image, ImageOps, ImageSequence
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import time

class ImageWatcher(FileSystemEventHandler):
    latest_image = None

    def on_created(self, event):
        """Ghi nhận ảnh mới khi nó được thêm vào thư mục"""
        if not event.is_directory and event.src_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            ImageWatcher.latest_image = event.src_path

class LoadLatestImagePerson:
    WATCHED_DIR = r"D:/image/person"  # Cập nhật đường dẫn của bạn

    def __init__(self):
        """Khởi tạo bộ theo dõi thư mục"""
        self.start_watching()

    def start_watching(self):
        """Bắt đầu theo dõi thư mục để tự động cập nhật ảnh mới"""
        event_handler = ImageWatcher()
        observer = Observer()
        observer.schedule(event_handler, self.WATCHED_DIR, recursive=False)
        observer_thread = threading.Thread(target=observer.start, daemon=True)
        observer_thread.start()

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_latest_image"

    def load_latest_image(self):
        """Load ảnh mới nhất từ thư mục hoặc từ bộ theo dõi"""
        image_path = ImageWatcher.latest_image or self.get_latest_image()
        if not image_path:
            raise FileNotFoundError("No valid images found in the directory")

        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)

        output_images = []
        output_masks = []
        w, h = None, None

        for i in ImageSequence.Iterator(img):
            i = i.convert("RGB")

            if len(output_images) == 0:
                w, h = i.size

            if i.size != (w, h):
                continue

            image = np.array(i).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((h, w), dtype=torch.float32, device="cpu")
            
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        output_image = torch.cat(output_images, dim=0) if len(output_images) > 1 else output_images[0]
        output_mask = torch.cat(output_masks, dim=0) if len(output_masks) > 1 else output_masks[0]

        return (output_image, output_mask)

    @classmethod
    def get_latest_image(cls):
        """Lấy ảnh mới nhất từ thư mục nếu không có ảnh nào từ watcher"""
        image_files = list(Path(cls.WATCHED_DIR).glob("*"))
        image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']]
        
        if not image_files:
            return None

        latest_image = max(image_files, key=os.path.getmtime)
        return latest_image

    @classmethod
    def IS_CHANGED(cls):
        """Kiểm tra ảnh có thay đổi không để kích hoạt lại node"""
        latest_image = cls.get_latest_image()
        if not latest_image:
            return None

        m = hashlib.sha256()
        with open(latest_image, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

NODE_CLASS_MAPPINGS = {"LoadLatestImagePerson": LoadLatestImagePerson}
NODE_DISPLAY_NAME_MAPPINGS = {"LoadLatestImagePerson": "Load Latest Image Person"}
