import sys
import threading
import aiohttp
from aiohttp import web
import numpy as np
from PIL import Image
import io
import logging
import asyncio
import torch
import requests
import time
import json

# Cấu hình logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("comfyui.log"),
        logging.handlers.RotatingFileHandler(
            "comfyui_debug.log",
            maxBytes=10*1024*1024,
            backupCount=5
        )
    ]
)
logger = logging.getLogger("HttpImageReceiver")

class HttpImageReceiver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "port": ("INT", {"default": 8765, "min": 1024, "max": 65535}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "receive_image"
    CATEGORY = "image"

    def __init__(self):
        self.last_image = None
        self.lock = threading.Lock()
        self.comfyui_api_url = "http://localhost:8188"
        self.port = 8765
        self.image_event = threading.Event()
        self.server_manager = ServerManager.get_instance()
        self.server_initialized = threading.Event()

    async def handle_image_upload(self, request):
        try:
            # Chờ server sẵn sàng
            if not self.server_initialized.wait(timeout=15):
                logger.error("Server initialization timeout")
                return web.Response(text="Server initialization failed", status=503)

            # Xử lý upload ảnh
            reader = await request.multipart()
            field = await reader.next()
            
            if field.name != 'file':
                return web.Response(text="Invalid field name", status=400)

            data = await field.read()
            logger.debug(f"Received image data: {len(data)} bytes")

            # Xử lý ảnh
            try:
                with Image.open(io.BytesIO(data)) as img:
                    img.verify()
                    img = Image.open(io.BytesIO(data)).convert('RGB')
                    image_array = np.array(img)
            except Exception as e:
                logger.error(f"Image processing failed: {str(e)}")
                return web.Response(text=f"Invalid image: {e}", status=400)

            # Lưu ảnh và kích hoạt workflow
            with self.lock:
                self.last_image = image_array
                self.image_event.set()

            self.trigger_workflow()
            return web.Response(text="Image processed successfully", status=200)

        except Exception as e:
            logger.error(f"Critical server error: {str(e)}", exc_info=True)
            return web.Response(text="Internal server error", status=500)

    def receive_image(self, port):
        try:
            # Khởi động server
            self.port = int(port)
            if not self.server_manager.is_running():
                self.server_manager.start_server(self.port)
                self.server_initialized.wait(timeout=30)

            # Chờ ảnh mới
            if self.image_event.wait(timeout=15):
                with self.lock:
                    image_data = self.last_image
                    self.last_image = None
                    self.image_event.clear()

                    # Chuyển đổi tensor
                    image_tensor = torch.from_numpy(image_data.astype(np.float32)/255.0)[None,]
                    logger.info(f"Returning image tensor: {image_tensor.shape}")
                    return (image_tensor,)
            
            # Fallback khi không có ảnh
            logger.warning("No image received, returning default")
            return (torch.zeros((1, 512, 512, 3)),)

        except Exception as e:
            logger.error(f"Image receiver error: {str(e)}")
            return (torch.zeros((1, 512, 512, 3)),)

    def trigger_workflow(self):
        try:
            # Tạo payload workflow
            workflow = {
                "prompt": {
                    "input_node": {
                        "class_type": "HttpImageReceiver",
                        "inputs": {"port": self.port}
                    },
                    "output_node": {
                        "class_type": "PreviewImage",
                        "inputs": {"images": ["input_node", 0]}
                    }
                }
            }

            # Gửi yêu cầu
            response = requests.post(
                f"{self.comfyui_api_url}/prompt",
                json=workflow,
                headers={"Content-Type": "application/json"},
                timeout=20
            )

            if response.status_code != 200:
                logger.error(f"Workflow trigger failed: {response.text}")

        except Exception as e:
            logger.error(f"Workflow error: {str(e)}")

class ServerManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.running = False
        self.server_thread = None
        self.port = 8765
        self.startup_event = threading.Event()
        self.loop = None

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def start_server(self, port):
        if not self.running:
            self.port = port
            self.server_thread = threading.Thread(
                target=self.run_server,
                daemon=True
            )
            self.server_thread.start()
            self.startup_event.wait(timeout=30)
            logger.info(f"Server initialized on port {self.port}")

    def run_server(self):
        # Cấu hình event loop cho Windows
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(
                asyncio.WindowsSelectorEventLoopPolicy()
            )

        # Khởi tạo event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            self.loop.run_until_complete(self.start_web_server())
            self.startup_event.set()
            self.running = True
            self.loop.run_forever()
        except Exception as e:
            logger.error(f"Server crashed: {str(e)}")
        finally:
            self.cleanup()

    async def start_web_server(self):
        app = web.Application()
        app.router.add_post('/upload', self.handle_upload)
        runner = web.AppRunner(app)
        
        try:
            await runner.setup()
            site = web.TCPSite(runner, 'localhost', self.port)
            await site.start()
            logger.debug(f"Server listening on {self.port}")
        except OSError as e:
            logger.error(f"Port {self.port} in use: {str(e)}")
            raise

    async def handle_upload(self, request):
        receiver = HttpImageReceiver()
        return await receiver.handle_image_upload(request)

    def is_running(self):
        return self.running

    def cleanup(self):
        if self.loop:
            # Dọn dẹp task
            tasks = [t for t in asyncio.all_tasks(self.loop) if not t.done()]
            for task in tasks:
                task.cancel()
            
            # Đóng loop
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()
            
        self.running = False
        self.startup_event.clear()
        logger.info("Server shutdown complete")

# Khởi tạo
server_manager = ServerManager.get_instance()

NODE_CLASS_MAPPINGS = {"HttpImageReceiver": HttpImageReceiver}
NODE_DISPLAY_NAME_MAPPINGS = {"HttpImageReceiver": "HTTP Image Receiver"}