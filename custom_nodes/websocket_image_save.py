from PIL import Image
import numpy as np
import asyncio
import websockets
import base64
import io
import time
import json

class SaveImageWebsocket:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "api/image"

    async def send_to_websocket(self, image_base64):
        """ Gửi ảnh base64 đến WebSocket server """
        ws_url = "ws://localhost:8766"
        try:
            # Kết nối, gửi, rồi đóng ngay. Không cần chờ phản hồi.
            async with websockets.connect(ws_url) as websocket:
                payload = {
                    # "status": "success", # Không thực sự cần thiết cho handle_comfyui
                    "image": image_base64,
                    # "message": "Ảnh từ ComfyUI" # Không thực sự cần thiết cho handle_comfyui
                }
                await websocket.send(json.dumps(payload))
                print(f"✅ Custom node đã gửi ảnh đến server 8766")
                # KHÔNG CẦN `await websocket.recv()` nữa
                # `async with` sẽ tự động đóng websocket khi thoát khỏi block này
        except ConnectionRefusedError:
            print(f"❌ Lỗi kết nối WebSocket từ custom node: Không thể kết nối tới {ws_url}. Server 8766 có đang chạy không?")
        except Exception as e:
            print(f"❌ Lỗi gửi WebSocket từ custom node: {e}")

    def save_images(self, images):
        # Không cần tạo event loop mới mỗi lần nếu ComfyUI đã chạy trong một event loop.
        # Tuy nhiên, custom nodes thường chạy trong thread riêng, nên cách làm này có thể vẫn cần thiết.
        # Nếu ComfyUI cung cấp cách lấy event loop hiện tại, hãy dùng nó.
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed(): # Nếu loop đã đóng, tạo loop mới
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError: # Nếu không có event loop nào được set cho thread này
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)


        for image_idx, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            print(f"🖼️ Custom node đang chuẩn bị gửi ảnh {image_idx + 1}/{len(images)}")
            # Gửi ảnh qua WebSocket
            loop.run_until_complete(self.send_to_websocket(img_base64))

        return {}

    @classmethod
    def IS_CHANGED(s, images):
        return time.time()


NODE_CLASS_MAPPINGS = {
    "SaveImageWebsocket": SaveImageWebsocket,
}
