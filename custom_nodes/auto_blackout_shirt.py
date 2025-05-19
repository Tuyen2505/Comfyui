import torch
import numpy as np
import cv2  # Import OpenCV để resize mask
from torchvision import transforms
from PIL import Image
import torchvision.models.segmentation as models

class AutoBlackoutShirt:
    def __init__(self):
        # Load mô hình DeepLabV3 pretrained
        self.model = models.deeplabv3_resnet101(pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Đầu vào có thể là PIL Image, Tensor, hoặc NumPy array
            }
        }

    # Trong trường hợp ComfyUI mong đợi IMAGE là tensor, ta trả về torch.Tensor
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"

    def process_image(self, image):
        # Bước 1: Chuyển đổi đầu vào về NumPy array với định dạng (H, W, 3)
        image_np = self._convert_to_numpy(image)

        # Bước 2: Chuyển về PIL Image để sử dụng transform của torchvision
        image_pil = Image.fromarray(image_np)
        image_tensor = self.transform(image_pil).unsqueeze(0)

        # Bước 3: Chạy mô hình phân vùng DeepLabV3
        with torch.no_grad():
            output = self.model(image_tensor)['out'][0]

        # Bước 4: Tạo mask (ban đầu có kích thước nhỏ hơn ảnh gốc)
        mask = output.argmax(0).byte().cpu().numpy()
        # Upsample mask về kích thước của ảnh gốc
        mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Bước 5: Tạo mask cho áo (giả định class ID = 5)
        shirt_mask = (mask == 5).astype(np.uint8) * 255

        # Bước 6: Áp dụng mask để bôi đen phần áo trên ảnh gốc
        image_np[shirt_mask > 128] = [0, 0, 0]

        # Bước 7: Chuyển kết quả từ NumPy array về torch.Tensor dạng (C, H, W) với giá trị từ 0 đến 1
        final_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        return (final_tensor,)

    def _convert_to_numpy(self, image):
        """
        Chuyển đổi đầu vào (Tensor / NumPy / PIL) thành NumPy array với định dạng (H, W, 3).
        Nếu số kênh > 3 (ví dụ: (1, 1, 940)), chỉ lấy 3 kênh đầu tiên.
        """
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:  # Nếu có batch dimension
                image = image[0]
            if image.dim() == 3:
                # Chuyển từ (C, H, W) sang (H, W, C)
                image_np = image.permute(1, 2, 0).cpu().numpy()
            elif image.dim() == 2:
                image_np = image.cpu().numpy()
            else:
                raise ValueError(f"Unsupported tensor shape: {image.shape}")
            image_np = np.clip(image_np, 0, 1)
            image_np = (image_np * 255).astype(np.uint8)
        elif isinstance(image, np.ndarray):
            image_np = image
        elif isinstance(image, Image.Image):
            image_np = np.array(image.convert("RGB"))
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Nếu số kênh > 3 (ví dụ: (1, 1, 940)), chỉ lấy 3 kênh đầu tiên
        if image_np.ndim == 3 and image_np.shape[2] > 3:
            image_np = image_np[..., :3]

        # Nếu ảnh là grayscale (2D) hoặc có 1 kênh, chuyển thành ảnh 3 kênh RGB
        if image_np.ndim == 2:
            image_np = np.stack([image_np]*3, axis=-1)
        elif image_np.ndim == 3 and image_np.shape[2] == 1:
            image_np = np.repeat(image_np, 3, axis=2)
        return image_np

NODE_CLASS_MAPPINGS = {
    "AutoBlackoutShirt": AutoBlackoutShirt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoBlackoutShirt": "Auto Blackout Shirt"
}
