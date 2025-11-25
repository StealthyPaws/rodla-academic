# federated/perturbation_engine.py
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import random
from typing import Dict, Tuple, List

class PubLayNetPerturbationEngine:
    """
    Perturbations used for inference-time robustness evaluation.
    Returns PIL.Image in RGB mode and a small aug_info dict describing what was applied.
    """

    def __init__(self, perturbation_type: str = 'random', severity_level: int = 2):
        self.perturbation_type = perturbation_type
        self.severity_level = severity_level  # 1,2,3
        self.perturbation_functions = {
            'background': self.apply_background,
            'defocus': self.apply_defocus,
            'illumination': self.apply_illumination,
            'ink_bleeding': self.apply_ink_bleeding,
            'ink_holdout': self.apply_ink_holdout,
            'keystoning': self.apply_keystoning,
            'rotation': self.apply_rotation,
            'speckle': self.apply_speckle,
            'texture': self.apply_texture,
            'vibration': self.apply_vibration,
            'warping': self.apply_warping,
            'watermark': self.apply_watermark
        }

    def get_available_perturbations(self) -> List[str]:
        return list(self.perturbation_functions.keys())

    def perturb(self, image: Image.Image, perturbation_type: str = None) -> Tuple[Image.Image, Dict]:
        """Apply the chosen perturbation and return (image, info)."""
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if perturbation_type is None:
            perturbation_type = self.perturbation_type

        if perturbation_type == 'random':
            perturbation_type = random.choice(self.get_available_perturbations())

        info = {
            'perturbation_type': perturbation_type,
            'severity_level': self.severity_level,
            'parameters': {}
        }

        func = self.perturbation_functions.get(perturbation_type, None)
        if func is None:
            return image, info

        out = func(image)
        if not isinstance(out, Image.Image):
            out = Image.fromarray(np.uint8(out))
        if out.mode != 'RGB':
            out = out.convert('RGB')

        info['final_size'] = out.size
        return out, info

    def apply_background(self, image: Image.Image) -> Image.Image:
        severity = {1: (10, 0.1), 2: (25, 0.3), 3: (50, 0.6)}[self.severity_level]
        color_var, tex_strength = severity
        img = np.array(image).astype(np.int16)
        shift = np.random.randint(-color_var, color_var + 1, 3)
        img = np.clip(img + shift, 0, 255).astype(np.uint8)

        if tex_strength > 0:
            noise = np.random.normal(0, tex_strength * 255, img.shape)
            img = np.clip(img.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)

        return Image.fromarray(img)

    def apply_defocus(self, image: Image.Image) -> Image.Image:
        radius = {1: 1.0, 2: 2.0, 3: 4.0}[self.severity_level]
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def apply_illumination(self, image: Image.Image) -> Image.Image:
        params = {1: (0.9, 0.9), 2: (0.7, 0.7), 3: (0.5, 0.5)}[self.severity_level]
        img = ImageEnhance.Brightness(image).enhance(params[0])
        img = ImageEnhance.Contrast(img).enhance(params[1])
        return img

    def apply_ink_bleeding(self, image: Image.Image) -> Image.Image:
        img = np.array(image)
        h, w = img.shape[:2]
        strength = {1: 0.1, 2: 0.2, 3: 0.4}[self.severity_level]
        kernel_size = max(1, int(max(h, w) * 0.01 * strength * 10))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
        out = np.empty_like(img)
        for c in range(img.shape[2]):
            out[:, :, c] = cv2.filter2D(img[:, :, c], -1, kernel)
        return Image.fromarray(out)

    def apply_ink_holdout(self, image: Image.Image) -> Image.Image:
        img = np.array(image)
        dropout = {1: 0.05, 2: 0.1, 3: 0.2}[self.severity_level]
        mask = np.random.random(img.shape[:2]) < dropout
        for c in range(img.shape[2]):
            img[:, :, c][mask] = 255
        return Image.fromarray(img)

    def apply_keystoning(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        distortion = {1: 0.05, 2: 0.1, 3: 0.15}[self.severity_level]
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        shift_x, shift_y = int(w * distortion), int(h * distortion)
        dst = np.float32([
            [0 + shift_x, 0 + int(shift_y * 0.2)],
            [w - shift_x, 0 + int(shift_y * 0.1)],
            [w - int(shift_x * 0.8), h - shift_y],
            [int(shift_x * 0.2), h - int(shift_y * 0.8)]
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        arr = np.array(image)
        warped = cv2.warpPerspective(arr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(warped)

    def apply_rotation(self, image: Image.Image) -> Image.Image:
        angle = {1: 2, 2: 5, 3: 10}[self.severity_level] * random.choice([-1, 1])
        return image.rotate(angle, resample=Image.BILINEAR, expand=False)

    def apply_speckle(self, image: Image.Image) -> Image.Image:
        lvl = {1: 0.05, 2: 0.1, 3: 0.2}[self.severity_level]
        arr = np.array(image).astype(np.float32) / 255.0
        noise = np.random.normal(0, lvl, arr.shape).astype(np.float32)
        out = np.clip(arr + arr * noise, 0, 1) * 255
        return Image.fromarray(out.astype(np.uint8))

    def apply_texture(self, image: Image.Image) -> Image.Image:
        opacity = {1: 0.1, 2: 0.25, 3: 0.4}[self.severity_level]
        w, h = image.size
        texture = np.random.randint(0, 50, (h, w, 3), dtype=np.uint8)
        texture_img = Image.fromarray(texture).convert('RGB').resize((w, h))
        return Image.blend(image, texture_img, opacity)

    def apply_vibration(self, image: Image.Image) -> Image.Image:
        kernel_size = {1: 3, 2: 5, 3: 8}[self.severity_level]
        arr = np.array(image).astype(np.float32)
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size, dtype=np.float32)
        kernel = kernel / kernel_size
        blurred = cv2.filter2D(arr, -1, kernel)
        return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8))

    def apply_warping(self, image: Image.Image) -> Image.Image:
        magnitude = {1: 5, 2: 10, 3: 20}[self.severity_level]
        w, h = image.size
        arr = np.array(image)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        dx = magnitude * np.sin(2 * np.pi * y / max(1, (h / 4.0)))
        dy = magnitude * np.cos(2 * np.pi * x / max(1, (w / 4.0)))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        warped = cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(warped)

    def apply_watermark(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        opacity = {1: 0.1, 2: 0.2, 3: 0.3}[self.severity_level]
        watermark = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(watermark)
        try:
            font = ImageFont.truetype("arial.ttf", max(12, min(w, h) // 12))
        except Exception:
            font = ImageFont.load_default()
        text = "CONFIDENTIAL"
        for i in range(3):
            x = int((w - 10) * (i / 2.0))
            y = int((h - 10) * (i / 2.0))
            draw.text((x, y), text, font=font, fill=(255, 255, 255, int(255 * opacity)))
        base = image.convert('RGBA')
        comp = Image.alpha_composite(base, watermark)
        return comp.convert('RGB')
