#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EvoLite AI Studio - Single-file app
Build: pyinstaller --noconfirm --clean --onedir --windowed --name EvoLiteAI EvoLiteAI.py
"""
import sys, os, traceback
from datetime import datetime


def _log_path():
    base = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, 'evolite_error.log')


def log_error(msg):
    try:
        with open(_log_path(), 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now()}] {msg}\n")
    except Exception:
        pass


import numpy as np

HAS_CV2 = HAS_QT = HAS_RAWPY = HAS_TIFF = HAS_MEDIAPIPE = False
HAS_GFPGAN = HAS_REALESRGAN = False

try:
    import cv2
    HAS_CV2 = True
except Exception as e:
    log_error(f"cv2: {e}")

try:
    import rawpy
    HAS_RAWPY = True
except Exception as e:
    log_error(f"rawpy: {e}")

try:
    import tifffile
    HAS_TIFF = True
except Exception as e:
    log_error(f"tifffile: {e}")

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception as e:
    log_error(f"mediapipe: {e}")

try:
    from gfpgan import GFPGANer
    HAS_GFPGAN = True
except Exception as e:
    log_error(f"gfpgan: {e}")

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    HAS_REALESRGAN = True
except Exception as e:
    log_error(f"realesrgan: {e}")

try:
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QLabel, QPushButton, QSplitter, QFileDialog, QSlider, QComboBox,
        QToolBar, QMessageBox, QProgressBar)
    from PySide6.QtCore import Qt, QThreadPool, QRunnable, QObject, Signal, Slot
    from PySide6.QtGui import QPixmap, QImage, QAction
    HAS_QT = True
except Exception as e:
    log_error(f"PySide6: {e}")


RAW_EXTS = {'.nef', '.cr2', '.cr3', '.arw', '.dng', '.raf', '.orf', '.rw2', '.pef', '.srw'}


def load_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXTS:
        if not HAS_RAWPY:
            raise RuntimeError("rawpy chua cai. pip install rawpy")
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, use_camera_wb=True)
        return rgb.astype(np.float32) / 65535.0, True
    if not HAS_CV2:
        raise RuntimeError("opencv chua cai")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Khong doc duoc: {path}")
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.dtype == np.uint16:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0, True
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0, False


def save_image(path, rgb, was_16bit):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff'):
        if not HAS_TIFF:
            raise RuntimeError("tifffile chua cai")
        out16 = np.clip(rgb * 65535.0, 0, 65535).astype(np.uint16)
        tifffile.imwrite(path, out16, photometric='rgb', compression='deflate')
        return
    if ext == '.png':
        out = np.clip(rgb * 65535.0, 0, 65535).astype(np.uint16) if was_16bit else np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        return
    out8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    q = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in ('.jpg', '.jpeg') else [cv2.IMWRITE_WEBP_QUALITY, 95]
    cv2.imwrite(path, cv2.cvtColor(out8, cv2.COLOR_RGB2BGR), q)


def apply_exposure(rgb, ev):
    if ev == 0:
        return rgb
    return np.clip(rgb * (2.0 ** ev), 0, 1)


def apply_contrast(rgb, amount):
    if amount == 0:
        return rgb
    return np.clip((rgb - 0.5) * (1 + amount) + 0.5, 0, 1)


def apply_high_shadow(rgb, hi, sh):
    if hi == 0 and sh == 0:
        return rgb
    lum = rgb.mean(axis=2, keepdims=True)
    out = rgb + hi * np.clip((lum - 0.5) * 2, 0, 1) * (1 - rgb) * 0.5
    out = out + sh * np.clip((0.5 - lum) * 2, 0, 1) * rgb * 0.5
    return np.clip(out, 0, 1)


def apply_vib_sat(rgb, vib, sat):
    if vib == 0 and sat == 0:
        return rgb
    gray = rgb.mean(axis=2, keepdims=True)
    out = rgb
    if vib:
        out = out + vib * (1 - np.abs(rgb - gray).max(axis=2, keepdims=True)) * (out - gray)
    if sat:
        out = out + sat * (out - gray)
    return np.clip(out, 0, 1)


def apply_wb(rgb, t, ti):
    if t == 0 and ti == 0:
        return rgb
    out = rgb.copy()
    out[..., 0] *= (1 + t)
    out[..., 2] *= (1 - t)
    out[..., 1] *= (1 + ti)
    return np.clip(out, 0, 1)


def natural_sharpen(rgb, amount=0.6, radius=1.2, threshold=0.02):
    if amount <= 0:
        return rgb
    blur = cv2.GaussianBlur(rgb, (0, 0), sigmaX=max(0.3, radius))
    detail = rgb - blur
    detail *= (np.abs(detail) > threshold).astype(np.float32)
    return np.clip(rgb + amount * detail, 0, 1)


def sharpen_protected(rgb, amount=0.6, radius=1.2, threshold=0.02):
    if amount <= 0:
        return rgb
    sharp = natural_sharpen(rgb, amount, radius, threshold)
    gray = (rgb.mean(axis=2) * 255).astype(np.uint8)
    edges = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    mask = np.clip((edges - 12) / 25, 0, 1)
    mask = cv2.GaussianBlur(mask, (0, 0), 2.0)[..., None]
    return rgb * (1 - mask) + sharp * mask


def noise_reduce(rgb, lum):
    if lum <= 0:
        return rgb
    gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    den = cv2.bilateralFilter(gray, 5, 30 * lum, 5)
    return np.clip(rgb + (den - gray)[..., None] * lum, 0, 1)


def freq_separation_skin(rgb, smooth_amount=0.0, texture_amount=1.0):
    if smooth_amount <= 0:
        return rgb
    low = cv2.GaussianBlur(rgb, (0, 0), sigmaX=6.0)
    high = rgb - low
    smooth_low = cv2.bilateralFilter(low, 9, 75, 75)
    out_low = low + smooth_amount * (smooth_low - low)
    out = out_low + texture_amount * high
    return np.clip(out, 0, 1)


def remove_blemishes(rgb, strength=0.0):
    if strength <= 0 or not HAS_CV2:
        return rgb
    gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    diff = np.abs(gray.astype(np.float32) - blur.astype(np.float32))
    mask = (diff > (15 + 20 * (1 - strength))).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    if mask.sum() == 0:
        return rgb
    out_bgr = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    out_bgr = cv2.inpaint(out_bgr, mask, 3, cv2.INPAINT_TELEA)
    return np.clip(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0, 0, 1)


_face_mesh = None


def _get_face_mesh():
    global _face_mesh
    if _face_mesh is None and HAS_MEDIAPIPE:
        _face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    return _face_mesh


def _landmarks_to_mask(shape, landmarks, idx_list):
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in idx_list], np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


EYE_IDX = [33, 133, 159, 145, 160, 144, 158, 153, 362, 263, 387, 373, 386, 374, 385, 380]
TEETH_IDX = [61, 291, 13, 14, 78, 308, 82, 312, 87, 317, 88, 318]


def retouch_eyes(rgb, brighten=0.0, under_eye=0.0):
    if brighten <= 0 and under_eye <= 0:
        return rgb
    fm = _get_face_mesh()
    if fm is None:
        return rgb
    res = fm.process((rgb * 255).astype(np.uint8))
    if not res.multi_face_landmarks:
        return rgb
    lm = res.multi_face_landmarks[0].landmark
    out = rgb.copy()
    if brighten > 0:
        mask = _landmarks_to_mask(rgb.shape, lm, EYE_IDX)
        mask = cv2.GaussianBlur(mask, (0, 0), 5).astype(np.float32)[..., None] / 255.0
        out = np.clip(out + brighten * 0.15 * mask * (1 - out), 0, 1)
    if under_eye > 0:
        mask = _landmarks_to_mask(rgb.shape, lm, EYE_IDX)
        M = np.float32([[1, 0, 0], [0, 1, int(rgb.shape[0] * 0.015)]])
        mask = cv2.warpAffine(mask, M, (rgb.shape[1], rgb.shape[0]))
        mask = cv2.GaussianBlur(mask, (0, 0), 7).astype(np.float32)[..., None] / 255.0
        out = np.clip(out - under_eye * 0.1 * mask * (1 - out), 0, 1)
    return out


def whiten_teeth(rgb, amount=0.0):
    if amount <= 0:
        return rgb
    fm = _get_face_mesh()
    if fm is None:
        return rgb
    res = fm.process((rgb * 255).astype(np.uint8))
    if not res.multi_face_landmarks:
        return rgb
    lm = res.multi_face_landmarks[0].landmark
    mask = _landmarks_to_mask(rgb.shape, lm, TEETH_IDX)
    mask = cv2.GaussianBlur(mask, (0, 0), 3).astype(np.float32)[..., None] / 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    L = np.clip(L + amount * 20 * mask[..., 0], 0, 100)
    b = np.clip(b - amount * 8 * mask[..., 0], -128, 127)
    return cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2RGB)


def apply_makeup(rgb, lip_color=None, eyeshadow_color=None, eyebrow_color=None):
    if lip_color is None and eyeshadow_color is None and eyebrow_color is None:
        return rgb
    fm = _get_face_mesh()
    if fm is None:
        return rgb
    res = fm.process((rgb * 255).astype(np.uint8))
    if not res.multi_face_landmarks:
        return rgb
    lm = res.multi_face_landmarks[0].landmark
    out = rgb.copy()
    LIP_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409]
    BROW_L = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    BROW_R = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
    for idx_list, color, alpha in [(LIP_IDX, lip_color, 0.35), (BROW_L, eyebrow_color, 0.5), (BROW_R, eyebrow_color, 0.5)]:
        if color is None:
            continue
        mask = _landmarks_to_mask(rgb.shape, lm, idx_list)
        mask = cv2.GaussianBlur(mask, (0, 0), 5).astype(np.float32)[..., None] / 255.0 * alpha
        out = np.clip(out * (1 - mask) + np.array(color, np.float32) / 255.0 * mask, 0, 1)
    return out


_gfpgan_restorer = None


def restore_face(rgb, weight=0.0):
    if weight <= 0 or not HAS_GFPGAN:
        return rgb
    global _gfpgan_restorer
    if _gfpgan_restorer is None:
        model_path = os.path.join(os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__)), "weights", "GFPGANv1.4.pth")
        if not os.path.exists(model_path):
            log_error(f"GFPGAN model khong ton tai: {model_path}")
            return rgb
        _gfpgan_restorer = GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)
    _, _, restored = _gfpgan_restorer.enhance((rgb * 255).astype(np.uint8), has_aligned=False, only_center_face=False)
    return np.clip((1 - weight) * rgb + weight * restored.astype(np.float32) / 255.0, 0, 1)


_realesrgan_upsampler = None


def ai_upscale(rgb, scale=2, denoise=0.0):
    if scale <= 1:
        return rgb
    if not HAS_REALESRGAN:
        h, w = rgb.shape[:2]
        return np.clip(cv2.resize(rgb, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4), 0, 1)
    global _realesrgan_upsampler
    if _realesrgan_upsampler is None:
        base = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base, "weights", "RealESRGAN_x2plus.pth")
        if not os.path.exists(model_path):
            h, w = rgb.shape[:2]
            return np.clip(cv2.resize(rgb, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4), 0, 1)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23)
        _realesrgan_upsampler = RealESRGANer(scale=scale, model_path=model_path, model=model, tile=512, tile_pad=16, pre_pad=0, half=False)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out, _ = _realesrgan_upsampler.enhance(bgr, outscale=scale)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if denoise > 0:
        out = cv2.bilateralFilter(out, 3, 5 * denoise, 3)
    return np.clip(out, 0, 1)


def _warp_radial(rgb, center, radius, scale):
    h, w = rgb.shape[:2]
    cx, cy = center
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    f = np.exp(-(dx * dx + dy * dy) / (2 * radius * radius))
    mx = xx - dx * (scale - 1) * f
    my = yy - dy * (scale - 1) * f
    return cv2.remap(rgb, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def _warp_horizontal(rgb, cy_frac, radius, strength):
    h, w = rgb.shape[:2]
    cx, cy = w / 2, cy_frac * h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    f = np.exp(-(dy * dy) / (2 * radius * radius))
    mx = xx - np.sign(dx) * strength * 30 * f
    my = yy
    return cv2.remap(rgb, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def detect_face(rgb):
    try:
        gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if cas.empty():
            return None
        faces = cas.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        if len(faces) == 0:
            return None
        return max(faces, key=lambda b: b[2] * b[3])
    except Exception as e:
        log_error(f"detect_face: {e}")
        return None


def retouch_face(rgb, p):
    bbox = detect_face(rgb)
    if bbox is None:
        return rgb
    x, y, fw, fh = bbox
    out = rgb
    if p.get('eye_size', 0):
        ey = y + fh * 0.4
        for ex in (x + fw * 0.3, x + fw * 0.7):
            out = _warp_radial(out, (ex, ey), fw * 0.15, 1 + p['eye_size'] * 0.3)
    if p.get('face_slim', 0):
        out = _warp_horizontal(out, (y + fh * 0.6) / rgb.shape[0], fh * 0.4, p['face_slim'] * 0.5)
    if p.get('lip_size', 0):
        out = _warp_radial(out, (x + fw * 0.5, y + fh * 0.75), fw * 0.12, 1 + p['lip_size'] * 0.2)
    return out


def retouch_body(rgb, p):
    if p.get('waist_slim', 0):
        return _warp_horizontal(rgb, 0.55, 0.2, p['waist_slim'] * 0.5)
    return rgb


def process_image(rgb, p, was_16bit):
    out = apply_wb(rgb, p.get('wb_temp', 0), p.get('wb_tint', 0))
    out = apply_exposure(out, p.get('exposure', 0))
    out = apply_high_shadow(out, p.get('highlights', 0), p.get('shadows', 0))
    out = apply_contrast(out, p.get('contrast', 0))
    out = apply_vib_sat(out, p.get('vibrance', 0), p.get('saturation', 0))
    out = freq_separation_skin(out, p.get('skin_smooth', 0), 1.0)
    out = remove_blemishes(out, p.get('blemish', 0))
    out = retouch_eyes(out, p.get('eye_brighten', 0), p.get('under_eye', 0))
    out = whiten_teeth(out, p.get('teeth_whiten', 0))
    lip = p.get('lip_color')
    eye_m = p.get('eye_color')
    brow = p.get('eyebrow_color')
    if any(c is not None for c in (lip, eye_m, brow)):
        out = apply_makeup(out, lip, eye_m, brow)
    out = restore_face(out, p.get('face_restore', 0))
    out = retouch_face(out, p)
    out = retouch_body(out, p)
    out = noise_reduce(out, p.get('nr_lum', 0))
    out = sharpen_protected(out, p.get('sharpen_amount', 0), p.get('sharpen_radius', 1.2), p.get('sharpen_threshold', 0.02))
    if p.get('upscale_scale', 1) > 1:
        out = ai_upscale(out, p['upscale_scale'], p.get('upscale_denoise', 0.3))
    return np.clip(out, 0, 1)


if HAS_QT:

    class WorkerSignals(QObject):
        finished = Signal(object, str)
        error = Signal(str)

    class ImageTask(QRunnable):
        def __init__(self, rgb, params, w16, path=""):
            super().__init__()
            self.rgb = rgb
            self.params = params
            self.w16 = w16
            self.path = path
            self.signals = WorkerSignals()

        @Slot()
        def run(self):
            try:
                out = process_image(self.rgb, self.params, self.w16)
                self.signals.finished.emit(out, self.path)
            except Exception as e:
                self.signals.error.emit(str(e) + "\n" + traceback.format_exc())

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("EvoLite AI Studio")
            self.resize(1500, 920)
            self.original = None
            self.current = None
            self.path = None
            self.w16 = False
            self.sliders = {}
            self.batch_files = []
            self.threadpool = QThreadPool()
            self.threadpool.setMaxThreadCount(max(1, os.cpu_count() or 2))
            self._build()
            self._style()

        def _build(self):
            tb = QToolBar("Main")
            self.addToolBar(tb)
            for label, slot in [("Mo anh", self.open), ("Mo nhieu (Batch)", self.open_batch),
                                ("Reset", self.reset), ("Xuat", self.export), ("Xuat Batch", self.export_batch)]:
                a = QAction(label, self)
                a.triggered.connect(slot)
                tb.addAction(a)
            self.progress = QProgressBar()
            self.progress.setMaximumWidth(200)
            self.progress.setVisible(False)
            tb.addWidget(self.progress)

            sp = QSplitter(Qt.Horizontal)
            self.preview = QLabel("Mo anh de bat dau\n\nJPG / PNG / TIFF / RAW (NEF, CR2, ARW, DNG...)")
            self.preview.setAlignment(Qt.AlignCenter)
            self.preview.setStyleSheet("background:#1a1a1a;color:#888;font-size:14px;")
            self.preview.setMinimumWidth(650)
            self.controls = self._controls()
            self.controls.setFixedWidth(380)
            sp.addWidget(self.preview)
            sp.addWidget(self.controls)
            sp.setStretchFactor(0, 1)
            self.setCentralWidget(sp)

        def _slider(self, layout, key, label, lo, hi, default=0, scale=1.0):
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 1, 0, 1)
            lbl = QLabel(label)
            lbl.setFixedWidth(150)
            s = QSlider(Qt.Horizontal)
            s.setRange(lo, hi)
            s.setValue(default)
            v = QLabel(str(default))
            v.setFixedWidth(45)
            s.valueChanged.connect(lambda x: v.setText(str(x)))
            h.addWidget(lbl)
            h.addWidget(s)
            h.addWidget(v)
            layout.addWidget(row)
            self.sliders[key] = (s, scale)
            return s

        def _combo(self, layout, key, label, options):
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 1, 0, 1)
            h.addWidget(QLabel(label))
            c = QComboBox()
            for txt, val in options:
                c.addItem(txt, val)
            h.addWidget(c)
            layout.addWidget(row)
            self.sliders[key] = (c, None)

        def _controls(self):
            w = QWidget()
            outer = QVBoxLayout(w)
            tabs = QTabWidget()
            tabs.setTabPosition(QTabWidget.West)

            r = QWidget()
            rl = QVBoxLayout(r)
            self._slider(rl, 'wb_temp', 'WB Temp', -100, 100)
            self._slider(rl, 'wb_tint', 'WB Tint', -100, 100)
            rl.addStretch()
            tabs.addTab(r, "RAW/WB")

            lt = QWidget()
            ll = QVBoxLayout(lt)
            for k, lab in [('exposure', 'Exposure'), ('contrast', 'Contrast'), ('highlights', 'Highlights'),
                           ('shadows', 'Shadows'), ('vibrance', 'Vibrance'), ('saturation', 'Saturation')]:
                scale = 100 if k == 'exposure' else 1
                lo, hi = (-300, 300) if k == 'exposure' else (-100, 100)
                self._slider(ll, k, lab, lo, hi, 0, scale)
            ll.addStretch()
            tabs.addTab(lt, "Anh sang")

            sk = QWidget()
            sl = QVBoxLayout(sk)
            self._slider(sl, 'skin_smooth', 'Skin Smooth', 0, 100, 0, 100)
            self._slider(sl, 'blemish', 'Blemish Remove', 0, 100, 0, 100)
            sl.addStretch()
            tabs.addTab(sk, "Skin")

            eh = QWidget()
            el = QVBoxLayout(eh)
            self._slider(el, 'eye_brighten', 'Eye Brighten', 0, 100, 0, 100)
            self._slider(el, 'under_eye', 'Under-eye', 0, 100, 0, 100)
            self._slider(el, 'teeth_whiten', 'Teeth Whiten', 0, 100, 0, 100)
            el.addStretch()
            tabs.addTab(eh, "Eyes/Teeth")

            mk = QWidget()
            ml = QVBoxLayout(mk)
            self._combo(ml, 'lip_color', 'Lipstick', [('None', None), ('Do', (200, 30, 40)), ('Hong', (230, 120, 140)), ('Cam', (240, 140, 90))])
            self._combo(ml, 'eyebrow_color', 'Eyebrow', [('None', None), ('Nau', (90, 60, 40)), ('Den', (30, 30, 30))])
            ml.addStretch()
            tabs.addTab(mk, "Makeup")

            fc = QWidget()
            fl = QVBoxLayout(fc)
            for k, lab in [('eye_size', 'Eye Size'), ('face_slim', 'Face Slim'), ('lip_size', 'Lip Size')]:
                self._slider(fl, k, lab, -100, 100, 0, 100)
            fl.addStretch()
            tabs.addTab(fc, "Mat")

            bd = QWidget()
            bl = QVBoxLayout(bd)
            self._slider(bl, 'waist_slim', 'Waist Slim', -100, 100, 0, 100)
            bl.addStretch()
            tabs.addTab(bd, "Body")

            dt = QWidget()
            dl = QVBoxLayout(dt)
            self._slider(dl, 'nr_lum', 'NR Luminance', 0, 100, 0, 100)
            self._slider(dl, 'sharpen_amount', 'Sharpen Amount', 0, 200, 0, 100)
            self._slider(dl, 'sharpen_radius', 'Sharpen Radius', 3, 50, 12, 10)
            self._slider(dl, 'sharpen_threshold', 'Sharpen Threshold', 0, 100, 2, 1000)
            dl.addStretch()
            tabs.addTab(dt, "Chi tiet")

            up = QWidget()
            ul = QVBoxLayout(up)
            self._slider(ul, 'face_restore', 'Face Restore (GFPGAN)', 0, 100, 0, 100)
            ul.addWidget(QLabel("AI Upscale:"))
            self.up = QComboBox()
            self.up.addItems(["Tat", "2x", "4x"])
            ul.addWidget(self.up)
            self._slider(ul, 'upscale_denoise', 'Upscale Denoise', 0, 100, 30, 100)
            ul.addStretch()
            tabs.addTab(up, "Restore/Upscale")

            outer.addWidget(tabs)
            btn = QPushButton("Ap dung & Xem truoc")
            btn.clicked.connect(self.apply)
            btn.setStyleSheet("padding:10px;font-weight:bold;")
            outer.addWidget(btn)
            return w

        def _style(self):
            self.setStyleSheet("""
                QMainWindow,QWidget{background:#252526;color:#ddd;}
                QTabBar::tab{background:#333;padding:6px;min-width:70px;}
                QTabBar::tab:selected{background:#094771;}
                QSlider::groove:horizontal{height:4px;background:#444;}
                QSlider::handle:horizontal{background:#0a84ff;width:14px;margin:-5px 0;border-radius:7px;}
                QPushButton{background:#0a84ff;color:white;border:none;border-radius:4px;}
                QToolBar{background:#2d2d30;}
            """)

        def params(self):
            p = {}
            for k, (w, sc) in self.sliders.items():
                if sc is None:
                    p[k] = w.currentData()
                else:
                    p[k] = w.value() / sc
            p['upscale_scale'] = [1, 2, 4][self.up.currentIndex()]
            return p

        def open(self):
            path, _ = QFileDialog.getOpenFileName(self, "Mo anh", "",
                "Anh (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp *.nef *.cr2 *.cr3 *.arw *.dng *.raf *.orf *.rw2)")
            if not path:
                return
            try:
                self.original, self.w16 = load_image(path)
                self.path = path
                self.current = self.original.copy()
                self.reset()
                self.update_preview(self.current)
                self.setWindowTitle(f"EvoLite AI - {os.path.basename(path)}")
            except Exception as e:
                QMessageBox.critical(self, "Loi", f"{e}\n\n{traceback.format_exc()}")

        def open_batch(self):
            paths, _ = QFileDialog.getOpenFileNames(self, "Chon nhieu anh", "",
                "Anh (*.jpg *.jpeg *.png *.tif *.tiff *.nef *.cr2 *.arw *.dng)")
            if paths:
                self.batch_files = paths
                QMessageBox.information(self, "Batch", f"Da chon {len(paths)} anh.")

        def update_preview(self, rgb):
            h, w = rgb.shape[:2]
            sc = min(900 / w, 700 / h, 1.0)
            disp = cv2.resize(rgb, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA) if sc < 1 else rgb
            d8 = np.clip(disp * 255, 0, 255).astype(np.uint8)
            img = QImage(d8.data, d8.shape[1], d8.shape[0], 3 * d8.shape[1], QImage.Format_RGB888)
            self.preview.setPixmap(QPixmap.fromImage(img.copy()))

        def apply(self):
            if self.original is None:
                QMessageBox.warning(self, "Chua co anh", "Mo anh truoc.")
                return
            p = self.params()
            p['upscale_scale'] = 1
            self.preview.setText("Dang xu ly...")
            QApplication.processEvents()
            task = ImageTask(self.original, p, self.w16)
            task.signals.finished.connect(lambda rgb, path: (setattr(self, 'current', rgb), self.update_preview(rgb)))
            task.signals.error.connect(lambda e: QMessageBox.critical(self, "Loi", e))
            self.threadpool.start(task)

        def reset(self):
            for w, sc in self.sliders.values():
                if sc is None:
                    w.setCurrentIndex(0)
                else:
                    w.setValue(0)
            self.up.setCurrentIndex(0)
            if self.original is not None:
                self.current = self.original.copy()
                self.update_preview(self.current)

        def export(self):
            if self.original is None:
                QMessageBox.warning(self, "Chua co anh", "Mo anh truoc.")
                return
            name = (os.path.splitext(os.path.basename(self.path))[0] + "_edited.tif") if self.path else "output.tif"
            path, _ = QFileDialog.getSaveFileName(self, "Xuat", name,
                "TIFF 16-bit (*.tif *.tiff);;PNG (*.png);;JPEG (*.jpg *.jpeg)")
            if not path:
                return
            try:
                self.preview.setText("Dang xuat...")
                QApplication.processEvents()
                out = process_image(self.original, self.params(), self.w16)
                save_image(path, out, self.w16)
                QMessageBox.information(self, "OK", f"Da xuat:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Loi xuat", f"{e}\n\n{traceback.format_exc()}")

        def export_batch(self):
            if not self.batch_files:
                QMessageBox.warning(self, "Batch", "Chua chon anh nao.")
                return
            outdir = QFileDialog.getExistingDirectory(self, "Chon thu muc xuat")
            if not outdir:
                return
            p = self.params()
            p['upscale_scale'] = 1
            self.progress.setVisible(True)
            self.progress.setMaximum(len(self.batch_files))
            self.progress.setValue(0)
            self._batch_remaining = len(self.batch_files)
            self._batch_outdir = outdir
            for f in self.batch_files:
                try:
                    rgb, w16 = load_image(f)
                    task = ImageTask(rgb, p, w16, f)
                    task.signals.finished.connect(self._batch_done)
                    task.signals.error.connect(self._batch_error)
                    self.threadpool.start(task)
                except Exception as e:
                    log_error(f"batch load {f}: {e}")

        def _batch_done(self, rgb, src):
            try:
                name = os.path.splitext(os.path.basename(src))[0] + "_edited.tif"
                save_image(os.path.join(self._batch_outdir, name), rgb, True)
            except Exception as e:
                log_error(f"batch save {src}: {e}")
            self._batch_remaining -= 1
            self.progress.setValue(self.progress.maximum() - self._batch_remaining)
            if self._batch_remaining == 0:
                self.progress.setVisible(False)
                QMessageBox.information(self, "Batch", "Da xuat xong.")

        def _batch_error(self, msg):
            log_error("batch: " + msg)
            self._batch_remaining -= 1
            self.progress.setValue(self.progress.maximum() - self._batch_remaining)


def main():
    if not HAS_QT:
        print("PySide6 chua cai. pip install PySide6\nLog: " + _log_path())
        try:
            input("Enter...")
        except Exception:
            pass
        return 1
    if not HAS_CV2:
        try:
            QMessageBox.critical(None, "Thieu thu vien", "opencv chua cai.\npip install opencv-python")
        except Exception:
            pass
        return 1
    try:
        app = QApplication(sys.argv)
        win = MainWindow()
        win.show()
        return app.exec()
    except Exception as e:
        log_error("FATAL: " + str(e) + "\n" + traceback.format_exc())
        try:
            QMessageBox.critical(None, "Loi khoi dong", f"{e}\n\nLog: {_log_path()}")
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())