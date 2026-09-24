import ast
import threading
import time
from pathlib import Path

import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed!")
    print("Install with: pip install onnxruntime")
    print("(For GPU acceleration: pip install onnxruntime-gpu)")
    exit(1)

# Dark mode color scheme
COLORS = {
    'bg': '#1e1e1e',
    'bg_light': '#2d2d2d',
    'bg_lighter': '#3d3d3d',
    'fg': '#ffffff',
    'fg_dim': '#b0b0b0',
    'accent': '#007acc',
    'accent_hover': '#005a9e',
    'success': '#4ec9b0',
    'warning': '#ce9178',
    'error': '#f48771',
    'border': '#404040',
    # Dark-mode red (Stop / Clear)
    'danger': '#c0392b',
    'danger_hover': '#992d22',
    # Dark-mode green (Save Result)
    'green': '#2ecc71',
    'green_hover': '#27ae60',
}

# A fixed color palette (BGR) used to draw boxes per class id
BOX_PALETTE = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255),
    (255, 255, 0), (0, 128, 255), (255, 128, 0), (128, 0, 255), (0, 255, 128),
    (128, 255, 0), (255, 0, 128), (0, 128, 128), (128, 128, 0), (128, 0, 128),
    (192, 192, 192), (64, 64, 255), (64, 255, 64), (255, 64, 64), (200, 200, 0),
]

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize + pad image to new_shape while keeping aspect ratio.
    Returns the padded image, the resize ratio, and (dw, dh) half-padding."""
    shape = im.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

class SimpleBox:
    """Mimics the small subset of ultralytics' Boxes API the UI code relies on."""

    def __init__(self, cls_id, conf, xyxy):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [np.array(xyxy, dtype=float)]

class SimpleResult:
    """Mimics the small subset of ultralytics' Results API the UI code relies on."""

    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names

class ONNXYOLO:
    """Thin ONNX Runtime wrapper that reproduces the predict() -> result interface
    the rest of the app expects, so the GUI code barely has to change."""

    def __init__(self, model_path):
        providers = []
        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')

        self.session = ort.InferenceSession(model_path, providers=providers)

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.input_h = shape[2] if isinstance(shape[2], int) else 640
        self.input_w = shape[3] if isinstance(shape[3], int) else 640

        self.names = self._load_names()

    def _load_names(self):
        names = {}
        try:
            meta = self.session.get_modelmeta()
            custom = meta.custom_metadata_map
            if custom and 'names' in custom:
                parsed = ast.literal_eval(custom['names'])
                if isinstance(parsed, dict):
                    names = {int(k): v for k, v in parsed.items()}
                elif isinstance(parsed, (list, tuple)):
                    names = {i: v for i, v in enumerate(parsed)}
        except Exception:
            names = {}

        if not names:
            names = {i: f"class{i}" for i in range(1000)}
        return names

    def _postprocess(self, pred, conf_thres, iou_thres, ratio, dw, dh, orig_shape):
        orig_h, orig_w = orig_shape[:2]

        pred = pred[0]
        # Normalize to shape (num_anchors, 4 + num_classes)
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        boxes_cxcywh = pred[:, :4]
        class_scores = pred[:, 4:]

        if class_scores.shape[1] == 0:
            return [], [], []

        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_ids)), class_ids]

        mask = scores > conf_thres
        boxes_cxcywh = boxes_cxcywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(scores) == 0:
            return [], [], []

        cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Undo letterbox padding/scale to map back to original image coordinates
        x1 = (x1 - dw) / ratio
        y1 = (y1 - dh) / ratio
        x2 = (x2 - dw) / ratio
        y2 = (y2 - dh) / ratio

        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        nms_scores = scores.tolist()

        indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, conf_thres, iou_thres)
        if indices is None or len(indices) == 0:
            return [], [], []
        indices = np.array(indices).flatten()

        final_boxes = [[float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])] for i in indices]
        final_scores = [float(scores[i]) for i in indices]
        final_class_ids = [int(class_ids[i]) for i in indices]

        return final_boxes, final_scores, final_class_ids

    def _draw(self, img_bgr, boxes):
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_id = box.cls[0]
            conf = box.conf[0]
            name = self.names.get(cls_id, f"class{cls_id}")
            color = BOX_PALETTE[cls_id % len(BOX_PALETTE)]

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf * 100:.1f}%"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty1 = max(0, y1 - th - baseline - 4)
            cv2.rectangle(img_bgr, (x1, ty1), (x1 + tw + 4, ty1 + th + baseline + 4), color, -1)
            cv2.putText(img_bgr, label, (x1 + 2, ty1 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return img_bgr

    def infer(self, image_bgr, conf=0.25, iou=0.45):
        """Run detection on a BGR numpy image. Returns (SimpleResult, annotated_bgr)."""
        letter_img, ratio, (dw, dh) = letterbox(image_bgr, (self.input_h, self.input_w))
        img_rgb = cv2.cvtColor(letter_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_chw = np.transpose(img_rgb, (2, 0, 1))
        input_tensor = np.expand_dims(img_chw, 0).astype(np.float32)

        outputs = self.session.run(None, {self.input_name: input_tensor})
        pred = outputs[0]

        boxes, scores, class_ids = self._postprocess(pred, conf, iou, ratio, dw, dh, image_bgr.shape)
        simple_boxes = [SimpleBox(class_ids[i], scores[i], boxes[i]) for i in range(len(boxes))]
        result = SimpleResult(simple_boxes, self.names)

        annotated = self._draw(image_bgr.copy(), simple_boxes)
        return result, annotated

class YOLODetectorApp:
    """
    YOLO Object Detection - Tkinter GUI Application (Dark Mode)
    Supports image and video detection using an ONNX model via ONNX Runtime.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Object Detection - Dark Mode (ONNX)")
        self.root.geometry("1400x850")
        self.root.configure(bg=COLORS['bg'])
        self.maximize_window()

        # Variables
        self.model = None
        self.model_path = None
        self.current_file = None
        self.video_thread = None
        self.stop_video = False
        self.is_processing = False
        self.detection_data = []
        self.placeholder_visible = True

        # Camera state
        self.available_cameras = [0]
        self.current_cam_index = 0
        self._swapping = False

        # Configure dark theme
        self.setup_dark_theme()

        # Setup UI
        self.setup_ui()

        # Auto-load model from models folder
        self.auto_load_model()

        # Detect cameras in the background so we don't block startup
        threading.Thread(target=self._detect_cameras_bg, daemon=True).start()

    def maximize_window(self):
        """Maximize the window immediately on launch, across platforms."""
        try:
            self.root.state('zoomed')  # Windows / some Linux window managers
        except tk.TclError:
            try:
                self.root.attributes('-zoomed', True)  # Most Linux WMs
            except tk.TclError:
                # Fallback (e.g. macOS): size window to the full screen
                w = self.root.winfo_screenwidth()
                h = self.root.winfo_screenheight()
                self.root.geometry(f"{w}x{h}+0+0")

    def setup_dark_theme(self):
        """Configure dark theme for ttk widgets"""
        style = ttk.Style()
        style.theme_use('clam')

        # Configure colors for all widgets
        style.configure('TFrame', background=COLORS['bg'])
        style.configure('TLabelframe', background=COLORS['bg'], foreground=COLORS['fg'],
                       bordercolor=COLORS['border'], relief='solid')
        style.configure('TLabelframe.Label', background=COLORS['bg'], foreground=COLORS['accent'],
                       font=('Arial', 10, 'bold'))
        style.configure('TLabel', background=COLORS['bg'], foreground=COLORS['fg'])
        style.configure('TButton', background=COLORS['bg_light'], foreground=COLORS['fg'],
                       bordercolor=COLORS['border'], focuscolor=COLORS['accent'])
        style.map('TButton', background=[('active', COLORS['accent_hover'])])

        style.configure('TScale', background=COLORS['bg'], troughcolor=COLORS['bg_light'],
                       bordercolor=COLORS['border'], darkcolor=COLORS['bg_light'],
                       lightcolor=COLORS['accent'])

        # Blue - used for Detect Objects and Swap Cam
        style.configure('Accent.TButton', background=COLORS['accent'], foreground=COLORS['fg'],
                       font=('Arial', 10, 'bold'))
        style.map('Accent.TButton',
                  background=[('active', COLORS['accent_hover']), ('disabled', COLORS['bg_lighter'])])

        # Red - used for Stop and Clear
        style.configure('Danger.TButton', background=COLORS['danger'], foreground=COLORS['fg'],
                       font=('Arial', 10, 'bold'))
        style.map('Danger.TButton',
                  background=[('active', COLORS['danger_hover']), ('disabled', COLORS['bg_lighter'])])

        # Green - used for Save Result
        style.configure('Green.TButton', background=COLORS['green'], foreground=COLORS['fg'],
                       font=('Arial', 10, 'bold'))
        style.map('Green.TButton',
                  background=[('active', COLORS['green_hover']), ('disabled', COLORS['bg_lighter'])])

        style.configure('Success.TLabel', foreground=COLORS['success'], font=('Arial', 10, 'bold'))
        style.configure('Error.TLabel', foreground=COLORS['error'], font=('Arial', 10, 'bold'))

    def setup_ui(self):
        """Setup the user interface"""

        # ============================================================
        # Top Frame - Model Selection & Controls
        # ============================================================
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        # Model info frame
        model_frame = ttk.LabelFrame(top_frame, text="🤖 ONNX Model", padding="10")
        model_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        model_info_frame = ttk.Frame(model_frame)
        model_info_frame.pack(fill=tk.X)

        ttk.Label(model_info_frame, text="Current Model:", font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        self.model_label = ttk.Label(model_info_frame, text="Loading...", style='Error.TLabel')
        self.model_label.pack(side=tk.LEFT, padx=5)

        ttk.Button(model_info_frame, text="📂 Change Model",
                  command=self.load_model).pack(side=tk.LEFT, padx=5)

        ttk.Button(model_info_frame, text="🔄 Reload",
                  command=self.reload_model).pack(side=tk.LEFT, padx=5)

        # Quick actions frame
        actions_frame = ttk.LabelFrame(top_frame, text="📁 Select Option", padding="10")
        actions_frame.pack(side=tk.RIGHT, padx=(5, 0))

        ttk.Button(actions_frame, text="🖼 Image",
                  command=self.upload_image, width=12, style='Accent.TButton').pack(side=tk.LEFT, padx=3)
        ttk.Button(actions_frame, text="🎬 Video",
                  command=self.upload_video, width=12, style='Accent.TButton').pack(side=tk.LEFT, padx=3)
        ttk.Button(actions_frame, text="📹 Webcam",
                  command=self.use_webcam, width=12, style='Accent.TButton').pack(side=tk.LEFT, padx=3)

        # ============================================================
        # Control Frame - Settings & Detection
        # ============================================================
        control_frame = ttk.LabelFrame(self.root, text="⚙️ Detection Settings", padding="10")
        control_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        # File info
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(file_frame, text="📄 Selected File:", font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        self.file_label = ttk.Label(file_frame, text="No file selected", style='Error.TLabel')
        self.file_label.pack(side=tk.LEFT, padx=5)

        # Settings
        settings_frame = ttk.Frame(control_frame)
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        # Confidence
        conf_frame = ttk.Frame(settings_frame)
        conf_frame.pack(side=tk.LEFT, padx=10)

        ttk.Label(conf_frame, text="Confidence Threshold:", font=('Arial', 9)).pack(anchor=tk.W)
        conf_slider_frame = ttk.Frame(conf_frame)
        conf_slider_frame.pack(fill=tk.X)

        self.conf_var = tk.DoubleVar(value=0.30)
        self.conf_scale = ttk.Scale(conf_slider_frame, from_=0.1, to=0.9,
                                    variable=self.conf_var, orient=tk.HORIZONTAL, length=250)
        self.conf_scale.pack(side=tk.LEFT, padx=(0, 10))
        self.conf_label = ttk.Label(conf_slider_frame, text="0.30",
                                    font=('Arial', 10, 'bold'), foreground=COLORS['accent'])
        self.conf_label.pack(side=tk.LEFT)
        self.conf_var.trace('w', self.update_conf_label)

        # IoU
        iou_frame = ttk.Frame(settings_frame)
        iou_frame.pack(side=tk.LEFT, padx=10)

        ttk.Label(iou_frame, text="IoU Threshold:", font=('Arial', 9)).pack(anchor=tk.W)
        iou_slider_frame = ttk.Frame(iou_frame)
        iou_slider_frame.pack(fill=tk.X)

        self.iou_var = tk.DoubleVar(value=0.30)
        self.iou_scale = ttk.Scale(iou_slider_frame, from_=0.1, to=0.9,
                                   variable=self.iou_var, orient=tk.HORIZONTAL, length=250)
        self.iou_scale.pack(side=tk.LEFT, padx=(0, 10))
        self.iou_label = ttk.Label(iou_slider_frame, text="0.30",
                                   font=('Arial', 10, 'bold'), foreground=COLORS['accent'])
        self.iou_label.pack(side=tk.LEFT)
        self.iou_var.trace('w', self.update_iou_label)

        # Action buttons
        action_frame = ttk.Frame(control_frame)
        action_frame.pack(fill=tk.X)

        self.detect_btn = ttk.Button(action_frame, text="🔍 Run Detection",
                                     command=self.detect_objects, state=tk.DISABLED,
                                     style='Accent.TButton')
        self.detect_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(action_frame, text="⏹ Stop",
                                   command=self.stop_detection, state=tk.DISABLED,
                                   style='Danger.TButton')
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = ttk.Button(action_frame, text="🗑 Clear",
                                    command=self.clear_display,
                                    style='Danger.TButton')
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = ttk.Button(action_frame, text="💾 Save Result",
                                   command=self.save_result, state=tk.DISABLED,
                                   style='Green.TButton')
        self.save_btn.pack(side=tk.LEFT, padx=5)

        self.swap_cam_btn = ttk.Button(action_frame, text="🔀 Swap Cam",
                                       command=self.swap_camera, state=tk.DISABLED,
                                       style='Accent.TButton')
        self.swap_cam_btn.pack(side=tk.LEFT, padx=5)

        # ============================================================
        # Main Content Area - Display & Results
        # ============================================================
        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Left side - Image/Video Display
        left_frame = ttk.LabelFrame(content_frame, text="📺 Display", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Canvas for image/video
        self.canvas = tk.Canvas(left_frame, bg=COLORS['bg_light'],
                               highlightbackground=COLORS['border'], highlightthickness=1)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Add placeholder text (kept centered dynamically via <Configure>)
        self.canvas_text = self.canvas.create_text(
            400, 300,
            text="📟 Makers - YOLO Object Detection\n\nClick 'Image', 'Video', or 'Webcam' to start",
            fill=COLORS['fg_dim'],
            font=('Arial', 16),
            justify=tk.CENTER
        )
        self.canvas.bind('<Configure>', self.on_canvas_resize)

        # Progress bar - only shown while actively processing, so it doesn't
        # sit under the display as a permanent, non-functional bar
        self.progress = ttk.Progressbar(left_frame, mode='indeterminate')

        # Right side - Detection Results
        right_frame = ttk.LabelFrame(content_frame, text="📊 Detection Results", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))

        # Results container with scrollbar
        result_container = ttk.Frame(right_frame)
        result_container.pack(fill=tk.BOTH, expand=True)

        result_scroll = ttk.Scrollbar(result_container)
        result_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.result_canvas = tk.Canvas(result_container, bg=COLORS['bg_light'],
                                       highlightbackground=COLORS['border'],
                                       highlightthickness=1, width=380,
                                       yscrollcommand=result_scroll.set)
        self.result_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        result_scroll.config(command=self.result_canvas.yview)

        # Frame inside canvas for results
        self.result_frame = ttk.Frame(self.result_canvas)
        self.result_canvas_window = self.result_canvas.create_window(
            (0, 0), window=self.result_frame, anchor=tk.NW
        )

        self.result_frame.bind('<Configure>', self.on_result_frame_configure)
        self.result_canvas.bind('<Configure>', self.on_result_canvas_configure)

        # Initial placeholder
        self.show_no_results()

        # ============================================================
        # Status Bar
        # ============================================================
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(0, 10))

        self.status_label = tk.Label(status_frame, text="Ready",
                                     bg=COLORS['bg_light'], fg=COLORS['fg'],
                                     anchor=tk.W, padx=10, pady=5,
                                     relief=tk.SOLID, borderwidth=1)
        self.status_label.pack(fill=tk.X)

    def on_canvas_resize(self, event):
        """Keep the placeholder text perfectly centered in the display canvas."""
        if self.placeholder_visible:
            try:
                self.canvas.coords(self.canvas_text, event.width // 2, event.height // 2)
            except tk.TclError:
                pass

    def on_result_frame_configure(self, event=None):
        """Reset the scroll region to encompass the inner frame"""
        self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all"))

    def on_result_canvas_configure(self, event):
        """Update the inner frame width to fill the canvas"""
        canvas_width = event.width
        self.result_canvas.itemconfig(self.result_canvas_window, width=canvas_width)

    def update_conf_label(self, *args):
        """Update confidence label"""
        self.conf_label.config(text=f"{self.conf_var.get():.2f}")

    def update_iou_label(self, *args):
        """Update IoU label"""
        self.iou_label.config(text=f"{self.iou_var.get():.2f}")

    def update_status(self, message, color=None):
        """Update status bar"""
        self.status_label.config(text=message)
        if color:
            self.status_label.config(fg=color)
        else:
            self.status_label.config(fg=COLORS['fg'])
        self.root.update_idletasks()

    # ============================================================
    # Camera detection / swapping
    # ============================================================
    def _detect_cameras_bg(self):
        """Probe a few camera indices in a background thread."""
        found = []
        for i in range(3):
            try:
                cap = cv2.VideoCapture(i)
                if cap is not None and cap.isOpened():
                    found.append(i)
                cap.release()
            except Exception:
                pass
        if not found:
            found = [0]
        self.root.after(0, lambda: self._on_cameras_detected(found))

    def _on_cameras_detected(self, cams):
        self.available_cameras = cams
        self.current_cam_index = cams[0]
        if len(cams) > 1:
            self.swap_cam_btn.config(state=tk.NORMAL)
        else:
            self.swap_cam_btn.config(state=tk.DISABLED)

    def swap_camera(self):
        """Cycle to the next detected camera index.

        This is done as a clean stop -> wait for the capture to fully
        release -> restart sequence, rather than just flipping the index,
        so the old and new camera handles never overlap (which is what
        was causing "both open" / "both closed" glitches).
        """
        if len(self.available_cameras) < 2:
            return
        if getattr(self, '_swapping', False):
            return  # a swap is already in progress; ignore repeat clicks

        try:
            idx = self.available_cameras.index(self.current_cam_index)
        except ValueError:
            idx = -1
        idx = (idx + 1) % len(self.available_cameras)
        new_index = self.available_cameras[idx]
        self.current_cam_index = new_index

        if not isinstance(self.current_file, int):
            # Webcam isn't the active source right now - just remember the
            # choice for the next time "Webcam" is selected.
            self.update_status(
                f"Camera set to index {new_index} (used next time you select Webcam)",
                COLORS['accent']
            )
            return

        self._swapping = True
        self.swap_cam_btn.config(state=tk.DISABLED)

        if self.is_processing:
            # A capture loop is currently running on the old camera. Ask it
            # to stop, then wait (off the UI thread) for it to fully release
            # the device before opening the new one.
            self.update_status(f"Switching to camera {new_index}...", COLORS['warning'])
            self.stop_video = True
            old_thread = self.video_thread
            threading.Thread(target=self._wait_and_restart_camera, args=(old_thread, new_index), daemon=True).start()
        else:
            # No active stream - safe to switch immediately.
            self.current_file = new_index
            self.file_label.config(text=f"📹 Webcam {new_index}", style='Success.TLabel')
            self.update_status(f"Switched to camera index {new_index}", COLORS['accent'])
            self._swapping = False
            self.swap_cam_btn.config(state=tk.NORMAL)

    def _wait_and_restart_camera(self, old_thread, new_index):
        """Runs in a background thread: waits for the previous capture
        thread to fully exit (and release its camera) before switching
        to the new camera and automatically resuming detection."""
        if old_thread is not None:
            old_thread.join(timeout=5)
        # Hand back to the UI thread to touch widgets / state safely.
        self.root.after(0, lambda: self._finish_camera_swap(new_index))

    def _finish_camera_swap(self, new_index):
        self.current_file = new_index
        self.file_label.config(text=f"📹 Webcam {new_index}", style='Success.TLabel')
        self._swapping = False
        self.swap_cam_btn.config(state=tk.NORMAL if len(self.available_cameras) > 1 else tk.DISABLED)
        self.update_status(f"Switched to camera index {new_index}", COLORS['accent'])
        # Seamlessly resume detection on the new camera
        self.detect_objects()

    # ============================================================
    # Model loading (ONNX)
    # ============================================================
    def auto_load_model(self):
        """Auto-load model from models folder"""
        models_folder = Path('models')
        model_path = models_folder / 'yolo11n.onnx'

        if model_path.exists():
            self.update_status("Loading default model from models/yolo11n.onnx...", COLORS['warning'])
            self.load_model_file(str(model_path))
        else:
            # Model folder doesn't exist or no model found
            self.update_status("No model found in 'models' folder. Please select a model.", COLORS['error'])
            self.model_label.config(text="No model loaded", style='Error.TLabel')

            # Ask user to select model
            if messagebox.askyesno("Model Not Found",
                                  "Model file 'models/yolo11n.onnx' not found.\n\n"
                                  "Would you like to select an ONNX model file now?"):
                self.load_model()

    def load_model(self):
        """Load YOLO ONNX model from file dialog"""
        file_path = filedialog.askopenfilename(
            title="Select YOLO ONNX Model File",
            filetypes=[("ONNX Models", "*.onnx"), ("All files", "*.*")],
            initialdir="models" if Path("models").exists() else "."
        )

        if not file_path:
            return

        self.load_model_file(file_path)

    def load_model_file(self, file_path):
        """Load ONNX model from given path"""
        try:
            self.update_status("Loading model...", COLORS['warning'])
            self.model = ONNXYOLO(file_path)
            self.model_path = file_path

            model_name = Path(file_path).name

            self.model_label.config(
                text=f"✓ {model_name}",
                style='Success.TLabel'
            )

            self.update_status(f"Model loaded successfully: {model_name}", COLORS['success'])

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.model_label.config(text="❌ Failed to load", style='Error.TLabel')
            self.update_status("Error loading model", COLORS['error'])

    def reload_model(self):
        """Reload the current model"""
        if self.model_path:
            self.load_model_file(self.model_path)
        else:
            messagebox.showinfo("Info", "No model to reload. Please load a model first.")

    def upload_image(self):
        """Upload and display image"""
        if not self.model:
            messagebox.showwarning("Warning", "Please load a model first!")
            return

        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )

        if not file_path:
            return

        self.current_file = file_path
        self.file_label.config(text=Path(file_path).name, style='Success.TLabel')

        # Display image
        self.display_image(file_path)

        self.detect_btn.config(state=tk.NORMAL)
        self.update_status(f"Image loaded: {Path(file_path).name}", COLORS['success'])

    def upload_video(self):
        """Upload video"""
        if not self.model:
            messagebox.showwarning("Warning", "Please load a model first!")
            return

        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )

        if not file_path:
            return

        self.current_file = file_path
        self.file_label.config(text=Path(file_path).name, style='Success.TLabel')

        self.detect_btn.config(state=tk.NORMAL)
        self.update_status(f"Video loaded: {Path(file_path).name}", COLORS['success'])

    def use_webcam(self):
        """Use webcam for detection"""
        if not self.model:
            messagebox.showwarning("Warning", "Please load a model first!")
            return

        self.current_file = self.current_cam_index
        self.file_label.config(text=f"📹 Webcam {self.current_cam_index}", style='Success.TLabel')
        self.detect_btn.config(state=tk.NORMAL)
        self.update_status(f"Webcam {self.current_cam_index} selected", COLORS['success'])

    def display_image(self, image_path):
        """Display image on canvas"""
        try:
            # Remove placeholder text
            self.placeholder_visible = False
            self.canvas.delete(self.canvas_text)

            # Load and resize image
            img = Image.open(image_path)

            # Calculate resize to fit canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            if canvas_width <= 1:
                canvas_width = 800
            if canvas_height <= 1:
                canvas_height = 600

            img.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)

            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(img)

            # Display on canvas
            self.canvas.delete("all")
            self.canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=self.photo,
                anchor=tk.CENTER
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image:\n{str(e)}")

    def show_no_results(self):
        """Show placeholder when no results"""
        # Clear existing widgets
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        placeholder = tk.Label(
            self.result_frame,
            text="No detections yet\n\nUpload an image\nand click 'Run Detection'",
            bg=COLORS['bg_light'],
            fg=COLORS['fg_dim'],
            font=('Arial', 12),
            justify=tk.CENTER,
            pady=50
        )
        placeholder.pack(fill=tk.BOTH, expand=True)

    def display_results(self, result, frame_num=None):
        """Display detection results in structured format"""
        # Clear existing widgets
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        boxes = result.boxes

        # Header
        if frame_num:
            header = tk.Label(
                self.result_frame,
                text=f"FRAME {frame_num}",
                bg=COLORS['bg_lighter'],
                fg=COLORS['accent'],
                font=('Arial', 12, 'bold'),
                pady=10
            )
            header.pack(fill=tk.X, pady=(0, 5))

        # Total detections card
        total_frame = tk.Frame(self.result_frame, bg=COLORS['accent'], pady=2)
        total_frame.pack(fill=tk.X, pady=(0, 10))

        total_inner = tk.Frame(total_frame, bg=COLORS['bg_lighter'], pady=15)
        total_inner.pack(fill=tk.BOTH, expand=True, padx=2)

        total_label = tk.Label(
            total_inner,
            text=f"{len(boxes)}",
            bg=COLORS['bg_lighter'],
            fg=COLORS['accent'],
            font=('Arial', 36, 'bold')
        )
        total_label.pack()

        total_text = tk.Label(
            total_inner,
            text="Objects Detected",
            bg=COLORS['bg_lighter'],
            fg=COLORS['fg_dim'],
            font=('Arial', 10)
        )
        total_text.pack()

        if len(boxes) == 0:
            no_detect = tk.Label(
                self.result_frame,
                text="No objects detected",
                bg=COLORS['bg_light'],
                fg=COLORS['fg_dim'],
                font=('Arial', 11),
                pady=20
            )
            no_detect.pack()
            return

        # Class summary
        summary_header = tk.Label(
            self.result_frame,
            text="📈 Summary by Class",
            bg=COLORS['bg_light'],
            fg=COLORS['accent'],
            font=('Arial', 11, 'bold'),
            anchor=tk.W,
            pady=8,
            padx=10
        )
        summary_header.pack(fill=tk.X, pady=(5, 0))

        # Count detections by class
        class_counts = {}
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = result.names[cls_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # Display summary cards
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            self.create_summary_card(class_name, count)

        # Separator
        separator = tk.Frame(self.result_frame, bg=COLORS['border'], height=2)
        separator.pack(fill=tk.X, pady=15)

        # Detailed detections
        detail_header = tk.Label(
            self.result_frame,
            text="🔍 Detailed Detections",
            bg=COLORS['bg_light'],
            fg=COLORS['accent'],
            font=('Arial', 11, 'bold'),
            anchor=tk.W,
            pady=8,
            padx=10
        )
        detail_header.pack(fill=tk.X)

        # Display individual detections
        for i, box in enumerate(boxes, 1):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = result.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0]

            self.create_detection_card(i, class_name, conf, x1, y1, x2, y2)

        # Update scroll region
        self.result_frame.update_idletasks()
        self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all"))

    def create_summary_card(self, class_name, count):
        """Create a summary card for each class"""
        card = tk.Frame(self.result_frame, bg=COLORS['bg_lighter'], pady=8, padx=10)
        card.pack(fill=tk.X, pady=2, padx=5)

        # Class name
        name_label = tk.Label(
            card,
            text=class_name.capitalize(),
            bg=COLORS['bg_lighter'],
            fg=COLORS['fg'],
            font=('Arial', 10, 'bold'),
            anchor=tk.W
        )
        name_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Count badge
        count_frame = tk.Frame(card, bg=COLORS['accent'], padx=12, pady=4)
        count_frame.pack(side=tk.RIGHT)

        count_label = tk.Label(
            count_frame,
            text=str(count),
            bg=COLORS['accent'],
            fg=COLORS['fg'],
            font=('Arial', 11, 'bold')
        )
        count_label.pack()

    def create_detection_card(self, index, class_name, conf, x1, y1, x2, y2):
        """Create a detection card for individual detection"""

        # Main card
        card = tk.Frame(
            self.result_frame,
            bg=COLORS['bg_lighter'],
            highlightbackground=COLORS['border'],
            highlightthickness=1
        )
        card.pack(fill=tk.X, pady=3, padx=5)

        # Header
        header = tk.Frame(card, bg=COLORS['bg_lighter'], padx=10, pady=8)
        header.pack(fill=tk.X)

        index_label = tk.Label(
            header,
            text=f"#{index}",
            bg=COLORS['bg_lighter'],
            fg=COLORS['accent'],
            font=('Arial', 9, 'bold'),
            width=4
        )
        index_label.pack(side=tk.LEFT)

        class_label = tk.Label(
            header,
            text=class_name.capitalize(),
            bg=COLORS['bg_lighter'],
            fg=COLORS['fg'],
            font=('Arial', 10, 'bold'),
            anchor=tk.W
        )
        class_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Confidence section
        conf_frame = tk.Frame(card, bg=COLORS['bg_lighter'], padx=10)
        conf_frame.pack(fill=tk.X)

        conf_label = tk.Label(
            conf_frame,
            text="Confidence:",
            bg=COLORS['bg_lighter'],
            fg=COLORS['fg_dim'],
            font=('Arial', 8)
        )
        conf_label.pack(side=tk.LEFT)

        # --- Fixed-width confidence bar -------------------------------
        # The background trough has a FIXED pixel width (BAR_WIDTH) and does
        # NOT expand to fill the row. The fill bar's width is computed as a
        # fraction of that same fixed width, so 100% confidence == BAR_WIDTH
        # pixels exactly (previously the trough expanded to fill available
        # space while the fill used raw percentage-as-pixels, so a "full"
        # bar visually overshot the trough, looking like ~200%).
        BAR_WIDTH = 120
        conf_bar_bg = tk.Frame(conf_frame, bg=COLORS['bg'], width=BAR_WIDTH, height=8)
        conf_bar_bg.pack(side=tk.LEFT, padx=5)
        conf_bar_bg.pack_propagate(False)

        conf_pct = max(0.0, min(1.0, conf))
        conf_width = max(1, int(conf_pct * BAR_WIDTH))
        conf_color = (
            COLORS['success'] if conf_pct > 0.7
            else COLORS['warning'] if conf_pct > 0.4
            else COLORS['error']
        )

        conf_bar = tk.Frame(conf_bar_bg, bg=conf_color, width=conf_width, height=8)
        conf_bar.pack(side=tk.LEFT, fill=tk.Y)
        # ----------------------------------------------------------------

        conf_percent = tk.Label(
            conf_frame,
            text=f"{conf_pct*100:.1f}%",
            bg=COLORS['bg_lighter'],
            fg=conf_color,
            font=('Arial', 9, 'bold')
        )
        conf_percent.pack(side=tk.LEFT, padx=5)

        # Bounding box info
        bbox_frame = tk.Frame(card, bg=COLORS['bg_lighter'], padx=10)
        bbox_frame.pack(fill=tk.X, pady=(5, 8))

        bbox_label = tk.Label(
            bbox_frame,
            text=f"Box: ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})",
            bg=COLORS['bg_lighter'],
            fg=COLORS['fg_dim'],
            font=('Courier', 8)
        )
        bbox_label.pack(anchor=tk.W)

    def detect_objects(self):
        """Run object detection"""
        if not self.model:
            messagebox.showwarning("Warning", "Please load a model first!")
            return

        if self.current_file is None:
            messagebox.showwarning("Warning", "Please select an image or video first!")
            return

        # Disable buttons during detection
        self.detect_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.is_processing = True
        self.stop_video = False

        # Check if image or video
        if isinstance(self.current_file, int) or str(self.current_file).endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # Video or webcam - run in thread
            self.video_thread = threading.Thread(target=self.process_video, daemon=True)
            self.video_thread.start()
        else:
            # Image - process directly
            self.progress.pack(fill=tk.X, pady=(5, 0))
            self.progress.start()
            self.root.after(100, self.process_image)

    def process_image(self):
        """Process single image"""
        try:
            self.update_status("Processing image...", COLORS['warning'])

            img_bgr = cv2.imread(str(self.current_file))
            if img_bgr is None:
                raise ValueError("Could not read the selected image file")

            # Run detection
            result, annotated_bgr = self.model.infer(
                img_bgr,
                conf=self.conf_var.get(),
                iou=self.iou_var.get()
            )

            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            # Display result
            self.placeholder_visible = False
            img = Image.fromarray(annotated_rgb)
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            if canvas_width <= 1:
                canvas_width = 800
            if canvas_height <= 1:
                canvas_height = 600

            img.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)

            self.photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=self.photo,
                anchor=tk.CENTER
            )

            # Display detection results - wrapped in after() to ensure canvas is ready
            self.root.after(100, lambda: self.display_results(result))

            # Save for later
            self.last_result = annotated_rgb
            self.save_btn.config(state=tk.NORMAL)

            self.update_status(f"✓ Detection complete: {len(result.boxes)} objects found", COLORS['success'])

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Detection failed:\n{str(e)}")
            self.update_status("Detection failed", COLORS['error'])

        finally:
            self.progress.stop()
            self.progress.pack_forget()
            self.detect_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.is_processing = False

    def process_video(self):
        """Process video or webcam"""
        try:
            cap = cv2.VideoCapture(self.current_file)

            if not cap.isOpened():
                messagebox.showerror("Error", "Failed to open video source!")
                return

            frame_count = 0
            total_detections = 0

            while cap.isOpened() and not self.stop_video:
                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                # Run detection
                result, annotated_bgr = self.model.infer(
                    frame,
                    conf=self.conf_var.get(),
                    iou=self.iou_var.get()
                )

                total_detections += len(result.boxes)

                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                # Display frame
                self.placeholder_visible = False
                img = Image.fromarray(annotated_rgb)
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                img.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)

                self.photo = ImageTk.PhotoImage(img)
                self.canvas.delete("all")
                self.canvas.create_image(
                    canvas_width // 2,
                    canvas_height // 2,
                    image=self.photo,
                    anchor=tk.CENTER
                )

                # Update status
                self.update_status(f"Frame {frame_count} - {len(result.boxes)} objects detected", COLORS['warning'])

                # Update results every 10 frames
                # if frame_count % 10 == 0:
                #     self.display_results(result, frame_count)

                # Small delay for display
                time.sleep(0.01)

            cap.release()

            if not self.stop_video:
                self.update_status(f"✓ Video complete: {frame_count} frames, {total_detections} total detections", COLORS['success'])
                messagebox.showinfo("Complete",
                                  f"Video processing complete!\n"
                                  f"Frames: {frame_count}\n"
                                  f"Total detections: {total_detections}")
            else:
                self.update_status("Video processing stopped", COLORS['warning'])

        except Exception as e:
            messagebox.showerror("Error", f"Video processing failed:\n{str(e)}")
            self.update_status("Video processing failed", COLORS['error'])

        finally:
            self.detect_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.is_processing = False

    def stop_detection(self):
        """Stop video processing"""
        self.stop_video = True
        self.update_status("Stopping...", COLORS['warning'])

    def save_result(self):
        """Save detection result"""
        if not hasattr(self, 'last_result'):
            messagebox.showwarning("Warning", "No result to save!")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Result",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            cv2.imwrite(file_path, cv2.cvtColor(self.last_result, cv2.COLOR_RGB2BGR))
            messagebox.showinfo("Success", f"Result saved to:\n{file_path}")
            self.update_status(f"✓ Result saved: {Path(file_path).name}", COLORS['success'])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save result:\n{str(e)}")

    def clear_display(self):
        """Clear display and results"""
        self.canvas.delete("all")
        self.placeholder_visible = True
        cw = self.canvas.winfo_width() or 800
        ch = self.canvas.winfo_height() or 600
        self.canvas_text = self.canvas.create_text(
            cw // 2, ch // 2,
            text="📟 Makers - YOLO Object Detection\n\nClick 'Image', 'Video', or 'Webcam' to start",
            fill=COLORS['fg_dim'],
            font=('Arial', 16),
            justify=tk.CENTER
        )

        self.show_no_results()

        self.current_file = None
        self.file_label.config(text="No file selected", style='Error.TLabel')
        self.detect_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.update_status("Display cleared", COLORS['fg_dim'])

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = YOLODetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()