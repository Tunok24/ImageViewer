import numpy as np
import SimpleITK as sitk
import cv2
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QSlider, QPushButton, QFileDialog
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import QMenuBar, QAction, QMessageBox


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.create_menu_bar()
        self.setWindowTitle("OpenCV Medical Image Viewer")
        self.setGeometry(100, 100, 1200, 900)

        self.current_image = None
        self.slice_idx = 0
        self.view = 'Axial'
        self.window = 1
        self.level = 0
        self.colormap = 'gray'
        self.shape = (128, 256, 256)
        self.row_idx = 0
        self.col_idx = 0

        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        self.image_label = QLabel("Load an image to begin")
        layout.addWidget(self.image_label)

        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.open_file)
        layout.addWidget(self.load_button)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.valueChanged.connect(self.update_slice)
        layout.addWidget(self.slice_slider)

        self.view_selector = QComboBox()
        self.view_selector.addItems(['Axial', 'Coronal', 'Sagittal'])
        self.view_selector.currentTextChanged.connect(self.set_view)
        layout.addWidget(self.view_selector)

        wl_layout = QHBoxLayout()

        # Window slider: 0 to 5 with 0.01 step → use 0 to 500
        self.window_slider = QSlider(Qt.Horizontal)
        self.window_slider.setRange(0, 500)
        self.window_slider.setValue(int(self.window * 100))  # scale to integer
        self.window_slider.valueChanged.connect(lambda val: self.set_window(val / 100.0))
        wl_layout.addWidget(QLabel("Window"))
        wl_layout.addWidget(self.window_slider)

        # Level slider: 0 to 1 with 0.001 step → use 0 to 1000
        self.level_slider = QSlider(Qt.Horizontal)
        self.level_slider.setRange(0, 1000)
        self.level_slider.setValue(int(self.level * 1000))  # scale to integer
        self.level_slider.valueChanged.connect(lambda val: self.set_level(val / 1000.0))
        wl_layout.addWidget(QLabel("Level"))
        wl_layout.addWidget(self.level_slider)

        layout.addLayout(wl_layout)

        self.colormap_selector = QComboBox()
        self.colormap_selector.addItems(['gray', 'viridis', 'jet'])
        self.colormap_selector.currentTextChanged.connect(self.set_colormap)
        layout.addWidget(self.colormap_selector)

        rc_slider_layout = QHBoxLayout()
        self.row_slider = QSlider(Qt.Horizontal)
        self.row_slider.valueChanged.connect(self.set_row)
        rc_slider_layout.addWidget(QLabel("Row"))
        rc_slider_layout.addWidget(self.row_slider)

        self.col_slider = QSlider(Qt.Horizontal)
        self.col_slider.valueChanged.connect(self.set_col)
        rc_slider_layout.addWidget(QLabel("Col"))
        rc_slider_layout.addWidget(self.col_slider)

        layout.addLayout(rc_slider_layout)

        self.profile_canvas = FigureCanvas(Figure(figsize=(8, 3)))
        self.ax_row = self.profile_canvas.figure.add_subplot(121)
        self.ax_col = self.profile_canvas.figure.add_subplot(122)
        layout.addWidget(self.profile_canvas)

        central_widget.setLayout(layout)
    
    def create_menu_bar(self):
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")
        open_action = QAction("Open", self)
        open_action.triggered.connect(lambda: print("Open clicked"))
        file_menu.addAction(open_action)

        save_action = QAction("Save", self)
        save_action.triggered.connect(lambda: print("Save clicked"))
        file_menu.addAction(save_action)

        # View menu
        view_menu = menu_bar.addMenu("View")
        toggle_axes_action = QAction("Toggle Axes", self)
        toggle_axes_action.triggered.connect(lambda: print("Toggle Axes clicked"))
        view_menu.addAction(toggle_axes_action)

        zoom_action = QAction("Zoom", self)
        zoom_action.triggered.connect(lambda: print("Zoom clicked"))
        view_menu.addAction(zoom_action)

        # Tools menu
        tools_menu = menu_bar.addMenu("Tools")
        denoise_action = QAction("Denoise Image", self)
        denoise_action.triggered.connect(self.setup_denoise_mode)
        tools_menu.addAction(denoise_action)

        edge_action = QAction("Edge Detection", self)
        edge_action.triggered.connect(lambda: print("Edge Detection clicked"))
        tools_menu.addAction(edge_action)

        noise_action = QAction("Noise Map", self)
        noise_action.triggered.connect(lambda: print("Noise Map clicked"))
        tools_menu.addAction(noise_action)

        # Analysis menu
        analysis_menu = menu_bar.addMenu("Analysis")
        nps_action = QAction("Show NPS", self)
        nps_action.triggered.connect(lambda: print("NPS Map clicked"))
        analysis_menu.addAction(nps_action)

        rms_action = QAction("RMS", self)
        rms_action.triggered.connect(lambda: print("RMS clicked"))
        analysis_menu.addAction(rms_action)

        compare_action = QAction("Compare Images", self)
        compare_action.triggered.connect(lambda: print("Compare Images clicked"))
        analysis_menu.addAction(compare_action)

        # Help menu
        help_menu = menu_bar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(lambda: QMessageBox.information(self, "About", "OpenCV Medical Image Viewer\nVersion 1.0"))
        help_menu.addAction(about_action)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.mha *.img)")
        if not path:
            return
        self.load_image(path)

    def load_image(self, path):
        if path.endswith('.mha'):
            img = sitk.ReadImage(path)
            array = sitk.GetArrayFromImage(img)
            modality = img.GetMetaData('0008|0060') if img.HasMetaDataKey('0008|0060') else 'Unknown'
        elif path.endswith('.img'):
            with open(path, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.float32)
                array = data.reshape(self.shape)
            modality = 'Raw'
        else:
            return

        self.current_image = array

        if modality.upper() == 'CT' or (np.min(array) < -500 and np.max(array) > 500):
            self.window = 2
            self.level = 0.2
        elif modality.upper() == 'MR' or (np.min(array) >= 0 and np.max(array) < 3000):
            self.window = np.max(array)
            self.level = self.window / 2
        else:
            self.window = float(np.max(array) - np.min(array))
            self.level = float((np.max(array) + np.min(array)) / 2)

        self.window_slider.setValue(int(self.window))
        self.level_slider.setValue(int(self.level))

        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(array.shape[0] - 1)
        self.slice_slider.setValue(self.slice_idx)

        self.row_slider.setMinimum(0)
        self.row_slider.setMaximum(array.shape[1] - 1)
        self.col_slider.setMinimum(0)
        self.col_slider.setMaximum(array.shape[2] - 1)

        print(f"[INFO] Loaded {path}")
        print(f"Modality: {modality}")
        print(f"Shape: {array.shape}, Min: {np.min(array)}, Max: {np.max(array)}")
        print(f"Window: {self.window}, Level: {self.level}")

        self.update_display()

    def get_slice(self):
        if self.current_image is None:
            return None
        if self.view == 'Axial':
            return self.current_image[self.slice_idx, :, :]
        elif self.view == 'Coronal':
            return self.current_image[:, self.slice_idx, :]
        elif self.view == 'Sagittal':
            return self.current_image[:, :, self.slice_idx]


    def apply_window_level(self, img, window=None, level=None):
        if window is None:window = self.window
        if level is None:level = self.level
        img = img.astype(np.float32)
        min_val = self.level - self.window / 2
        max_val = self.level + self.window / 2
        img = np.clip((img - min_val) / (max_val - min_val), 0, 1)
        img = (img * 255).astype(np.uint8)
        return img

    def colorize(self, img):
        if self.colormap == 'gray':
            return img
        elif self.colormap == 'viridis':
            return cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
        elif self.colormap == 'jet':
            return cv2.applyColorMap(img, cv2.COLORMAP_JET)
        return img

    def update_display(self):
        slice_img = self.get_slice()
        if slice_img is None:
            return
        wl_img = self.apply_window_level(slice_img)
        color_img = self.colorize(wl_img)

        if len(color_img.shape) == 3:
            rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = color_img

        h, w = rgb_img.shape[:2]
        bytes_per_line = 3 * w if len(rgb_img.shape) == 3 else w
        q_img = QImage(rgb_img.data, w, h, bytes_per_line,
                       QImage.Format_RGB888 if len(rgb_img.shape) == 3 else QImage.Format_Grayscale8)
        # draw crosshair on slice image
        slice_img = self.get_slice()
        if 0 <= self.row_idx < slice_img.shape[0]:
            cv2.line(color_img, (0, self.row_idx), (color_img.shape[1]-1, self.row_idx), (255, 0, 0), 1)
        if 0 <= self.col_idx < slice_img.shape[1]:
            cv2.line(color_img, (self.col_idx, 0), (self.col_idx, color_img.shape[0]-1), (255, 0, 0), 1)

        # convert to RGB if needed
        if len(color_img.shape) == 3:
            rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = color_img

        h, w = rgb_img.shape[:2]
        bytes_per_line = 3 * w if len(rgb_img.shape) == 3 else w
        q_img = QImage(rgb_img.data, w, h, bytes_per_line,
                       QImage.Format_RGB888 if len(rgb_img.shape) == 3 else QImage.Format_Grayscale8)
        self.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

        # update profiles
        if 0 <= self.row_idx < slice_img.shape[0] and 0 <= self.col_idx < slice_img.shape[1]:
            row_profile = slice_img[self.row_idx, 20:-20]
            col_profile = slice_img[20:-20, self.col_idx]
            self.ax_row.clear()
            self.ax_row.plot(row_profile)
            self.ax_row.set_title(f"Row {self.row_idx} Profile")
            self.ax_row.set_xlabel("X-axis")
            self.ax_col.clear()
            self.ax_col.plot(col_profile)
            self.ax_col.set_title(f"Col {self.col_idx} Profile")
            self.ax_col.set_xlabel("Y-axis")
            self.profile_canvas.draw()

    def update_slice(self, value):
        self.slice_idx = value
        self.update_display()

    def set_view(self, value):
        self.view = value
        if self.current_image is not None:
            if self.view == 'Axial':
                self.slice_slider.setMaximum(self.current_image.shape[0] - 1)
            elif self.view == 'Coronal':
                self.slice_slider.setMaximum(self.current_image.shape[1] - 1)
            elif self.view == 'Sagittal':
                self.slice_slider.setMaximum(self.current_image.shape[2] - 1)
        self.update_display()

    def set_window(self, value):
        self.window = value
        self.update_display()

    def set_level(self, value):
        self.level = value
        self.update_display()

    def set_colormap(self, value):
        self.colormap = value
        self.update_display()

    def set_row(self, value):
        self.row_idx = value
        self.update_display()

    def set_col(self, value):
        self.col_idx = value
        self.update_display()

    def setup_bm3d_slider(self, layout):
        # BM3D Sigma slider setup
        self.bm3d_sigma_label = QLabel("σ_psd (BM3D)")
        self.bm3d_sigma_slider = QSlider(Qt.Horizontal)
        self.bm3d_sigma_slider.setRange(1, 50)  # maps to 0.01–0.5
        self.bm3d_sigma_slider.setValue(15)
        self.bm3d_sigma_slider.valueChanged.connect(self.update_denoise_display)

        layout.addWidget(self.bm3d_sigma_label)
        layout.addWidget(self.bm3d_sigma_slider)

    def update_filter_visibility(self):
        is_bilateral = self.filter_selector.currentText() == 'Bilateral'

        self.bilateral_d_slider.setVisible(is_bilateral)
        self.bilateral_sigma_color_slider.setVisible(is_bilateral)
        self.bilateral_sigma_space_slider.setVisible(is_bilateral)
        self.bilateral_d_label.setVisible(is_bilateral)
        self.bilateral_sigma_color_label.setVisible(is_bilateral)
        self.bilateral_sigma_space_label.setVisible(is_bilateral)

        self.bm3d_sigma_slider.setVisible(not is_bilateral)
        self.bm3d_sigma_label.setVisible(not is_bilateral)

    def setup_denoise_mode(self):
        if self.current_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image before entering Denoise Mode.")
            return

        cw = self.centralWidget()
        if cw is None:
            cw = QWidget()
            self.setCentralWidget(cw)

        layout = cw.layout()
        if layout is not None:
            for i in reversed(range(layout.count())):
                item = layout.itemAt(i)
                if item.widget():
                    item.widget().setParent(None)
        else:
            layout = QVBoxLayout()
            cw.setLayout(layout)

        # --- Image comparison display ---
        img_layout = QHBoxLayout()

        # Input image side
        left_col = QVBoxLayout()
        left_col.addWidget(QLabel("Input Image"))
        self.original_img_label = QLabel()
        left_col.addWidget(self.original_img_label)

        # Denoised image side
        right_col = QVBoxLayout()
        right_col.addWidget(QLabel("De-noised Image"))
        self.denoised_img_label = QLabel()
        right_col.addWidget(self.denoised_img_label)

        img_layout.addLayout(left_col)
        img_layout.addLayout(right_col)
        layout.addLayout(img_layout)

        # --- Slice selection ---
        self.denoise_slice_slider = QSlider(Qt.Horizontal)
        self.denoise_slice_slider.setMinimum(0)
        self.denoise_slice_slider.setMaximum(self.current_image.shape[0] - 1)
        self.denoise_slice_slider.setValue(0)
        self.denoise_slice_slider.valueChanged.connect(self.update_denoise_display)
        layout.addWidget(self.denoise_slice_slider)

        # --- Filter selector ---
        self.filter_selector = QComboBox()
        self.filter_selector.addItems(['Bilateral', 'BM3D'])
        self.filter_selector.currentTextChanged.connect(self.update_filter_visibility)
        self.filter_selector.currentTextChanged.connect(self.update_denoise_display)
        layout.addWidget(self.filter_selector)

        # --- Bilateral filter sliders ---
        bilateral_layout = QHBoxLayout()

        self.bilateral_d_label = QLabel("d")
        self.bilateral_d_slider = QSlider(Qt.Horizontal)
        self.bilateral_d_slider.setRange(1, 15)
        self.bilateral_d_slider.setValue(9)
        self.bilateral_d_slider.valueChanged.connect(self.update_denoise_display)

        self.bilateral_sigma_color_label = QLabel("σColor")
        self.bilateral_sigma_color_slider = QSlider(Qt.Horizontal)
        self.bilateral_sigma_color_slider.setRange(1, 150)
        self.bilateral_sigma_color_slider.setValue(75)
        self.bilateral_sigma_color_slider.valueChanged.connect(self.update_denoise_display)

        self.bilateral_sigma_space_label = QLabel("σSpace")
        self.bilateral_sigma_space_slider = QSlider(Qt.Horizontal)
        self.bilateral_sigma_space_slider.setRange(1, 150)
        self.bilateral_sigma_space_slider.setValue(75)
        self.bilateral_sigma_space_slider.valueChanged.connect(self.update_denoise_display)

        bilateral_layout.addWidget(self.bilateral_d_label)
        bilateral_layout.addWidget(self.bilateral_d_slider)
        bilateral_layout.addWidget(self.bilateral_sigma_color_label)
        bilateral_layout.addWidget(self.bilateral_sigma_color_slider)
        bilateral_layout.addWidget(self.bilateral_sigma_space_label)
        bilateral_layout.addWidget(self.bilateral_sigma_space_slider)

        layout.addLayout(bilateral_layout)

        # BM3D slider (hidden unless selected)
        self.bm3d_sigma_label = QLabel("σ_psd (BM3D)")
        self.bm3d_sigma_slider = QSlider(Qt.Horizontal)
        self.bm3d_sigma_slider.setRange(1, 50)  # maps to 0.01 – 0.5
        self.bm3d_sigma_slider.setValue(15)
        self.bm3d_sigma_slider.valueChanged.connect(self.update_denoise_display)

        layout.addWidget(self.bm3d_sigma_label)
        layout.addWidget(self.bm3d_sigma_slider)

        # Finish setup
        self.update_filter_visibility()
        self.update_denoise_display()

        # Residual display row (stacked under image row)
        residual_row = QVBoxLayout()
        residual_row.addWidget(QLabel("Residual (Input - Denoised)"))
        self.residual_img_label = QLabel()
        residual_row.addWidget(self.residual_img_label)
        layout.addLayout(residual_row)

    def update_denoise_display(self):
        slice_idx = self.denoise_slice_slider.value()
        slice_img = self.current_image[slice_idx, :, :].astype(np.float32)
        wl_img = self.apply_window_level(slice_img, self.window, self.level)
        original_display = self.colorize(wl_img)

        # Apply denoising
        if self.filter_selector.currentText() == 'Bilateral':
            d = self.bilateral_d_slider.value()
            sigma_color = self.bilateral_sigma_color_slider.value()
            sigma_space = self.bilateral_sigma_space_slider.value()
            denoised = cv2.bilateralFilter(wl_img, d, sigma_color, sigma_space)

        elif self.filter_selector.currentText() == 'BM3D':
            try:
                import bm3d
                sigma_val = self.bm3d_sigma_slider.value() / 100.0
                normalized = slice_img / 255.0
                denoised = bm3d.bm3d(normalized, sigma_val)
                denoised = np.clip(denoised, 0, 1) * 255
                denoised = denoised.astype(np.uint8)
            except ImportError:
                denoised = cv2.GaussianBlur(wl_img, (5, 5), 1)


        denoised_display = self.colorize(denoised)

        def show_on_label(label, image):
            if len(image.shape) == 2:
                qimg = QImage(image.data, image.shape[1], image.shape[0], image.shape[1], QImage.Format_Grayscale8)
            else:
                rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb_img.data, rgb_img.shape[1], rgb_img.shape[0], 3 * rgb_img.shape[1], QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(qimg).scaled(label.width(), label.height(), Qt.KeepAspectRatio))

        show_on_label(self.original_img_label, original_display)
        show_on_label(self.denoised_img_label, denoised_display)
