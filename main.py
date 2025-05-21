import sys
import time
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit,QTextBrowser,
    QPushButton, QTextEdit, QVBoxLayout, QHBoxLayout, QFileDialog
)
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from transformers import logging as transformers_logging
import logging
from DetectionWorker import DetectionWorker
from yolo import yolo
from trOCR_ import *
from device import device
from FolderDetectWorker import *
from collections import deque
import os
from urllib.parse import quote
cwd = os.path.dirname(os.path.abspath(__file__))


excel_path = os.path.join(cwd,r"Book1.xlsx")

# Set log level
transformers_logging.set_verbosity_error()
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Device
trocr_base.to(device)
trocr_score.to(device)

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Phần mềm nhập điểm tự động")
        self.resize(1600, 800)
        self.file_path = ""
        self.folder_path = ""
        self.init_ui()
        self.detection_worker = None
        self.stop_request = False

    def clickable_path(file_path):
        url = f"file:///{quote(file_path.replace(os.sep, '/'))}"
        return f'<a href="{url}">{file_path}</a>'

    def left_Layout(self):
        # Log
        self.log = QTextBrowser()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("font-size: 14pt;")
        self.log.setOpenExternalLinks(True)

        # Error Log
        self.error_log = QTextBrowser()
        self.error_log.setReadOnly(True)
        self.error_log.setStyleSheet("font-size: 14pt;")
        self.error_log.setOpenExternalLinks(True)

        self.left_layout = QVBoxLayout()
        self.left_layout.addWidget(self.title)
        self.left_layout.addWidget(self.Input_btn)
        self.left_layout.addWidget(self.InputFolder_btn)
        self.left_layout.addWidget(self.DetectFile_btn)
        self.left_layout.addWidget(self.DetectFolder_btn)
        self.left_layout.addWidget(self.Stop_btn)
        self.left_layout.addWidget(self.log)
        self.left_layout.addStretch()
        self.left_layout.addWidget(self.error_log)
        self.left_layout.addStretch()

    def right_Layout(self):
        self.right_layout_title = QLabel("Thông tin")
        self.right_layout_title.setStyleSheet("font-size: 20pt; font-weight: bold;")

        self.right_layout_info = QLabel()

        self.right_layout = QVBoxLayout()
        self.right_layout.addWidget(self.right_layout_title,1)
        self.right_layout.addWidget(self.right_layout_info,10)

    def layout(self):
        self.left_Layout()
        self.right_Layout()
        main_layout = QHBoxLayout(self)
        main_layout.addLayout(self.left_layout, 3)
        main_layout.addLayout(self.right_layout, 1)
    
    def buttons(self):
        self.Input_btn          = QPushButton("Chọn file ảnh")
        self.InputFolder_btn    = QPushButton("Chọn folder chứa ảnh")
        self.DetectFile_btn     = QPushButton("Nhập điểm từ file")
        self.DetectFolder_btn   = QPushButton("Nhập điểm từ folder")
        self.Stop_btn           = QPushButton("Hủy")
        
        for btn in (self.Input_btn, self.InputFolder_btn, self.DetectFile_btn, self.DetectFolder_btn, self.Stop_btn):
            btn.setStyleSheet("font-size: 14pt; padding: 10px;")

    def init_ui(self):
        self.title = QLabel("PHẦN MỀM NHẬP ĐIỂM TỰ ĐỘNG")
        self.title.setStyleSheet("font-size: 24pt; font-weight: bold;")

        self.buttons()
        self.layout()

        self.DetectFolder_btn.setEnabled(False)
        self.DetectFile_btn.setEnabled(False)


        self.Input_btn.clicked.connect(self.open_file_dialog)
        self.InputFolder_btn.clicked.connect(self.open_folder_dialog)

        self.DetectFolder_btn.clicked.connect(self.folder_detect)
        self.DetectFile_btn.clicked.connect(self.file_detect)
        
        self.Stop_btn.clicked.connect(self.stop_processing)

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.file_path = file_path
            self.DetectFile_btn.setEnabled(True)
            self.DetectFolder_btn.setEnabled(False)
            self.stop_request = False

    def open_folder_dialog(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.folder_path = folder_path
            self.DetectFile_btn.setEnabled(False)
            self.DetectFolder_btn.setEnabled(True)
            self.right_layout_info.setText(folder_path)
            self.stop_request = False

    def file_detect(self):

        image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}

        ext = os.path.splitext(self.file_path)[1].lower()
        if ext in image_extensions:
            self.detection_worker = DetectionWorker(self.file_path, excel_path)
            self.detection_worker.notification.connect(self.log.append)
            self.detection_worker.error_msg.connect(self.error_log.append)

            self.detection_worker.start()
        else:
            self.error_log.append(f"File không hợp lệ {self.clickable_path(self.file_path)}")

    def folder_detect(self):
        self.Enable_btn(False)

        self.file_queue = deque()
        self.folder_worker = FolderDetectionWorker(self.folder_path)
        self.folder_worker.result.connect(self.start_detection_queue)
        self.folder_worker.error_msg.connect(self.error_log.append)
        self.folder_worker.start()

    def start_detection_queue(self, file_list):
        self.file_queue = deque(file_list)
        self.process_next_file()

    def process_next_file(self):
        if self.stop_request:
            return
        
        if not self.file_queue:
            self.log.append("✅ All files processed.")
            self.Enable_btn(True)
            return

        file_path = self.file_queue.popleft()
        self.file_path = file_path

        self.detection_worker = DetectionWorker(self.file_path, excel_path)
        self.detection_worker.notification.connect(self.log.append)
        self.detection_worker.error_msg.connect(self.error_log.append)
        self.detection_worker.finished.connect(self.process_next_file)
        self.detection_worker.start()

    def stop_processing(self):
        self.stop_request = True 
        self.Enable_btn(True)

    def Enable_btn(self, con=True):
        self.DetectFolder_btn.setEnabled(not con)
        self.Input_btn.setEnabled(con)
        self.InputFolder_btn.setEnabled(con)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = App()
    win.show()
    sys.exit(app.exec_())
