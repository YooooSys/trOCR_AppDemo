from PyQt5.QtCore import QThread, pyqtSignal
import os

class FolderDetectionWorker(QThread):
    result = pyqtSignal(list)
    error_msg = pyqtSignal(str)

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path

    def run(self):
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
        valid_files = []

        for root, _, files in os.walk(self.folder_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                ext = os.path.splitext(file_path)[1].lower()

                if ext in image_extensions:
                    valid_files.append(file_path)
                else:
                    self.error_msg.emit(f"❌ Bỏ qua file không hợp lệ: {file_path}")

        self.result.emit(valid_files)