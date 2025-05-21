from PyQt5.QtCore import QThread, pyqtSignal, Qt
import cv2
import pandas as pd
from yolo import yolo
from trOCR_ import *
from fuzzywuzzy import process

class DetectionWorker(QThread):
    error_msg = pyqtSignal(str)
    notification = pyqtSignal(str)
    def __init__(self, file_path, excel_path, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.excel_path = excel_path
        self.df = pd.read_excel(self.excel_path, engine='openpyxl')
        self.mssv_list = self.df['MSSV'].astype(str).tolist()

    def lexicon_search(self, mssv_text, threshold=90):
        if not mssv_text or not mssv_text.isdigit():
            return None, None
        match = process.extractOne(mssv_text, self.mssv_list, score_cutoff=threshold)
        if match:
            return match[0], match[1]
        return None, None

    def run(self):
        
        results = yolo(self.file_path, imgsz=960, conf=0.6)
        frame = cv2.imread(self.file_path)

        boxes = results[0].boxes
        if not boxes:
            self.error_msg.emit(f"Không thể đọc {self.file_path}")
            return

        mssv_text, score_text = None, None
        for box in boxes:
            cls_id   = int(box.cls)
            cls_name = yolo.names[cls_id].lower()
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            crop = frame[y1:y2, x1:x2]
            if cls_name == "mssv":
                mssv_text = ocr_trocr(crop, processor_base, trocr_base)
            elif cls_name == "score":
                score_text = ocr_trocr(crop, processor_score, trocr_score)

        success = False
        if mssv_text:
            matched_mssv, match_score = self.lexicon_search(mssv_text)
            if matched_mssv:
                mask = self.df['MSSV'].astype(str) == matched_mssv
                if mask.any():
                    self.df.loc[mask, 'Điểm'] = score_text
                    self.df.to_excel(self.excel_path, index=False, engine='openpyxl')
                    success = True

            
        if success and score_text:
            print(f"MSSV: {mssv_text} \t Điểm: {score_text}")
            self.notification.emit(f"✔️ MSSV: {mssv_text} \t Điểm: {score_text}")
        elif success and not score_text:
            self.notification.emit(f"❌ MSSV: {mssv_text} \t Điểm: Not found")