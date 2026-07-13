import os
import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class SemaphoreDetector:
    # 6 landmark penting: Bahu (11, 12), Siku (13, 14), Pergelangan Tangan (15, 16)
    IMPORTANT_LANDMARKS = [11, 12, 13, 14, 15, 16]

    def __init__(self):
        self.pose_model_path = os.getenv("POSE_MODEL_PATH", "data/models/pose_landmarker_full.task")
        self.rf_model_path = os.getenv("SEMAPHORE_MODEL_PATH", "data/models/semaphore_rf.pkl")
        
        self.landmarker = self._init_landmarker()
        self.classifier = self._init_classifier()

    def _init_landmarker(self):
        if not os.path.exists(self.pose_model_path):
            raise FileNotFoundError(f"Pose landmarker model tidak ditemukan di: {self.pose_model_path}")
            
        base_options = python.BaseOptions(model_asset_path=self.pose_model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.6,
            min_pose_tracking_confidence=0.6
        )
        return vision.PoseLandmarker.create_from_options(options)

    def _init_classifier(self):
        if not os.path.exists(self.rf_model_path):
            raise FileNotFoundError(f"Random Forest model tidak ditemukan di: {self.rf_model_path}")
        return joblib.load(self.rf_model_path)

    @staticmethod
    def arm_direction_angle(shoulder_x, shoulder_y, wrist_x, wrist_y):
        """
        Menghitung sudut kemiringan arah lengan dalam derajat (degrees)
        menggunakan koordinat bahu dan pergelangan tangan.
        """
        return np.degrees(
            np.arctan2(
                wrist_y - shoulder_y,
                wrist_x - shoulder_x
            )
        )

    def _extract_features(self, pose_landmarks):
        """
        Mengekstrak 20 fitur:
        - 18 fitur dari koordinat x, y, z pada 6 titik landmark penting (11 s.d. 16)
        - 2 fitur sudut kemiringan lengan kiri dan kanan
        """
        landmarks = pose_landmarks[0]
        features = []

        # 1. Ambil koordinat x, y, z dari 6 landmark penting
        for idx in self.IMPORTANT_LANDMARKS:
            lm = landmarks[idx]
            features.extend([lm.x, lm.y, lm.z])

        # 2. Hitung sudut arah lengan kiri (bahu kiri 11 ke pergelangan kiri 15)
        left_dir = self.arm_direction_angle(
            landmarks[11].x, landmarks[11].y,
            landmarks[15].x, landmarks[15].y
        )

        # 3. Hitung sudut arah lengan kanan (bahu kanan 12 ke pergelangan kanan 16)
        right_dir = self.arm_direction_angle(
            landmarks[12].x, landmarks[12].y,
            landmarks[16].x, landmarks[16].y
        )

        features.append(left_dir)
        features.append(right_dir)

        return features

    def predict_from_image_bytes(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_cv2 is None:
            return {"status": "error", "message": "Format gambar tidak valid atau rusak"}

        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        detection_result = self.landmarker.detect(mp_image)

        if not detection_result.pose_landmarks:
            return {
                "status": "success",
                "detected": False,
                "prediction": None,
                "confidence": 0.0,
                "message": "Postur tubuh tidak terdeteksi dalam frame kamera"
            }

        # Ekstraksi fitur (menghasilkan list berisi 20 angka)
        features = self._extract_features(detection_result.pose_landmarks)
        
        # Bungkus ke DataFrame agar kompatibel dengan model scikit-learn
        X = pd.DataFrame([features])

        prediction = self.classifier.predict(X)[0]
        
        confidence = 1.0
        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(X)
            confidence = float(np.max(probs))

        return {
            "status": "success",
            "detected": True,
            "prediction": str(prediction),
            "confidence": round(confidence, 4),
            "message": "Semafore berhasil dideteksi"
        }