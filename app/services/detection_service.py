import uuid
from app.module.detection import SemaphoreDetector
from config.database import get_db

# Inisialisasi detector secara global
detector = None

def get_detector():
    global detector
    if detector is None:
        detector = SemaphoreDetector()
    return detector

class DetectionService:
    @staticmethod
    def process_semaphore_detection(image_file, user_id=None, ip_address=None, user_agent=None):
        if not image_file:
            return {"status": "error", "message": "File gambar tidak ditemukan dalam request"}, 400

        image_bytes = image_file.read()
        
        # Eksekusi prediksi AI
        detector_instance = get_detector()
        result = detector_instance.predict_from_image_bytes(image_bytes)

        # Simpan log aktivitas ke Supabase jika postur terdeteksi DAN user_id disertakan
        if result.get("status") == "success" and result.get("detected") and user_id:
            try:
                # Validasi format UUID agar tidak memicu error tipe data di PostgreSQL
                val_user_id = str(uuid.UUID(user_id))
                
                db = get_db()
                
                # Format deskripsi aktivitas sesuai dengan hasil deteksi
                letter = result["prediction"]
                conf_pct = round(result["confidence"] * 100, 1)
                activity_desc = f"Deteksi Semafore menghasilkan huruf: {letter} (Confidence: {conf_pct}%)"
                
                # Payload sesuai skema tabel public.activity_logs
                log_data = {
                    "user_id": val_user_id,
                    "activity": activity_desc,
                    "ip_address": ip_address,
                    "user_agent": user_agent
                }
                
                # Insert data ke Supabase
                db.table("activity_logs").insert(log_data).execute()
                
            except ValueError:
                print(f"[Supabase Log Warning]: Format user_id '{user_id}' bukan UUID yang valid. Log diabaikan.")
            except Exception as e:
                # Error logging tidak membatalkan response deteksi ke aplikasi mobile
                print(f"[Supabase Log Error]: {str(e)}")

        return result, 200