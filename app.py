import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from config.database import init_db
from app.services.detection_service import DetectionService

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Mengaktifkan CORS untuk aplikasi Flutter/Mobile

# Inisialisasi Supabase Client saat aplikasi startup
init_db()

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "app": os.getenv("APP_NAME", "Scoutify Backend"),
        "environment": os.getenv("ENV", "development"),
        "database": "Supabase PostgreSQL",
        "status": "running",
        "version": "1.0.0"
    }), 200

@app.route("/api/deteksi/semaphore", methods=["POST"])
def detect_semaphore():
    """
    Endpoint deteksi semafore.
    Menerima form-data:
      - image: File gambar (Wajib)
      - user_id: UUID pengguna dari tabel public.users (Opsional untuk testing, Wajib untuk pencatatan log)
    """
    if "image" not in request.files:
        return jsonify({
            "status": "error",
            "message": "Field 'image' wajib disertakan dalam format multipart/form-data"
        }), 400

    image_file = request.files["image"]
    user_id = request.form.get("user_id", None)
    
    # Menangkap IP Address dan User-Agent pengguna untuk kebutuhan activity_logs
    ip_address = request.headers.get("X-Forwarded-For", request.remote_addr)
    if ip_address and "," in ip_address:
        ip_address = ip_address.split(",")[0].strip()
        
    user_agent = request.headers.get("User-Agent", "Unknown Device")

    # Proses deteksi dan logging
    result, status_code = DetectionService.process_semaphore_detection(
        image_file=image_file, 
        user_id=user_id,
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    return jsonify(result), status_code

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    # Membaca mode environment dari variabel ENV
    is_dev = os.getenv("ENV", "development") == "development"
    
    app.run(host="0.0.0.0", port=port, debug=is_dev)