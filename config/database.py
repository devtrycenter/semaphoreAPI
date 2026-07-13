import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Inisialisasi variabel global untuk client Supabase
supabase: Client = None

def init_db():
    global supabase
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if not url or not key:
        raise ValueError("SUPABASE_URL dan SUPABASE_KEY wajib dikonfigurasi di file .env")
        
    supabase = create_client(url, key)
    return supabase

def get_db() -> Client:
    global supabase
    if supabase is None:
        return init_db()
    return supabase