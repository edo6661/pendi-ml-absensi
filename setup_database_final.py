import psycopg2
from psycopg2 import sql

# --- KONFIGURASI DATABASE ---
DB_NAME = "skripsi_db"
DB_USER = "postgres"
DB_PASS = "12345"       
DB_HOST = "db" # <-- Ubah baris ini menjadi 'db'

def create_tables():
    commands = (
        # 1. Tabel DIVISI (Kunci logika Marketing vs Lainnya)
        """
        CREATE TABLE IF NOT EXISTS divisi (
            id SERIAL PRIMARY KEY,
            nama_divisi VARCHAR(50) UNIQUE NOT NULL
        )
        """,
        
        # 2. Tabel USERS (Data Karyawan & Admin)
        """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            nama_lengkap VARCHAR(100) NOT NULL,
            username VARCHAR(50) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL, -- Nanti akan di-hash
            role VARCHAR(20) DEFAULT 'Staff', -- 'Admin' atau 'Staff'
            divisi_id INTEGER,
            foto_wajah VARCHAR(255), -- Path file foto dataset
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (divisi_id) REFERENCES divisi (id) ON DELETE SET NULL
        )
        """,
        
        # 3. Tabel LOKASI KANTOR (Pusat Geofencing)
        """
        CREATE TABLE IF NOT EXISTS lokasi_kantor (
            id SERIAL PRIMARY KEY,
            nama_lokasi VARCHAR(100),
            latitude DECIMAL(10,8),
            longitude DECIMAL(11,8),
            radius_meter INTEGER DEFAULT 50
        )
        """,

        # 4. Tabel ABSENSI (Transaksi Harian)
        # Satu user, satu baris per hari (Update jam_pulang nanti)
        """
        CREATE TABLE IF NOT EXISTS absensi (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            tanggal DATE DEFAULT CURRENT_DATE,
            jam_masuk TIME,
            jam_pulang TIME,
            foto_masuk VARCHAR(255), -- Bukti foto saat absen
            foto_pulang VARCHAR(255),
            lokasi_masuk VARCHAR(100), -- Koordinat teks
            lokasi_pulang VARCHAR(100),
            status_kehadiran VARCHAR(20), -- 'Hadir', 'Terlambat'
            durasi_kerja VARCHAR(20), -- Hasil hitungan jam
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
        """,

        # 5. Tabel PENGAJUAN IZIN (Skenario B)
        """
        CREATE TABLE IF NOT EXISTS pengajuan_izin (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            tanggal_mulai DATE NOT NULL,
            tanggal_selesai DATE NOT NULL,
            tipe_izin VARCHAR(20), -- 'Sakit', 'Cuti', 'Keperluan Pribadi'
            keterangan TEXT,
            bukti_foto VARCHAR(255), -- Path upload surat dokter
            status_approval VARCHAR(20) DEFAULT 'Pending', -- 'Pending', 'Disetujui', 'Ditolak'
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
        """
    )

    conn = None
    try:
        conn = psycopg2.connect(database=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)
        cur = conn.cursor()
        
        print("[INFO] Menghapus tabel lama (jika ada) agar bersih...")
        # Urutan drop penting karena Foreign Key!
        cur.execute("DROP TABLE IF EXISTS pengajuan_izin")
        cur.execute("DROP TABLE IF EXISTS absensi")
        cur.execute("DROP TABLE IF EXISTS users")
        cur.execute("DROP TABLE IF EXISTS divisi")
        cur.execute("DROP TABLE IF EXISTS lokasi_kantor")

        print("[INFO] Membuat struktur tabel baru...")
        for command in commands:
            cur.execute(command)
        
        # --- SEEDING DATA (Isi Data Awal) ---
        print("[INFO] Mengisi data default...")
        
        # 1. Isi Divisi
        cur.execute("INSERT INTO divisi (nama_divisi) VALUES ('IT'), ('Marketing'), ('Keuangan'), ('HRD')")
        
        # 2. Isi Lokasi Kantor (Contoh: Monas)
        cur.execute("INSERT INTO lokasi_kantor (nama_lokasi, latitude, longitude, radius_meter) VALUES ('Kampus Pusat', -6.175392, 106.827153, 50)")

        # 3. Isi User Admin (ID 1) -> Masuk Divisi IT
        # Password '123' (Nanti di aplikasi kita buat enkripsinya, ini data mentah dulu)
        cur.execute("""
            INSERT INTO users (nama_lengkap, username, password, role, divisi_id) 
            VALUES ('Super Admin', 'admin', '123', 'Admin', 1)
        """)
        
        # 4. Isi User Staff Marketing (ID 2) -> Untuk tes fitur bebas lokasi
        cur.execute("""
            INSERT INTO users (nama_lengkap, username, password, role, divisi_id) 
            VALUES ('Budi Marketing', 'budi', '123', 'Staff', 2)
        """)

        cur.close()
        conn.commit()
        print("[INFO] SELESAI! Database skripsi_db sudah siap dengan struktur final.")
        
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"[ERROR] {error}")
    finally:
        if conn is not None:
            conn.close()

if __name__ == '__main__':
    create_tables()