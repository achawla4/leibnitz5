# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 20:21:02 2026

@author: acer
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory, session, flash
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans Devanagari', 'Nirmala UI', 'Sanskrit Text', 'DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from scipy import signal
import uuid
from datetime import datetime, timedelta
import json
import re
import shutil

# Import Signal Processing Suite
from SignalProcessingSuite.blocks import get_block, list_blocks, timed_run

app = Flask(__name__)
app.secret_key = 'leibnitz-super-secret-key-2026'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)
app.url_map.strict_slashes = False

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'txt', 'csv', 'npy', 'wav'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Demo user and database setup
DEMO_USER = {"username": "demo", "password": "demo123"}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, 'users.json')

def load_users():
    if not os.path.exists(USERS_FILE):
        users = {DEMO_USER['username']: DEMO_USER['password']}
        save_users(users)
        return users
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading users: {e}")
        return {DEMO_USER['username']: DEMO_USER['password']}

def save_users(users):
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f, indent=4)
    except Exception as e:
        print(f"Error saving users: {e}")

# PostgreSQL setup
DATABASE_URL = os.environ.get('DATABASE_URL')

# Try reading from postgreconnection.txt if DATABASE_URL is not set
if not DATABASE_URL:
    conn_file = os.path.join(BASE_DIR, 'postgreconnection.txt')
    if os.path.exists(conn_file):
        try:
            with open(conn_file, 'r') as f:
                DATABASE_URL = f.read().strip()
        except Exception as e:
            print(f"Error reading postgreconnection.txt: {e}")

def init_db():
    if not DATABASE_URL:
        print("DATABASE_URL not set. Running in JSON mode.")
        return
    import psycopg2
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                is_paid BOOLEAN NOT NULL DEFAULT FALSE,
                payment_utr VARCHAR(32)
            );
        """)
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_paid BOOLEAN NOT NULL DEFAULT FALSE;")
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS payment_utr VARCHAR(32);")
        # Insert demo user if not exists
        cur.execute("SELECT 1 FROM users WHERE username = 'demo';")
        if not cur.fetchone():
            cur.execute("INSERT INTO users (username, password) VALUES ('demo', 'demo123');")
            print("Default demo user inserted into PostgreSQL database.")
        conn.commit()
        cur.close()
        conn.close()
        print("PostgreSQL database initialized successfully.")
    except Exception as e:
        print(f"Error initializing PostgreSQL database: {e}")



def get_user_password(username):
    if DATABASE_URL:
        import psycopg2
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            cur.execute("SELECT password FROM users WHERE username = %s;", (username,))
            row = cur.fetchone()
            cur.close()
            conn.close()
            if row:
                return row[0]
            return None
        except Exception as e:
            print(f"Database error in get_user_password: {e}. Falling back to JSON.")
    
    # JSON fallback
    users = load_users()
    user_record = users.get(username)
    if isinstance(user_record, dict):
        return user_record.get('password')
    return user_record

def add_user(username, password):
    if DATABASE_URL:
        import psycopg2
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            cur.execute("INSERT INTO users (username, password) VALUES (%s, %s);", (username, password))
            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"Database error in add_user: {e}. Falling back to JSON.")
            
    # JSON fallback
    users = load_users()
    users[username] = {
        "password": password,
        "is_paid": False,
        "payment_utr": None
    }
    save_users(users)
    return True

def is_paid_user(username):
    if DATABASE_URL:
        import psycopg2
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            cur.execute("SELECT is_paid FROM users WHERE username = %s;", (username,))
            row = cur.fetchone()
            cur.close()
            conn.close()
            return bool(row[0]) if row else False
        except Exception as e:
            print(f"Database error in is_paid_user: {e}. Falling back to JSON.")

    users = load_users()
    user_record = users.get(username)
    if isinstance(user_record, dict):
        return bool(user_record.get('is_paid', False))
    return False

def mark_user_paid(username, utr=None):
    if DATABASE_URL:
        import psycopg2
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            cur.execute(
                "UPDATE users SET is_paid = TRUE, payment_utr = COALESCE(%s, payment_utr) WHERE username = %s;",
                (utr, username)
            )
            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"Database error in mark_user_paid: {e}. Falling back to JSON.")

    users = load_users()
    user_record = users.get(username)
    if isinstance(user_record, dict):
        user_record['is_paid'] = True
        if utr:
            user_record['payment_utr'] = utr
    else:
        user_record = {
            "password": user_record,
            "is_paid": True,
            "payment_utr": utr
        }
    users[username] = user_record
    save_users(users)
    return True

def has_unlimited_access():
    return bool(session.get('is_paid')) or is_paid_user(session.get('user'))

def ensure_session_user():
    """Ensure a user exists in the session. If not, instantiate an anonymous guest user."""
    if 'user' not in session or not session.get('user'):
        guest_user = f"guest_{uuid.uuid4().hex[:8]}"
        add_user(guest_user, "guest")
        session['user'] = guest_user
        session['is_paid'] = False
        session['usage_count'] = 0
    return session['user']

# ====================== File Metadata Database (PostgreSQL) ======================

def init_files_table():
    """Initialize the files table in PostgreSQL"""
    if not DATABASE_URL:
        print("DATABASE_URL not set. File tracking requires PostgreSQL.")
        return
    
    import psycopg2
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255) UNIQUE NOT NULL,
                original_name VARCHAR(255) NOT NULL,
                username VARCHAR(100) NOT NULL,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR(100) DEFAULT 'Uploaded',
                processing_count INT DEFAULT 0,
                FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_files_username ON files(username);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_files_uploaded_at ON files(uploaded_at DESC);")
        conn.commit()
        cur.close()
        conn.close()
        print("Files table initialized successfully.")
    except Exception as e:
        print(f"Error initializing files table: {e}")

# Initialize Database
init_db()
init_files_table()


def add_file_metadata(filename, original_filename, username):
    """Add metadata for an uploaded file to PostgreSQL"""
    if not DATABASE_URL:
        print("File tracking requires DATABASE_URL")
        return
    
    import psycopg2
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO files (filename, original_name, username, status)
            VALUES (%s, %s, %s, 'Uploaded')
            ON CONFLICT (filename) DO NOTHING;
        """, (filename, original_filename, username))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error adding file metadata: {e}")

def get_user_files(username, limit=10):
    """Get recent files for a specific user from PostgreSQL"""
    if not DATABASE_URL:
        return []
    
    import psycopg2
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            SELECT filename, original_name, uploaded_at, status, processing_count
            FROM files
            WHERE username = %s
            ORDER BY uploaded_at DESC
            LIMIT %s;
        """, (username, limit))
        
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        formatted_files = []
        for filename, original_name, uploaded_at, status, processing_count in rows:
            # Format time difference
            time_diff = datetime.now() - uploaded_at.replace(tzinfo=None)
            
            if time_diff.days > 0:
                time_str = f"{time_diff.days}d ago"
            elif time_diff.seconds > 3600:
                time_str = f"{time_diff.seconds // 3600}h ago"
            elif time_diff.seconds > 60:
                time_str = f"{time_diff.seconds // 60}m ago"
            else:
                time_str = "now"
            
            formatted_files.append({
                'filename': filename,
                'name': original_name,
                'uploaded_at': time_str,
                'status': status or 'Ready',
                'processing_count': processing_count
            })
        
        return formatted_files
    except Exception as e:
        print(f"Error fetching user files: {e}")
        return []

def update_file_status(filename, operation_name):
    """Update file status after processing"""
    if not DATABASE_URL:
        return
    
    import psycopg2
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            UPDATE files
            SET status = %s, processing_count = processing_count + 1
            WHERE filename = %s;
        """, (f"Processed • {operation_name}", filename))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error updating file status: {e}")

def get_wav_sample_rate(filepath):
    import wave
    try:
        with wave.open(filepath, 'rb') as w:
            return w.getframerate()
    except Exception as e:
        print(f"Error reading WAV header: {e}")
        return None

def detect_csv_properties(filepath):
    try:
        with open(filepath, 'r') as f:
            lines = [f.readline() for _ in range(101)]
        start_idx = 1 if any(c.isalpha() for c in lines[0]) else 0
        data_lines = lines[start_idx:]
        rows = []
        for line in data_lines:
            if not line.strip():
                continue
            parts = line.strip().split(',')
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                pass
        if len(rows) < 5:
            return None
        data = np.array(rows)
        if data.ndim == 2 and data.shape[1] > 1:
            col0 = data[:, 0]
            diffs = np.diff(col0)
            if len(diffs) > 0 and np.all(diffs > 0) and np.std(diffs) < 1e-3 * np.mean(diffs):
                mean_diff = np.mean(diffs)
                if mean_diff > 0:
                    return float(round(1.0 / mean_diff, 2))
    except Exception as e:
        print(f"Error detecting CSV properties: {e}")
    return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_signal_file(filepath, filename):
    detected_rate = None

    if filename.lower().endswith('.wav'):
        from scipy.io import wavfile
        sample_rate_wav, signal_data = wavfile.read(filepath)
        detected_rate = float(sample_rate_wav)

        # Ensure 1D (mono)
        if len(signal_data.shape) > 1:
            signal_data = signal_data[:, 0]

        # Normalize to [-1.0, 1.0] if integer type
        if signal_data.dtype.kind in 'iu':
            max_val = np.iinfo(signal_data.dtype).max
            signal_data = signal_data.astype(np.float32) / max_val

    elif filename.lower().endswith('.npy'):
        signal_data = np.load(filepath)

    elif filename.lower().endswith('.csv'):
        # Load with delimiter
        signal_data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        # Detect rate and extract signal column if 2D
        if len(signal_data.shape) > 1 and signal_data.shape[1] > 1:
            col0 = signal_data[:, 0]
            diffs = np.diff(col0)
            if len(diffs) > 0 and np.all(diffs > 0) and np.std(diffs) < 1e-3 * np.mean(diffs):
                mean_diff = np.mean(diffs)
                if mean_diff > 0:
                    detected_rate = float(round(1.0 / mean_diff, 2))
                # Column 0 is time, Column 1 is signal
                signal_data = signal_data[:, 1]
            else:
                signal_data = signal_data[:, 0]
        else:
            signal_data = signal_data.flatten()

    else:  # txt or others
        signal_data = np.loadtxt(filepath)
        if len(signal_data.shape) > 1:
            signal_data = signal_data[:, 0] if signal_data.shape[1] > 1 else signal_data.flatten()

    return np.asarray(signal_data, dtype=float).flatten(), detected_rate

def get_request_sample_rate(data, detected_rate):
    sample_rate = detected_rate if detected_rate is not None else 1000.0
    user_rate = data.get('sample_rate')
    if user_rate is not None:
        try:
            val = float(user_rate)
            if val > 0:
                sample_rate = val
        except (ValueError, TypeError):
            pass
    return sample_rate

def normalize_block_specs(data):
    blocks = data.get('blocks')
    if blocks is not None:
        if not isinstance(blocks, list) or not blocks:
            raise ValueError("blocks must be a non-empty list")
        normalized = []
        for index, spec in enumerate(blocks):
            if not isinstance(spec, dict):
                raise ValueError(f"blocks[{index}] must be an object")
            block_id = spec.get('id') or spec.get('operation')
            if not block_id:
                raise ValueError(f"blocks[{index}] is missing id")
            normalized.append({
                "id": block_id,
                "params": spec.get('params') or {}
            })
        return normalized

    operation = data.get('operation')
    if not operation:
        raise ValueError("Missing operation or blocks")
    legacy_params = {
        key: value
        for key, value in data.items()
        if key not in {'filename', 'operation', 'sample_rate'}
    }
    return [{"id": operation, "params": legacy_params}]

# ====================== ROUTES ======================

@app.route('/')
def index():
    return render_template('leibnitz.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        return redirect(url_for('dashboard'))
        
    active_tab = 'register'
    username_val = ''
    
    if request.method == 'POST':
        action = request.form.get('action', 'login')
        username_val = request.form.get('username', '').strip()
        
        if action == 'register':
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            
            if not username_val or not password:
                flash('प्रयोक्तानाम गुप्तशब्दश्च उभावपि आवश्यकौ। (Prayōktr̥-nāma gupta-śabdaśca ubhāvapi āvaśyakau.)', 'error')
                return render_template('login.html', active_tab='register', username=username_val)
                
            if password != confirm_password:
                flash('गुप्तशब्दौ न समेते। (Gupta-śabdau na samētē.)', 'error')
                return render_template('login.html', active_tab='register', username=username_val)
                
            if get_user_password(username_val) is not None:
                flash('प्रयोक्तानाम पूर्वमेव विद्यते। (Prayōktr̥-nāma pūrvamēva vidyatē.)', 'error')
                return render_template('login.html', active_tab='register', username=username_val)
                
            add_user(username_val, password)
            
            # Autologin and set permanent session
            session.permanent = True
            session['user'] = username_val
            session['is_paid'] = False
            flash('पञ्जीकरणं सफलम्! लायब्निट्ज़-मध्ये स्वागतम्। (Pañjīkaraṇaṁ saphalam! Lāyabniṭza-madhyē svāgatam.)', 'success')
            return redirect(url_for('dashboard'))
            
        else:  # action == 'login'
            password = request.form.get('password')
            db_password = get_user_password(username_val)
            
            if db_password and db_password == password:
                session.permanent = True
                session['user'] = username_val
                session['is_paid'] = is_paid_user(username_val)
                flash('प्रवेशः सफलः! (Pravēśaḥ saphalaḥ!)', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('अवैधानि प्रमाणपत्राणि। (Avaidhāni pramāṇapatrāṇi.)', 'error')
                return render_template('login.html', active_tab='login', username=username_val)
                
    return render_template('login.html', active_tab=active_tab, username=username_val)


@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('is_paid', None)
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/payment')
def payment_page():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('payment.html', usage_count=session.get('usage_count', 0))

# ====================== API ENDPOINTS ======================

@app.route('/api/usage')
def get_usage():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify({
        "usage_count": session.get('usage_count', 0),
        "limit": None if has_unlimited_access() else 50,
        "unlimited_access": has_unlimited_access()
    })

@app.route('/api/payment/confirm', methods=['POST'])
def confirm_payment():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    utr = str(data.get('utr', '')).strip()
    if utr and not re.fullmatch(r'\d{12}', utr):
        return jsonify({"error": "Invalid UTR number"}), 400

    mark_user_paid(session['user'], utr or None)
    session['is_paid'] = True
    session['usage_count'] = 0
    return jsonify({
        "success": True,
        "unlimited_access": True,
        "message": "शुल्कशोधनं प्रमाणीकृतम्! असीमितप्रवेशः उद्घाटितः। (Śulkaśōdhanaṁ pramāṇīkr̥tam! Asīmita-pravēśaḥ udghāṭitaḥ.)"
    })

@app.route('/api/blocks', methods=['GET'])
def available_blocks():
    return jsonify({"blocks": list_blocks()})

@app.route('/api/demo-signal/<preset_type>', methods=['GET'])
def get_demo_signal(preset_type):
    try:
        user = ensure_session_user()
        preset_type = (preset_type or '').lower().strip()
        uid = uuid.uuid4().hex[:8]

        if preset_type == 'sinusoid':
            original_name = "sinusoidal_12Hz.csv"
            filename = f"demo_sinusoid_12Hz_{uid}.csv"
            source_file = os.path.join(BASE_DIR, 'sinusoidal_12Hz.csv')
            target_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(source_file):
                shutil.copyfile(source_file, target_file)
            else:
                sample_rate = 1000.0
                t = np.linspace(0, 1.0, 1000, endpoint=False)
                sig = np.sin(2 * np.pi * 12.0 * t) + 0.25 * np.sin(2 * np.pi * 60.0 * t)
                np.savetxt(target_file, np.column_stack((t, sig)), delimiter=',', header='Time (s),Signal', comments='')
            
            detected_rate = 1000.0
            description = "१२ हर्ट्ज़ नादसंवादः ध्वनिप्रदूषणेन सह (12 Hz Sinusoid with harmonics & noise)"

        elif preset_type == 'chirp':
            original_name = "frequency_chirp_10-200Hz.csv"
            filename = f"demo_chirp_{uid}.csv"
            target_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            sample_rate = 1000.0
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            sig = np.sin(2 * np.pi * (10.0 + (200.0 - 10.0) * t / (2 * duration)) * t)
            sig += 0.04 * np.random.normal(0, 1, len(t))
            np.savetxt(target_file, np.column_stack((t, sig)), delimiter=',', header='Time (s),Signal', comments='')
            
            detected_rate = 1000.0
            description = "रैखिक-आवृत्ति-चीत्कार-प्रसारः (१० Hz → २०० Hz) (Linear frequency sweep 10-200 Hz)"

        elif preset_type == 'ecg':
            original_name = "synthetic_ecg_trace.csv"
            filename = f"demo_ecg_{uid}.csv"
            target_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            sample_rate = 500.0
            duration = 2.0
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            period = 1.0 / 1.2
            phase = np.mod(t, period)
            
            p_wave = 0.15 * np.exp(-((phase - 0.20) ** 2) / (2 * (0.025 ** 2)))
            q_wave = -0.15 * np.exp(-((phase - 0.27) ** 2) / (2 * (0.012 ** 2)))
            r_wave = 1.20 * np.exp(-((phase - 0.30) ** 2) / (2 * (0.015 ** 2)))
            s_wave = -0.30 * np.exp(-((phase - 0.34) ** 2) / (2 * (0.015 ** 2)))
            t_wave = 0.30 * np.exp(-((phase - 0.48) ** 2) / (2 * (0.045 ** 2)))
            
            baseline = 0.08 * np.sin(2 * np.pi * 0.25 * t)
            noise = 0.02 * np.random.normal(0, 1, len(t))
            ecg_sig = p_wave + q_wave + r_wave + s_wave + t_wave + baseline + noise
            np.savetxt(target_file, np.column_stack((t, ecg_sig)), delimiter=',', header='Time (s),Signal', comments='')
            
            detected_rate = 500.0
            description = "कृत्रिम-हृद्लेख-स्पन्दः (५०० Hz) (Synthetic Cardiac ECG Trace 500 Hz)"
        else:
            return jsonify({"error": f"अज्ञातं निदर्शनम् '{preset_type}'"}), 400

        add_file_metadata(filename, original_name, user)

        return jsonify({
            "success": True,
            "filename": filename,
            "original_name": original_name,
            "detected_sample_rate": detected_rate,
            "description": description,
            "message": f"निदर्शनसङ्केतः स्थापितः: {description}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    user = ensure_session_user()

    if not has_unlimited_access() and session.get('usage_count', 0) >= 50:
        return jsonify({"error": "Usage limit reached", "redirect": "/payment"}), 402

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Track file metadata for dashboard
        add_file_metadata(filename, file.filename, session.get('user'))
        
        detected_rate = None
        if filename.lower().endswith('.wav'):
            detected_rate = get_wav_sample_rate(filepath)
        elif filename.lower().endswith('.csv'):
            detected_rate = detect_csv_properties(filepath)
            
        if not has_unlimited_access():
            session['usage_count'] = session.get('usage_count', 0) + 1
        return jsonify({
            "success": True,
            "filename": filename,
            "message": "सञ्चिका सफ़लतया आरोपिता (Sañcikā saphalatayā ārōpitā)",
            "detected_sample_rate": detected_rate,
            "usage_count": session.get('usage_count', 0)
        })
    
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/api/recent-files', methods=['GET'])
def get_recent_files():
    """Get recent files for the current user"""
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    username = session.get('user')
    files = get_user_files(username, limit=10)
    
    return jsonify({
        "files": files
    })

@app.route('/api/process', methods=['POST'])
def process_signal():
    user = ensure_session_user()

    if not has_unlimited_access() and session.get('usage_count', 0) >= 50:
        return jsonify({"error": "Usage limit reached", "redirect": "/payment"}), 402

    data = request.get_json() or {}
    filename = data.get('filename')

    if not filename:
        return jsonify({"error": "Missing parameters"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    try:
        signal_data, detected_rate = load_signal_file(filepath, filename)
        sample_rate = get_request_sample_rate(data, detected_rate)
        block_specs = normalize_block_specs(data)

        current_signal = signal_data
        pipeline_results = []

        for index, spec in enumerate(block_specs):
            block = get_block(spec["id"])
            params = spec.get("params") or {}
            block_input = current_signal
            run_result, elapsed_ms = timed_run(block, block_input, sample_rate, params)
            plot_path = generate_suite_plot(filename, block.id, block_input, run_result.plot_data, sample_rate=sample_rate)

            pipeline_results.append({
                "index": index,
                "id": block.id,
                "name": block.name,
                "category": block.category,
                "params": params,
                "result": run_result.result,
                "metadata": run_result.metadata,
                "warnings": run_result.warnings,
                "elapsed_ms": round(elapsed_ms, 3),
                "plot_url": f"/processed/{plot_path}" if plot_path else None,
            })

            if run_result.output_signal is not None:
                current_signal = np.asarray(run_result.output_signal, dtype=float).flatten()

        first_result = pipeline_results[0] if pipeline_results else {}
        last_result = pipeline_results[-1] if pipeline_results else {}

        # Update file status in PostgreSQL
        if first_result.get('name'):
            update_file_status(filename, first_result.get('name'))

        # Increment usage count only for users still on the free tier.
        if not has_unlimited_access():
            session['usage_count'] = session.get('usage_count', 0) + 1

        # Generate downloadable CSV of the processed / filtered signal or spectrum data
        processed_csv_filename = f"processed_{first_result.get('id', 'signal')}_{uuid.uuid4().hex[:8]}.csv"
        processed_csv_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_csv_filename)

        if first_result.get('id') == 'fft' and isinstance(last_result.get('result'), dict) and 'frequencies' in last_result['result']:
            freqs = np.array(last_result['result']['frequencies'])
            mags = np.array(last_result['result'].get('magnitude', last_result['result'].get('magnitudes', [])))
            min_len = min(len(freqs), len(mags))
            export_data = np.column_stack((freqs[:min_len], mags[:min_len]))
            np.savetxt(processed_csv_path, export_data, delimiter=',', header='Frequency (Hz),Magnitude', comments='')
            download_label = "वर्णक्रमदत्तांशः (.csv)"
        else:
            time_axis = np.arange(len(current_signal)) / float(sample_rate)
            export_data = np.column_stack((time_axis, current_signal))
            np.savetxt(processed_csv_path, export_data, delimiter=',', header='Time (s),Signal', comments='')
            download_label = "संसाधितः सङ्केतः (.csv)"

        report_json_filename = f"report_{first_result.get('id', 'summary')}_{uuid.uuid4().hex[:8]}.json"
        report_json_path = os.path.join(app.config['PROCESSED_FOLDER'], report_json_filename)
        with open(report_json_path, 'w') as f:
            json.dump({
                "operation": first_result.get("id"),
                "sample_rate": sample_rate,
                "timestamp": datetime.utcnow().isoformat(),
                "result": last_result.get("result"),
                "pipeline": pipeline_results
            }, f, indent=2, default=str)

        return jsonify({
            "success": True,
            "operation": first_result.get("id"),
            "result": last_result.get("result"),
            "plot_url": last_result.get("plot_url"),
            "pipeline": pipeline_results,
            "sample_rate": sample_rate,
            "download_url": f"/api/download/{processed_csv_filename}",
            "download_label": download_label,
            "report_url": f"/api/download/{report_json_filename}",
            "usage_count": session.get('usage_count', 0)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_suite_plot(original_filename, operation, original_signal, result_data, sample_rate=1000.0):
    if operation == 'davincitron':
        canvas = result_data.get('canvas')
        if canvas:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"{operation}_{timestamp}_{uuid.uuid4().hex[:8]}.png"
            plot_path = os.path.join(app.config['PROCESSED_FOLDER'], plot_filename)
            canvas.save(plot_path)
            return plot_filename
        return None

    if operation == 'timefeaturetron':
        canvas = result_data.get('canvas')
        if canvas:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"{operation}_{timestamp}_{uuid.uuid4().hex[:8]}.png"
            plot_path = os.path.join(app.config['PROCESSED_FOLDER'], plot_filename)
            canvas.save(plot_path)
            return plot_filename
        return None


    from SignalProcessingSuite.visualization import plot_time, plot_frequency, plot_ifft, plot_wavelet_coefficients
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0a0e27')
    ax.set_facecolor('#0a0e27')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"{operation}_{timestamp}_{uuid.uuid4().hex[:8]}.png"
    plot_path = os.path.join(app.config['PROCESSED_FOLDER'], plot_filename)

    if operation == 'fft':
        fig.clf()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), facecolor='#0a0e27')
        ax1.set_facecolor('#0a0e27')
        ax2.set_facecolor('#0a0e27')
        
        plot_time(original_signal[:1000], sample_rate=sample_rate, ax=ax1, title="समयक्षेत्रसङ्केतः (Time Domain)")
        ax1.set_xlabel("कालः (s)", color='#b0b8cc')
        ax1.set_ylabel("आयामः", color='#b0b8cc')
        ax1.get_lines()[0].set_color('#00ff88')
        
        plot_frequency(original_signal, sample_rate=sample_rate, ax=ax2, db=False, title="आवृत्तिवर्णक्रमः (द्रुत-फूर्ये FFT)")
        ax2.set_xlabel("आवृत्तिः (Hz)", color='#b0b8cc')
        ax2.set_ylabel("परिमाणम्", color='#b0b8cc')
        ax2.get_lines()[0].set_color('#00d4ff')
        ax2.set_xlim(0, sample_rate / 2.0)

    elif operation == 'ifft':
        fig.clf()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), facecolor='#0a0e27')
        ax1.set_facecolor('#0a0e27')
        ax2.set_facecolor('#0a0e27')
        
        plot_frequency(original_signal, sample_rate=sample_rate, ax=ax1, db=False, title="आवृत्तिवर्णक्रमः (द्रुत-फूर्ये FFT)")
        ax1.set_xlabel("आवृत्तिः (Hz)", color='#b0b8cc')
        ax1.set_ylabel("परिमाणम्", color='#b0b8cc')
        ax1.get_lines()[0].set_color('#00d4ff')
        ax1.set_xlim(0, sample_rate / 2.0)

        plot_ifft(
            original_signal,
            sample_rate=sample_rate,
            ax=ax2,
            db=False,
            title="समयक्षेत्रे प्रतिलोम-द्रुत-फूर्ये (IFFT)"
        )
        ax2.set_xlabel("प्रतिचयन-क्रमाङ्कः", color='#b0b8cc')
        ax2.set_ylabel("आयामः", color='#b0b8cc')

        lines = ax2.get_lines()
        if lines:
            lines[0].set_color('#00d4ff')

        ax2.set_xlim(0, 1000)

    elif operation == 'filter':
        times = np.arange(min(len(original_signal), 1000)) / sample_rate
        ax.plot(times, original_signal[:1000], label='मूलसङ्केतः', alpha=0.7, color='#ff3344')
        ax.plot(times, result_data['filtered'][:1000], label='शोधितसङ्केतः', color='#00ff88')
        ax.set_title('निम्नगामि-शोधितसङ्केतः (Filtered Signal)')
        ax.set_xlabel('कालः (s)')
        ax.set_ylabel('आयामः')
        ax.legend()
        ax.grid(True, alpha=0.3)

    elif operation == 'wavelet':
        fig.clf()
        fig, ax1 = plt.subplots(figsize=(12, 6), facecolor='#0a0e27')
        ax1.set_facecolor('#0a0e27')
        
        coeffs = result_data['_coeffs_obj']
        plot_wavelet_coefficients(coeffs, ax=ax1)
        ax1.set_title("वीचिका-गुणाङ्काः (Wavelet Coefficients)")
        ax1.set_xlabel("गुणाङ्क-निर्देशाङ्कः")
        ax1.set_ylabel("प्रसामान्यीकृत-स्तरः")
        ax1.grid(True, alpha=0.3)

    elif operation == 'sft':
        fig.clf()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8), facecolor='#0a0e27')
        ax1.set_facecolor('#0a0e27')
        ax2.set_facecolor('#0a0e27')
        ax3.set_facecolor('#0a0e27')
        ax4.set_facecolor('#0a0e27')

        N = result_data['N']
        psi_history = result_data['psi_history']
        collapsed_history = result_data['collapsed_history']
        fidelity_proj_history = result_data['fidelity_proj_history']
        fidelity_mod_history = result_data['fidelity_mod_history']
        freqs = np.array(result_data['freqs'])
        final_P_FT = np.array(result_data['final_P_FT'])
        final_P_proj_FT = np.array(result_data['final_P_proj_FT'])
        final_P_mod_FT = np.array(result_data['final_P_mod_FT'])
        final_P_col_FT = np.array(result_data['final_P_col_FT'])
        steps = result_data['steps']
        alpha = result_data['alpha']
        dt = result_data['dt']
        operator_matrix_mag = np.array(result_data['operator_matrix_magnitude'])

        # Plot 1: Wavefunction Position Probability Evolution over time
        steps_to_plot = [0, steps // 4, 2 * steps // 4, 3 * steps // 4, steps]
        colors = ['#ff3344', '#ff8800', '#ffd700', '#00d4ff', '#00ff88']
        color_idx = 0
        for step_idx in steps_to_plot:
            if step_idx < len(psi_history):
                psi_t = np.array(psi_history[step_idx])
                P_t = np.abs(psi_t) ** 2
                t_val = step_idx * dt
                col = colors[color_idx % len(colors)]
                ax1.plot(P_t, label=f"t = {t_val:.2f}", color=col, alpha=0.8)
                color_idx += 1
        ax1.set_title("श्रोडिङ्गर-स्थान-तरङ्गफलन-विकासः", color='#ffffff', fontsize=11, pad=8)
        ax1.set_xlabel("स्थान-निर्देशाङ्कः (x)", color='#b0b8cc', fontsize=9)
        ax1.set_ylabel("सम्भाव्यता-सान्द्रता", color='#b0b8cc', fontsize=9)
        ax1.legend(loc="upper right", facecolor='#0f1535', edgecolor='#1a2847', fontsize=8)
        ax1.grid(True, alpha=0.1)
        ax1.tick_params(colors='#b0b8cc', labelsize=8)

        # Plot 2: 2D Semiclassical Operator Matrix Heatmap
        im = ax2.imshow(operator_matrix_mag, cmap='inferno', extent=[0, N, 0, N], origin='lower')
        cbar = fig.colorbar(im, ax=ax2, shrink=0.8)
        cbar.ax.yaxis.set_tick_params(color='#b0b8cc', labelsize=8)
        cbar.ax.set_ylabel('प्रचालक-व्यूह-परिमाणम् |P_sc[j,k]|', color='#b0b8cc', rotation=270, labelpad=15, fontsize=8)
        ax2.set_title("अर्धशास्त्रीय-प्रचालक-सान्द्रता-व्यूहः $|P_{sc}|$", color='#ffffff', fontsize=11, pad=8)
        ax2.set_xlabel("अवस्था-निर्देशाङ्कः (k)", color='#b0b8cc', fontsize=9)
        ax2.set_ylabel("अवस्था-निर्देशाङ्कः (j)", color='#b0b8cc', fontsize=9)
        ax2.tick_params(colors='#b0b8cc', labelsize=8)

        # Plot 3: Frequency-domain Fourier probabilities comparison
        ax3.plot(freqs, final_P_FT, label="प्रमात्र-फूर्ये (QFT)", color='#00ff88', linewidth=2)
        ax3.fill_between(freqs, final_P_FT, alpha=0.1, color='#00ff88')
        ax3.plot(freqs, final_P_proj_FT, label=f"प्रक्षेप-SFT (alpha={alpha})", color='#ffd700', linewidth=1.5)
        ax3.plot(freqs, final_P_mod_FT, label="विकार-SFT", color='#00d4ff', linewidth=1.2, linestyle=':')
        ax3.plot(freqs, final_P_col_FT, label="शास्त्रीय-संपाती-DFT", color='#ff3344', linestyle='--', alpha=0.7, linewidth=1.2)
        ax3.set_title("आवृत्तिक्षेत्रीय-फूर्ये-सम्भाव्यताः", color='#ffffff', fontsize=11, pad=8)
        ax3.set_xlabel("आवृत्तिः (Hz)", color='#b0b8cc', fontsize=9)
        ax3.set_ylabel("सम्भाव्यता-सान्द्रता", color='#b0b8cc', fontsize=9)
        ax3.legend(loc="upper right", facecolor='#0f1535', edgecolor='#1a2847', fontsize=8)
        ax3.grid(True, alpha=0.1)
        ax3.tick_params(colors='#b0b8cc', labelsize=8)

        # Plot 4: Spectral Fidelities over Time Steps
        time_axis = np.arange(1, len(fidelity_proj_history) + 1) * dt
        ax4.plot(time_axis, fidelity_proj_history, color='#ffd700', marker='.', linewidth=1.5, label="प्रक्षेप-SFT विश्वसनीयता")
        ax4.plot(time_axis, fidelity_mod_history, color='#00d4ff', marker='x', linewidth=1.2, label="विकार-SFT विश्वसनीयता", alpha=0.8)
        ax4.set_title("कालविकासे वर्णक्रम-विश्वसनीयता", color='#ffffff', fontsize=11, pad=8)
        ax4.set_xlabel("कालः (s)", color='#b0b8cc', fontsize=9)
        ax4.set_ylabel("वर्णक्रम-विश्वसनीयता", color='#b0b8cc', fontsize=9)
        ax4.set_ylim(-0.05, 1.05)
        ax4.grid(True, alpha=0.1)
        ax4.legend(loc="lower right", facecolor='#0f1535', edgecolor='#1a2847', fontsize=8)
        ax4.tick_params(colors='#b0b8cc', labelsize=8)

    else:
        times = np.arange(min(len(original_signal), 1000)) / sample_rate
        ax.plot(times, original_signal[:1000], color='#ffd700')
        ax.set_title('सङ्केत-चित्रीकरणम् (Signal Visualization)')
        ax.set_xlabel('कालः (s)')
        ax.set_ylabel('आयामः')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=100, facecolor='#0a0e27')
    plt.close(fig)
    
    return plot_filename

@app.route('/processed/<filename>')
def serve_processed(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/api/download/<path:filename>')
def download_output(filename):
    clean_filename = os.path.basename(filename)
    filepath = os.path.join(app.config['PROCESSED_FOLDER'], clean_filename)
    if os.path.exists(filepath):
        return send_from_directory(app.config['PROCESSED_FOLDER'], clean_filename, as_attachment=True)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], clean_filename)
    if os.path.exists(upload_path):
        return send_from_directory(app.config['UPLOAD_FOLDER'], clean_filename, as_attachment=True)
    return jsonify({"error": "File not found"}), 404

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    tb = traceback.format_exc()
    return jsonify({
        "success": False,
        "error": str(e),
        "traceback": tb
    }), 500

if __name__ == '__main__':
    print("[Leibnitz Signal Processing Backend Started]")
    print("Demo Login: username: demo | password: demo123")
    app.run(debug=True, host='0.0.0.0', port=5000)
