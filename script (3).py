# Create the backend Python Flask application
backend_code = '''
"""
Behavioral Biometric Authentication System - Backend
Processes and authenticates behavioral patterns using machine learning
"""

from flask import Flask, request, jsonify, session
from flask_cors import CORS
import sqlite3
import json
import hashlib
import numpy as np
from datetime import datetime
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
import logging

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiometricAuthenticator:
    def __init__(self, db_path='biometric_auth.db'):
        self.db_path = db_path
        self.init_database()
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.load_models()

    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                device_fingerprint TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create behavioral patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS behavioral_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                pattern_type TEXT NOT NULL,
                features TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create authentication logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS auth_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                username TEXT,
                success BOOLEAN,
                confidence_score REAL,
                patterns_matched INTEGER,
                ip_address TEXT,
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def extract_features(self, patterns):
        """Extract numerical features from behavioral patterns"""
        features = []
        feature_names = []
        
        # Device features
        if 'device' in patterns:
            device = patterns['device']
            features.extend([
                hash(device.get('userAgent', '')) % 10000,
                hash(device.get('platform', '')) % 1000,
                device.get('screenResolution', '1920x1080').count('x'),
                device.get('colorDepth', 24),
                device.get('hardwareConcurrency', 4),
                len(device.get('plugins', [])),
                hash(device.get('canvasFingerprint', '')) % 100000
            ])
            feature_names.extend([
                'user_agent_hash', 'platform_hash', 'screen_resolution_complexity',
                'color_depth', 'hardware_concurrency', 'plugins_count', 'canvas_hash'
            ])
        
        # Keystroke features
        if 'keystroke' in patterns and patterns['keystroke']:
            keystroke = patterns['keystroke']
            
            # Average dwell times for common keys
            common_keys = ['a', 'e', 'i', 'o', 'u', 'n', 't', 'r', 's', 'l']
            for key in common_keys:
                avg_dwell = keystroke.get('avgDwellTimes', {}).get(key, 100)
                features.append(avg_dwell)
                feature_names.append(f'dwell_time_{key}')
            
            # Typing speed
            features.append(keystroke.get('typingSpeed', 0))
            feature_names.append('typing_speed')
            
            # Flight times for common key pairs
            common_pairs = ['th', 'he', 'in', 'er', 'an']
            for pair in common_pairs:
                flight_time = keystroke.get('avgFlightTimes', {}).get(f'{pair[0]}_{pair[1]}', 50)
                features.append(flight_time)
                feature_names.append(f'flight_time_{pair}')
        else:
            # Pad with zeros if no keystroke data
            features.extend([0] * 16)  # 10 dwell + 1 speed + 5 flight times
            feature_names.extend([f'dwell_time_{k}' for k in ['a', 'e', 'i', 'o', 'u', 'n', 't', 'r', 's', 'l']])
            feature_names.extend(['typing_speed'])
            feature_names.extend([f'flight_time_{p}' for p in ['th', 'he', 'in', 'er', 'an']])
        
        # Mouse features
        if 'mouse' in patterns and patterns['mouse']:
            mouse = patterns['mouse']
            features.extend([
                mouse.get('avgMouseSpeed', 0),
                mouse.get('avgMouseAcceleration', 0),
                mouse.get('totalMouseDistance', 0),
                mouse.get('mouseMovements', 0),
                mouse.get('clickCount', 0)
            ])
            feature_names.extend([
                'avg_mouse_speed', 'avg_mouse_acceleration', 
                'total_mouse_distance', 'mouse_movements', 'click_count'
            ])
        else:
            features.extend([0] * 5)
            feature_names.extend([
                'avg_mouse_speed', 'avg_mouse_acceleration',
                'total_mouse_distance', 'mouse_movements', 'click_count'
            ])
        
        # Touch features
        if 'touch' in patterns and patterns['touch']:
            touch = patterns['touch']
            features.extend([
                touch.get('swipeCount', 0),
                touch.get('avgSwipeLength', 0),
                touch.get('avgSwipeDuration', 0),
                touch.get('avgSwipePressure', 0)
            ])
            feature_names.extend([
                'swipe_count', 'avg_swipe_length', 
                'avg_swipe_duration', 'avg_swipe_pressure'
            ])
        else:
            features.extend([0] * 4)
            feature_names.extend([
                'swipe_count', 'avg_swipe_length',
                'avg_swipe_duration', 'avg_swipe_pressure'
            ])
        
        return np.array(features), feature_names

    def register_user(self, username, password, patterns):
        """Register a new user with their behavioral patterns"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Hash password
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            device_fingerprint = json.dumps(patterns.get('device', {}))
            
            # Insert user
            cursor.execute('''
                INSERT INTO users (username, password_hash, device_fingerprint)
                VALUES (?, ?, ?)
            ''', (username, password_hash, device_fingerprint))
            
            user_id = cursor.lastrowid
            
            # Extract and store behavioral patterns
            features, feature_names = self.extract_features(patterns)
            
            # Store individual pattern types
            for pattern_type in ['keystroke', 'mouse', 'touch']:
                if pattern_type in patterns:
                    cursor.execute('''
                        INSERT INTO behavioral_patterns (user_id, pattern_type, features)
                        VALUES (?, ?, ?)
                    ''', (user_id, pattern_type, json.dumps(patterns[pattern_type])))
            
            # Store combined features
            cursor.execute('''
                INSERT INTO behavioral_patterns (user_id, pattern_type, features)
                VALUES (?, ?, ?)
            ''', (user_id, 'combined_features', json.dumps({
                'features': features.tolist(),
                'feature_names': feature_names
            })))
            
            conn.commit()
            conn.close()
            
            logger.info(f"User {username} registered successfully")
            return True, "Registration successful"
            
        except sqlite3.IntegrityError:
            return False, "Username already exists"
        except Exception as e:
            logger.error(f"Registration error: {str(e)}")
            return False, f"Registration failed: {str(e)}"

    def authenticate_user(self, username, password, patterns):
        """Authenticate user based on password and behavioral patterns"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Verify password
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            cursor.execute('''
                SELECT id, device_fingerprint FROM users 
                WHERE username = ? AND password_hash = ?
            ''', (username, password_hash))
            
            user_result = cursor.fetchone()
            if not user_result:
                conn.close()
                return False, 0.0, "Invalid credentials"
            
            user_id, stored_device_fp = user_result
            
            # Get stored behavioral patterns
            cursor.execute('''
                SELECT features FROM behavioral_patterns 
                WHERE user_id = ? AND pattern_type = 'combined_features'
                ORDER BY created_at DESC
            ''', (user_id,))
            
            stored_patterns = cursor.fetchall()
            if not stored_patterns:
                conn.close()
                return False, 0.0, "No behavioral patterns found"
            
            # Extract features from current patterns
            current_features, _ = self.extract_features(patterns)
            
            # Calculate similarity scores with stored patterns
            similarity_scores = []
            for pattern_data in stored_patterns[:5]:  # Use last 5 patterns
                stored_data = json.loads(pattern_data[0])
                stored_features = np.array(stored_data['features'])
                
                # Ensure same feature count
                min_len = min(len(current_features), len(stored_features))
                current_trimmed = current_features[:min_len]
                stored_trimmed = stored_features[:min_len]
                
                # Calculate euclidean distance similarity
                distance = euclidean(current_trimmed, stored_trimmed)
                max_possible_distance = np.sqrt(len(current_trimmed)) * 1000  # Normalize
                similarity = max(0, 1 - (distance / max_possible_distance))
                similarity_scores.append(similarity)
            
            # Average similarity score
            avg_similarity = np.mean(similarity_scores)
            
            # Device fingerprint check
            device_similarity = 0.5  # Default moderate similarity
            if stored_device_fp:
                stored_device = json.loads(stored_device_fp)
                current_device = patterns.get('device', {})
                
                # Check critical device attributes
                device_matches = 0
                device_checks = 0
                
                for key in ['userAgent', 'platform', 'screenResolution', 'colorDepth']:
                    if key in stored_device and key in current_device:
                        device_checks += 1
                        if stored_device[key] == current_device[key]:
                            device_matches += 1
                
                device_similarity = device_matches / device_checks if device_checks > 0 else 0.5
            
            # Combined confidence score
            confidence_score = (avg_similarity * 0.7 + device_similarity * 0.3)
            
            # Authentication threshold
            auth_threshold = 0.65
            is_authenticated = confidence_score >= auth_threshold
            
            # Log authentication attempt
            cursor.execute('''
                INSERT INTO auth_logs (user_id, username, success, confidence_score, 
                                     patterns_matched, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, username, is_authenticated, confidence_score, 
                  len(similarity_scores), 
                  request.environ.get('REMOTE_ADDR', 'unknown'),
                  request.environ.get('HTTP_USER_AGENT', 'unknown')))
            
            # If authenticated, store new pattern for learning
            if is_authenticated:
                cursor.execute('''
                    INSERT INTO behavioral_patterns (user_id, pattern_type, features)
                    VALUES (?, ?, ?)
                ''', (user_id, 'combined_features', json.dumps({
                    'features': current_features.tolist(),
                    'feature_names': _
                })))
            
            conn.commit()
            conn.close()
            
            status_msg = "Authentication successful" if is_authenticated else "Behavioral patterns do not match"
            logger.info(f"Authentication attempt for {username}: {is_authenticated} (confidence: {confidence_score:.3f})")
            
            return is_authenticated, confidence_score, status_msg
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False, 0.0, f"Authentication failed: {str(e)}"

    def get_user_statistics(self, username):
        """Get authentication statistics for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_attempts,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_attempts,
                    AVG(confidence_score) as avg_confidence,
                    MAX(confidence_score) as max_confidence,
                    MIN(confidence_score) as min_confidence
                FROM auth_logs 
                WHERE username = ? 
                AND created_at > datetime('now', '-30 days')
            ''', (username,))
            
            stats = cursor.fetchone()
            conn.close()
            
            if stats and stats[0] > 0:
                return {
                    'total_attempts': stats[0],
                    'successful_attempts': stats[1],
                    'success_rate': (stats[1] / stats[0]) * 100,
                    'avg_confidence': stats[2] or 0,
                    'max_confidence': stats[3] or 0,
                    'min_confidence': stats[4] or 0
                }
            else:
                return {
                    'total_attempts': 0,
                    'successful_attempts': 0,
                    'success_rate': 0,
                    'avg_confidence': 0,
                    'max_confidence': 0,
                    'min_confidence': 0
                }
        except Exception as e:
            logger.error(f"Statistics error: {str(e)}")
            return {}

    def save_models(self):
        """Save trained models to disk"""
        try:
            with open('biometric_models.pkl', 'wb') as f:
                pickle.dump({
                    'scaler': self.scaler,
                    'model': self.model,
                    'anomaly_detector': self.anomaly_detector
                }, f)
            logger.info("Models saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")

    def load_models(self):
        """Load trained models from disk"""
        try:
            if os.path.exists('biometric_models.pkl'):
                with open('biometric_models.pkl', 'rb') as f:
                    models = pickle.load(f)
                    self.scaler = models['scaler']
                    self.model = models['model']
                    self.anomaly_detector = models['anomaly_detector']
                logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")

# Initialize authenticator
authenticator = BiometricAuthenticator()

@app.route('/api/register', methods=['POST'])
def register():
    """User registration endpoint"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        patterns = data.get('patterns', {})
        
        if not username or not password:
            return jsonify({'success': False, 'error': 'Username and password required'})
        
        success, message = authenticator.register_user(username, password, patterns)
        return jsonify({'success': success, 'message': message})
        
    except Exception as e:
        logger.error(f"Registration endpoint error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        patterns = data.get('patterns', {})
        
        if not username or not password:
            return jsonify({'success': False, 'error': 'Username and password required'})
        
        authenticated, confidence, message = authenticator.authenticate_user(username, password, patterns)
        
        if authenticated:
            session['username'] = username
            session['authenticated'] = True
        
        return jsonify({
            'success': authenticated,
            'authenticated': authenticated,
            'confidence': confidence,
            'message': message
        })
        
    except Exception as e:
        logger.error(f"Login endpoint error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/biometric-auth', methods=['POST'])
def biometric_auth():
    """Main biometric authentication endpoint"""
    try:
        data = request.get_json()
        patterns = data.get('patterns', {})
        is_registration = data.get('isRegistration', False)
        
        if is_registration:
            # This is a signup - would need username/password from session or form
            return jsonify({'success': True, 'message': 'Patterns captured for registration'})
        else:
            # This is a login attempt - would need to get username from session
            username = session.get('username')
            if not username:
                return jsonify({'success': False, 'error': 'No active session'})
            
            # For demo purposes, we'll skip password verification here
            # In real implementation, password should be verified first
            authenticated, confidence, message = authenticator.authenticate_user(username, 'dummy_password', patterns)
            
            return jsonify({
                'success': authenticated,
                'authenticated': authenticated,
                'confidence': confidence,
                'message': message
            })
        
    except Exception as e:
        logger.error(f"Biometric auth endpoint error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/network-info', methods=['GET'])
def network_info():
    """Get network information for device fingerprinting"""
    try:
        ip_address = request.environ.get('REMOTE_ADDR', 'unknown')
        user_agent = request.environ.get('HTTP_USER_AGENT', 'unknown')
        
        # In a real implementation, you would use external services to get more info
        connection_type = 'unknown'
        if 'Mobile' in user_agent:
            connection_type = 'mobile'
        elif 'Tablet' in user_agent:
            connection_type = 'tablet'
        else:
            connection_type = 'desktop'
        
        return jsonify({
            'ip': ip_address,
            'connectionType': connection_type,
            'userAgent': user_agent
        })
        
    except Exception as e:
        logger.error(f"Network info error: {str(e)}")
        return jsonify({'ip': 'unknown', 'connectionType': 'unknown'})

@app.route('/api/stats/<username>', methods=['GET'])
def get_stats(username):
    """Get user authentication statistics"""
    try:
        stats = authenticator.get_user_statistics(username)
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Stats endpoint error: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
'''

with open('backend_biometric_auth.py', 'w') as f:
    f.write(backend_code)

print("✓ Created backend Python Flask application")