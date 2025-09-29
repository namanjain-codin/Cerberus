#!/usr/bin/env python3
"""
Integrated Behavioral Biometric Authentication Server
Serves both the API backend and HTML frontend
"""

from flask import Flask, request, jsonify, session, send_from_directory
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
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=20.0)
            # Use WAL mode for better concurrency and prevent locking
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            
            cursor = conn.cursor()

            # Create users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    device_fingerprint TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create behavioral patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS behavioral_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    pattern_type TEXT NOT NULL,
                    features TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)

            # Create authentication logs table
            cursor.execute("""
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
            """)

            conn.commit()
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
            raise
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass

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
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=20.0)
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()

            # Hash password
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            device_fingerprint = json.dumps(patterns.get('device', {}))

            # Insert user
            cursor.execute("""
                INSERT INTO users (username, password_hash, device_fingerprint)
                VALUES (?, ?, ?)
            """, (username, password_hash, device_fingerprint))

            user_id = cursor.lastrowid

            # Extract and store behavioral patterns
            features, feature_names = self.extract_features(patterns)

            # Store individual pattern types
            for pattern_type in ['keystroke', 'mouse', 'touch']:
                if pattern_type in patterns:
                    cursor.execute("""
                        INSERT INTO behavioral_patterns (user_id, pattern_type, features)
                        VALUES (?, ?, ?)
                    """, (user_id, pattern_type, json.dumps(patterns[pattern_type])))

            # Store combined features
            cursor.execute("""
                INSERT INTO behavioral_patterns (user_id, pattern_type, features)
                VALUES (?, ?, ?)
            """, (user_id, 'combined_features', json.dumps({
                'features': features.tolist(),
                'feature_names': feature_names
            })))

            conn.commit()

            logger.info(f"User {username} registered successfully")
            return True, "Registration successful"

        except sqlite3.IntegrityError:
            return False, "Username already exists"
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                logger.error(f"Database locked during registration for {username}")
                return False, "Database temporarily unavailable, please try again"
            else:
                logger.error(f"Database error during registration: {str(e)}")
                return False, f"Database error: {str(e)}"
        except Exception as e:
            logger.error(f"Registration error: {str(e)}")
            return False, f"Registration failed: {str(e)}"
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass

    def authenticate_user(self, username, password, patterns):
        """Authenticate user based on password and behavioral patterns"""
        conn = None
        try:
            # Use timeout and better connection handling
            conn = sqlite3.connect(self.db_path, timeout=20.0)
            conn.execute("PRAGMA journal_mode=WAL")  # Use WAL mode for better concurrency
            cursor = conn.cursor()

            # Verify password if provided
            if password:
                password_hash = hashlib.sha256(password.encode()).hexdigest()
                cursor.execute("""
                    SELECT id, device_fingerprint FROM users 
                    WHERE username = ? AND password_hash = ?
                """, (username, password_hash))
            else:
                # For biometric-only auth, just get user ID
                cursor.execute("""
                    SELECT id, device_fingerprint FROM users 
                    WHERE username = ?
                """, (username,))

            user_result = cursor.fetchone()
            if not user_result:
                return False, 0.0, "Invalid credentials"

            user_id, stored_device_fp = user_result

            # Get stored behavioral patterns
            cursor.execute("""
                SELECT features FROM behavioral_patterns 
                WHERE user_id = ? AND pattern_type = 'combined_features'
                ORDER BY created_at DESC
            """, (user_id,))

            stored_patterns = cursor.fetchall()
            if not stored_patterns:
                return False, 0.0, "No behavioral patterns found"

            # Extract features from current patterns
            current_features, current_feature_names = self.extract_features(patterns)

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
            if stored_device_fp and stored_device_fp.strip():
                try:
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

                    if device_checks > 0:
                        device_similarity = device_matches / device_checks
                        logger.info(f"Device fingerprint match: {device_matches}/{device_checks} = {device_similarity:.3f}")
                    else:
                        device_similarity = 0.5
                        logger.warning("No device attributes found for comparison")
                except json.JSONDecodeError:
                    logger.warning("Invalid device fingerprint JSON, using default similarity")
                    device_similarity = 0.5
            else:
                logger.warning("No stored device fingerprint found, using default similarity")
                device_similarity = 0.5

            # Improved confidence score calculation
            # If behavioral similarity is very low, reduce its weight
            if avg_similarity < 0.1:
                # Very low behavioral similarity - rely more on device fingerprint
                confidence_score = (avg_similarity * 0.3 + device_similarity * 0.7)
                logger.info(f"Low behavioral similarity ({avg_similarity:.3f}), using device-weighted calculation")
            else:
                # Normal calculation
                confidence_score = (avg_similarity * 0.7 + device_similarity * 0.3)
            
            # Ensure confidence score is reasonable (not too low due to poor data)
            if confidence_score < 0.1:
                confidence_score = 0.1  # Minimum reasonable confidence
                logger.warning(f"Confidence score too low, adjusted to {confidence_score}")

            # Enhanced authentication threshold with stricter requirements
            auth_threshold = 0.85  # Increased from 0.65 to 0.85 for better security
            is_authenticated = confidence_score >= auth_threshold
            
            # Additional security checks
            security_flags = []
            
            # Check for suspicious patterns
            if avg_similarity < 0.3:
                security_flags.append("LOW_BEHAVIORAL_SIMILARITY")
            
            if device_similarity < 0.4:
                security_flags.append("DEVICE_MISMATCH")
            
            # If behavioral auth fails but confidence is above 0.5, flag for enhanced verification
            if not is_authenticated and confidence_score >= 0.5:
                security_flags.append("REQUIRES_ENHANCED_VERIFICATION")
            
            logger.info(f"Confidence calculation: behavioral={avg_similarity:.3f}, device={device_similarity:.3f}, final={confidence_score:.3f}")

            # Log authentication attempt
            cursor.execute("""
                INSERT INTO auth_logs (user_id, username, success, confidence_score, 
                                     patterns_matched, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, username, is_authenticated, confidence_score, 
                  len(similarity_scores), 
                  'unknown',  # Would get from request in real app
                  'unknown'))  # Would get from request in real app

            # If authenticated, store new pattern for learning
            if is_authenticated:
                cursor.execute("""
                    INSERT INTO behavioral_patterns (user_id, pattern_type, features)
                    VALUES (?, ?, ?)
                """, (user_id, 'combined_features', json.dumps({
                    'features': current_features.tolist(),
                    'feature_names': current_feature_names
                })))

            conn.commit()

            # Determine authentication result and next steps
            if is_authenticated:
                status_msg = "Authentication successful"
            elif "REQUIRES_ENHANCED_VERIFICATION" in security_flags:
                status_msg = "REQUIRES_ENHANCED_VERIFICATION"
            else:
                status_msg = "Behavioral patterns do not match"
            
            logger.info(f"Authentication attempt for {username}: {is_authenticated} (confidence: {confidence_score:.3f})")

            return is_authenticated, confidence_score, status_msg, security_flags

        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                logger.error(f"Database locked during authentication for {username}")
                return False, 0.0, "Database temporarily unavailable, please try again"
            else:
                logger.error(f"Database error during authentication: {str(e)}")
                return False, 0.0, f"Database error: {str(e)}"
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False, 0.0, f"Authentication failed: {str(e)}"
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass

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

# Create Flask app instance
app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'
CORS(app)

# Initialize the authenticator
authenticator = BiometricAuthenticator()

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'biometric_auth_demo.html')

@app.route('/frontend_biometric_capture.js')
def serve_js():
    """Serve the JavaScript file"""
    return send_from_directory('.', 'frontend_biometric_capture.js', mimetype='application/javascript')

@app.route('/favicon.ico')
def serve_favicon():
    """Serve a default favicon or return 204 No Content"""
    return '', 204

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'database': 'connected' if os.path.exists(authenticator.db_path) else 'disconnected'
    })

@app.route('/api/register', methods=['POST'])
def register_user():
    """User registration endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({'success': False, 'error': 'Username and password required'}), 400
        
        username = data['username']
        password = data['password']
        patterns = data.get('patterns', {})
        
        # Register the user
        success, message = authenticator.register_user(username, password, patterns)
        
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'username': username
            })
        else:
            return jsonify({
                'success': False,
                'error': message
            }), 400
            
    except Exception as e:
        logger.error(f"Registration endpoint error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login_user():
    """User login endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({'success': False, 'error': 'Username and password required'}), 400
        
        username = data['username']
        password = data['password']
        patterns = data.get('patterns', {})
        
        # Authenticate the user
        auth_result = authenticator.authenticate_user(username, password, patterns)
        
        if len(auth_result) == 4:
            is_authenticated, confidence_score, message, security_flags = auth_result
        else:
            # Handle old return format for backward compatibility
            is_authenticated, confidence_score, message = auth_result
            security_flags = []
        
        if is_authenticated:
            # Set session
            session['user_id'] = username
            session['authenticated'] = True
            
            return jsonify({
                'success': True,
                'message': message,
                'confidence_score': confidence_score,
                'username': username
            })
        elif message == "REQUIRES_ENHANCED_VERIFICATION":
            # Set session for enhanced verification
            session['user_id'] = username
            session['needs_enhanced_verification'] = True
            
            return jsonify({
                'success': False,
                'requires_enhanced_verification': True,
                'message': 'Enhanced verification required for security',
                'confidence_score': confidence_score,
                'security_flags': security_flags,
                'redirect_url': '/enhanced-auth'
            }), 202  # 202 Accepted - requires additional verification
        else:
            return jsonify({
                'success': False,
                'error': message,
                'confidence_score': confidence_score
            }), 401
            
    except Exception as e:
        logger.error(f"Login endpoint error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/authenticate', methods=['POST'])
def authenticate_biometric():
    """Biometric authentication endpoint"""
    try:
        if not session.get('authenticated'):
            return jsonify({'success': False, 'error': 'No active session'}), 401
        
        data = request.get_json()
        username = session.get('user_id')
        patterns = data.get('patterns', {})
        
        if not patterns:
            return jsonify({'success': False, 'error': 'No behavioral patterns provided'}), 400
        
        # Perform biometric authentication
        is_authenticated, confidence_score, message = authenticator.authenticate_user(
            username, None, patterns
        )
        
        if is_authenticated:
            return jsonify({
                'success': True,
                'message': message,
                'confidence_score': confidence_score,
                'username': username
            })
        else:
            return jsonify({
                'success': False,
                'error': message,
                'confidence_score': confidence_score
            }), 401
            
    except Exception as e:
        logger.error(f"Biometric auth endpoint error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    """Logout endpoint"""
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        # Get basic stats from the database
        conn = sqlite3.connect(authenticator.db_path)
        cursor = conn.cursor()
        
        # User count
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        # Authentication attempts
        cursor.execute("SELECT COUNT(*) FROM auth_logs")
        total_auths = cursor.fetchone()[0]
        
        # Successful authentications
        cursor.execute("SELECT COUNT(*) FROM auth_logs WHERE success = 1")
        successful_auths = cursor.fetchone()[0]
        
        # Success rate
        success_rate = (successful_auths / total_auths * 100) if total_auths > 0 else 0
        
        conn.close()
        
        return jsonify({
            'success': True,
            'stats': {
                'total_users': user_count,
                'total_authentications': total_auths,
                'successful_authentications': successful_auths,
                'success_rate': round(success_rate, 2)
            }
        })
        
    except Exception as e:
        logger.error(f"Stats endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/network-info', methods=['GET'])
def get_network_info():
    """Get network and request information"""
    try:
        return jsonify({
            'success': True,
            'network_info': {
                'ip_address': request.remote_addr,
                'user_agent': request.headers.get('User-Agent', 'Unknown'),
                'timestamp': datetime.now().isoformat()
            }
        })
    except Exception as e:
        logger.error(f"Network info error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/enhanced-auth')
def enhanced_auth_redirect():
    """Redirect to enhanced authentication page"""
    if not session.get('needs_enhanced_verification'):
        return jsonify({'error': 'Enhanced verification not required'}), 400
    
    # Return HTML page for enhanced authentication
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Enhanced Authentication Required</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 20px;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .auth-container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                padding: 40px;
                max-width: 600px;
                width: 100%;
                text-align: center;
            }
            .auth-header h1 {
                color: #333;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .auth-header p {
                color: #666;
                font-size: 1.1em;
            }
            .btn {
                background: linear-gradient(45deg, #007bff, #0056b3);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 25px;
                font-size: 1.1em;
                cursor: pointer;
                transition: all 0.3s ease;
                margin: 10px;
                text-decoration: none;
                display: inline-block;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(0,123,255,0.3);
            }
            .security-info {
                background: #e3f2fd;
                border: 1px solid #bbdefb;
                border-radius: 10px;
                padding: 15px;
                margin: 20px 0;
                text-align: left;
            }
        </style>
    </head>
    <body>
        <div class="auth-container">
            <div class="auth-header">
                <h1>🔐 Enhanced Authentication Required</h1>
                <p>Additional verification needed for security</p>
            </div>
            
            <div class="security-info">
                <h4>🛡️ Why Enhanced Verification?</h4>
                <p>Your behavioral patterns didn't match our security requirements. To ensure your account security, we need additional verification through:</p>
                <ul>
                    <li>Face recognition scan</li>
                    <li>Voice pattern analysis</li>
                    <li>Location and device verification</li>
                </ul>
            </div>
            
            <a href="http://localhost:5001/enhanced-auth" class="btn">
                Start Enhanced Verification
            </a>
            
            <p style="margin-top: 20px; color: #666;">
                This process takes about 2-3 minutes and helps protect your account.
            </p>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    print("Starting Integrated Behavioral Biometric Authentication Server...")
    print("Frontend available at: http://localhost:5000")
    print("API endpoints available at: http://localhost:5000/api/")
    print("Health check: http://localhost:5000/api/health")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
