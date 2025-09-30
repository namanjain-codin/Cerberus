#!/usr/bin/env python3
"""
Enhanced Authentication System with Face, Voice, and Location Verification
Fallback authentication for users who fail behavioral biometric verification
"""

import cv2
import numpy as np
import base64
import io
import json
import hashlib
import sqlite3
import requests
from datetime import datetime
from flask import Flask, request, jsonify, session, render_template_string
from flask_cors import CORS
import logging
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    FACE_IMPORT_ERROR = None
except Exception as _face_import_error:  # ImportError or other env errors on Windows
    FACE_RECOGNITION_AVAILABLE = False
    FACE_IMPORT_ERROR = str(_face_import_error)
import librosa
import soundfile as sf
from scipy.spatial.distance import cosine
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedAuthenticator:
    def __init__(self, db_path='enhanced_auth.db'):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize database for enhanced authentication"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Face biometrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_biometrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                face_encoding BLOB,
                face_image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # Voice biometrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS voice_biometrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                voice_features BLOB,
                voice_sample_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # Location verification table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS location_verification (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                ip_address TEXT,
                country TEXT,
                city TEXT,
                latitude REAL,
                longitude REAL,
                is_vpn BOOLEAN,
                risk_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # Enhanced auth sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS enhanced_auth_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_token TEXT,
                face_verified BOOLEAN,
                voice_verified BOOLEAN,
                location_verified BOOLEAN,
                ip_verified BOOLEAN,
                overall_verified BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def extract_face_features(self, image_data):
        """Extract face features from image"""
        try:
            if not FACE_RECOGNITION_AVAILABLE:
                return None, f"Face recognition not available: {FACE_IMPORT_ERROR}"
            # Decode base64 image
            if isinstance(image_data, str):
                image_data = base64.b64decode(image_data.split(',')[1])
            
            # Convert to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convert to RGB for face_recognition
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Find face locations
            face_locations = face_recognition.face_locations(rgb_image)
            
            if not face_locations:
                return None, "No face detected"
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if not face_encodings:
                return None, "Could not extract face features"
            
            # Return the first face encoding
            return face_encodings[0], "Face features extracted successfully"
            
        except Exception as e:
            logger.error(f"Face feature extraction error: {str(e)}")
            return None, f"Face extraction failed: {str(e)}"
    
    def extract_voice_features(self, audio_data):
        """Extract voice features from audio"""
        try:
            # Decode base64 audio
            if isinstance(audio_data, str):
                audio_data = base64.b64decode(audio_data)
            
            # Save to temporary file
            temp_path = f"temp_audio_{datetime.now().timestamp()}.wav"
            with open(temp_path, 'wb') as f:
                f.write(audio_data)
            
            # Load audio with librosa
            y, sr = librosa.load(temp_path, sr=16000)
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Extract other features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # Combine features
            features = np.concatenate([
                np.mean(mfccs, axis=1),
                np.mean(spectral_centroids),
                np.mean(zero_crossing_rate),
                np.mean(chroma, axis=1)
            ])
            
            # Clean up temp file
            os.remove(temp_path)
            
            return features, "Voice features extracted successfully"
            
        except Exception as e:
            logger.error(f"Voice feature extraction error: {str(e)}")
            return None, f"Voice extraction failed: {str(e)}"
    
    def verify_face(self, user_id, face_encoding):
        """Verify user's face against stored biometrics"""
        try:
            if not FACE_RECOGNITION_AVAILABLE:
                return False, 0.0, f"Face recognition not available: {FACE_IMPORT_ERROR}"
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get stored face encodings for user
            cursor.execute("""
                SELECT face_encoding FROM face_biometrics 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT 5
            """, (user_id,))
            
            stored_encodings = cursor.fetchall()
            if not stored_encodings:
                conn.close()
                return False, 0.0, "No face biometrics found for user"
            
            # Compare with stored encodings
            best_match = 0.0
            for stored_encoding in stored_encodings:
                stored_face = pickle.loads(stored_encoding[0])
                # Calculate face distance
                face_distance = face_recognition.face_distance([stored_face], face_encoding)[0]
                # Convert distance to similarity (lower distance = higher similarity)
                similarity = max(0, 1 - face_distance)
                best_match = max(best_match, similarity)
            
            conn.close()
            
            # Face verification threshold
            face_threshold = 0.6
            is_verified = best_match >= face_threshold
            
            return is_verified, best_match, "Face verification completed"
            
        except Exception as e:
            logger.error(f"Face verification error: {str(e)}")
            return False, 0.0, f"Face verification failed: {str(e)}"
    
    def verify_voice(self, user_id, voice_features):
        """Verify user's voice against stored biometrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get stored voice features for user
            cursor.execute("""
                SELECT voice_features FROM voice_biometrics 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT 5
            """, (user_id,))
            
            stored_features = cursor.fetchall()
            if not stored_features:
                conn.close()
                return False, 0.0, "No voice biometrics found for user"
            
            # Compare with stored features
            best_match = 0.0
            for stored_feature in stored_features:
                stored_voice = pickle.loads(stored_feature[0])
                # Calculate cosine similarity
                similarity = 1 - cosine(voice_features, stored_voice)
                best_match = max(best_match, similarity)
            
            conn.close()
            
            # Voice verification threshold
            voice_threshold = 0.7
            is_verified = best_match >= voice_threshold
            
            return is_verified, best_match, "Voice verification completed"
            
        except Exception as e:
            logger.error(f"Voice verification error: {str(e)}")
            return False, 0.0, f"Voice verification failed: {str(e)}"
    
    def verify_location(self, user_id, ip_address):
        """Verify user's location and IP address"""
        try:
            # Get IP geolocation (using a free service)
            try:
                response = requests.get(f"http://ip-api.com/json/{ip_address}", timeout=5)
                location_data = response.json()
                
                if location_data.get('status') == 'success':
                    country = location_data.get('country', 'Unknown')
                    city = location_data.get('city', 'Unknown')
                    lat = location_data.get('lat', 0)
                    lon = location_data.get('lon', 0)
                    isp = location_data.get('isp', 'Unknown')
                    
                    # Check if IP is from VPN/Proxy
                    is_vpn = any(keyword in isp.lower() for keyword in ['vpn', 'proxy', 'hosting', 'datacenter'])
                    
                    # Calculate risk score
                    risk_score = 0.0
                    if is_vpn:
                        risk_score += 0.3
                    if country == 'Unknown':
                        risk_score += 0.2
                    
                    # Store location data
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        INSERT INTO location_verification 
                        (user_id, ip_address, country, city, latitude, longitude, is_vpn, risk_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (user_id, ip_address, country, city, lat, lon, is_vpn, risk_score))
                    
                    conn.commit()
                    conn.close()
                    
                    # Location verification (basic check)
                    location_verified = risk_score < 0.5
                    
                    return location_verified, 1 - risk_score, {
                        'country': country,
                        'city': city,
                        'is_vpn': is_vpn,
                        'risk_score': risk_score
                    }
                else:
                    return False, 0.0, "Could not determine location"
                    
            except requests.RequestException:
                return False, 0.0, "Location service unavailable"
                
        except Exception as e:
            logger.error(f"Location verification error: {str(e)}")
            return False, 0.0, f"Location verification failed: {str(e)}"
    
    def register_face_biometric(self, user_id, face_encoding, face_image_path=None):
        """Register face biometric for user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Store face encoding
            face_blob = pickle.dumps(face_encoding)
            cursor.execute("""
                INSERT INTO face_biometrics (user_id, face_encoding, face_image_path)
                VALUES (?, ?, ?)
            """, (user_id, face_blob, face_image_path))
            
            conn.commit()
            conn.close()
            
            return True, "Face biometric registered successfully"
            
        except Exception as e:
            logger.error(f"Face registration error: {str(e)}")
            return False, f"Face registration failed: {str(e)}"
    
    def register_voice_biometric(self, user_id, voice_features, voice_sample_path=None):
        """Register voice biometric for user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Store voice features
            voice_blob = pickle.dumps(voice_features)
            cursor.execute("""
                INSERT INTO voice_biometrics (user_id, voice_features, voice_sample_path)
                VALUES (?, ?, ?)
            """, (user_id, voice_blob, voice_sample_path))
            
            conn.commit()
            conn.close()
            
            return True, "Voice biometric registered successfully"
            
        except Exception as e:
            logger.error(f"Voice registration error: {str(e)}")
            return False, f"Voice registration failed: {str(e)}"
    
    def perform_enhanced_verification(self, user_id, face_data=None, voice_data=None, ip_address=None):
        """Perform comprehensive enhanced verification"""
        verification_results = {
            'face_verified': False,
            'voice_verified': False,
            'location_verified': False,
            'overall_verified': False,
            'confidence_scores': {},
            'security_flags': []
        }
        
        try:
            # Face verification
            if face_data and FACE_RECOGNITION_AVAILABLE:
                face_encoding, face_msg = self.extract_face_features(face_data)
                if face_encoding is not None:
                    face_verified, face_confidence, face_result = self.verify_face(user_id, face_encoding)
                    verification_results['face_verified'] = face_verified
                    verification_results['confidence_scores']['face'] = face_confidence
                    if not face_verified:
                        verification_results['security_flags'].append("FACE_VERIFICATION_FAILED")
                else:
                    verification_results['security_flags'].append("FACE_EXTRACTION_FAILED")
            elif face_data and not FACE_RECOGNITION_AVAILABLE:
                logger.warning(f"Face verification skipped: module unavailable ({FACE_IMPORT_ERROR})")
                verification_results['security_flags'].append("FACE_MODULE_MISSING")
            
            # Voice verification
            if voice_data:
                voice_features, voice_msg = self.extract_voice_features(voice_data)
                if voice_features is not None:
                    voice_verified, voice_confidence, voice_result = self.verify_voice(user_id, voice_features)
                    verification_results['voice_verified'] = voice_verified
                    verification_results['confidence_scores']['voice'] = voice_confidence
                    if not voice_verified:
                        verification_results['security_flags'].append("VOICE_VERIFICATION_FAILED")
                else:
                    verification_results['security_flags'].append("VOICE_EXTRACTION_FAILED")
            
            # Location verification
            if ip_address:
                location_verified, location_confidence, location_data = self.verify_location(user_id, ip_address)
                verification_results['location_verified'] = location_verified
                verification_results['confidence_scores']['location'] = location_confidence
                if not location_verified:
                    verification_results['security_flags'].append("LOCATION_VERIFICATION_FAILED")
            
            # Overall verification decision
            verification_count = sum([
                verification_results['face_verified'],
                verification_results['voice_verified'],
                verification_results['location_verified']
            ])
            
            # Require at least 2 out of 3 verifications to pass
            verification_results['overall_verified'] = verification_count >= 2
            
            return verification_results
            
        except Exception as e:
            logger.error(f"Enhanced verification error: {str(e)}")
            verification_results['security_flags'].append(f"VERIFICATION_ERROR: {str(e)}")
            return verification_results

# Enhanced authentication HTML page
ENHANCED_AUTH_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Authentication - Face & Voice Verification</title>
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
        .auth-header {
            margin-bottom: 30px;
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
        .verification-step {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            border-left: 5px solid #007bff;
        }
        .verification-step h3 {
            color: #333;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .camera-container, .voice-container {
            margin: 20px 0;
        }
        .camera-preview {
            width: 100%;
            max-width: 300px;
            border-radius: 10px;
            border: 2px solid #ddd;
        }
        .voice-visualizer {
            height: 100px;
            background: #f0f0f0;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 15px 0;
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
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,123,255,0.3);
        }
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        .status {
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-weight: bold;
        }
        .status.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .status.warning {
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 20px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(45deg, #28a745, #20c997);
            transition: width 0.3s ease;
            border-radius: 10px;
        }
        .security-info {
            background: #e3f2fd;
            border: 1px solid #bbdefb;
            border-radius: 10px;
            padding: 15px;
            margin: 20px 0;
            text-align: left;
        }
        .security-info h4 {
            color: #1976d2;
            margin-bottom: 10px;
        }
        .security-info ul {
            margin: 0;
            padding-left: 20px;
        }
        .security-info li {
            color: #424242;
            margin: 5px 0;
        }
    </style>
</head>
<body>
    <div class="auth-container">
        <div class="auth-header">
            <h1>🔐 Enhanced Authentication</h1>
            <p>Additional verification required for security</p>
        </div>
        
        <div class="security-info">
            <h4>🛡️ Security Verification</h4>
            <ul>
                <li>Face recognition for identity verification</li>
                <li>Voice pattern analysis for authentication</li>
                <li>Location and IP address verification</li>
                <li>Device fingerprinting for additional security</li>
            </ul>
        </div>
        
        <div class="progress-bar">
            <div class="progress-fill" id="progressFill" style="width: 0%"></div>
        </div>
        
        <div class="verification-step">
            <h3>📷 Face Verification</h3>
            <div class="camera-container">
                <video id="cameraPreview" class="camera-preview" autoplay></video>
                <br>
                <button id="captureFace" class="btn">Capture Face</button>
                <button id="startCamera" class="btn">Start Camera</button>
            </div>
            <div id="faceStatus" class="status" style="display: none;"></div>
        </div>
        
        <div class="verification-step">
            <h3>🎤 Voice Verification</h3>
            <div class="voice-container">
                <div class="voice-visualizer" id="voiceVisualizer">
                    <p>Click "Record Voice" to start</p>
                </div>
                <button id="recordVoice" class="btn">Record Voice</button>
                <button id="stopVoice" class="btn" style="display: none;">Stop Recording</button>
            </div>
            <div id="voiceStatus" class="status" style="display: none;"></div>
        </div>
        
        <div class="verification-step">
            <h3>🌍 Location Verification</h3>
            <p>Verifying your location and IP address...</p>
            <div id="locationStatus" class="status" style="display: none;"></div>
        </div>
        
        <button id="submitVerification" class="btn" style="display: none; background: linear-gradient(45deg, #28a745, #20c997);">
            Submit Verification
        </button>
        
        <div id="finalStatus" class="status" style="display: none;"></div>
    </div>

    <script>
        class EnhancedAuth {
            constructor() {
                this.mediaStream = null;
                this.audioRecorder = null;
                this.audioChunks = [];
                this.verificationData = {
                    face: null,
                    voice: null,
                    location: null
                };
                this.init();
            }
            
            init() {
                this.setupEventListeners();
                this.verifyLocation();
            }
            
            setupEventListeners() {
                document.getElementById('startCamera').addEventListener('click', () => this.startCamera());
                document.getElementById('captureFace').addEventListener('click', () => this.captureFace());
                document.getElementById('recordVoice').addEventListener('click', () => this.startVoiceRecording());
                document.getElementById('stopVoice').addEventListener('click', () => this.stopVoiceRecording());
                document.getElementById('submitVerification').addEventListener('click', () => this.submitVerification());
            }
            
            async startCamera() {
                try {
                    this.mediaStream = await navigator.mediaDevices.getUserMedia({ 
                        video: { width: 640, height: 480 },
                        audio: false 
                    });
                    document.getElementById('cameraPreview').srcObject = this.mediaStream;
                    document.getElementById('captureFace').disabled = false;
                    this.showStatus('faceStatus', 'Camera started. Please position your face in the frame.', 'warning');
                } catch (error) {
                    this.showStatus('faceStatus', 'Camera access denied. Please allow camera access.', 'error');
                }
            }
            
            captureFace() {
                const video = document.getElementById('cameraPreview');
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0);
                
                const imageData = canvas.toDataURL('image/jpeg', 0.8);
                this.verificationData.face = imageData;
                
                this.showStatus('faceStatus', 'Face captured successfully!', 'success');
                this.updateProgress(33);
                this.checkVerificationComplete();
            }
            
            async startVoiceRecording() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    this.audioRecorder = new MediaRecorder(stream);
                    this.audioChunks = [];
                    
                    this.audioRecorder.ondataavailable = (event) => {
                        this.audioChunks.push(event.data);
                    };
                    
                    this.audioRecorder.onstop = () => {
                        const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
                        const reader = new FileReader();
                        reader.onload = () => {
                            this.verificationData.voice = reader.result;
                            this.showStatus('voiceStatus', 'Voice recorded successfully!', 'success');
                            this.updateProgress(66);
                            this.checkVerificationComplete();
                        };
                        reader.readAsDataURL(audioBlob);
                    };
                    
                    this.audioRecorder.start();
                    document.getElementById('recordVoice').style.display = 'none';
                    document.getElementById('stopVoice').style.display = 'inline-block';
                    this.showStatus('voiceStatus', 'Recording... Please speak clearly.', 'warning');
                } catch (error) {
                    this.showStatus('voiceStatus', 'Microphone access denied. Please allow microphone access.', 'error');
                }
            }
            
            stopVoiceRecording() {
                if (this.audioRecorder && this.audioRecorder.state === 'recording') {
                    this.audioRecorder.stop();
                    document.getElementById('recordVoice').style.display = 'inline-block';
                    document.getElementById('stopVoice').style.display = 'none';
                }
            }
            
            async verifyLocation() {
                try {
                    const response = await fetch('/api/network-info');
                    const data = await response.json();
                    
                    if (data.success) {
                        this.verificationData.location = data.network_info;
                        this.showStatus('locationStatus', 'Location verified successfully!', 'success');
                        this.updateProgress(100);
                        this.checkVerificationComplete();
                    } else {
                        this.showStatus('locationStatus', 'Location verification failed.', 'error');
                    }
                } catch (error) {
                    this.showStatus('locationStatus', 'Location verification error.', 'error');
                }
            }
            
            checkVerificationComplete() {
                const hasFace = this.verificationData.face !== null;
                const hasVoice = this.verificationData.voice !== null;
                const hasLocation = this.verificationData.location !== null;
                
                if (hasFace && hasVoice && hasLocation) {
                    document.getElementById('submitVerification').style.display = 'inline-block';
                }
            }
            
            async submitVerification() {
                try {
                    const response = await fetch('/api/enhanced-verification', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            face_data: this.verificationData.face,
                            voice_data: this.verificationData.voice,
                            location_data: this.verificationData.location
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (result.success && result.verified) {
                        this.showStatus('finalStatus', '✅ Enhanced verification successful! Access granted.', 'success');
                        setTimeout(() => {
                            window.location.href = '/dashboard';
                        }, 2000);
                    } else {
                        this.showStatus('finalStatus', '❌ Verification failed. Please try again or contact support.', 'error');
                    }
                } catch (error) {
                    this.showStatus('finalStatus', 'Verification submission failed. Please try again.', 'error');
                }
            }
            
            showStatus(elementId, message, type) {
                const element = document.getElementById(elementId);
                element.textContent = message;
                element.className = `status ${type}`;
                element.style.display = 'block';
            }
            
            updateProgress(percentage) {
                document.getElementById('progressFill').style.width = percentage + '%';
            }
        }
        
        // Initialize enhanced authentication
        new EnhancedAuth();
    </script>
</body>
</html>
"""

# Flask app for enhanced authentication
app = Flask(__name__)
app.secret_key = 'enhanced-auth-secret-key'
CORS(app)

# Initialize enhanced authenticator
enhanced_auth = EnhancedAuthenticator()

@app.route('/enhanced-auth')
def enhanced_auth_page():
    """Serve the enhanced authentication page"""
    return render_template_string(ENHANCED_AUTH_HTML)

@app.route('/api/enhanced-verification', methods=['POST'])
def enhanced_verification():
    """Handle enhanced verification request"""
    try:
        data = request.get_json()
        user_id = session.get('user_id')
        
        if not user_id:
            return jsonify({'success': False, 'error': 'No active session'}), 401
        
        # Get IP address
        ip_address = request.remote_addr
        
        # Perform enhanced verification
        verification_results = enhanced_auth.perform_enhanced_verification(
            user_id=user_id,
            face_data=data.get('face_data'),
            voice_data=data.get('voice_data'),
            ip_address=ip_address
        )
        
        if verification_results['overall_verified']:
            # Create enhanced auth session
            session['enhanced_verified'] = True
            session['verification_timestamp'] = datetime.now().isoformat()
            
            return jsonify({
                'success': True,
                'verified': True,
                'message': 'Enhanced verification successful',
                'results': verification_results
            })
        else:
            return jsonify({
                'success': False,
                'verified': False,
                'message': 'Enhanced verification failed',
                'results': verification_results
            }), 401
            
    except Exception as e:
        logger.error(f"Enhanced verification error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/register-biometrics', methods=['POST'])
def register_biometrics():
    """Register face and voice biometrics for a user"""
    try:
        data = request.get_json()
        user_id = session.get('user_id')
        
        if not user_id:
            return jsonify({'success': False, 'error': 'No active session'}), 401
        
        results = {}
        
        # Register face biometric
        if data.get('face_data'):
            face_encoding, face_msg = enhanced_auth.extract_face_features(data['face_data'])
            if face_encoding is not None:
                success, message = enhanced_auth.register_face_biometric(user_id, face_encoding)
                results['face'] = {'success': success, 'message': message}
            else:
                results['face'] = {'success': False, 'message': face_msg}
        
        # Register voice biometric
        if data.get('voice_data'):
            voice_features, voice_msg = enhanced_auth.extract_voice_features(data['voice_data'])
            if voice_features is not None:
                success, message = enhanced_auth.register_voice_biometric(user_id, voice_features)
                results['voice'] = {'success': success, 'message': message}
            else:
                results['voice'] = {'success': False, 'message': voice_msg}
        
        return jsonify({
            'success': True,
            'message': 'Biometric registration completed',
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Biometric registration error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced Authentication Server...")
    print("📱 Enhanced Auth Page: http://localhost:5001/enhanced-auth")
    print("🔌 API endpoints available at: http://localhost:5001/api/")
    
    app.run(debug=True, host='0.0.0.0', port=5001)

