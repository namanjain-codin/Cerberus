# Behavioral Biometric Authentication System - Integrated Frontend & Backend

This system integrates the `demo_biometric_system.py` backend with the `biometric_auth_demo.html` frontend to create a complete web-based behavioral biometric authentication application.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Integrated Server
```bash
python server.py
```

### 3. Access the Application
Open your browser and go to: `http://localhost:5000`

## 📁 File Structure

- **`server.py`** - Main integrated server (Flask + HTML serving)
- **`biometric_auth_demo.html`** - Frontend interface
- **`backend_biometric_auth.py`** - Backend API (standalone version)
- **`demo_biometric_system.py`** - Demo script (standalone version)
- **`accuracy_checker.py`** - Accuracy analysis module
- **`biometric_auth.db`** - SQLite database

## 🔧 How It Works

### Frontend (HTML/JavaScript)
- **User Registration**: Captures behavioral patterns during signup
- **User Login**: Authenticates with password + behavioral patterns
- **Biometric Auth**: Pure behavioral pattern authentication
- **Real-time Capture**: Mouse movements, keystrokes, device fingerprinting

### Backend (Python/Flask)
- **API Endpoints**: RESTful API for all operations
- **Pattern Analysis**: Machine learning-based behavioral analysis
- **Database Storage**: SQLite for users, patterns, and logs
- **Security**: Password hashing, session management

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main HTML page |
| `/api/health` | GET | System health check |
| `/api/register` | POST | User registration |
| `/api/login` | POST | User login |
| `/api/authenticate` | POST | Biometric authentication |
| `/api/logout` | POST | User logout |
| `/api/stats` | GET | System statistics |
| `/api/network-info` | GET | Network information |

## 📱 User Workflow

### 1. Registration
1. Fill in username and password
2. Click "Start Capture" to begin behavioral pattern collection
3. Type naturally in the form fields
4. Click "Complete Registration" to submit

### 2. Login
1. Enter username and password
2. Click "Start Capture" to collect behavioral patterns
3. Type naturally in the form fields
4. Click "Complete Login" to authenticate

### 3. Biometric Authentication
1. Click "Authenticate Biometrically" (requires active session)
2. System captures behavioral patterns
3. Compares with stored patterns for verification

## 🔒 Security Features

- **Password Hashing**: SHA-256 encryption
- **Behavioral Profiling**: Multi-modal pattern analysis
- **Device Fingerprinting**: Browser/device identification
- **Confidence Scoring**: Authentication confidence metrics
- **Session Management**: Secure user sessions

## 📊 Behavioral Patterns Captured

- **Keystroke Dynamics**: Typing speed, dwell times, flight times
- **Mouse Behavior**: Movement patterns, click patterns, acceleration
- **Device Fingerprint**: Browser, platform, screen, hardware info
- **Touch Patterns**: Swipe gestures, pressure, duration (mobile)

## 🛠️ Development

### Running in Development Mode
```bash
python server.py
```

### Testing the API
```bash
# Health check
curl http://localhost:5000/api/health

# Register user
curl -X POST http://localhost:5000/api/register \
  -H "Content-Type: application/json" \
  -d '{"username":"test","password":"test123","patterns":{}}'
```

### Database Inspection
```bash
sqlite3 biometric_auth.db
.tables
SELECT * FROM users;
SELECT * FROM behavioral_patterns;
SELECT * FROM auth_logs;
```

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Find process using port 5000
   netstat -ano | findstr :5000
   # Kill the process
   taskkill /PID <PID> /F
   ```

2. **Database Errors**
   ```bash
   # Remove corrupted database
   rm biometric_auth.db
   # Restart server (will recreate database)
   ```

3. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   ```

### Debug Mode
The server runs in debug mode by default. Check the console for detailed error messages and logs.

## 🔮 Future Enhancements

- **Multi-factor Authentication**: SMS, email verification
- **Advanced ML Models**: Deep learning for pattern recognition
- **Real-time Monitoring**: Live authentication attempt monitoring
- **Mobile App**: Native mobile application
- **Cloud Deployment**: AWS/Azure deployment options

## 📝 License

This project is for educational and research purposes. Please ensure compliance with local privacy and security regulations when deploying in production.

## 🤝 Support

For issues or questions:
1. Check the console logs for error messages
2. Verify all dependencies are installed
3. Ensure the database file is writable
4. Check firewall/antivirus settings

---

**Note**: This is a demonstration system. For production use, implement additional security measures, proper error handling, and comprehensive testing.
