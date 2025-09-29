#!/usr/bin/env python3
"""
Enhanced Biometric Authentication System Startup Script
Runs both the main server and enhanced authentication server
"""

import subprocess
import sys
import time
import threading
import os
from pathlib import Path

def run_server(script_name, port, description):
    """Run a server in a separate process"""
    try:
        print(f"🚀 Starting {description} on port {port}...")
        process = subprocess.Popen([
            sys.executable, script_name
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Monitor the process
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"[{description}] {output.strip()}")
        
        return_code = process.poll()
        if return_code != 0:
            print(f"❌ {description} failed with return code {return_code}")
            stderr = process.stderr.read()
            if stderr:
                print(f"Error: {stderr}")
        else:
            print(f"✅ {description} stopped gracefully")
            
    except Exception as e:
        print(f"❌ Error running {description}: {str(e)}")

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'flask', 'flask_cors', 'numpy', 'pandas', 'scikit_learn', 
        'scipy', 'opencv-python', 'face_recognition', 'librosa', 
        'soundfile', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'face_recognition':
                import face_recognition
            elif package == 'scikit_learn':
                import sklearn
            elif package == 'flask_cors':
                import flask_cors
            else:
                __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All required dependencies are installed")
    return True

def main():
    """Main startup function"""
    print("🔐 Enhanced Biometric Authentication System")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('server.py') or not os.path.exists('enhanced_auth_system.py'):
        print("❌ Please run this script from the project root directory")
        print("   Make sure server.py and enhanced_auth_system.py are present")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before starting the system")
        sys.exit(1)
    
    print("\n🚀 Starting Enhanced Biometric Authentication System...")
    print("=" * 50)
    
    # Start main server (port 5000)
    main_server_thread = threading.Thread(
        target=run_server, 
        args=('server.py', 5000, 'Main Biometric Server'),
        daemon=True
    )
    
    # Start enhanced auth server (port 5001)
    enhanced_server_thread = threading.Thread(
        target=run_server, 
        args=('enhanced_auth_system.py', 5001, 'Enhanced Auth Server'),
        daemon=True
    )
    
    try:
        # Start both servers
        main_server_thread.start()
        time.sleep(2)  # Give main server time to start
        enhanced_server_thread.start()
        
        print("\n✅ Both servers are starting...")
        print("\n📱 Access Points:")
        print("   Main System:     http://localhost:5000")
        print("   Enhanced Auth:   http://localhost:5001/enhanced-auth")
        print("   API Health:      http://localhost:5000/api/health")
        print("\n🔧 Features:")
        print("   • Improved behavioral biometric accuracy (85% threshold)")
        print("   • Face recognition verification")
        print("   • Voice pattern analysis")
        print("   • IP location verification")
        print("   • Enhanced security layers")
        print("\n⚠️  Press Ctrl+C to stop both servers")
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down servers...")
        print("✅ Enhanced Biometric Authentication System stopped")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ System error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

