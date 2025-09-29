#!/usr/bin/env python3
"""
Simplified Enhanced Biometric Authentication System Startup Script
Runs the main server and simplified enhanced authentication server
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
        print(f"Starting {description} on port {port}...")
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
            print(f"ERROR: {description} failed with return code {return_code}")
            stderr = process.stderr.read()
            if stderr:
                print(f"Error: {stderr}")
        else:
            print(f"SUCCESS: {description} stopped gracefully")
            
    except Exception as e:
        print(f"ERROR: Error running {description}: {str(e)}")

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'flask', 'flask_cors', 'numpy', 'pandas', 'scikit_learn', 
        'scipy', 'opencv-python', 'librosa', 'soundfile', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'scikit_learn':
                import sklearn
            elif package == 'flask_cors':
                import flask_cors
            else:
                __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("ERROR: Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("SUCCESS: All required dependencies are installed")
    return True

def main():
    """Main startup function"""
    print("Simplified Enhanced Biometric Authentication System")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('server.py') or not os.path.exists('enhanced_auth_system_simple.py'):
        print("ERROR: Please run this script from the project root directory")
        print("   Make sure server.py and enhanced_auth_system_simple.py are present")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("\nERROR: Please install missing dependencies before starting the system")
        print("Try running: python install_enhanced_system.py")
        sys.exit(1)
    
    print("\nStarting Simplified Enhanced Biometric Authentication System...")
    print("=" * 60)
    
    # Start main server (port 5000)
    main_server_thread = threading.Thread(
        target=run_server, 
        args=('server.py', 5000, 'Main Biometric Server'),
        daemon=True
    )
    
    # Start simplified enhanced auth server (port 5001)
    enhanced_server_thread = threading.Thread(
        target=run_server, 
        args=('enhanced_auth_system_simple.py', 5001, 'Simplified Enhanced Auth Server'),
        daemon=True
    )
    
    try:
        # Start both servers
        main_server_thread.start()
        time.sleep(2)  # Give main server time to start
        enhanced_server_thread.start()
        
        print("\nSUCCESS: Both servers are starting...")
        print("\nAccess Points:")
        print("   Main System:     http://localhost:5000")
        print("   Enhanced Auth:   http://localhost:5001/enhanced-auth")
        print("   API Health:      http://localhost:5000/api/health")
        print("\nFeatures:")
        print("   • Improved behavioral biometric accuracy (85% threshold)")
        print("   • Voice pattern analysis (simplified)")
        print("   • IP location verification")
        print("   • Enhanced security layers")
        print("\nPress Ctrl+C to stop both servers")
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nShutting down servers...")
        print("SUCCESS: Simplified Enhanced Biometric Authentication System stopped")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: System error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
