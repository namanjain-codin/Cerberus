#!/usr/bin/env python3
"""
Enhanced Biometric Authentication System - Installation Helper
Handles dependency installation with fallback options for Windows
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and return success status"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ {description} error: {str(e)}")
        return False

def check_cmake():
    """Check if CMake is installed"""
    try:
        result = subprocess.run(['cmake', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CMake is installed")
            return True
        else:
            print("❌ CMake is not installed")
            return False
    except FileNotFoundError:
        print("❌ CMake is not installed")
        return False

def install_cmake_windows():
    """Install CMake on Windows using chocolatey or direct download"""
    print("🔧 Installing CMake for Windows...")
    
    # Try chocolatey first
    if run_command("choco install cmake -y", "Installing CMake via Chocolatey"):
        return True
    
    # Try winget
    if run_command("winget install Kitware.CMake", "Installing CMake via Winget"):
        return True
    
    print("❌ Automatic CMake installation failed")
    print("📥 Please install CMake manually:")
    print("   1. Download from: https://cmake.org/download/")
    print("   2. Install and add to PATH")
    print("   3. Restart your terminal")
    return False

def install_basic_dependencies():
    """Install basic dependencies that don't require compilation"""
    basic_deps = [
        "flask==2.3.3",
        "flask-cors==4.0.0", 
        "numpy==1.24.3",
        "pandas==2.0.3",
        "scikit-learn==1.3.0",
        "scipy==1.11.1",
        "requests==2.31.0",
        "matplotlib==3.7.2",
        "seaborn==0.12.2"
    ]
    
    print("📦 Installing basic dependencies...")
    for dep in basic_deps:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            print(f"⚠️  Failed to install {dep}, continuing...")
    
    return True

def install_audio_dependencies():
    """Install audio processing dependencies"""
    audio_deps = [
        "librosa==0.10.1",
        "soundfile==0.12.1"
    ]
    
    print("🎵 Installing audio processing dependencies...")
    for dep in audio_deps:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            print(f"⚠️  Failed to install {dep}, continuing...")
    
    return True

def install_opencv():
    """Install OpenCV"""
    print("📷 Installing OpenCV...")
    if run_command("pip install opencv-python==4.8.1.78", "Installing OpenCV"):
        return True
    else:
        print("⚠️  OpenCV installation failed, trying alternative...")
        return run_command("pip install opencv-python-headless", "Installing OpenCV Headless")

def create_alternative_requirements():
    """Create alternative requirements file without face-recognition"""
    alt_requirements = """flask==2.3.3
flask-cors==4.0.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
scipy==1.11.1
requests==2.31.0
opencv-python==4.8.1.78
librosa==0.10.1
soundfile==0.12.1
matplotlib==3.7.2
seaborn==0.12.2

# Face recognition dependencies (install manually if needed)
# dlib
# face-recognition
"""
    
    with open('requirements_alternative.txt', 'w') as f:
        f.write(alt_requirements)
    
    print("📝 Created requirements_alternative.txt without face-recognition")

def create_enhanced_system_without_face():
    """Create a version of the enhanced system without face recognition"""
    print("🔧 Creating enhanced system without face recognition...")
    
    # Read the original enhanced_auth_system.py
    with open('enhanced_auth_system.py', 'r') as f:
        content = f.read()
    
    # Replace face recognition imports with alternatives
    content = content.replace(
        "import face_recognition",
        "# import face_recognition  # Disabled - requires CMake"
    )
    content = content.replace(
        "face_recognition.face_locations",
        "# face_recognition.face_locations  # Disabled"
    )
    content = content.replace(
        "face_recognition.face_encodings", 
        "# face_recognition.face_encodings  # Disabled"
    )
    content = content.replace(
        "face_recognition.face_distance",
        "# face_recognition.face_distance  # Disabled"
    )
    
    # Create alternative version
    with open('enhanced_auth_system_no_face.py', 'w') as f:
        f.write(content)
    
    print("✅ Created enhanced_auth_system_no_face.py")

def main():
    """Main installation process"""
    print("🔐 Enhanced Biometric Authentication System - Installation Helper")
    print("=" * 70)
    
    system = platform.system().lower()
    print(f"🖥️  Detected system: {system}")
    
    # Step 1: Install basic dependencies
    print("\n📦 Step 1: Installing basic dependencies...")
    install_basic_dependencies()
    
    # Step 2: Install audio dependencies
    print("\n🎵 Step 2: Installing audio processing dependencies...")
    install_audio_dependencies()
    
    # Step 3: Install OpenCV
    print("\n📷 Step 3: Installing OpenCV...")
    install_opencv()
    
    # Step 4: Handle face recognition
    print("\n👤 Step 4: Handling face recognition dependencies...")
    
    if system == "windows":
        print("🪟 Windows detected - checking CMake...")
        if not check_cmake():
            print("\n🔧 CMake not found. Installing CMake...")
            if install_cmake_windows():
                print("✅ CMake installed successfully")
            else:
                print("⚠️  CMake installation failed")
                print("📝 Creating alternative system without face recognition...")
                create_enhanced_system_without_face()
                create_alternative_requirements()
                print("\n✅ Alternative system created!")
                print("📁 Files created:")
                print("   - enhanced_auth_system_no_face.py")
                print("   - requirements_alternative.txt")
                print("\n🚀 You can now run:")
                print("   pip install -r requirements_alternative.txt")
                print("   python enhanced_auth_system_no_face.py")
                return True
        
        # Try to install face recognition
        print("👤 Attempting to install face recognition...")
        if run_command("pip install dlib", "Installing dlib"):
            if run_command("pip install face-recognition", "Installing face-recognition"):
                print("✅ Face recognition installed successfully!")
            else:
                print("⚠️  Face recognition installation failed")
                create_enhanced_system_without_face()
        else:
            print("⚠️  dlib installation failed")
            create_enhanced_system_without_face()
    else:
        # Linux/Mac - try direct installation
        print("🐧 Linux/Mac detected - attempting direct installation...")
        if run_command("pip install dlib", "Installing dlib"):
            if run_command("pip install face-recognition", "Installing face-recognition"):
                print("✅ Face recognition installed successfully!")
            else:
                print("⚠️  Face recognition installation failed")
                create_enhanced_system_without_face()
        else:
            print("⚠️  dlib installation failed")
            create_enhanced_system_without_face()
    
    # Step 5: Test installation
    print("\n🧪 Step 5: Testing installation...")
    try:
        import flask
        import numpy
        import cv2
        print("✅ Basic dependencies working")
        
        try:
            import librosa
            print("✅ Audio processing working")
        except ImportError:
            print("⚠️  Audio processing not available")
        
        try:
            import face_recognition
            print("✅ Face recognition working")
        except ImportError:
            print("⚠️  Face recognition not available - using alternative system")
            
    except ImportError as e:
        print(f"❌ Basic dependencies failed: {str(e)}")
        return False
    
    print("\n🎉 Installation completed!")
    print("\n📱 Next steps:")
    print("   1. Run: python start_enhanced_system.py")
    print("   2. Or run: python test_enhanced_system.py")
    print("   3. Access: http://localhost:5000")
    
    return True

if __name__ == "__main__":
    main()
