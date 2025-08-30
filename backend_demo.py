#!/usr/bin/env python3
"""
Backend Biometric Authentication System Demo
Demonstrates feature extraction, authentication, and accuracy metrics
"""

import json
import hashlib
import numpy as np
from datetime import datetime
import sqlite3
from backend_biometric_auth import BiometricAuthenticator

def create_sample_biometric_data():
    """Create realistic sample biometric data for demonstration"""
    
    # Sample device fingerprint
    device_data = {
        'userAgent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'platform': 'Win32',
        'screenResolution': '1920x1080',
        'colorDepth': 24,
        'hardwareConcurrency': 8,
        'plugins': ['Chrome PDF Plugin', 'Chrome PDF Viewer', 'Native Client'],
        'canvasFingerprint': 'canvas_hash_12345'
    }
    
    # Sample keystroke dynamics
    keystroke_data = {
        'avgDwellTimes': {
            'a': 120, 'e': 95, 'i': 110, 'o': 105, 'u': 115,
            'n': 100, 't': 125, 'r': 90, 's': 105, 'l': 110
        },
        'typingSpeed': 45,  # words per minute
        'avgFlightTimes': {
            'th': 85, 'he': 75, 'in': 80, 'er': 70, 'an': 90
        }
    }
    
    # Sample mouse movement patterns
    mouse_data = {
        'avgMouseSpeed': 1250,  # pixels per second
        'avgMouseAcceleration': 85,
        'totalMouseDistance': 4500,  # pixels
        'mouseMovements': 45,
        'clickCount': 12
    }
    
    # Sample touch patterns (for mobile devices)
    touch_data = {
        'swipeCount': 8,
        'avgSwipeLength': 120,  # pixels
        'avgSwipeDuration': 350,  # milliseconds
        'avgSwipePressure': 0.75  # normalized pressure
    }
    
    return {
        'device': device_data,
        'keystroke': keystroke_data,
        'mouse': mouse_data,
        'touch': touch_data
    }

def demonstrate_feature_extraction(authenticator):
    """Demonstrate how features are extracted from biometric data"""
    print("\n" + "="*80)
    print("FEATURE EXTRACTION DEMONSTRATION")
    print("="*80)
    
    # Create sample data
    sample_patterns = create_sample_biometric_data()
    
    print("\n📱 SAMPLE BIOMETRIC DATA:")
    print(json.dumps(sample_patterns, indent=2))
    
    # Extract features
    features, feature_names = authenticator.extract_features(sample_patterns)
    
    print(f"\n🔍 EXTRACTED FEATURES:")
    print(f"Total Features: {len(features)}")
    print(f"Feature Vector Shape: {features.shape}")
    
    print(f"\n📊 FEATURE BREAKDOWN:")
    
    # Device features (first 7 features)
    print(f"\n🖥️  DEVICE FEATURES ({len(features[:7])} features):")
    for i, name in enumerate(feature_names[:7]):
        print(f"  {name}: {features[i]:.2f}")
    
    # Keystroke features (next 16 features)
    print(f"\n⌨️  KEYSTROKE FEATURES ({len(features[7:23])} features):")
    for i, name in enumerate(feature_names[7:23]):
        print(f"  {name}: {features[i]:.2f}")
    
    # Mouse features (next 5 features)
    print(f"\n🖱️  MOUSE FEATURES ({len(features[23:28])} features):")
    for i, name in enumerate(feature_names[23:28]):
        print(f"  {name}: {features[i]:.2f}")
    
    # Touch features (last 4 features)
    print(f"\n👆 TOUCH FEATURES ({len(features[28:])} features):")
    for i, name in enumerate(feature_names[28:]):
        print(f"  {name}: {features[i]:.2f}")
    
    return features, feature_names, sample_patterns

def demonstrate_user_registration(authenticator, features, feature_names, sample_patterns):
    """Demonstrate user registration process"""
    print("\n" + "="*80)
    print("USER REGISTRATION DEMONSTRATION")
    print("="*80)
    
    # Register a test user
    username = "demo_user_001"
    password = "secure_password_123"
    
    print(f"\n👤 REGISTERING USER: {username}")
    print(f"Password: {password}")
    
    # Register the user
    success, message = authenticator.register_user(username, password, sample_patterns)
    
    if success:
        print(f"✅ Registration: {message}")
        
        # Show what was stored in the database
        print(f"\n💾 STORED IN DATABASE:")
        print(f"  - Username: {username}")
        print(f"  - Password Hash: {hashlib.sha256(password.encode()).hexdigest()[:16]}...")
        print(f"  - Device Fingerprint: {json.dumps(sample_patterns['device'], indent=2)}")
        print(f"  - Behavioral Features: {len(features)} features stored")
        
        return username
    else:
        print(f"❌ Registration Failed: {message}")
        return None

def demonstrate_authentication(authenticator, username, sample_patterns):
    """Demonstrate authentication process"""
    print("\n" + "="*80)
    print("AUTHENTICATION DEMONSTRATION")
    print("="*80)
    
    print(f"\n🔐 AUTHENTICATING USER: {username}")
    
    # Test 1: Correct password + correct patterns
    print(f"\n🧪 TEST 1: Correct Password + Correct Patterns")
    success, confidence, message = authenticator.authenticate_user(
        username, "secure_password_123", sample_patterns
    )
    
    print(f"  Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"  Confidence Score: {confidence:.3f}")
    print(f"  Message: {message}")
    
    # Test 2: Correct password + modified patterns (simulating different user)
    print(f"\n🧪 TEST 2: Correct Password + Modified Patterns (Different User)")
    
    modified_patterns = sample_patterns.copy()
    modified_patterns['keystroke']['typingSpeed'] = 80  # Different typing speed
    modified_patterns['mouse']['avgMouseSpeed'] = 2000  # Different mouse speed
    
    success2, confidence2, message2 = authenticator.authenticate_user(
        username, "secure_password_123", modified_patterns
    )
    
    print(f"  Result: {'✅ SUCCESS' if success2 else '❌ FAILED'}")
    print(f"  Confidence Score: {confidence2:.3f}")
    print(f"  Message: {message2}")
    
    # Test 3: Biometric-only authentication (no password)
    print(f"\n🧪 TEST 3: Biometric-Only Authentication (No Password)")
    success3, confidence3, message3 = authenticator.authenticate_user(
        username, None, sample_patterns
    )
    
    print(f"  Result: {'✅ SUCCESS' if success3 else '❌ FAILED'}")
    print(f"  Confidence Score: {confidence3:.3f}")
    print(f"  Message: {message3}")
    
    return {
        'test1': (success, confidence),
        'test2': (success2, confidence2),
        'test3': (success3, confidence3)
    }

def demonstrate_database_analysis(authenticator):
    """Demonstrate database analysis and statistics"""
    print("\n" + "="*80)
    print("DATABASE ANALYSIS & STATISTICS")
    print("="*80)
    
    try:
        conn = sqlite3.connect(authenticator.db_path)
        cursor = conn.cursor()
        
        # User statistics
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM behavioral_patterns")
        pattern_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM auth_logs")
        auth_count = cursor.fetchone()[0]
        
        print(f"\n📊 DATABASE STATISTICS:")
        print(f"  Total Users: {user_count}")
        print(f"  Total Behavioral Patterns: {pattern_count}")
        print(f"  Total Authentication Attempts: {auth_count}")
        
        # Recent authentication attempts
        cursor.execute("""
            SELECT username, success, confidence_score, created_at 
            FROM auth_logs 
            ORDER BY created_at DESC 
            LIMIT 5
        """)
        
        recent_auths = cursor.fetchall()
        
        if recent_auths:
            print(f"\n🕒 RECENT AUTHENTICATION ATTEMPTS:")
            for username, success, confidence, timestamp in recent_auths:
                status = "✅ SUCCESS" if success else "❌ FAILED"
                print(f"  {username}: {status} (Confidence: {confidence:.3f}) at {timestamp}")
        
        # Feature analysis
        cursor.execute("""
            SELECT pattern_type, COUNT(*) 
            FROM behavioral_patterns 
            GROUP BY pattern_type
        """)
        
        pattern_types = cursor.fetchall()
        
        if pattern_types:
            print(f"\n🔍 PATTERN TYPE DISTRIBUTION:")
            for pattern_type, count in pattern_types:
                print(f"  {pattern_type}: {count} patterns")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database analysis error: {str(e)}")

def demonstrate_accuracy_metrics(authenticator):
    """Demonstrate accuracy and performance metrics"""
    print("\n" + "="*80)
    print("ACCURACY & PERFORMANCE METRICS")
    print("="*80)
    
    try:
        conn = sqlite3.connect(authenticator.db_path)
        cursor = conn.cursor()
        
        # Calculate basic metrics
        cursor.execute("SELECT COUNT(*) FROM auth_logs")
        total_auths = cursor.fetchone()[0]
        
        if total_auths == 0:
            print("\n📊 No authentication data available yet.")
            print("   Complete at least one authentication to see metrics.")
            conn.close()
            return
        
        cursor.execute("SELECT COUNT(*) FROM auth_logs WHERE success = 1")
        successful_auths = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(confidence_score) FROM auth_logs")
        avg_confidence = cursor.fetchone()[0] or 0
        
        # SQLite doesn't have STDDEV, so we calculate it manually
        cursor.execute("SELECT confidence_score FROM auth_logs")
        confidence_scores = [row[0] for row in cursor.fetchall() if row[0] is not None]
        confidence_std = np.std(confidence_scores) if confidence_scores else 0
        
        # Success rate
        success_rate = (successful_auths / total_auths) * 100 if total_auths > 0 else 0
        
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"  Total Authentication Attempts: {total_auths}")
        print(f"  Successful Authentications: {successful_auths}")
        print(f"  Failed Authentications: {total_auths - successful_auths}")
        print(f"  Success Rate: {success_rate:.2f}%")
        print(f"  Average Confidence Score: {avg_confidence:.3f}")
        print(f"  Confidence Standard Deviation: {confidence_std:.3f}")
        
        # Threshold analysis
        threshold = 0.65
        cursor.execute("SELECT COUNT(*) FROM auth_logs WHERE confidence_score >= ?", (threshold,))
        above_threshold = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM auth_logs WHERE confidence_score < ?", (threshold,))
        below_threshold = cursor.fetchone()[0]
        
        print(f"\n🎯 THRESHOLD ANALYSIS (Threshold: {threshold}):")
        print(f"  Above Threshold: {above_threshold} attempts")
        print(f"  Below Threshold: {below_threshold} attempts")
        
        # False positive/negative analysis
        cursor.execute("""
            SELECT COUNT(*) FROM auth_logs 
            WHERE confidence_score >= ? AND success = 0
        """, (threshold,))
        false_positives = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM auth_logs 
            WHERE confidence_score < ? AND success = 1
        """, (threshold,))
        false_negatives = cursor.fetchone()[0]
        
        print(f"\n🚨 ERROR ANALYSIS:")
        print(f"  False Positives (FAR): {false_positives} ({false_positives/total_auths*100:.2f}%)")
        print(f"  False Negatives (FRR): {false_negatives} ({false_negatives/total_auths*100:.2f}%)")
        
        # Calculate EER approximation
        far_rate = false_positives / total_auths if total_auths > 0 else 0
        frr_rate = false_negatives / total_auths if total_auths > 0 else 0
        eer = (far_rate + frr_rate) / 2
        
        print(f"  Equal Error Rate (EER): {eer:.3f}")
        
        # Performance assessment
        if eer < 0.05:
            performance = "EXCELLENT"
        elif eer < 0.10:
            performance = "GOOD"
        elif eer < 0.15:
            performance = "FAIR"
        else:
            performance = "NEEDS IMPROVEMENT"
        
        print(f"\n🏆 OVERALL PERFORMANCE: {performance}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Accuracy metrics error: {str(e)}")

def main():
    """Main demonstration function"""
    print("🚀 BEHAVIORAL BIOMETRIC AUTHENTICATION SYSTEM - BACKEND DEMO")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize the authenticator
        print("\n🔧 Initializing Biometric Authenticator...")
        authenticator = BiometricAuthenticator()
        print("✅ Authenticator initialized successfully!")
        
        # Demonstrate feature extraction
        features, feature_names, sample_patterns = demonstrate_feature_extraction(authenticator)
        
        # Demonstrate user registration
        username = demonstrate_user_registration(authenticator, features, feature_names, sample_patterns)
        
        if username:
            # Demonstrate authentication
            auth_results = demonstrate_authentication(authenticator, username, sample_patterns)
            
            # Demonstrate database analysis
            demonstrate_database_analysis(authenticator)
            
            # Demonstrate accuracy metrics
            demonstrate_accuracy_metrics(authenticator)
            
            print("\n" + "="*80)
            print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
            print("="*80)
            
            print(f"\n📋 SUMMARY:")
            print(f"  - Feature extraction: {len(features)} features extracted")
            print(f"  - User registration: {'✅ Success' if username else '❌ Failed'}")
            print(f"  - Authentication tests: 3 tests completed")
            print(f"  - Database analysis: Completed")
            print(f"  - Accuracy metrics: Calculated")
            
        else:
            print("\n❌ Cannot proceed with authentication demo - user registration failed")
            
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
