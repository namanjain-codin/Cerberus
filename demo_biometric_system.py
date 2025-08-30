
"""
Behavioral Biometric Authentication System - Demo and Test Script
Complete demonstration of the system capabilities
"""

import json
import random
import time
import os
from datetime import datetime
from backend_biometric_auth import BiometricAuthenticator
from accuracy_checker import BiometricAccuracyChecker

class BiometricDemo:
    def __init__(self):
        self.authenticator = BiometricAuthenticator()
        self.accuracy_checker = BiometricAccuracyChecker()

    def generate_sample_patterns(self, user_id=1, variation=0.1):
        """Generate sample behavioral patterns for testing"""
        base_patterns = {
            'device': {
                'userAgent': f'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0.{4400 + user_id}.0 Safari/537.36',
                'language': 'en-US',
                'platform': 'Win32',
                'screenResolution': '1920x1080',
                'colorDepth': 24,
                'timezone': 'America/New_York',
                'hardwareConcurrency': 8,
                'deviceMemory': 8,
                'cookieEnabled': True,
                'doNotTrack': None,
                'plugins': ['Chrome PDF Plugin', 'Native Client'],
                'canvasFingerprint': f'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/{user_id}AAAAASUVORK5CYII=',
                'timestamp': int(time.time() * 1000)
            },
            'keystroke': {
                'avgDwellTimes': {
                    'a': 120 + random.gauss(0, 20 * variation),
                    'e': 115 + random.gauss(0, 18 * variation),
                    'i': 125 + random.gauss(0, 22 * variation),
                    'o': 130 + random.gauss(0, 25 * variation),
                    'u': 135 + random.gauss(0, 20 * variation),
                    'n': 110 + random.gauss(0, 15 * variation),
                    't': 105 + random.gauss(0, 18 * variation),
                    'r': 140 + random.gauss(0, 30 * variation),
                    's': 118 + random.gauss(0, 20 * variation),
                    'l': 145 + random.gauss(0, 25 * variation)
                },
                'avgFlightTimes': {
                    'th': 80 + random.gauss(0, 15 * variation),
                    'he': 75 + random.gauss(0, 12 * variation),
                    'in': 85 + random.gauss(0, 18 * variation),
                    'er': 90 + random.gauss(0, 20 * variation),
                    'an': 78 + random.gauss(0, 15 * variation)
                },
                'typingSpeed': 65 + random.gauss(0, 10 * variation)
            },
            'mouse': {
                'avgMouseSpeed': 120 + random.gauss(0, 30 * variation),
                'avgMouseAcceleration': 15 + random.gauss(0, 5 * variation),
                'totalMouseDistance': 2500 + random.gauss(0, 500 * variation),
                'mouseMovements': 150 + int(random.gauss(0, 30 * variation)),
                'clickCount': 8 + int(random.gauss(0, 3 * variation))
            },
            'touch': {
                'swipeCount': 12 + int(random.gauss(0, 4 * variation)),
                'avgSwipeLength': 200 + random.gauss(0, 50 * variation),
                'avgSwipeDuration': 350 + random.gauss(0, 80 * variation),
                'avgSwipePressure': 0.6 + random.gauss(0, 0.1 * variation)
            }
        }

        return base_patterns

    def demo_user_registration(self):
        """Demonstrate user registration process"""
        print("\n" + "="*60)
        print("DEMONSTRATION: USER REGISTRATION")
        print("="*60)

        # Create sample users
        users = [
            {'username': 'alice_smith', 'password': 'SecurePass123!'},
            {'username': 'bob_jones', 'password': 'MyPassword456@'},
            {'username': 'charlie_brown', 'password': 'StrongAuth789#'}
        ]

        for i, user in enumerate(users):
            print(f"\nRegistering user: {user['username']}")

            # Generate unique patterns for this user
            patterns = self.generate_sample_patterns(user_id=i+1, variation=0.05)

            success, message = self.authenticator.register_user(
                user['username'], 
                user['password'], 
                patterns
            )

            if success:
                print(f"✅ Registration successful: {message}")
            else:
                print(f"❌ Registration failed: {message}")

            # Small delay to simulate real timing
            time.sleep(0.5)

        print(f"\n📊 Total users registered: {len(users)}")

    def demo_authentication_attempts(self):
        """Demonstrate authentication attempts with various scenarios"""
        print("\n" + "="*60)
        print("DEMONSTRATION: AUTHENTICATION ATTEMPTS")
        print("="*60)

        test_scenarios = [
            # Legitimate user attempts
            {
                'username': 'alice_smith',
                'password': 'SecurePass123!',
                'user_id': 1,
                'variation': 0.08,
                'description': 'Legitimate user - normal session'
            },
            {
                'username': 'alice_smith',
                'password': 'SecurePass123!', 
                'user_id': 1,
                'variation': 0.15,
                'description': 'Legitimate user - tired/different conditions'
            },
            {
                'username': 'bob_jones',
                'password': 'MyPassword456@',
                'user_id': 2,
                'variation': 0.06,
                'description': 'Legitimate user - consistent patterns'
            },
            # Fraudulent attempts
            {
                'username': 'alice_smith',
                'password': 'SecurePass123!',
                'user_id': 99,  # Different user ID = different patterns
                'variation': 0.3,
                'description': 'FRAUD: Someone else using Alice\'s credentials'
            },
            {
                'username': 'charlie_brown',
                'password': 'StrongAuth789#',
                'user_id': 88,  # Different user ID = different patterns
                'variation': 0.4,
                'description': 'FRAUD: Attacker with stolen credentials'
            }
        ]

        for i, scenario in enumerate(test_scenarios):
            print(f"\n🔍 Test {i+1}: {scenario['description']}")
            print(f"Username: {scenario['username']}")

            # Generate patterns based on scenario
            patterns = self.generate_sample_patterns(
                user_id=scenario['user_id'],
                variation=scenario['variation']
            )

            success, confidence, message = self.authenticator.authenticate_user(
                scenario['username'],
                scenario['password'],
                patterns
            )

            status_emoji = "✅" if success else "❌"
            fraud_indicator = "🚨 POTENTIAL FRAUD" if "FRAUD" in scenario['description'] and success else ""

            print(f"{status_emoji} Authentication: {message}")
            print(f"📊 Confidence Score: {confidence:.3f}")
            print(f"🎯 Expected Result: {'SUCCESS' if 'FRAUD' not in scenario['description'] else 'REJECTION'}")
            if fraud_indicator:
                print(fraud_indicator)

            time.sleep(0.5)

    def demo_accuracy_analysis(self):
        """Demonstrate accuracy analysis capabilities"""
        print("\n" + "="*60)
        print("DEMONSTRATION: ACCURACY ANALYSIS")
        print("="*60)

        # Run the accuracy checker
        metrics = self.accuracy_checker.generate_accuracy_report()

        # Additional robustness testing
        robustness_score = self.accuracy_checker.test_system_robustness()

        print(f"\n📈 SUMMARY:")
        print(f"System Robustness Score: {robustness_score}%")

        if robustness_score >= 75:
            print("🎉 System Performance: EXCELLENT")
        elif robustness_score >= 60:
            print("👍 System Performance: GOOD")
        elif robustness_score >= 40:
            print("⚠️  System Performance: NEEDS IMPROVEMENT")
        else:
            print("❌ System Performance: POOR")

    def demo_feature_extraction(self):
        """Demonstrate feature extraction process"""
        print("\n" + "="*60)
        print("DEMONSTRATION: FEATURE EXTRACTION")
        print("="*60)

        # Generate sample patterns
        patterns = self.generate_sample_patterns(user_id=1)

        # Extract features
        features, feature_names = self.authenticator.extract_features(patterns)

        print(f"📊 Extracted {len(features)} behavioral features:")
        print()

        for i, (name, value) in enumerate(zip(feature_names, features)):
            print(f"{i+1:2d}. {name:25s}: {value:8.2f}")

        print(f"\n🔢 Feature vector shape: {features.shape}")
        print(f"📈 Feature statistics:")
        print(f"   Mean: {features.mean():.2f}")
        print(f"   Std:  {features.std():.2f}")
        print(f"   Min:  {features.min():.2f}")
        print(f"   Max:  {features.max():.2f}")

    def run_complete_demo(self):
        """Run the complete demonstration"""
        print("🚀 BEHAVIORAL BIOMETRIC AUTHENTICATION SYSTEM DEMO")
        print("=" * 60)
        print(f"Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # 1. Feature extraction demo
            self.demo_feature_extraction()

            # 2. User registration demo
            self.demo_user_registration()

            # 3. Authentication attempts demo
            self.demo_authentication_attempts()

            # 4. Accuracy analysis demo
            self.demo_accuracy_analysis()

            print("\n" + "="*60)
            print("✅ DEMO COMPLETED SUCCESSFULLY")
            print("="*60)
            print("\nKey Points Demonstrated:")
            print("• Multi-modal biometric pattern capture")
            print("• Secure user registration with behavioral profiling")
            print("• Real-time authentication with confidence scoring")
            print("• Fraud detection capabilities")
            print("• Comprehensive accuracy assessment")
            print("• Robust feature extraction pipeline")

        except Exception as e:
            print(f"\n❌ Demo failed with error: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """Main execution function"""
    demo = BiometricDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main()
