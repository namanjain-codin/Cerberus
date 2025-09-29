#!/usr/bin/env python3
"""
Test script for Enhanced Biometric Authentication System
Verifies that the enhanced security features work correctly
"""

import requests
import json
import time
import random
import numpy as np
from datetime import datetime

class EnhancedSystemTester:
    def __init__(self):
        self.main_server = "http://localhost:5000"
        self.enhanced_server = "http://localhost:5001"
        self.test_results = []
        
    def test_server_health(self):
        """Test if both servers are running"""
        print("🔍 Testing server health...")
        
        try:
            # Test main server
            response = requests.get(f"{self.main_server}/api/health", timeout=5)
            if response.status_code == 200:
                print("✅ Main server (port 5000) is running")
                main_healthy = True
            else:
                print("❌ Main server health check failed")
                main_healthy = False
        except requests.RequestException:
            print("❌ Main server is not responding")
            main_healthy = False
        
        try:
            # Test enhanced server
            response = requests.get(f"{self.enhanced_server}/enhanced-auth", timeout=5)
            if response.status_code == 200:
                print("✅ Enhanced server (port 5001) is running")
                enhanced_healthy = True
            else:
                print("❌ Enhanced server health check failed")
                enhanced_healthy = False
        except requests.RequestException:
            print("❌ Enhanced server is not responding")
            enhanced_healthy = False
        
        return main_healthy and enhanced_healthy
    
    def generate_test_patterns(self, user_type="legitimate", variation=0.1):
        """Generate test behavioral patterns"""
        base_patterns = {
            'device': {
                'userAgent': f'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0.{4400 + random.randint(1, 100)}.0 Safari/537.36',
                'platform': 'Win32',
                'screenResolution': '1920x1080',
                'colorDepth': 24,
                'hardwareConcurrency': 8,
                'canvasFingerprint': f'fingerprint_{random.randint(1000, 9999)}'
            },
            'keystroke': {
                'avgDwellTimes': {
                    'a': 120 + random.gauss(0, 20 * variation),
                    'e': 115 + random.gauss(0, 18 * variation),
                    'i': 125 + random.gauss(0, 22 * variation),
                    'o': 130 + random.gauss(0, 25 * variation),
                    'u': 135 + random.gauss(0, 20 * variation)
                },
                'typingSpeed': 65 + random.gauss(0, 10 * variation)
            },
            'mouse': {
                'avgMouseSpeed': 120 + random.gauss(0, 30 * variation),
                'avgMouseAcceleration': 15 + random.gauss(0, 5 * variation),
                'totalMouseDistance': 2500 + random.gauss(0, 500 * variation)
            }
        }
        
        return base_patterns
    
    def test_user_registration(self):
        """Test user registration with behavioral patterns"""
        print("\n📝 Testing user registration...")
        
        test_user = {
            'username': f'test_user_{int(time.time())}',
            'password': 'TestPassword123!',
            'patterns': self.generate_test_patterns()
        }
        
        try:
            response = requests.post(
                f"{self.main_server}/api/register",
                json=test_user,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print(f"✅ User registration successful: {test_user['username']}")
                    return test_user['username']
                else:
                    print(f"❌ Registration failed: {result.get('error')}")
                    return None
            else:
                print(f"❌ Registration request failed: {response.status_code}")
                return None
                
        except requests.RequestException as e:
            print(f"❌ Registration error: {str(e)}")
            return None
    
    def test_legitimate_authentication(self, username):
        """Test legitimate user authentication"""
        print(f"\n🔐 Testing legitimate authentication for {username}...")
        
        auth_data = {
            'username': username,
            'password': 'TestPassword123!',
            'patterns': self.generate_test_patterns(variation=0.05)  # Low variation = legitimate user
        }
        
        try:
            response = requests.post(
                f"{self.main_server}/api/login",
                json=auth_data,
                timeout=10
            )
            
            result = response.json()
            
            if response.status_code == 200 and result.get('success'):
                print(f"✅ Legitimate authentication successful")
                print(f"   Confidence score: {result.get('confidence_score', 0):.3f}")
                return True
            elif result.get('requires_enhanced_verification'):
                print(f"⚠️  Enhanced verification required (confidence: {result.get('confidence_score', 0):.3f})")
                print(f"   Security flags: {result.get('security_flags', [])}")
                return "enhanced_required"
            else:
                print(f"❌ Authentication failed: {result.get('error')}")
                return False
                
        except requests.RequestException as e:
            print(f"❌ Authentication error: {str(e)}")
            return False
    
    def test_suspicious_authentication(self, username):
        """Test suspicious user authentication (should trigger enhanced verification)"""
        print(f"\n🚨 Testing suspicious authentication for {username}...")
        
        auth_data = {
            'username': username,
            'password': 'TestPassword123!',
            'patterns': self.generate_test_patterns(variation=0.3)  # High variation = suspicious
        }
        
        try:
            response = requests.post(
                f"{self.main_server}/api/login",
                json=auth_data,
                timeout=10
            )
            
            result = response.json()
            
            if result.get('requires_enhanced_verification'):
                print(f"✅ Suspicious behavior detected - Enhanced verification required")
                print(f"   Confidence score: {result.get('confidence_score', 0):.3f}")
                print(f"   Security flags: {result.get('security_flags', [])}")
                return True
            elif result.get('success'):
                print(f"⚠️  Suspicious behavior not detected (confidence: {result.get('confidence_score', 0):.3f})")
                return False
            else:
                print(f"❌ Authentication failed: {result.get('error')}")
                return False
                
        except requests.RequestException as e:
            print(f"❌ Authentication error: {str(e)}")
            return False
    
    def test_enhanced_verification_page(self):
        """Test if enhanced verification page is accessible"""
        print("\n🔍 Testing enhanced verification page...")
        
        try:
            response = requests.get(f"{self.enhanced_server}/enhanced-auth", timeout=5)
            
            if response.status_code == 200:
                print("✅ Enhanced verification page is accessible")
                return True
            else:
                print(f"❌ Enhanced verification page failed: {response.status_code}")
                return False
                
        except requests.RequestException as e:
            print(f"❌ Enhanced verification page error: {str(e)}")
            return False
    
    def test_network_info(self):
        """Test network information endpoint"""
        print("\n🌍 Testing network information...")
        
        try:
            response = requests.get(f"{self.main_server}/api/network-info", timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    network_info = result.get('network_info', {})
                    print(f"✅ Network info retrieved:")
                    print(f"   IP: {network_info.get('ip_address', 'Unknown')}")
                    print(f"   User Agent: {network_info.get('user_agent', 'Unknown')[:50]}...")
                    return True
                else:
                    print("❌ Network info request failed")
                    return False
            else:
                print(f"❌ Network info request failed: {response.status_code}")
                return False
                
        except requests.RequestException as e:
            print(f"❌ Network info error: {str(e)}")
            return False
    
    def run_comprehensive_test(self):
        """Run comprehensive test suite"""
        print("🧪 Enhanced Biometric Authentication System Test Suite")
        print("=" * 60)
        print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test 1: Server Health
        if not self.test_server_health():
            print("\n❌ Server health check failed. Please start the servers first:")
            print("   python start_enhanced_system.py")
            return False
        
        # Test 2: User Registration
        username = self.test_user_registration()
        if not username:
            print("\n❌ User registration failed. Cannot continue tests.")
            return False
        
        # Test 3: Legitimate Authentication
        legit_result = self.test_legitimate_authentication(username)
        
        # Test 4: Suspicious Authentication
        suspicious_result = self.test_suspicious_authentication(username)
        
        # Test 5: Enhanced Verification Page
        enhanced_page_result = self.test_enhanced_verification_page()
        
        # Test 6: Network Information
        network_result = self.test_network_info()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        tests = [
            ("Server Health", True),
            ("User Registration", username is not None),
            ("Legitimate Authentication", legit_result in [True, "enhanced_required"]),
            ("Suspicious Detection", suspicious_result),
            ("Enhanced Verification Page", enhanced_page_result),
            ("Network Information", network_result)
        ]
        
        passed = 0
        for test_name, result in tests:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:25s}: {status}")
            if result:
                passed += 1
        
        print(f"\n📈 Overall Score: {passed}/{len(tests)} tests passed")
        
        if passed == len(tests):
            print("🎉 All tests passed! Enhanced system is working correctly.")
        elif passed >= len(tests) * 0.8:
            print("👍 Most tests passed. System is mostly functional.")
        else:
            print("⚠️  Several tests failed. Please check the system configuration.")
        
        return passed >= len(tests) * 0.8

def main():
    """Main test execution"""
    tester = EnhancedSystemTester()
    success = tester.run_comprehensive_test()
    
    if success:
        print("\n✅ Enhanced Biometric Authentication System is ready!")
        print("\n📱 Access Points:")
        print("   Main System:     http://localhost:5000")
        print("   Enhanced Auth:   http://localhost:5001/enhanced-auth")
    else:
        print("\n❌ System test failed. Please check the configuration.")
        sys.exit(1)

if __name__ == "__main__":
    main()

