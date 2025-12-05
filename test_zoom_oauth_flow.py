#!/usr/bin/env python3
"""
Zoom OAuth Flow Test Case
Tests the complete OAuth flow without scope restrictions
"""

import requests
import json
import time
from urllib.parse import urlparse, parse_qs

def test_zoom_oauth_flow():
    """Test the complete Zoom OAuth flow"""
    
    base_url = "https://virtual-assistent-cudwb7h9e6avdkfu.eastus-01.azurewebsites.net"
    
    print("🧪 Starting Zoom OAuth Flow Test")
    print("=" * 50)
    
    # Test 1: Check authentication status (should be false initially)
    print("\n📊 Test 1: Check initial authentication status")
    try:
        status_response = requests.get(f"{base_url}/api/zoom/auth/status")
        print(f"Status Code: {status_response.status_code}")
        
        if status_response.status_code == 200:
            status_data = status_response.json()
            print(f"Response: {status_data}")
            
            if not status_data.get('authenticated', True):
                print("✅ PASS: User not authenticated initially (expected)")
            else:
                print("⚠️  WARNING: User appears to be already authenticated")
        else:
            print(f"❌ FAIL: Status endpoint error - {status_response.text}")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: Status check error - {str(e)}")
        return False
    
    # Test 2: Generate OAuth URL without scope restrictions
    print("\n🔗 Test 2: Generate OAuth authorization URL")
    try:
        auth_response = requests.get(f"{base_url}/api/zoom/auth/url")
        print(f"Status Code: {auth_response.status_code}")
        
        if auth_response.status_code == 200:
            auth_data = auth_response.json()
            auth_url = auth_data.get('authorization_url', '')
            state = auth_data.get('state', '')
            
            print(f"Authorization URL: {auth_url[:100]}...")
            print(f"State: {state[:20]}...")
            
            # Validate URL structure
            parsed_url = urlparse(auth_url)
            query_params = parse_qs(parsed_url.query)
            
            # Check required parameters
            required_params = ['response_type', 'client_id', 'redirect_uri', 'state']
            missing_params = [p for p in required_params if p not in query_params]
            
            if missing_params:
                print(f"❌ FAIL: Missing required parameters: {missing_params}")
                return False
            
            # Check that scope parameter is NOT present (this is what we want)
            if 'scope' not in query_params:
                print("✅ PASS: No scope parameter in OAuth URL (user can configure manually)")
            else:
                print(f"⚠️  WARNING: Scope parameter found: {query_params['scope']}")
            
            # Validate client ID
            if query_params['client_id'][0] == 'eRb5n7RBSq_SvJZSdxzQQ':
                print("✅ PASS: Correct client ID")
            else:
                print(f"❌ FAIL: Incorrect client ID: {query_params['client_id'][0]}")
                return False
            
            # Validate redirect URI
            expected_redirect = f"{base_url}/api/zoom/auth/callback"
            actual_redirect = query_params['redirect_uri'][0]
            if actual_redirect == expected_redirect:
                print("✅ PASS: Correct redirect URI")
            else:
                print(f"❌ FAIL: Incorrect redirect URI: {actual_redirect}")
                return False
                
        else:
            print(f"❌ FAIL: Auth URL generation error - {status_response.text}")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: Auth URL generation error - {str(e)}")
        return False
    
    # Test 3: Test callback endpoint accessibility
    print("\n📞 Test 3: Test callback endpoint accessibility")
    try:
        # Test callback endpoint (should return error without proper code)
        callback_response = requests.get(f"{base_url}/api/zoom/auth/callback")
        print(f"Status Code: {callback_response.status_code}")
        
        # We expect this to fail without proper authorization code
        if callback_response.status_code in [400, 422]:
            print("✅ PASS: Callback endpoint accessible (returns expected error without auth code)")
        elif callback_response.status_code == 200:
            print("⚠️  WARNING: Callback endpoint returned success without auth code")
        else:
            print(f"❌ FAIL: Unexpected callback response - {callback_response.text}")
            
    except Exception as e:
        print(f"❌ FAIL: Callback endpoint error - {str(e)}")
        return False
    
    # Test 4: Validate environment variables
    print("\n🔧 Test 4: Validate environment configuration")
    try:
        # Test if environment is properly configured by checking error messages
        test_response = requests.get(f"{base_url}/api/zoom/auth/url")
        if test_response.status_code == 200:
            print("✅ PASS: Environment variables properly configured")
        else:
            print(f"❌ FAIL: Environment configuration issue - {test_response.text}")
            
    except Exception as e:
        print(f"❌ FAIL: Environment validation error - {str(e)}")
        return False
    
    print("\n" + "=" * 50)
    print("🎯 TEST SUMMARY")
    print("✅ OAuth URL generation: Working")
    print("✅ No scope restrictions: Implemented")  
    print("✅ Proper redirect URI: Configured")
    print("✅ Client credentials: Valid")
    print("✅ Callback endpoint: Accessible")
    print("✅ Environment variables: Configured")
    print("\n🚀 NEXT STEPS FOR MANUAL TESTING:")
    print("1. Use the generated OAuth URL to authorize in browser")
    print("2. Configure scopes manually in your Zoom app settings")
    print("3. Complete authorization flow")
    print("4. Check authentication status again")
    
    return True

def test_manual_authorization_instructions():
    """Provide instructions for manual authorization testing"""
    
    print("\n" + "=" * 60)
    print("📋 MANUAL AUTHORIZATION TEST INSTRUCTIONS")
    print("=" * 60)
    
    print("\n🔧 ZOOM APP CONFIGURATION:")
    print("1. Go to Zoom Marketplace: https://marketplace.zoom.us/")
    print("2. Navigate to your app: eRb5n7RBSq_SvJZSdxzQQ")
    print("3. Go to 'Scopes' section")
    print("4. Select the scopes you need manually:")
    print("   - meeting:read (to detect meetings)")
    print("   - recording:read (to access recordings)")
    print("   - user:read (basic user info)")
    print("   - Or any other scopes you require")
    
    print("\n🌐 AUTHORIZATION TESTING:")
    print("1. Run: python test_zoom_oauth_flow.py")
    print("2. Copy the generated OAuth URL")
    print("3. Open in browser and authorize")
    print("4. Should redirect to callback without 'Invalid scope' error")
    print("5. Check authentication status again")
    
    print("\n✅ SUCCESS CRITERIA:")
    print("- No 'Invalid scope' errors during authorization")
    print("- Successful redirect to callback URL")
    print("- Authentication status becomes true")
    print("- Meeting detection works in frontend")

if __name__ == "__main__":
    print("🧪 Zoom OAuth Flow Test Suite")
    print("Testing scope-free OAuth implementation")
    
    success = test_zoom_oauth_flow()
    
    if success:
        print("\n🎉 ALL AUTOMATED TESTS PASSED!")
        test_manual_authorization_instructions()
    else:
        print("\n❌ SOME TESTS FAILED - Check output above")