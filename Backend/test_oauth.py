#!/usr/bin/env python3
"""
Script đơn giản để test OAuth và lấy refresh token
"""

import json
import os
from google_auth_oauthlib.flow import InstalledAppFlow

# Scopes cho Gmail API
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def test_oauth():
    """Test OAuth với nhiều port khác nhau"""
    
    print("🚀 Testing OAuth Flow...")
    
    # Thử các port khác nhau
    ports_to_try = [8080, 8000, 3000, 5000]
    
    for port in ports_to_try:
        try:
            print(f"\n🔄 Trying port {port}...")
            
            # Khởi tạo OAuth flow với port cụ thể
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', 
                SCOPES,
                redirect_uri=f'http://localhost:{port}'
            )
            
            print(f"📱 Opening browser for authentication on port {port}...")
            print(f"⚠️  Make sure http://localhost:{port} is in your Google Console redirect URIs")
            
            # Chạy authentication
            creds = flow.run_local_server(port=port, open_browser=True)
            
            # Lưu credentials
            token_data = json.loads(creds.to_json())
            
            with open('token.json', 'w') as f:
                json.dump(token_data, f, indent=2)
            
            print(f"✅ Authentication successful on port {port}!")
            print("\n" + "="*60)
            print("📋 SECRETS FOR HUGGINGFACE:")
            print("="*60)
            print(f"GOOGLE_CLIENT_ID: {token_data.get('client_id')}")
            print(f"GOOGLE_CLIENT_SECRET: {token_data.get('client_secret')}")
            print(f"GOOGLE_REFRESH_TOKEN: {token_data.get('refresh_token')}")
            print("="*60)
            
            return True
            
        except Exception as e:
            print(f"❌ Port {port} failed: {str(e)}")
            continue
    
    print("❌ All ports failed. Please check your Google Console settings.")
    return False

if __name__ == "__main__":
    print("🔐 EVEMASK OAuth Token Generator")
    print("="*50)
    
    # Kiểm tra file credentials.json
    if not os.path.exists('credentials.json'):
        print("❌ credentials.json not found!")
        exit(1)
    
    # Chạy OAuth test
    success = test_oauth()
    
    if success:
        print("\n🎉 Success! Use the tokens above in HuggingFace secrets.")
    else:
        print("\n💥 Failed. Please check Google Console settings.")
