#!/usr/bin/env python3
"""
Script để lấy refresh token cho Gmail API
Chạy script này một lần để lấy refresh token, sau đó sử dụng token đó trong production
"""

import json
import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Scopes cần thiết cho Gmail API
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def get_refresh_token():
    """Lấy refresh token từ Google OAuth"""
    
    # Đọc credentials từ file
    if not os.path.exists('credentials.json'):
        print("❌ Không tìm thấy file credentials.json")
        return None
    
    try:
        # Khởi tạo OAuth flow
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json', SCOPES)
        
        # Chạy local server để xác thực
        print("🔄 Đang mở browser để xác thực...")
        print("⚠️  Đảm bảo bạn đã thêm http://localhost:8080 vào redirect URIs trong Google Console")
        
        # Sử dụng port 8080 như trong redirect_uris
        creds = flow.run_local_server(port=8080)
        
        # Lưu credentials bao gồm refresh token
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
        
        print("✅ Đã lưu credentials vào token.json")
        
        # Hiển thị thông tin cần thiết
        print("\n" + "="*60)
        print("📋 THÔNG TIN CẦN THIẾT CHO HUGGINGFACE SECRETS:")
        print("="*60)
        
        creds_data = json.loads(creds.to_json())
        
        print(f"GOOGLE_CLIENT_ID: {creds_data.get('client_id', 'N/A')}")
        print(f"GOOGLE_CLIENT_SECRET: {creds_data.get('client_secret', 'N/A')}")
        print(f"GOOGLE_REFRESH_TOKEN: {creds_data.get('refresh_token', 'N/A')}")
        
        # Test Gmail API
        print("\n🧪 Testing Gmail API...")
        service = build('gmail', 'v1', credentials=creds)
        profile = service.users().getProfile(userId='me').execute()
        sender_email = profile.get('emailAddress')
        
        print(f"SENDER_EMAIL: {sender_email}")
        print(f"✅ Gmail API test thành công!")
        
        print("\n" + "="*60)
        print("🔧 HƯỚNG DẪN:")
        print("1. Copy các giá trị trên")
        print("2. Vào HuggingFace Space Settings > Repository secrets")
        print("3. Thêm từng secret với tên và giá trị tương ứng")
        print("4. Không commit file token.json lên git!")
        print("="*60)
        
        return creds_data
        
    except Exception as e:
        print(f"❌ Lỗi khi lấy refresh token: {str(e)}")
        return None

def test_existing_token():
    """Test refresh token hiện tại nếu có"""
    if os.path.exists('token.json'):
        try:
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            
            # Refresh token nếu cần
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            
            # Test Gmail API
            service = build('gmail', 'v1', credentials=creds)
            profile = service.users().getProfile(userId='me').execute()
            
            print(f"✅ Token hiện tại vẫn hoạt động tốt!")
            print(f"📧 Email: {profile.get('emailAddress')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Token hiện tại không hoạt động: {str(e)}")
            return False
    
    return False

if __name__ == "__main__":
    print("🚀 EVEMASK - Gmail API Token Generator")
    print("="*50)
    
    # Kiểm tra token hiện tại trước
    if test_existing_token():
        choice = input("\n❓ Token hiện tại vẫn hoạt động. Bạn có muốn tạo mới? (y/N): ")
        if choice.lower() != 'y':
            print("👍 Sử dụng token hiện tại.")
            exit(0)
    
    # Lấy refresh token mới
    result = get_refresh_token()
    
    if result:
        print("\n🎉 Hoàn thành! Bạn có thể sử dụng các thông tin trên để cấu hình HuggingFace secrets.")
    else:
        print("\n💥 Có lỗi xảy ra. Vui lòng kiểm tra lại.")
