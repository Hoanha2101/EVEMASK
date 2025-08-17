# HƯỚNG DẪN LẤY REFRESH TOKEN BẰNG GOOGLE OAUTH 2.0 PLAYGROUND

## ⚠️ QUAN TRỌNG: Khắc phục lỗi "Google verification process"

Nếu gặp lỗi "evemask has not completed the Google verification process", làm theo:

### Bước A: Thêm Test Users
1. Vào https://console.cloud.google.com/
2. Chọn project "evemask"
3. APIs & Services → **OAuth consent screen**
4. Cuộn xuống "Test users" 
5. Click **"ADD USERS"**
6. Thêm email: `evemask.ai@gmail.com`
7. Save

### Bước B: Hoặc tạo OAuth Client mới
1. APIs & Services → Credentials
2. **+ CREATE CREDENTIALS** → **OAuth 2.0 Client ID**
3. Application type: **Web application**
4. Name: **Gmail API Playground**
5. Authorized redirect URIs: `https://developers.google.com/oauthplayground`
6. **CREATE** và copy Client ID + Client Secret mới

## Cách 1: Sử dụng Google OAuth 2.0 Playground (Sau khi sửa lỗi trên)

1. Vào https://developers.google.com/oauthplayground/

2. Click vào Settings (⚙️) ở góc phải

3. Tick ✅ "Use your own OAuth credentials"

4. Nhập thông tin từ credentials.json:
   - OAuth Client ID: 725317207385-k54i22mi3igk436kddg36cesjk5giano.apps.googleusercontent.com
   - OAuth Client Secret: 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

5. Ở bước 1, tìm và chọn "Gmail API v1" → "https://www.googleapis.com/auth/gmail.send"
   ⚠️  **QUAN TRỌNG**: Chỉ chọn scope này, không chọn thêm scope khác
   ⚠️  **Đảm bảo**: Tick vào chính xác "https://www.googleapis.com/auth/gmail.send"

6. Click "Authorize APIs"

7. Đăng nhập với Gmail account của bạn (evemask.ai@gmail.com)

8. Ở bước 2, click "Exchange authorization code for tokens"

9. Copy "Refresh token" từ kết quả

## Cách 2: Cập nhật Google Console trước (BẮT BUỘC)

1. Vào https://console.cloud.google.com/
2. Chọn project "evemask" 
3. APIs & Services → Credentials
4. Click OAuth 2.0 Client ID (ID: 725317207385-k54i22mi3igk436kddg36cesjk5giano.apps.googleusercontent.com)
5. Thêm vào "Authorized redirect URIs":
   - https://developers.google.com/oauthplayground ← **QUAN TRỌNG NHẤT**
   - http://localhost:8080
   - http://localhost:8000
   - http://127.0.0.1:8080

6. Click **Save** và đợi vài phút để Google cập nhật

## Thông tin cần cho HuggingFace Secrets:

```
GOOGLE_CLIENT_ID=''
GOOGLE_CLIENT_SECRET=''
GOOGLE_REFRESH_TOKEN=[Token từ OAuth Playground]
SENDER_EMAIL=evemask.ai@gmail.com
SENDER_NAME=EVEMASK Team
```

## Troubleshooting:

- Nếu lỗi "redirect_uri_mismatch": Thêm https://developers.google.com/oauthplayground vào redirect URIs
- Nếu lỗi "access_denied": Đảm bảo Gmail API đã được enable trong Google Console
- Nếu lỗi "invalid_scope": Kiểm tra scope là "https://www.googleapis.com/auth/gmail.send"
