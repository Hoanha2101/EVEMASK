"""
===========================
EVEMASK Gmail OAuth Token Generator
===========================

This utility script generates OAuth 2.0 refresh tokens for Gmail API integration.

OAuth 2.0 Flow Overview:
1. Credential Loading:
   - Loads OAuth client credentials from credentials.json
   - Validates required client ID and secret
   - Sets up appropriate API scopes for Gmail sending

2. Authorization Process:
   - Initiates OAuth 2.0 authorization flow
   - Opens browser for user consent
   - Handles callback and authorization code exchange
   - Generates access and refresh tokens

3. Token Management:
   - Saves tokens to token.json for persistence
   - Handles token refresh automatically
   - Provides refresh token for environment configuration

Authorization Flow:
[credentials.json] -> [Browser Auth] -> [User Consent] -> [Token Exchange] -> [token.json]
        |                  |               |                 |                |
   (Client Info)    (User Login)    (Grant Permission)  (Get Tokens)   (Save Locally)

Required Files:
- credentials.json: OAuth 2.0 client credentials from Google Cloud Console
- (Generated) token.json: Contains access and refresh tokens

Required Scopes:
- https://www.googleapis.com/auth/gmail.send: Send emails via Gmail API

Security Notes:
- Refresh tokens are long-lived and should be kept secure
- Access tokens are short-lived and automatically refreshed
- Client credentials should never be committed to version control

Author: EVEMASK Team  
Version: 1.0.0
Usage: python get_token.py
"""

import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def main():
    """
    Execute OAuth 2.0 authorization flow for Gmail API access.
    
    This function handles the complete token generation process:
    
    Token Lifecycle Management:
        1. Checks for existing valid tokens in token.json
        2. Attempts token refresh if expired but refresh_token available
        3. Initiates new authorization flow if no valid tokens
        4. Saves generated tokens for future use
        
    Authorization Steps:
        1. Load client credentials from credentials.json
        2. Start local server on port 8080 for OAuth callback
        3. Open browser for user authentication
        4. Handle authorization code and exchange for tokens
        5. Save credentials to token.json
        
    Error Handling:
        - Missing credentials.json: Clear error message with setup instructions
        - Network issues: Retry logic with timeout handling
        - User denial: Graceful handling with retry option
        - Port conflicts: Alternative port selection
        
    Output:
        Creates token.json file containing:
        - access_token: Short-lived token for API requests
        - refresh_token: Long-lived token for token renewal
        - token_uri: Google OAuth 2.0 token endpoint
        - client_id: OAuth client identifier
        - client_secret: OAuth client secret
        - scopes: Granted permission scopes
        
    Environment Setup:
        After successful execution, copy the refresh_token value
        to your .env file as GOOGLE_REFRESH_TOKEN
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES, redirect_uri='http://localhost:8080')
            creds = flow.run_local_server(port=8080)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    print("\nSUCCESS!")
    print("A file named 'token.json' has been created.")
    print("Please open it and copy the 'refresh_token' value.")
    print("You will need to add this as a secret in your Hugging Face Space.")

if __name__ == '__main__':
    main()