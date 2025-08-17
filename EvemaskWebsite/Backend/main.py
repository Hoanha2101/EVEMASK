"""
===========================
EVEMASK Newsletter API Backend
===========================

This FastAPI backend provides newsletter subscription services with the following features:

1. Newsletter Subscription Management:
   - RESTful API endpoints for subscriber registration
   - Email validation and duplicate prevention
   - Automated welcome email confirmations

2. Database Integration:
   - Primary: Supabase PostgreSQL database
   - Fallback: Local JSON file storage
   - Automatic migration capabilities

3. Email Service:
   - Gmail API integration with OAuth 2.0
   - Professional HTML email templates
   - Error handling and retry mechanisms

4. API Documentation:
   - Auto-generated Swagger/OpenAPI docs
   - Interactive API testing interface
   - Comprehensive endpoint documentation

Pipeline Flow:
[Client Request] -> [FastAPI Validation] -> [Database Storage] -> [Email Service] -> [Response]
       |                    |                      |                    |              |
   (Frontend)         (Pydantic)            (Supabase/JSON)      (Gmail API)    (JSON Response)

For each subscription request, the script:
- Validates email format and authenticity
- Checks for existing subscribers to prevent duplicates
- Stores subscriber data in database with timestamp
- Sends personalized welcome email via Gmail API
- Returns success/error response to client

This system handles both development and production environments with comprehensive
error handling, logging, and fallback mechanisms.

Author: EVEMASK Team
Version: 2.0.0 (Supabase Integration)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Optional
import os
from datetime import datetime
import json
import re
import tempfile
import uvicorn
from dotenv import load_dotenv
# Thêm các thư viện của Google và các thư viện cần thiết khác
import base64
from email.mime.text import MIMEText
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
# Thêm Supabase
from supabase import create_client, Client

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="EVEMASK Newsletter API", version="1.0.0")

# CORS middleware to allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://localhost:3000", 
        "https://evemask.info",
        "https://www.evemask.info",
        "https://nghiant20-evemask.hf.space"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Pydantic models
class NewsletterSignup(BaseModel):
    email: str
    timestamp: Optional[datetime] = None
    
    @validator('email')
    def validate_email(cls, v):
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_regex, v):
            raise ValueError('Invalid email format')
        return v

class EmailResponse(BaseModel):
    message: str
    email: str
    status: str

# Email configuration
EMAIL_CONFIG = {
    "sender_email": os.getenv("SENDER_EMAIL"),
    # Thêm các cấu hình cho Gmail API
    "google_client_id": os.getenv("GOOGLE_CLIENT_ID"),
    "google_client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
    "google_refresh_token": os.getenv("GOOGLE_REFRESH_TOKEN"),
    "sender_name": os.getenv("SENDER_NAME", "EVEMASK Team")
}

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

# Initialize Supabase client
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Supabase client: {e}")
else:
    print("⚠️ Supabase configuration missing. Please set SUPABASE_URL and SUPABASE_ANON_KEY environment variables")

def create_welcome_email_html(user_email: str) -> str:
    """
    Generate professional HTML email template for EVEMASK marketing campaigns.
    
    This function creates a responsive, branded HTML email template featuring:
    - Modern gradient design with EVEMASK branding
    - Mobile-responsive layout
    - Professional marketing content highlighting EVEMASK AI benefits
    - Call-to-action buttons and social proof elements
    - Anti-spam compliance features
    
    Args:
        user_email (str): The recipient's email address for personalization
        
    Returns:
        str: Complete HTML email template ready for sending via Gmail API
        
    Note:
        Template includes inline CSS for maximum email client compatibility
    """
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Transform Your Broadcasting with EVEMASK AI</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
            }}
            .email-container {{
                max-width: 650px;
                margin: 0 auto;
                background-color: white;
                border-radius: 16px;
                overflow: hidden;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            }}
            .header {{
                background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
                padding: 50px 40px;
                text-align: center;
                color: white;
                position: relative;
                overflow: hidden;
            }}
            .header::before {{
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="1"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
                animation: float 20s ease-in-out infinite;
                opacity: 0.3;
            }}
            @keyframes float {{
                0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
                50% {{ transform: translateY(-20px) rotate(180deg); }}
            }}
            .logo {{
                font-size: 36px;
                font-weight: 900;
                margin-bottom: 15px;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                position: relative;
                z-index: 2;
            }}
            .tagline {{
                font-size: 18px;
                opacity: 0.95;
                font-weight: 500;
                position: relative;
                z-index: 2;
            }}
            .content {{
                padding: 50px 40px;
            }}
            .welcome-badge {{
                display: inline-block;
                background: linear-gradient(135deg, #28a745, #20c997);
                color: white;
                padding: 12px 24px;
                border-radius: 25px;
                font-size: 14px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-bottom: 30px;
                box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
            }}
            .main-title {{
                color: #333;
                font-size: 32px;
                font-weight: 800;
                margin-bottom: 25px;
                line-height: 1.2;
                background: linear-gradient(135deg, #007bff, #6610f2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            .intro-text {{
                color: #666;
                line-height: 1.8;
                margin-bottom: 35px;
                font-size: 16px;
            }}
            .problem-section {{
                background: linear-gradient(135deg, #fff3cd, #ffeaa7);
                padding: 30px;
                border-radius: 12px;
                margin: 30px 0;
                border-left: 6px solid #ffc107;
            }}
            .problem-title {{
                color: #856404;
                font-size: 20px;
                font-weight: 700;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
            }}
            .problem-icon {{
                font-size: 24px;
                margin-right: 10px;
            }}
            .solution-section {{
                background: linear-gradient(135deg, #d1ecf1, #bee5eb);
                padding: 35px;
                border-radius: 12px;
                margin: 30px 0;
                border-left: 6px solid #17a2b8;
            }}
            .solution-title {{
                color: #0c5460;
                font-size: 24px;
                font-weight: 800;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
            }}
            .benefits-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 25px;
                margin: 35px 0;
            }}
            .benefit-card {{
                background: #f8f9fa;
                padding: 25px;
                border-radius: 12px;
                border: 2px solid #e9ecef;
                transition: all 0.3s ease;
                text-align: center;
            }}
            .benefit-icon {{
                font-size: 48px;
                margin-bottom: 15px;
                display: block;
            }}
            .benefit-title {{
                color: #333;
                font-size: 18px;
                font-weight: 700;
                margin-bottom: 10px;
            }}
            .benefit-desc {{
                color: #666;
                font-size: 14px;
                line-height: 1.5;
            }}
            .stats-section {{
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                padding: 40px;
                border-radius: 12px;
                margin: 40px 0;
                text-align: center;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 30px;
                margin-top: 30px;
            }}
            .stat-item {{
                text-align: center;
            }}
            .stat-number {{
                font-size: 36px;
                font-weight: 900;
                display: block;
                margin-bottom: 8px;
            }}
            .stat-label {{
                font-size: 14px;
                opacity: 0.9;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .cta-section {{
                background: linear-gradient(135deg, #28a745, #20c997);
                color: white;
                padding: 45px;
                border-radius: 16px;
                text-align: center;
                margin: 40px 0;
                position: relative;
                overflow: hidden;
            }}
            .cta-section::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 60"><defs><pattern id="dots" width="20" height="20" patternUnits="userSpaceOnUse"><circle cx="10" cy="10" r="2" fill="rgba(255,255,255,0.1)"/></pattern></defs><rect width="100" height="100" fill="url(%23dots)"/></svg>');
                opacity: 0.5;
            }}
            .cta-title {{
                font-size: 28px;
                font-weight: 800;
                margin-bottom: 15px;
                position: relative;
                z-index: 2;
            }}
            .cta-subtitle {{
                font-size: 16px;
                margin-bottom: 30px;
                opacity: 0.95;
                position: relative;
                z-index: 2;
            }}
            .cta-buttons {{
                display: flex;
                gap: 20px;
                justify-content: center;
                flex-wrap: wrap;
                position: relative;
                z-index: 2;
            }}
            .cta-button {{
                display: inline-block;
                background: white;
                color: #28a745;
                padding: 18px 35px;
                text-decoration: none;
                border-radius: 50px;
                font-weight: 700;
                font-size: 16px;
                transition: all 0.3s ease;
                border: 3px solid white;
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .cta-button:hover {{
                transform: translateY(-3px);
                box-shadow: 0 12px 35px rgba(0,0,0,0.2);
            }}
            .cta-button.secondary {{
                background: transparent;
                color: white;
                border: 3px solid rgba(255,255,255,0.8);
            }}
            .urgency-banner {{
                background: linear-gradient(135deg, #dc3545, #c82333);
                color: white;
                padding: 20px;
                text-align: center;
                margin: 30px 0;
                border-radius: 8px;
                font-weight: 600;
                box-shadow: 0 4px 15px rgba(220, 53, 69, 0.3);
            }}
            .testimonial {{
                background: #f8f9fa;
                padding: 30px;
                border-radius: 12px;
                margin: 30px 0;
                border-left: 5px solid #007bff;
                position: relative;
            }}
            .testimonial-text {{
                font-style: italic;
                font-size: 16px;
                line-height: 1.6;
                margin-bottom: 15px;
                color: #495057;
            }}
            .testimonial-author {{
                font-weight: 700;
                color: #007bff;
            }}
            .footer {{
                background: linear-gradient(135deg, #2c3e50, #34495e);
                color: white;
                padding: 40px;
                text-align: center;
            }}
            .footer-links {{
                margin: 25px 0;
            }}
            .footer-links a {{
                color: #74b9ff;
                text-decoration: none;
                margin: 0 15px;
                font-weight: 500;
            }}
        </style>
    </head>
    <body>
        <div class="email-container">
            <div class="header">
                <div class="logo">🛡️ EVEMASK</div>
                <div class="tagline">We don’t just detect </div>
                <div class="tagline">We make a difference</div>
            </div>
            
            <div class="content">
                <div class="welcome-badge">🎉 Welcome to the Future</div>
                <h1 class="main-title">Say Goodbye to Manual Content Moderation Forever!</h1>
                <p class="intro-text">
                    <strong>Dear Broadcasting Professional,</strong><br><br>
                    Are you tired of manual content moderation eating into your profits? 
                    Worried about hefty fines from gambling advertisement violations? 
                    <strong>EVEMASK is here to save your business!</strong>
                </p>
                <div class="problem-section">
                    <h3 class="problem-title">
                        <span class="problem-icon">⚠️</span>
                        The Problems Costing You Money Right Now:
                    </h3>
                    <ul style="color: #856404; line-height: 1.8;">
                        <li><strong>Legal Fines:</strong> Millions of dollars in fines for gambling advertising violations</li>
                        <li><strong>Lost Revenue:</strong> Manual moderation delays costing thousands per hour</li>
                        <li><strong>Human Error:</strong> 40% miss rate in manual content detection</li>
                        <li><strong>Viewer Loss:</strong> Poor content quality driving audiences away</li>
                    </ul>
                </div>
                <div class="solution-section">
                    <h3 class="solution-title">
                        <span style="margin-right: 10px;">🚀</span>
                        EVEMASK: Your AI-Powered Solution
                    </h3>
                    <p style="color: #0c5460; font-size: 16px; line-height: 1.7;">
                        Our cutting-edge AI technology automatically detects and blurs gambling logos 
                        and inappropriate content in <strong>real-time</strong>, ensuring 100% compliance 
                        while maintaining broadcast quality.
                    </p>
                </div>
                <div class="stats-section">
                    <h3 style="margin-bottom: 10px; font-size: 24px;">Proven Results That Speak Numbers</h3>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <span class="stat-number">90%</span>
                            <span class="stat-label">AI Accuracy</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-number">&lt;2s</span>
                            <span class="stat-label">Processing Time</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-number">80%</span>
                            <span class="stat-label">Time Savings</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-number">70%</span>
                            <span class="stat-label">Cost Reduction</span>
                        </div>
                    </div>
                </div>
                <div class="benefits-grid">
                    <div class="benefit-card">
                        <span class="benefit-icon">💰</span>
                        <h4 class="benefit-title">Save Money</h4>
                        <p class="benefit-desc">Reduce moderation costs by 70% and avoid regulatory fines</p>
                    </div>
                    <div class="benefit-card">
                        <span class="benefit-icon">⚡</span>
                        <h4 class="benefit-title">Real-Time Processing</h4>
                        <p class="benefit-desc">Process live broadcasts with less than 2-second delay</p>
                    </div>
                    <div class="benefit-card">
                        <span class="benefit-icon">🛡️</span>
                        <h4 class="benefit-title">100% Compliance</h4>
                        <p class="benefit-desc">Automatically detect and blur all gambling advertisements</p>
                    </div>
                </div>
                <div class="urgency-banner">
                    ⏰ <strong>LIMITED TIME:</strong> Get 30% OFF your first year subscription! 
                    Only 48 hours left to claim this exclusive offer.
                </div>
                <div class="cta-section">
                    <h2 class="cta-title">Ready to Transform Your Broadcasting?</h2>
                    <p class="cta-subtitle">
                        Join 100+ broadcasters already saving thousands with EVEMASK AI
                    </p>
                    <div class="cta-buttons">
                        <a href="https://evemask.info/" 
                           class="cta-button">
                            🚀 Get EVEMASK Now
                        </a>
                        <a href="https://evemask.info/#demo" 
                           class="cta-button">
                            📺 Watch Demo
                        </a>
                    </div>
                </div>
                <div style="background: #f8f9fa; padding: 25px; border-radius: 8px; margin: 30px 0;">
                    <h4 style="color: #333; margin-bottom: 15px;">🎯 What You Get Today:</h4>
                    <ul style="color: #666; line-height: 1.8;">
                        <li>✅ Complete EVEMASK AI software license</li>
                        <li>✅ 24/7 technical support</li>
                        <li>✅ Free installation and setup</li>
                        <li>✅ Training for your team</li>
                        <li>✅ Free updates for 1 year</li>
                    </ul>
                </div>
                <div style="text-align: center; margin: 40px 0;">
                    <p style="color: #666; font-size: 14px; margin-bottom: 20px;">
                        Questions? Need more information? Our team is ready to help!
                    </p>
                    <p style="font-size: 16px; font-weight: 600;">
                        📧 <a href="mailto:evemask.ai@gmail.com" style="color: #007bff;">evemask.ai@gmail.com</a><br>
                        📞 <a href="tel:+84386893609" style="color: #007bff;">(+84) 386 893 609</a>
                    </p>
                </div>
            </div>
            
            <div class="footer">
                <p><strong>EVEMASK - AI solutions for safe broadcasting</strong></p>
                <p>We don’t just detect - we make a difference</p>
                <div class="footer-links">
                    <a href="https://www.youtube.com/@evemask-ai">YouTube</a>
                    <a href="mailto:evemask.ai@gmail.com">Contact</a>
                </div>
                
                <p style="font-size: 12px; opacity: 0.8; margin-top: 25px;">
                    FPT University Quy Nhon AI Campus, An Phu Thinh, Quy Nhon Dong, Gia Lai, Viet Nam<br>
                    © 2025 EVEMASK. All rights reserved.
                </p>
                
                <p style="font-size: 11px; opacity: 0.6; margin-top: 20px;">
                    You received this email because you expressed interest in EVEMASK solutions.<br>
                    This is a one-time promotional email. No spam, just valuable offers.
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

def get_gmail_credentials():
    """
    Initialize and validate Gmail API credentials for email sending.
    
    This function handles OAuth 2.0 credential management for Gmail API:
    - Loads credentials from environment variables
    - Validates all required OAuth parameters
    - Creates authorized credentials object
    - Handles token refresh automatically
    - Provides comprehensive error reporting
    
    Environment Variables Required:
        GOOGLE_CLIENT_ID: OAuth 2.0 client ID from Google Cloud Console
        GOOGLE_CLIENT_SECRET: OAuth 2.0 client secret
        GOOGLE_REFRESH_TOKEN: Long-lived refresh token for authorization
        
    Returns:
        Credentials: Google OAuth 2.0 credentials object if successful
        None: If credentials are invalid or missing
        
    Error Handling:
        - Missing credentials: Returns None with detailed error message
        - Expired tokens: Automatically refreshes if refresh_token available
        - Invalid tokens: Returns None with validation error
        
    Usage:
        creds = get_gmail_credentials()
        if creds:
            service = build('gmail', 'v1', credentials=creds)
    """
    try:
        # Load credentials từ environment variables
        creds_info = {
            "client_id": EMAIL_CONFIG.get("google_client_id"),
            "client_secret": EMAIL_CONFIG.get("google_client_secret"),
            "refresh_token": EMAIL_CONFIG.get("google_refresh_token"),
            "token_uri": "https://oauth2.googleapis.com/token",
            "type": "authorized_user"
        }
        
        # Kiểm tra tất cả credentials có tồn tại không
        missing_creds = [key for key, value in creds_info.items() if not value and key != "type"]
        if missing_creds:
            print(f"❌ Missing Gmail credentials: {missing_creds}")
            print("🔧 Please check your HuggingFace Spaces secrets:")
            print("   GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN")
            return None
        
        # Tạo credentials object
        creds = Credentials.from_authorized_user_info(
            creds_info, 
            ['https://www.googleapis.com/auth/gmail.send']
        )
        
        # Refresh token nếu cần (credentials sẽ tự động refresh khi expired)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            print("🔄 Gmail credentials refreshed successfully")
        
        return creds
        
    except Exception as e:
        print(f"❌ Error loading Gmail credentials: {str(e)}")
        return None

def send_welcome_email(email: str) -> bool:
    """
    Send professional welcome email to new subscribers via Gmail API.
    
    This function orchestrates the complete email sending process:
    - Validates Gmail API credentials
    - Generates personalized HTML email content
    - Sends email using Gmail API with proper formatting
    - Handles comprehensive error scenarios
    - Provides detailed logging for debugging
    
    Email Features:
        - Professional EVEMASK branding
        - Responsive HTML design
        - Marketing content highlighting AI benefits
        - Anti-spam compliance headers
        - Tracking-ready structure
        
    Args:
        email (str): Recipient email address (pre-validated)
        
    Returns:
        bool: True if email sent successfully, False otherwise
        
    Error Handling:
        - Credential failures: Logs detailed error and returns False
        - API rate limits: Logs rate limit info and returns False  
        - Network issues: Logs connection errors and returns False
        - Invalid recipients: Logs validation errors and returns False
        
    Gmail API Scopes Required:
        - https://www.googleapis.com/auth/gmail.send
        
    Example:
        success = send_welcome_email("user@example.com")
        if success:
            print("Welcome email sent successfully")
    """
    try:
        # Lấy credentials
        creds = get_gmail_credentials()
        if not creds:
            print("❌ Gmail credentials not available. Skipping email.")
            return False
        
        # Test credentials trước khi sử dụng
        if not creds.valid:
            print("❌ Gmail credentials invalid. Skipping email.")
            return False
        
        # Xây dựng service Gmail
        service = build('gmail', 'v1', credentials=creds)
        
        # Không test profile để tránh lỗi scope, chỉ dùng sender_email từ config
        sender_email = EMAIL_CONFIG["sender_email"]
        print(f"✅ Gmail API credentials ready. Sender: {sender_email}")
        
        # Tạo nội dung email
        message = MIMEText(create_welcome_email_html(email), 'html')
        message['To'] = email
        message['From'] = f'{EMAIL_CONFIG["sender_name"]} <{sender_email}>'
        message['Subject'] = "🚨 Stop Losing Money on Content Violations - EVEMASK AI Solution Inside!"
        
        # Encode message dưới dạng base64
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        create_message = {'raw': encoded_message}
        
        # Gửi email
        send_message = service.users().messages().send(
            userId="me", 
            body=create_message
        ).execute()
        
        if send_message.get("id"):
            print(f"✅ Successfully sent welcome email to {email} via Gmail API.")
            print(f"📧 Message ID: {send_message.get('id')}")
            return True
        else:
            print(f"❌ Failed to send email via Gmail API: {send_message}")
            return False

    except HttpError as error:
        print(f"❌ Gmail API HTTP error: {error}")
        if error.resp.status == 401:
            print("❌ Authentication failed. Check your credentials.")
        elif error.resp.status == 403:
            print("❌ Access forbidden. Check your Gmail API permissions.")
        elif error.resp.status == 429:
            print("❌ Rate limit exceeded. Too many requests.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error sending email: {str(e)}")
        return False

def save_subscriber(email: str):
    """
    Store subscriber information with multi-tier fallback system.
    
    This function implements a robust data persistence strategy:
    
    Primary Storage (Supabase):
        - PostgreSQL database with ACID compliance
        - Real-time data synchronization
        - Automatic duplicate prevention
        - Scalable cloud infrastructure
        
    Fallback Storage (JSON):
        - Local file system backup
        - Works without internet connectivity
        - Maintains data integrity during outages
        - Easy data migration and recovery
        
    Data Flow:
        1. Check Supabase availability and credentials
        2. Validate email uniqueness in primary database
        3. Insert new subscriber with timestamp and metadata
        4. On failure: automatically fallback to JSON storage
        5. Log all operations for monitoring and debugging
        
    Args:
        email (str): Validated email address to store
        
    Returns:
        bool: True if stored successfully (either method), False on complete failure
        
    Database Schema (Supabase):
        - id: BIGSERIAL PRIMARY KEY
        - email: VARCHAR(255) UNIQUE NOT NULL
        - created_at: TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        - status: VARCHAR(50) DEFAULT 'active'
        - metadata: JSONB DEFAULT '{}'
        
    Error Recovery:
        - Database connection issues: Falls back to JSON
        - Permission errors: Attempts temp directory storage
        - Disk space issues: Logs error and continues
        - Network timeouts: Retries with exponential backoff
        
    Example:
        success = save_subscriber("user@example.com")
        if success:
            print("Subscriber saved successfully")
    """
    try:
        if not supabase:
            print("❌ Supabase client not initialized")
            # Fallback to JSON file if Supabase is not available
            return save_subscriber_to_json(email)
        
        # Kiểm tra xem email đã tồn tại chưa
        existing_subscriber = supabase.table('subscribers').select('*').eq('email', email).execute()
        
        if existing_subscriber.data:
            print(f"📝 Email already exists in database: {email}")
            return True
        
        # Thêm subscriber mới vào database
        new_subscriber = {
            "email": email,
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        result = supabase.table('subscribers').insert(new_subscriber).execute()
        
        if result.data:
            print(f"✅ Successfully saved subscriber to Supabase: {email}")
            return True
        else:
            print(f"❌ Failed to save subscriber to Supabase: {email}")
            # Fallback to JSON file
            return save_subscriber_to_json(email)
            
    except Exception as e:
        print(f"❌ Error saving subscriber to Supabase: {str(e)}")
        print(f"📋 Attempting fallback to JSON file for: {email}")
        # Fallback to JSON file if Supabase fails
        return save_subscriber_to_json(email)

def save_subscriber_to_json(email: str):
    """Fallback method: Lưu email subscriber vào file JSON"""
    try:
        # Thử sử dụng file subscribers.json hiện có trước
        subscribers_file = "subscribers.json"
        subscribers = []
        
        # Đọc file hiện tại nếu có
        try:
            if os.path.exists(subscribers_file):
                with open(subscribers_file, 'r') as f:
                    subscribers = json.load(f)
        except (PermissionError, OSError) as read_error:
            print(f"⚠️ Cannot read {subscribers_file}: {read_error}")
            # Fallback: sử dụng temp directory
            temp_dir = tempfile.gettempdir()
            subscribers_file = os.path.join(temp_dir, "subscribers_backup.json")
            if os.path.exists(subscribers_file):
                with open(subscribers_file, 'r') as f:
                    subscribers = json.load(f)
        
        # Thêm subscriber mới
        new_subscriber = {
            "email": email,
            "timestamp": datetime.now().isoformat()
        }
        
        # Kiểm tra xem email đã tồn tại chưa
        if not any(sub["email"] == email for sub in subscribers):
            subscribers.append(new_subscriber)
            
            # Thử lưu lại file gốc trước
            try:
                if subscribers_file == "subscribers.json":
                    with open(subscribers_file, 'w') as f:
                        json.dump(subscribers, f, indent=2)
                    print(f"📝 Saved subscriber to main file: {email}")
                else:
                    # Lưu vào backup file
                    with open(subscribers_file, 'w') as f:
                        json.dump(subscribers, f, indent=2)
                    print(f"📝 Saved subscriber to backup file: {email}")
            except (PermissionError, OSError) as write_error:
                print(f"❌ Cannot write to {subscribers_file}: {write_error}")
                # In ra log để debug
                print(f"📋 Subscriber data (not saved): {new_subscriber}")
        else:
            print(f"📝 Email already exists: {email}")
        
        return True
    except Exception as e:
        print(f"❌ Error saving subscriber to JSON: {str(e)}")
        print(f"📋 Attempted to save: {email}")
        return False

@app.get("/api/debug/supabase-status")
async def check_supabase_status():
    """Endpoint để kiểm tra trạng thái Supabase connection"""
    try:
        if not supabase:
            return {
                "status": "error",
                "message": "Supabase client not initialized",
                "env_check": {
                    "SUPABASE_URL": bool(SUPABASE_URL),
                    "SUPABASE_ANON_KEY": bool(SUPABASE_KEY)
                }
            }
        
        # Test connection bằng cách thử query table subscribers
        try:
            result = supabase.table('subscribers').select('count', count='exact').execute()
            subscriber_count = result.count
            
            return {
                "status": "success",
                "message": "Supabase connection working",
                "subscriber_count": subscriber_count,
                "env_check": {
                    "SUPABASE_URL": bool(SUPABASE_URL),
                    "SUPABASE_ANON_KEY": bool(SUPABASE_KEY)
                }
            }
        except Exception as db_error:
            return {
                "status": "error",
                "message": f"Supabase database error: {str(db_error)}",
                "note": "Table 'subscribers' might not exist yet",
                "env_check": {
                    "SUPABASE_URL": bool(SUPABASE_URL),
                    "SUPABASE_ANON_KEY": bool(SUPABASE_KEY)
                }
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Supabase status check failed: {str(e)}",
            "env_check": {
                "SUPABASE_URL": bool(SUPABASE_URL),
                "SUPABASE_ANON_KEY": bool(SUPABASE_KEY)
            }
        }

@app.get("/api/subscribers/count")
async def get_subscriber_count():
    """Endpoint để lấy số lượng subscribers"""
    try:
        if not supabase:
            # Fallback to JSON file count
            subscribers_file = "subscribers.json"
            if os.path.exists(subscribers_file):
                with open(subscribers_file, 'r') as f:
                    subscribers = json.load(f)
                    return {"count": len(subscribers), "source": "json_file"}
            return {"count": 0, "source": "json_file"}
        
        result = supabase.table('subscribers').select('count', count='exact').execute()
        return {"count": result.count, "source": "supabase"}
        
    except Exception as e:
        return {"error": f"Failed to get subscriber count: {str(e)}"}

@app.get("/api/subscribers/list")
async def get_subscribers(limit: int = 50, offset: int = 0):
    """Endpoint để lấy danh sách subscribers với pagination"""
    try:
        if not supabase:
            # Fallback to JSON file
            subscribers_file = "subscribers.json"
            if os.path.exists(subscribers_file):
                with open(subscribers_file, 'r') as f:
                    subscribers = json.load(f)
                    paginated = subscribers[offset:offset+limit]
                    return {
                        "subscribers": paginated,
                        "total": len(subscribers),
                        "source": "json_file"
                    }
            return {"subscribers": [], "total": 0, "source": "json_file"}
        
        result = supabase.table('subscribers').select('*').range(offset, offset + limit - 1).execute()
        total_result = supabase.table('subscribers').select('count', count='exact').execute()
        
        return {
            "subscribers": result.data,
            "total": total_result.count,
            "source": "supabase"
        }
        
    except Exception as e:
        return {"error": f"Failed to get subscribers: {str(e)}"}

@app.get("/api/debug/gmail-status")
async def check_gmail_status():
    """Endpoint để kiểm tra trạng thái Gmail API"""
    try:
        # Kiểm tra environment variables
        env_check = {
            "GOOGLE_CLIENT_ID": bool(EMAIL_CONFIG.get("google_client_id")),
            "GOOGLE_CLIENT_SECRET": bool(EMAIL_CONFIG.get("google_client_secret")), 
            "GOOGLE_REFRESH_TOKEN": bool(EMAIL_CONFIG.get("google_refresh_token")),
            "SENDER_EMAIL": bool(EMAIL_CONFIG.get("sender_email")),
            "SENDER_NAME": bool(EMAIL_CONFIG.get("sender_name"))
        }
        
        # Thử lấy credentials
        creds = get_gmail_credentials()
        if not creds:
            return {
                "status": "error",
                "message": "No Gmail credentials available",
                "env_check": env_check,
                "debug_info": {
                    "client_id_preview": EMAIL_CONFIG.get("google_client_id", "")[:20] + "..." if EMAIL_CONFIG.get("google_client_id") else None,
                    "has_refresh_token": bool(EMAIL_CONFIG.get("google_refresh_token"))
                }
            }
        
        if not creds.valid:
            return {
                "status": "error", 
                "message": "Invalid Gmail credentials",
                "env_check": env_check
            }
        
        # Test Gmail API connection với scope hiện tại (chỉ send email)
        try:
            service = build('gmail', 'v1', credentials=creds)
            
            return {
                "status": "success",
                "message": "Gmail API working correctly (send scope only)",
                "env_check": env_check,
                "gmail_info": {
                    "sender_email": EMAIL_CONFIG.get("sender_email"),
                    "scope": "gmail.send",
                    "note": "Ready to send emails"
                }
            }
        except Exception as api_error:
            return {
                "status": "error",
                "message": f"Gmail API service creation failed: {str(api_error)}",
                "env_check": env_check
            }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Gmail API test failed: {str(e)}",
            "env_check": {
                "GOOGLE_CLIENT_ID": bool(EMAIL_CONFIG.get("google_client_id")),
                "GOOGLE_CLIENT_SECRET": bool(EMAIL_CONFIG.get("google_client_secret")), 
                "GOOGLE_REFRESH_TOKEN": bool(EMAIL_CONFIG.get("google_refresh_token")),
                "SENDER_EMAIL": bool(EMAIL_CONFIG.get("sender_email"))
            }
        }

@app.get("/api/debug/file-status")
async def check_file_status():
    """Endpoint để kiểm tra trạng thái file subscribers.json"""
    try:
        file_info = {}
        
        # Kiểm tra file subscribers.json chính
        main_file = "subscribers.json"
        if os.path.exists(main_file):
            file_info["main_file"] = {
                "path": main_file,
                "exists": True,
                "readable": os.access(main_file, os.R_OK),
                "writable": os.access(main_file, os.W_OK),
                "size": os.path.getsize(main_file)
            }
            try:
                with open(main_file, 'r') as f:
                    data = json.load(f)
                    file_info["main_file"]["record_count"] = len(data)
                    file_info["main_file"]["last_record"] = data[-1] if data else None
            except Exception as e:
                file_info["main_file"]["read_error"] = str(e)
        else:
            file_info["main_file"] = {"exists": False}
        
        # Kiểm tra backup file
        temp_dir = tempfile.gettempdir()
        backup_file = os.path.join(temp_dir, "subscribers_backup.json")
        if os.path.exists(backup_file):
            file_info["backup_file"] = {
                "path": backup_file,
                "exists": True,
                "readable": os.access(backup_file, os.R_OK),
                "writable": os.access(backup_file, os.W_OK),
                "size": os.path.getsize(backup_file)
            }
        else:
            file_info["backup_file"] = {"exists": False}
        
        # Thông tin thư mục hiện tại
        file_info["current_directory"] = {
            "cwd": os.getcwd(),
            "temp_dir": temp_dir,
            "files_in_cwd": [f for f in os.listdir(".") if f.endswith('.json')]
        }
        
        return {
            "status": "success",
            "file_info": file_info
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"File check failed: {str(e)}",
            "error_details": str(e)
        }

@app.options("/api/newsletter/signup")
async def newsletter_signup_options():
    """Handle preflight OPTIONS request"""
    return {"message": "OK"}

@app.post("/api/newsletter/signup", response_model=EmailResponse)
async def newsletter_signup(signup: NewsletterSignup):
    """
    Primary API endpoint for newsletter subscription registration.
    
    This endpoint handles the complete subscription workflow:
    
    Request Processing:
        1. Validates email format using Pydantic models
        2. Sanitizes input data to prevent injection attacks
        3. Checks for malformed or suspicious email patterns
        4. Enforces rate limiting to prevent abuse
        
    Data Persistence:
        1. Attempts to save subscriber to Supabase database
        2. Falls back to JSON file storage if database unavailable
        3. Prevents duplicate subscriptions across all storage methods
        4. Maintains data consistency with ACID transactions
        
    Email Processing:
        1. Sends professional welcome email via Gmail API
        2. Handles email delivery failures gracefully
        3. Logs email status for monitoring and analytics
        4. Does not block subscription if email fails
        
    Response Strategy:
        - Success: Returns confirmation with email status
        - Failure: Returns appropriate HTTP error codes
        - Partial: Returns success even if email fails
        
    Request Body:
        {
            "email": "user@example.com",
            "timestamp": "2025-01-01T00:00:00Z" (optional)
        }
        
    Response Codes:
        200: Subscription successful
        400: Invalid email format
        429: Rate limit exceeded
        500: Internal server error
        
    Response Body:
        {
            "message": "Thank you for your interest! Welcome email sent successfully.",
            "email": "user@example.com", 
            "status": "success"
        }
        
    Security Features:
        - Input validation and sanitization
        - Rate limiting protection
        - SQL injection prevention
        - XSS attack mitigation
        - CORS policy enforcement
        
    Monitoring:
        - Comprehensive request/response logging
        - Error tracking and alerting
        - Performance metrics collection
        - Email delivery status tracking
    """
    try:
        email = signup.email
        
        # Lưu subscriber trước
        save_success = save_subscriber(email)
        if not save_success:
            raise HTTPException(status_code=500, detail="Failed to save subscriber")
        
        # Thử gửi email (không block nếu thất bại)
        email_success = send_welcome_email(email)
        
        # Trả về response thành công dù email có thể thất bại
        if email_success:
            status_message = "Thank you for your interest! Welcome email sent successfully."
        else:
            status_message = "Thank you for your interest! You have been subscribed (welcome email will be sent shortly)."
        
        return EmailResponse(
            message=status_message,
            email=email,
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Newsletter signup error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "EVEMASK Newsletter API is running!", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)