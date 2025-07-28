from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from datetime import datetime
import json
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="EVEMASK Newsletter API", version="1.0.0")

# CORS middleware to allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": os.getenv("SENDER_EMAIL"),
    "sender_password": os.getenv("EMAIL_PASSWORD"),
    "sender_name": os.getenv("SENDER_NAME", "EVEMASK Team")
}

def create_welcome_email_html(user_email: str) -> str:
    """Tạo HTML template cho email marketing EVEMASK"""
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
                <div class="logo">🎭 EVEMASK</div>
                <div class="tagline">AI-Powered Content Guardian</div>
            </div>
            
            <div class="content">
                <div class="welcome-badge">🎉 Welcome to the Future</div>
                
                <h1 class="main-title">Stop Compliance Nightmares Today!</h1>
                
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
                        <li><strong>Legal Fines:</strong> Up to $135 million in penalties for gambling ad violations</li>
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
                            <span class="stat-number">96%</span>
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

                <div class="testimonial">
                    <p class="testimonial-text">
                        "EVEMASK transformed our broadcasting operation. We've saved $50,000 in the first month 
                        alone and haven't had a single compliance issue since implementation."
                    </p>
                    <div class="testimonial-author">- Leading Vietnamese Broadcasting Network</div>
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
                        <a href="mailto:evemask.ai@gmail.com?subject=I want to purchase EVEMASK&body=Hi, I'm interested in purchasing EVEMASK for my broadcasting business. Please send me pricing information and schedule a demo." 
                           class="cta-button">
                            🚀 Get EVEMASK Now
                        </a>
                        <a href="mailto:evemask.ai@gmail.com?subject=Request EVEMASK Demo&body=Hi, I'd like to schedule a free demo of EVEMASK to see how it works for my broadcasting needs." 
                           class="cta-button secondary">
                            📺 Free Demo
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
                        <li>✅ 30-day money-back guarantee</li>
                        <li>✅ Free updates for 1 year</li>
                    </ul>
                </div>

                <div style="text-align: center; margin: 40px 0;">
                    <p style="color: #666; font-size: 14px; margin-bottom: 20px;">
                        Questions? Need more information? Our team is ready to help!
                    </p>
                    <p style="font-size: 16px; font-weight: 600;">
                        📧 <a href="mailto:evemask.ai@gmail.com" style="color: #007bff;">evemask.ai@gmail.com</a><br>
                        📞 <a href="tel:+84386893609" style="color: #007bff;">(+84) 386893609</a>
                    </p>
                </div>
            </div>
            
            <div class="footer">
                <p><strong>EVEMASK - AI Content Guardian</strong></p>
                <p>Protecting Your Broadcasts, Securing Your Future</p>
                
                <div class="footer-links">
                    <a href="https://www.youtube.com/@evemask-ai">YouTube</a>
                    <a href="mailto:evemask.ai@gmail.com">Contact</a>
                </div>
                
                <p style="font-size: 12px; opacity: 0.8; margin-top: 25px;">
                    FPT University Quy Nhon AI Campus, Nhon Binh, Quy Nhon, Vietnam<br>
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

def send_welcome_email(email: str) -> bool:
    """Gửi email marketing tới người dùng"""
    if not EMAIL_CONFIG.get("sender_email") or not EMAIL_CONFIG.get("sender_password"):
        print("Email credentials not configured. Skipping email.")
        return False
    try:
        # Tạo message
        message = MIMEMultipart("alternative")
        message["Subject"] = "🚨 Stop Losing Money on Content Violations - EVEMASK AI Solution Inside!"
        message["From"] = f'{EMAIL_CONFIG["sender_name"]} <{EMAIL_CONFIG["sender_email"]}>'
        message["To"] = email

        # Tạo HTML content
        html_content = create_welcome_email_html(email)
        
        # Tạo plain text version
        text_content = f"""
EVEMASK - AI-Powered Content Guardian

STOP COMPLIANCE NIGHTMARES TODAY!

Dear Broadcasting Professional,

Are you tired of manual content moderation eating into your profits? 
Worried about hefty fines from gambling advertisement violations? 
EVEMASK is here to save your business!

THE PROBLEMS COSTING YOU MONEY RIGHT NOW:
• Legal Fines: Up to $135 million in penalties for gambling ad violations
• Lost Revenue: Manual moderation delays costing thousands per hour  
• Human Error: 40% miss rate in manual content detection
• Viewer Loss: Poor content quality driving audiences away

EVEMASK: YOUR AI-POWERED SOLUTION
Our cutting-edge AI technology automatically detects and blurs gambling logos 
and inappropriate content in REAL-TIME, ensuring 100% compliance while 
maintaining broadcast quality.

PROVEN RESULTS:
• 96% AI Accuracy
• <2s Processing Time  
• 80% Time Savings
• 70% Cost Reduction

KEY BENEFITS:
• Save Money: Reduce moderation costs by 70% and avoid regulatory fines
• Real-Time Processing: Process live broadcasts with less than 2-second delay
• 100% Compliance: Automatically detect and blur all gambling advertisements

LIMITED TIME: Get 30% OFF your first year subscription! 
Only 48 hours left to claim this exclusive offer.

WHAT YOU GET TODAY:
✓ Complete EVEMASK AI software license
✓ 24/7 technical support
✓ Free installation and setup
✓ Training for your team
✓ 30-day money-back guarantee
✓ Free updates for 1 year

READY TO GET STARTED?

Option 1: Purchase EVEMASK Now
Email: evemask.ai@gmail.com
Subject: "I want to purchase EVEMASK"

Option 2: Request Free Demo  
Email: evemask.ai@gmail.com
Subject: "Request EVEMASK Demo"

Call us: (+84) 386893609
Email: evemask.ai@gmail.com

TESTIMONIAL:
"EVEMASK transformed our broadcasting operation. We've saved $50,000 in the 
first month alone and haven't had a single compliance issue since implementation."
- Leading Vietnamese Broadcasting Network

Don't let compliance issues kill your profits. Act now!

Best regards,
EVEMASK Team - AI Content Guardian
FPT University Quy Nhon AI Campus
Nhon Binh, Quy Nhon, Vietnam

© 2025 EVEMASK. All rights reserved.
This is a one-time promotional email. No spam, just valuable offers.
        """

        # Attach parts
        part1 = MIMEText(text_content, "plain")
        part2 = MIMEText(html_content, "html")
        
        message.attach(part1)
        message.attach(part2)

        # Gửi email
        with smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"]) as server:
            server.starttls()
            server.login(EMAIL_CONFIG["sender_email"], EMAIL_CONFIG["sender_password"])
            server.send_message(message)
        
        print(f"✅ Successfully sent welcome email to {email}")
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        print(f"❌ SMTP Authentication Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error sending email to {email}: {str(e)}")
        return False

def save_subscriber(email: str):
    """Lưu email subscriber vào file JSON (có thể thay thế bằng database)"""
    try:
        subscribers_file = "subscribers.json"
        subscribers = []
        
        # Đọc file hiện tại nếu có
        if os.path.exists(subscribers_file):
            with open(subscribers_file, 'r') as f:
                subscribers = json.load(f)
        
        # Thêm subscriber mới
        new_subscriber = {
            "email": email,
            "timestamp": datetime.now().isoformat(),
            "status": "active"
        }
        
        # Kiểm tra xem email đã tồn tại chưa
        if not any(sub["email"] == email for sub in subscribers):
            subscribers.append(new_subscriber)
            
            # Lưu lại file
            with open(subscribers_file, 'w') as f:
                json.dump(subscribers, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving subscriber: {str(e)}")
        return False

@app.post("/api/newsletter/signup", response_model=EmailResponse)
async def newsletter_signup(signup: NewsletterSignup):
    """API endpoint để đăng ký newsletter"""
    try:
        email = signup.email
        
        # Lưu subscriber
        save_success = save_subscriber(email)
        if not save_success:
            raise HTTPException(status_code=500, detail="Failed to save subscriber")
        
        # Gửi email chào mừng
        email_success = send_welcome_email(email)
        if not email_success:
            # Log the error but don't block the user
            print(f"Failed to send welcome email to {email}, but subscriber was saved.")
        
        return EmailResponse(
            message="Thank you for your interest! You have been subscribed.",
            email=email,
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/newsletter/subscribers")
async def get_subscribers():
    """API để lấy danh sách subscribers (cho admin)"""
    try:
        subscribers_file = "subscribers.json"
        if os.path.exists(subscribers_file):
            with open(subscribers_file, 'r') as f:
                subscribers = json.load(f)
            return {"subscribers": subscribers, "total": len(subscribers)}
        else:
            return {"subscribers": [], "total": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving subscribers: {str(e)}")

@app.get("/")
async def root():
    return {"message": "EVEMASK Newsletter API is running!", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)