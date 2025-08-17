import React, { useState, useEffect } from 'react';
import './Navbar/Navbar.css';
import Navbar from './Navbar/Navbar';

const FullSiteComponent: React.FC = () => {
  const [email, setEmail] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitMessage, setSubmitMessage] = useState('');

  const handleNewsletterSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email) return;
    
    setIsSubmitting(true);
    setSubmitMessage('');
    
    try {
      // Real API call to HuggingFace Space
      const apiUrl = "https://nghiant20-evemask.hf.space/api/newsletter/signup";
      
      console.log('Calling API URL:', apiUrl);
      console.log('Email:', email);
      
      const response = await fetch(apiUrl, {
        method: "POST",
        mode: "cors",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify({ email: email })
      });
      
      console.log('Response status:', response.status);
      
      if (response.ok) {
        const data = await response.json();
        console.log('Success response:', data);
        setSubmitMessage('Thank you for subscribing!');
        setEmail('');
      } else {
        const data = await response.json();
        console.log('Error response:', data);
        setSubmitMessage(data.detail || 'Subscription failed. Please try again.');
      }
    } catch (error) {
      console.error('Network error:', error);
      setSubmitMessage('Network error. Please check your connection and try again.');
    } finally {
      setIsSubmitting(false);
      setTimeout(() => setSubmitMessage(''), 5000);
    }
  };

  useEffect(() => {
    // Add custom styles for newsletter message
    const style = document.createElement('style');
    style.textContent = `
      .newsletter-message {
        margin-top: 8px;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 14px;
        text-align: center;
      }
      .newsletter-message.success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
      }
      .newsletter-message.error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
      }
      .newsletter-send-btn:disabled {
        opacity: 0.6;
        cursor: not-allowed;
      }
    `;
    document.head.appendChild(style);
    
    return () => {
      document.head.removeChild(style);
    };
  }, []);

  return (
    <>
      <Navbar />
      
      {/* Hero Section */}
      <section className="section section-large hero" id="hero">
        <div className="container">
          <div className="hero-content">
            <div className="hero-text">
              <div className="tagline-wrapper">
                <span className="tagline">🎭 AI-Powered</span>
              </div>
              <h1 className="hero-heading">
                <span className="primary-text">Automate Logo Detection</span>
                <span className="secondary-text">& Content Moderation</span>
              </h1>
              <p className="hero-description">
                EVEMASK uses cutting-edge AI to automatically detect and blur gambling logos in real-time broadcasts. Protect your content, ensure compliance, and maintain viewer trust with our intelligent content moderation solution.
              </p>
              <div className="hero-actions">
                <a href="#demo" className="btn btn-primary">
                  <span>🎬 Watch Demo</span>
                </a>
                <a href="#contact" className="btn btn-secondary">
                  <span>📞 Get Started</span>
                </a>
              </div>
            </div>
            <div className="hero-media">
              <div className="hero-video-container">
                <video
                  className="hero-video"
                  autoPlay
                  muted
                  loop
                  playsInline
                  poster="assets/images/hero.png"
                >
                  <source src="assets/videos/video_evenmask.mp4" type="video/mp4" />
                  Your browser does not support the video tag.
                </video>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Why Choose Section */}
      <section className="section" id="why-choose">
        <div className="section-title section-title-center">
          <div className="tagline-wrapper tagline-wrapper-center">
            <span className="tagline">🌟 Why Choose</span>
          </div>
          <div className="section-content">
            <h2 className="section-heading">The Smart Choice for Content Moderation</h2>
            <p className="section-text">
              Experience next-generation content protection with AI that understands context, adapts to changes, and delivers results in real-time.
            </p>
          </div>
        </div>
        <div className="container">
          <div className="why-choose-grid">
            <div className="why-choose-item">
              <div className="why-choose-icon">
                <svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <circle cx="32" cy="32" r="28" fill="#2563eb" fillOpacity="0.1"/>
                  <path d="M32 8l6.928 14.056L56 25.528l-12 11.7L46.856 56 32 48.944 17.144 56 20 37.228l-12-11.7 17.072-3.472L32 8z" fill="#2563eb"/>
                </svg>
              </div>
              <h3 className="why-choose-title">99.8% Accuracy</h3>
              <p className="why-choose-description">
                Advanced AI algorithms deliver industry-leading detection accuracy, minimizing false positives and ensuring reliable content protection.
              </p>
            </div>
            <div className="why-choose-item">
              <div className="why-choose-icon">
                <svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <circle cx="32" cy="32" r="28" fill="#10b981" fillOpacity="0.1"/>
                  <path d="M32 12v20l14 14-4 4-14-14V12h4z" fill="#10b981"/>
                  <circle cx="32" cy="32" r="24" stroke="#10b981" strokeWidth="2" fill="none"/>
                </svg>
              </div>
              <h3 className="why-choose-title">Real-Time Processing</h3>
              <p className="why-choose-description">
                Process live streams with minimal latency. Our optimized engine ensures seamless content moderation without affecting broadcast quality.
              </p>
            </div>
            <div className="why-choose-item">
              <div className="why-choose-icon">
                <svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <circle cx="32" cy="32" r="28" fill="#f59e0b" fillOpacity="0.1"/>
                  <path d="M20 32c0-6.627 5.373-12 12-12s12 5.373 12 12" stroke="#f59e0b" strokeWidth="3" fill="none"/>
                  <path d="M16 36c0-8.837 7.163-16 16-16s16 7.163 16 16" stroke="#f59e0b" strokeWidth="2" fill="none"/>
                  <circle cx="32" cy="32" r="4" fill="#f59e0b"/>
                </svg>
              </div>
              <h3 className="why-choose-title">Self-Learning AI</h3>
              <p className="why-choose-description">
                Continuously adapts to new logo variations and designs. The more it processes, the smarter it becomes at detecting emerging patterns.
              </p>
            </div>
            <div className="why-choose-item">
              <div className="why-choose-icon">
                <svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <circle cx="32" cy="32" r="28" fill="#8b5cf6" fillOpacity="0.1"/>
                  <path d="M24 28h16v8H24V28z" fill="#8b5cf6"/>
                  <path d="M20 24h24v16H20V24z" stroke="#8b5cf6" strokeWidth="2" fill="none"/>
                  <path d="M28 24V20h8v4M32 40v4" stroke="#8b5cf6" strokeWidth="2"/>
                </svg>
              </div>
              <h3 className="why-choose-title">Easy Integration</h3>
              <p className="why-choose-description">
                Plug-and-play solution that integrates seamlessly with existing broadcast infrastructure. Deploy in minutes, not months.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Challenges Section */}
      <section className="section challenges-section" id="challenges">
        <div className="section-title section-title-center">
          <div className="tagline-wrapper tagline-wrapper-center">
            <span className="tagline">⚠️ Challenge</span>
          </div>
          <div className="section-content">
            <h2 className="section-heading">The Growing Content Moderation Crisis</h2>
            <p className="section-text">
              Traditional content moderation methods are failing to keep up with the evolving landscape of digital advertising and regulatory requirements.
            </p>
          </div>
        </div>
        <div className="container">
          <div className="challenges-grid">
            <div className="challenge-item">
              <div className="challenge-icon">
                <svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <circle cx="32" cy="32" r="28" fill="#ef4444" fillOpacity="0.1"/>
                  <path d="M24 24l16 16M40 24L24 40" stroke="#ef4444" strokeWidth="3" strokeLinecap="round"/>
                  <circle cx="32" cy="32" r="24" stroke="#ef4444" strokeWidth="2" fill="none"/>
                </svg>
              </div>
              <h3 className="challenge-title">Manual Oversight Failures</h3>
              <p className="challenge-description">
                Human moderators miss up to 23% of prohibited content due to fatigue, inconsistency, and the sheer volume of content requiring review.
              </p>
            </div>
            
            <div className="challenge-item">
              <div className="challenge-icon">
                <svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <circle cx="32" cy="32" r="28" fill="#f97316" fillOpacity="0.1"/>
                  <path d="M32 16v16l12 8-4 6-12-8V16h4z" fill="#f97316"/>
                  <path d="M18 20l8 8M46 20l-8 8M18 44l8-8M46 44l-8-8" stroke="#FF9800" strokeWidth="1.5" strokeLinecap="round"/>
                </svg>
              </div>
              <h3 className="challenge-title">Challenging Evolving Ad Technologies</h3>
              <p className="challenge-description">
                Gambling sites constantly change logos, use sophisticated designs, and place them in hard-to-detect locations. Keeping moderation teams updated is a never-ending race.
              </p>
            </div>
          </div>
          
          {/* News Articles Section */}
          <div className="news-articles">
            <div className="section-content">
              <h3 className="articles-heading">Media Coverage</h3>
              <p className="articles-description">
                Media outlets have repeatedly reported on the serious consequences of violating gambling advertising regulations on television.
              </p>
            </div>
            <div className="articles-grid">
              <div className="article-card">
                <div className="article-image">
                  <img src="assets/images/baibao1.png" alt="News article about gambling advertising violations" />
                </div>
                <div className="article-content">
                  <h4 className="article-title">VTV and Other Channels Face Heavy Fines</h4>
                  <p className="article-excerpt">Television stations have been fined billions of VND for broadcasting gambling advertisements during prime time hours.</p>
                </div>
              </div>
              <div className="article-card">
                <div className="article-image">
                  <img src="assets/images/baibao2.png" alt="Research on gambling advertisement impact" />
                </div>
                <div className="article-content">
                  <h4 className="article-title">Impact on Vulnerable Populations</h4>
                  <p className="article-excerpt">Research shows that gambling ads have a negative impact on society, especially on vulnerable groups such as youth.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Solution Section */}
      <section className="section" id="solution">
        <div className="section-title section-title-center">
          <div className="tagline-wrapper tagline-wrapper-center">
            <span className="tagline">🛠️ Guardian</span>
          </div>
          <div className="section-content">
            <h2 className="section-heading">Your Automated Content Guardian</h2>
            <p className="section-text">
              EVEMASK is your ultimate solution for content protection. With
              real-time logo detection and blurring, we ensure compliance and
              safeguard your broadcasts.
            </p>
          </div>
        </div>

        <div className="workflow-comparison workflow-comparison-large">
          <div className="workflow-grid">
            <div className="workflow-section">
              <div className="workflow-badge before">
                <span>⚠️</span>
                Previously
              </div>
              <h3 className="workflow-title">Manual Moderation Workflow</h3>
              <p className="workflow-description">
                Complex, time-consuming, and error-prone process
              </p>
              <div className="workflow-image-container">
                <img src="assets/images/Existed_Workflow.png" alt="Traditional moderation workflow" className="workflow-image workflow-image-large" />
              </div>
            </div>

            <div className="workflow-section evemask">
              <div className="workflow-badge after">
                <span>✨</span>
                EVEMASK
              </div>
              <h3 className="workflow-title">Intelligent Automated Workflow</h3>
              <p className="workflow-description">
                AI processes in real time, accurately and efficiently
              </p>
              <div className="workflow-image-container">
                <img src="assets/images/Evemask_Workflow.png" alt="EVEMASK automated workflow" className="workflow-image workflow-image-large" />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Demo Section */}
      <section className="section" id="demo">
        <div className="section-title section-title-center">
          <div className="tagline-wrapper tagline-wrapper-center">
            <span className="tagline">🎬 Demo</span>
          </div>
          <div className="section-content">
            <h2 className="section-heading">See EVEMASK in Action</h2>
            <p className="section-text">
              Watch how our AI detects and blurs gambling logos in real-time, ensuring your broadcasts stay compliant and viewer-friendly.
            </p>
          </div>
        </div>
        <div className="container">
          <div className="demo-content">
            <div className="demo-video-container">
              <video 
                className="demo-video" 
                controls 
                poster="assets/images/hero.png"
                preload="metadata"
              >
                <source src="assets/videos/highlights.mp4" type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            </div>
            <div className="demo-features">
              <div className="demo-feature">
                <div className="demo-feature-icon">
                  <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="16" cy="16" r="14" fill="#2563eb" fillOpacity="0.1"/>
                    <path d="M12 16l4 4 8-8" stroke="#2563eb" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </div>
                <div className="demo-feature-content">
                  <h4 className="demo-feature-title">Real-time Detection</h4>
                  <p className="demo-feature-description">Instantly identifies gambling logos as they appear on screen</p>
                </div>
              </div>
              <div className="demo-feature">
                <div className="demo-feature-icon">
                  <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="16" cy="16" r="14" fill="#10b981" fillOpacity="0.1"/>
                    <circle cx="16" cy="16" r="6" fill="none" stroke="#10b981" strokeWidth="2"/>
                    <circle cx="16" cy="16" r="2" fill="#10b981"/>
                  </svg>
                </div>
                <div className="demo-feature-content">
                  <h4 className="demo-feature-title">Precise Blurring</h4>
                  <p className="demo-feature-description">Selectively blurs only the detected content, preserving video quality</p>
                </div>
              </div>
              <div className="demo-feature">
                <div className="demo-feature-icon">
                  <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="16" cy="16" r="14" fill="#f59e0b" fillOpacity="0.1"/>
                    <path d="M8 16h16M16 8v16" stroke="#f59e0b" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                </div>
                <div className="demo-feature-content">
                  <h4 className="demo-feature-title">Seamless Integration</h4>
                  <p className="demo-feature-description">Works with existing broadcast systems without interruption</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Team Section */}
      <section className="section" id="team">
        <div className="section-title section-title-center">
          <div className="tagline-wrapper tagline-wrapper-center">
            <span className="tagline">👥 Team</span>
          </div>
          <div className="section-content">
            <h2 className="section-heading">Meet Our Expert Team</h2>
            <p className="section-text">
              Our diverse team of AI researchers, engineers, and industry experts
              is dedicated to revolutionizing content moderation technology.
            </p>
          </div>
        </div>
        <div className="container">
          <div className="team-grid">
            <div className="team-member">
              <div className="team-member-image">
                <img src="assets/images/trongnghia.jpg" alt="Trong Nghia" />
              </div>
              <div className="team-member-info">
                <h4 className="team-member-name">Trong Nghia</h4>
                <p className="team-member-role">AI Engineer & Team Lead</p>
                <p className="team-member-description">
                  Specializes in computer vision and deep learning algorithms for real-time content analysis.
                </p>
              </div>
            </div>
            <div className="team-member">
              <div className="team-member-image">
                <img src="assets/images/thunguyen.jpg" alt="Thu Nguyen" />
              </div>
              <div className="team-member-info">
                <h4 className="team-member-name">Thu Nguyen</h4>
                <p className="team-member-role">Full-Stack Developer</p>
                <p className="team-member-description">
                  Expert in scalable web architectures and API development for seamless system integration.
                </p>
              </div>
            </div>
            <div className="team-member">
              <div className="team-member-image">
                <img src="assets/images/chaubui.jpg" alt="Chau Bui" />
              </div>
              <div className="team-member-info">
                <h4 className="team-member-name">Chau Bui</h4>
                <p className="team-member-role">UI/UX Designer</p>
                <p className="team-member-description">
                  Creates intuitive user experiences and ensures seamless interaction with our AI-powered platform.
                </p>
              </div>
            </div>
            <div className="team-member">
              <div className="team-member-image">
                <img src="assets/images/khaihoan.jpg" alt="Khai Hoan" />
              </div>
              <div className="team-member-info">
                <h4 className="team-member-name">Khai Hoan</h4>
                <p className="team-member-role">Machine Learning Engineer</p>
                <p className="team-member-description">
                  Develops and optimizes neural networks for high-performance logo detection and classification.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Contact Section */}
      <section className="section" id="contact">
        <div className="container">
          <div className="contact-content">
            <div className="contact-info">
              <div className="section-title">
                <div className="tagline-wrapper">
                  <span className="tagline">✉️ Contact</span>
                </div>
                <div className="section-content">
                  <h2 className="section-heading">Get in Touch</h2>
                  <p className="section-text">
                    We're here to help you secure your broadcasts. Reach out for a
                    personalized demo today!
                  </p>
                </div>
              </div>
            </div>
            <div className="contact-details">
              <div className="contact-item">
                <svg
                  className="contact-icon"
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <polyline
                    points="22,6 12,13 2,6"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                <div className="contact-item-content">
                  <h4 className="contact-item-title">Email Us</h4>
                  <p className="contact-item-description">evemask.ai@gmail.com</p>
                </div>
              </div>
              <div className="contact-item">
                <svg
                  className="contact-icon"
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                <div className="contact-item-content">
                  <h4 className="contact-item-title">Call Us</h4>
                  <p className="contact-item-description">(+84) 386893609</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="section section-medium footer">
        <div className="container">
          <div className="footer-content">
            <div className="footer-links">
              <div className="footer-column">
                <img
                  src="assets/images/EveMask-logo.png"
                  alt="EVEMASK Logo"
                  className="footer-logo"
                />
              </div>
              <div className="footer-column">
                <h4 className="footer-heading">Quick Links</h4>
                <div className="footer-links-list">
                  <div className="footer-link-item">
                    <a href="#hero" className="footer-link">🏠 Home</a>
                  </div>
                  <div className="footer-link-item">
                    <a href="#why-choose" className="footer-link">🌟 Why Choose</a>
                  </div>
                  <div className="footer-link-item">
                    <a href="#solution" className="footer-link">🛠️ Solutions</a>
                  </div>
                  <div className="footer-link-item">
                    <a href="#contact" className="footer-link">✉️ Contact Us</a>
                  </div>
                  <div className="footer-link-item">
                    <a href="#demo" className="footer-link">🎬 Demo</a>
                  </div>
                </div>
              </div>
              <div className="footer-column">
                <h4 className="footer-heading">Stay Connected</h4>
                <div className="footer-links-list">
                  <div className="footer-link-item">
                    <a href="mailto:evemask.ai@gmail.com" className="footer-link">
                      <img src="assets/images/email.svg" alt="Email" className="footer-link-icon" />
                      evemask.ai@gmail.com
                    </a>
                  </div>
                  <div className="footer-link-item">
                    <a href="https://www.facebook.com/evemask.ai" className="footer-link" target="_blank" rel="noopener noreferrer">
                      <img src="assets/images/facebook.svg" alt="Facebook" className="footer-link-icon" />
                      Facebook
                    </a>
                  </div>
                  <div className="footer-link-item">
                    <a href="https://www.instagram.com/evemask.ai" className="footer-link" target="_blank" rel="noopener noreferrer">
                      <img src="assets/images/instagram.svg" alt="Instagram" className="footer-link-icon" />
                      Instagram
                    </a>
                  </div>
                  <div className="footer-link-item">
                    <a href="tel:+84386893609" className="footer-link">
                      <img src="assets/images/phone.svg" alt="Phone" className="footer-link-icon" />
                      (+84) 386893609
                    </a>
                  </div>
                  <div className="footer-link-item">
                    <a href="https://www.youtube.com/@evemask-ai" className="footer-link" target="_blank" rel="noopener noreferrer">
                      <img src="assets/images/youtube.svg" alt="YouTube" className="footer-link-icon" />
                      YouTube
                    </a>
                  </div>
                </div>
              </div>
            </div>
            <div className="newsletter">
              <div className="newsletter-content">
                <h4 className="newsletter-heading">Join</h4>
                <p className="newsletter-text">
                  Stay updated on our latest features and product releases.
                </p>
              </div>
              <form className="newsletter-form" onSubmit={handleNewsletterSubmit}>
                <div className="form-group">
                  <input
                    id="main-newsletter-email"
                    name="email"
                    type="email"
                    className="form-input"
                    placeholder="Your Email"
                    aria-label="Email address"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    disabled={isSubmitting}
                    required
                  />
                  <button 
                    type="submit" 
                    className="btn btn-secondary newsletter-send-btn"
                    disabled={isSubmitting || !email}
                  >
                    <span>{isSubmitting ? 'Sending...' : 'Send'}</span>
                  </button>
                </div>
                {submitMessage && (
                  <p className={`newsletter-message ${submitMessage.includes('Thank you') ? 'success' : 'error'}`}>
                    {submitMessage}
                  </p>
                )}
                <p className="newsletter-disclaimer">
                  By subscribing, you agree to our Privacy Policy and consent to
                  updates.
                </p>
              </form>
            </div>
          </div>
          <div className="footer-credits">
            <div className="footer-divider"></div>
            <div className="footer-bottom">
              <div className="footer-legal">
                <span>© 2025 EVEMASK. All rights reserved. - React Version</span>
                <a href="#privacy" className="footer-legal-link">Privacy Policy</a>
                <a href="#terms" className="footer-legal-link">Terms of Service</a>
                <a href="#cookies" className="footer-legal-link">Cookie Settings</a>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </>
  );
};

export default FullSiteComponent;
