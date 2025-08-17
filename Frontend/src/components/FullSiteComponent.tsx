/**
 * FullSiteComponent - Complete EVEMASK Website Layout
 * 
 * This component contains the entire EVEMASK website structure including:
 * - Hero section with main value proposition
 * - Problem/Challenge section
 * - Solution section with key features
 * - Team section
 * - Newsletter subscription functionality
 * - Contact information and footer
 * 
 * Features:
 * - Newsletter subscription with HuggingFace API integration
 * - Responsive design for all device types
 * - Smooth scroll animations and interactions
 * - Dynamic stats animation
 * - Form validation and error handling
 * 
 * Author: EVEMASK Team
 * Version: 1.0.0
 */

import React, { useState, useEffect } from 'react';
import './Navbar/Navbar.css';
import Navbar from './Navbar/Navbar';

const FullSiteComponent: React.FC = () => {
  // Newsletter subscription state management
  const [email, setEmail] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitMessage, setSubmitMessage] = useState('');

  /**
   * Handles newsletter subscription form submission
   * Integrates with HuggingFace Space API for email collection
   * @param e - Form submission event
   */
  const handleNewsletterSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email) return;
    
    setIsSubmitting(true);
    setSubmitMessage('');
    
    try {
      // API call to HuggingFace Space for newsletter signup
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
      
      // Handle successful subscription
      if (response.ok) {
        const data = await response.json();
        console.log('Success response:', data);
        setSubmitMessage('Thank you for subscribing!');
        setEmail('');
      } else {
        // Handle API error responses
        const data = await response.json();
        console.log('Error response:', data);
        setSubmitMessage(data.detail || 'Subscription failed. Please try again.');
      }
    } catch (error) {
      // Handle network errors
      console.error('Network error:', error);
      setSubmitMessage('Network error. Please check your connection and try again.');
    } finally {
      setIsSubmitting(false);
      // Clear message after 5 seconds
      setTimeout(() => setSubmitMessage(''), 5000);
    }
  };

  /**
   * Component initialization and setup
   * - Adds custom styles for newsletter messages
   * - Sets up scroll event listeners
   * - Initializes stats animations
   * - Cleanup on component unmount
   */
  useEffect(() => {
    // TODO: Uncomment when CSS component testing is complete
    // const linkElement = document.createElement('link');
    // linkElement.rel = 'stylesheet';
    // linkElement.href = '/assets/css/styles.css';
    // linkElement.id = 'evemask-styles';
    
    // if (!document.getElementById('evemask-styles')) {
    //   document.head.appendChild(linkElement);
    // }
    
    // Add custom styles for newsletter message feedback
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

    // Scroll effect cho các thành phần khác nếu cần
    const handleScroll = () => {
      // Code xử lý scroll cho các phần khác ngoài navbar
    };

    window.addEventListener('scroll', handleScroll);

    // Stats animation
    const animateStats = () => {
      const statNumbers = document.querySelectorAll('.stat-number');
      statNumbers.forEach((stat) => {
        const target = parseInt(stat.getAttribute('data-value') || '0');
        const increment = target / 100;
        let current = 0;
        
        const updateCounter = () => {
          if (current < target) {
            current += increment;
            stat.textContent = Math.ceil(current).toString();
            setTimeout(updateCounter, 20);
          } else {
            stat.textContent = target.toString();
          }
        };
        
        updateCounter();
      });
    };

    // Trigger animation when hero section is in view
    const heroSection = document.querySelector('.hero');
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          animateStats();
        }
      });
    });

    if (heroSection) {
      observer.observe(heroSection);
    }

    return () => {
      window.removeEventListener('scroll', handleScroll);
      if (heroSection) {
        observer.unobserve(heroSection);
      }
    };
  }, []);

  return (
    <>
      {/* Sử dụng component Navbar */}
      <Navbar />

      <main>
        {/* Hero Section */}
        <section className="section hero">
          <div className="hero-background">
            <div className="hero-particles"></div>
            <div className="hero-gradient"></div>
          </div>
          <div className="hero-container">
            <div className="hero-content">
              <div className="hero-text">
                <div className="hero-badge">
                  <span className="badge-text">🚀 AI TECHNOLOGY</span>
                </div>
                <h1 className="hero-title">
                  <span className="title-highlight">EVEMASK</span>
                  <span className="title-subtitle">Smart AI Solution for Broadcast Compliance</span>
                </h1>
                <p className="hero-description">
                  Advanced AI technology automatically detects and blurs gambling advertisement logos in real time, ensuring legal compliance and protecting your brand integrity.
                </p>
                <div className="hero-stats">
                  <div className="stat-item">
                    <div className="stat-number" data-value="90">0</div>
                    <div className="stat-label">Accuracy ( % )</div>
                  </div>
                  <div className="stat-item">
                    <div className="stat-number" data-value="60">0</div>
                    <div className="stat-label">Real-time Processing Up To ( FPS )</div>
                  </div>
                  <div className="stat-item">
                    <div className="stat-number" data-value="100">0</div>
                    <div className="stat-label">Image Quality ( % )</div>
                  </div>
                </div>
                <div className="hero-actions">
                  <a href="#demo" className="btn btn-primary btn-large hero-cta">
                    <span>🎬 Demo</span>
                    <div className="btn-shine"></div>
                  </a>
                  <a href="assets/images/EVEMASK_Leaflet.pdf" target="_blank" rel="noopener noreferrer" className="btn btn-primary btn-large hero-cta-secondary">
                    <span>📄 Leaflet</span>
                  </a>
                </div>
                <div className="hero-trust">
                  <div className="trust-indicators">
                    <div className="trust-item">⚡ Fast</div>
                    <div className="trust-item">🎯 Accurate</div>
                    <div className="trust-item">🛡️ Secure</div>
                  </div>
                </div>
              </div>
              <div className="hero-visual">
                <div className="hero-image-container">
                  <div className="image-glow"></div>
                  <img src="assets/images/img_evenmask.png" alt="EVEMASK Technology" className="hero-image" />
                  <div className="floating-elements">
                    <div className="floating-element element-1">🤖</div>
                    <div className="floating-element element-2">⚡</div>
                    <div className="floating-element element-3">🎯</div>
                    <div className="floating-element element-4">🛡️</div>
                  </div>
                </div>
                <div className="hero-image-container secondary">
                  <div className="image-glow"></div>
                  <img src="assets/images/gilariver_evemask.png" alt="Gilariver EVEMASK Demo" className="hero-image secondary-image" />
                  <div className="floating-elements">
                    <div className="floating-element element-5">🔍</div>
                    <div className="floating-element element-6">🎥</div>
                    <div className="floating-element element-7">💎</div>
                    <div className="floating-element element-8">🚀</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Challenges Section */}
        <section className="section" id="challenges">
          <div className="container">
            <div className="section-title section-title-center">
              <div className="tagline-wrapper tagline-wrapper-center">
                <span className="tagline"> 🏷️ Costs</span>
              </div>
              <div className="section-content">
                <h2 className="section-heading">
                  Major Challenges in the Broadcasting Industry
                </h2>
                <p className="section-text">
                  These challenges are causing significant financial and reputational losses for broadcasters. Understanding the problem is the first step to finding an effective solution.
                </p>
              </div>
            </div>
            
            <div className="challenges-grid">
              <div className="challenge-card">
                <div className="challenge-icon">
                  <svg width="48" height="48" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="32" cy="32" r="30" fill="#FFE5E5"/>
                    <path d="M25 28h14v8H25z" fill="#FF6B6B"/>
                    <path d="M27 30h10v4H27z" fill="#FF4757"/>
                    <path d="M32 24v-4M32 44v4M24 32h-4M44 32h4" stroke="#FF3838" strokeWidth="2" strokeLinecap="round"/>
                    <circle cx="32" cy="32" r="2" fill="#FF3838"/>
                  </svg>
                </div>
                <h3 className="challenge-title">Legal Risks & Financial Losses</h3>
                <p className="challenge-description">
                  Accidentally broadcasting gambling advertisement logos can result in heavy administrative fines under Vietnamese law.
                </p>
              </div>
              
              <div className="challenge-card">
                <div className="challenge-icon">
                  <svg width="48" height="48" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="32" cy="32" r="30" fill="#FFF3E0"/>
                    <circle cx="32" cy="32" r="20" fill="none" stroke="#FF9800" strokeWidth="3"/>
                    <path d="M32 18v14l10 6" stroke="#FF9800" strokeWidth="3" strokeLinecap="round"/>
                    <path d="M25 25l-3-3M39 25l3-3" stroke="#FF9800" strokeWidth="2" strokeLinecap="round"/>
                    <path d="M32 8v4M32 52v4M8 32h4M52 32h4" stroke="#FF9800" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                </div>
                <h3 className="challenge-title">Inefficient Manual Moderation</h3>
                <p className="challenge-description">
                  Manual moderation is time-consuming, costly, and error-prone due to fatigue. For live broadcasts, this is nearly impossible.
                </p>
              </div>
              
              <div className="challenge-card">
                <div className="challenge-icon">
                  <svg width="48" height="48" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="32" cy="32" r="30" fill="#FFF3E0"/>
                    <path d="M20 32c0-6.6 5.4-12 12-12s12 5.4 12 12" fill="none" stroke="#FF9800" strokeWidth="3"/>
                    <path d="M28 32c0-2.2 1.8-4 4-4s4 1.8 4 4" fill="none" stroke="#FF9800" strokeWidth="3"/>
                    <circle cx="32" cy="32" r="2" fill="#FF9800"/>
                    <path d="M32 36v8M24 44h16" stroke="#FF9800" strokeWidth="2" strokeLinecap="round"/>
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
                    <a href="https://vnexpress.net/hai-don-vi-phat-tran-viet-nam-indonesia-bi-phat-135-trieu-dong-4735757.html" target="_blank" rel="noopener noreferrer">
                      <img src="assets/images/baibao1.png" alt="Bài báo về vi phạm quảng cáo cá cược" className="article-img" />
                    </a>
                    <div className="article-overlay">
                      <div className="article-badge">News</div>
                    </div>
                  </div>
                  <div className="article-content">
                    <h4 className="article-title">Penalties for Gambling Ad Violations</h4>
                    <p className="article-excerpt">Authorities have issued strict penalties for violations of gambling advertising regulations on television.</p>
                  </div>
                </div>
                
                <div className="article-card">
                  <div className="article-image">
                    <a href="https://abei.gov.vn/phat-thanh-truyen-hinh/quang-cao-co-bac-ca-do-khong-duoc-phep-xuat-hien-tren-song-truyen-hinh/118505#:~:text=15%2F05%2F2024-,Qu%E1%BA%A3ng%20c%C3%A1o%20c%E1%BB%9D%20b%E1%BA%A1c%2C%20c%C3%A1%20%C4%91%E1%BB%96" target="_blank" rel="noopener noreferrer">
                      <img src="assets/images/baibao2.png" alt="Bài báo về tác động của quảng cáo cá cược" className="article-img" />
                    </a>
                    <div className="article-overlay">
                      <div className="article-badge">News</div>
                    </div>
                  </div>
                  <div className="article-content">
                    <h4 className="article-title">Negative Impact of Gambling Ads</h4>
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

        </section>

        {/* Why Choose Section */}
        <section className="section" id="why-choose">
          <div className="container">
            <div className="section-title section-title-center">
              <div className="tagline-wrapper tagline-wrapper-center">
                <span className="tagline">� Why Choose</span>
              </div>
              <div className="section-content">
                <h2 className="section-heading">Why Choose EVEMASK?</h2>
                <p className="section-text">
                  EVEMASK delivers a comprehensive solution with advanced AI technology, ensuring excellent broadcast quality and effective regulatory compliance.
                </p>
              </div>
            </div>

            <div className="why-choose-bubbles">
              <div className="bubble-section">
                <div className="bubble-section-icon">🤖</div>
                <div className="bubble-section-value">90%</div>
                <div className="bubble-section-title">AI Accuracy</div>
                <div className="bubble-section-description">Detects and blurs logos or violating content with high precision</div>
              </div>

              <div className="bubble-section">
                <div className="bubble-section-icon">🚀</div>
                <div className="bubble-section-value">&lt; 2s</div>
                <div className="bubble-section-title">Real-time Processing</div>
                <div className="bubble-section-description">Minimal latency during live broadcasts</div>
              </div>

              <div className="bubble-section">
                <div className="bubble-section-icon">💡</div>
                <div className="bubble-section-value">80%</div>
                <div className="bubble-section-title">Time Saving</div>
                <div className="bubble-section-description">Reduces manual moderation time compared to traditional methods</div>
              </div>

              <div className="bubble-section">
                <div className="bubble-section-icon">🎬</div>
                <div className="bubble-section-value">10+</div>
                <div className="bubble-section-title">Video Formats</div>
                <div className="bubble-section-description">Supports various formats: MP4, AVI, MOV, WEBM, and more</div>
              </div>

              <div className="bubble-section">
                <div className="bubble-section-icon">📸</div>
                <div className="bubble-section-value">100%</div>
                <div className="bubble-section-title">Image Quality</div>
                <div className="bubble-section-description">Preserves original video quality after processing and blurring</div>
              </div>

              <div className="bubble-section">
                <div className="bubble-section-icon">💰</div>
                <div className="bubble-section-value">70%</div>
                <div className="bubble-section-title">Reduced Operational Costs</div>
                <div className="bubble-section-description">Significantly saves on personnel and moderation costs</div>
              </div>
            </div>
          </div>
        </section>

        {/* Demo Section */}
        <section className="section" id="demo">
          <div className="container">
            <div className="section-title section-title-center">
              <div className="tagline-wrapper tagline-wrapper-center">
                <span className="tagline">🎬 Demo</span>
              </div>
              <div className="section-content">
                <h2 className="section-heading">Experience EVEMASK in Action</h2>
                <p className="section-text">
                  Watch EVEMASK's AI technology in action, automatically detecting and blurring logos in real time.
                </p>
              </div>
            </div>
            
            <div className="demo-video-container">
              <video
                src="assets/videos/video_evenmask.mp4"
                className="demo-video"
                controls
                autoPlay
                muted
                loop
              >
                Your browser does not support the video tag.
              </video>
            </div>
          </div>
        </section>

        {/* Team Section */}
        <section className="section" id="team">
          <div className="container">
            <div className="section-title section-title-center">
              <div className="tagline-wrapper tagline-wrapper-center">
                <span className="tagline"> 👥 Team</span>
              </div>
              <div className="section-content">
                <h2 className="section-heading">The Team Behind The Technology</h2>
                <p className="section-text">
                  Meet the dedicated professionals driving innovation at
                  EVEMASK. Our team's expertise ensures that your broadcasts are
                  secure and compliant.
                </p>
              </div>
            </div>
            
            <div className="team-grid">
              <div className="team-member">
                <div className="team-member-image">
                  <img
                    src="assets/images/chaubui.jpg"
                    alt="Dang Phuc Bao Chau"
                    className="member-photo"
                  />
                </div>
                <div className="team-member-info">
                  <h3 className="member-name">Dang Phuc Bao Chau</h3>
                  <p className="member-role">AI Engineer</p>
                </div>
              </div>
              
              <div className="team-member">
                <div className="team-member-image">
                  <img
                    src="assets/images/khaihoan.jpg"
                    alt="Ha Khai Hoan"
                    className="member-photo"
                  />
                </div>
                <div className="team-member-info">
                  <h3 className="member-name">Ha Khai Hoan</h3>
                  <p className="member-role">AI Engineer</p>
                </div>
              </div>
              
              <div className="team-member">
                <div className="team-member-image">
                  <img
                    src="assets/images/trongnghia.jpg"
                    alt="Nguyen Trong Nghia"
                    className="member-photo"
                  />
                </div>
                <div className="team-member-info">
                  <h3 className="member-name">Nguyen Trong Nghia</h3>
                  <p className="member-role">AI Engineer</p>
                </div>
              </div>
              
              <div className="team-member">
                <div className="team-member-image">
                  <img
                    src="assets/images/thunguyen.jpg"
                    alt="Nguyen Van Thu"
                    className="member-photo"
                  />
                </div>
                <div className="team-member-info">
                  <h3 className="member-name">Nguyen Van Thu</h3>
                  <p className="member-role">AI Engineer</p>
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
                      d="M4 4H20C21.1 4 22 4.9 22 6V18C22 19.1 21.1 20 20 20H4C2.9 20 2 19.1 2 18V6C2 4.9 2.9 4 4 4Z"
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
                  <div className="contact-info-content">
                    <h4 className="contact-info-heading">Email</h4>
                    <a href="mailto:evemask.ai@gmail.com" className="contact-link">
                      evemask.ai@gmail.com
                    </a>
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
                      d="M22 16.92V19.92C22.0011 20.1985 21.9441 20.4742 21.8325 20.7293C21.7209 20.9845 21.5573 21.2136 21.3521 21.4019C21.1468 21.5901 20.9046 21.7335 20.6407 21.8227C20.3769 21.9119 20.0974 21.9451 19.82 21.92C16.7428 21.5856 13.787 20.5341 11.19 18.85C8.77382 17.3147 6.72533 15.2662 5.18999 12.85C3.49997 10.2412 2.44824 7.27099 2.11999 4.18C2.095 3.90347 2.12787 3.62476 2.21649 3.36162C2.30512 3.09849 2.44756 2.85669 2.63476 2.65162C2.82196 2.44655 3.0498 2.28271 3.30379 2.17052C3.55777 2.05833 3.83233 2.00026 4.10999 2H7.10999C7.59532 1.99522 8.06579 2.16708 8.43376 2.48353C8.80173 2.79999 9.04207 3.23945 9.10999 3.72C9.23662 4.68007 9.47144 5.62273 9.80999 6.53C9.94454 6.88792 9.97366 7.27691 9.89391 7.65088C9.81415 8.02485 9.62886 8.36811 9.35999 8.64L8.08999 9.91C9.51355 12.4135 11.5865 14.4864 14.09 15.91L15.36 14.64C15.6319 14.3711 15.9751 14.1858 16.3491 14.1061C16.7231 14.0263 17.1121 14.0555 17.47 14.19C18.3773 14.5286 19.3199 14.7634 20.28 14.89C20.7658 14.9585 21.2094 15.2032 21.5265 15.5775C21.8437 15.9518 22.0122 16.4296 22 16.92Z"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  <div className="contact-info-content">
                    <h4 className="contact-info-heading">Phone</h4>
                    <a href="tel:(+84) 386 893 609" className="contact-link">
                      (+84) 386 893 609
                    </a>
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
                      d="M21 10C21 17 12 23 12 23S3 17 3 10C3 7.61305 3.94821 5.32387 5.63604 3.63604C7.32387 1.94821 9.61305 1 12 1C14.3869 1 16.6761 1.94821 18.364 3.63604C20.0518 5.32387 21 7.61305 21 10Z"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <circle
                      cx="12"
                      cy="10"
                      r="3"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  <div className="contact-info-content">
                    <h4 className="contact-info-heading">Office</h4>
                    <p className="contact-text">FPT University Quy Nhon AI Campus, An Phu Thinh, Quy Nhon Dong, Gia Lai, Viet Nam</p>
                  </div>
                </div>
              </div>
            </div>
            <div className="map-container">
              <iframe 
                src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3873.08943300018!2d109.2191454!3d13.8038844!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x316f6bf778c80973%3A0x8a7d0b5aa0af29c7!2sFPT%20University%20Quy%20Nhon%20AI%20Campus!5e0!3m2!1sen!2s!4v1678886700006!5m2!1sen!2s" 
                width="100%" 
                height="100%" 
                style={{border: 0, borderRadius: '12px'}}
                allowFullScreen={true}
                loading="lazy" 
                referrerPolicy="no-referrer-when-downgrade"
                title="FPT University Quy Nhon AI Campus"
              />
            </div>
          </div>
        </section>
      </main>

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
                    <a href="#challenges" className="footer-link">🚀 Home</a>
                  </div>
                  <div className="footer-link-item">
                    <a href="#team" className="footer-link">👥 About Us</a>
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
                    <a href="tel:+84386893609" className="footer-link">
                      <img src="assets/images/phone.svg" alt="Phone" className="footer-link-icon" />
                      (+84) 386 893 609
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
                    type="email"
                    className="form-input"
                    placeholder="Your Email"
                    aria-label="Email address"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
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
