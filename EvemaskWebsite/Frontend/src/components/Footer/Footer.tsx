/**
 * Footer Component - Website Footer with Newsletter Subscription
 * 
 * This component serves as the website footer, providing comprehensive
 * information about EVEMASK and offering newsletter subscription functionality.
 * 
 * Features:
 * - Newsletter subscription with HuggingFace API integration
 * - Company information and branding
 * - Social media links and contact information
 * - Copyright and legal information
 * - Professional footer layout with multiple sections
 * - Form validation and error handling
 * - Responsive design for all screen sizes
 * 
 * Newsletter functionality:
 * - Email validation and submission
 * - Integration with HuggingFace Space API
 * - Success/error message handling
 * - Automatic message clearing after 5 seconds
 * - Loading states during submission
 * 
 * Footer sections include:
 * - Company branding and description
 * - Newsletter subscription form
 * - Social media links
 * - Contact information
 * - Copyright notice
 * 
 * Author: EVEMASK Team
 * Version: 1.0.0
 */

import React, { FormEvent, useState } from 'react';
import './Footer.css';

const Footer: React.FC = () => {
  // Newsletter subscription state management
  const [email, setEmail] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitMessage, setSubmitMessage] = useState('');

  /**
   * Handles newsletter subscription form submission
   * Integrates with HuggingFace Space API for email collection
   * @param e - Form submission event
   */
  const handleNewsletterSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!email) return;
    
    setIsSubmitting(true);
    setSubmitMessage('');
    
    try {
      // API call to HuggingFace Space for newsletter signup
      const apiUrl = "https://nghiant20-evemask.hf.space/api/newsletter/signup";
      
      console.log('Footer - Calling API URL:', apiUrl);
      console.log('Footer - Email:', email);
      
      const response = await fetch(apiUrl, {
        method: "POST",
        mode: "cors",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify({ email: email })
      });
      
      console.log('Footer - Response status:', response.status);
      
      // Handle successful subscription
      if (response.ok) {
        const data = await response.json();
        console.log('Footer - Success response:', data);
        setSubmitMessage('Thank you for subscribing!');
        setEmail('');
      } else {
        // Handle API error responses
        const data = await response.json();
        console.log('Footer - Error response:', data);
        setSubmitMessage(data.detail || 'Subscription failed. Please try again.');
      }
    } catch (error) {
      // Handle network errors
      console.error('Footer - Network error:', error);
      setSubmitMessage('Network error. Please check your connection and try again.');
    } finally {
      setIsSubmitting(false);
      // Clear message after 5 seconds
      setTimeout(() => setSubmitMessage(''), 5000);
    }
  };

  return (
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
                  <a href="#" className="footer-link">🚀 Home</a>
                </div>
                <div className="footer-link-item">
                  <a href="#team" className="footer-link">👥 About Us</a>
                </div>
                <div className="footer-link-item">
                  <a href="#why-choose" className="footer-link">🌟 Why Choose</a>
                </div>
                <div className="footer-link-item">
                  <a href="#about" className="footer-link">🛠️ Solutions</a>
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
                    (+84) 386893609
                  </a>
                </div>
                <div className="footer-link-item">
                  <a href="https://www.youtube.com/@evemask-ai" className="footer-link" target="_blank" rel="noopener">
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
                  disabled={isSubmitting}
                />
                <button type="submit" className="btn btn-secondary newsletter-send-btn" disabled={isSubmitting || !email}>
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
              <span>© 2025 EVEMASK. All rights reserved.</span>
              <a href="#" className="footer-legal-link">Privacy Policy</a>
              <a href="#" className="footer-legal-link">Terms of Service</a>
              <a href="#" className="footer-legal-link">Cookie Settings</a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
