/**
 * ContactSection Component - Contact Information & CTA
 * 
 * This component provides comprehensive contact information and serves as
 * the final call-to-action for potential customers. It includes multiple
 * communication channels and professional contact details.
 * 
 * Features:
 * - Multiple contact methods (email, phone, social media)
 * - Professional contact icons using SVG for scalability
 * - Clean, accessible contact information layout
 * - Interactive contact links with proper protocols
 * - Social media integration for broader engagement
 * - Responsive design for all devices
 * 
 * Contact channels include:
 * - Email contact for business inquiries
 * - Phone number for direct communication
 * - Social media links for community engagement
 * - Professional presentation of contact options
 * 
 * The component follows accessibility best practices with:
 * - Proper link protocols (mailto:, tel:)
 * - Clear visual hierarchy
 * - Hover states for interactive elements
 * 
 * Author: EVEMASK Team
 * Version: 1.0.0
 */

import React from 'react';
import './ContactSection.css';

const ContactSection: React.FC = () => {
  return (
    <section className="section" id="contact">
      <div className="container">
        <div className="contact-content">
          {/* Contact Information Header */}
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
          
          {/* Contact Details List */}
          <div className="contact-details">
            {/* Email Contact */}
            <div className="contact-item">
              {/* Email SVG Icon */}
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
                {/* Email link with proper mailto protocol */}
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
                <a href="tel:+84386893609" className="contact-link">
                  (+84) 386893609
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
                <p className="contact-text">FPT University Quy Nhon AI Campus, Nhon Binh, Quy Nhon, Viet Nam</p>
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
  );
};

export default ContactSection;
