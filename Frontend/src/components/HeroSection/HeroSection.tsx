/**
 * HeroSection Component - Main Landing Section
 * 
 * This component renders the hero section of the EVEMASK website, which includes:
 * - Main value proposition and headline
 * - Call-to-action buttons (Demo and Learn More)
 * - Statistics display (accuracy, speed, efficiency)
 * - Trust indicators and social proof
 * - Responsive mobile optimization with forced styling
 * 
 * Features:
 * - Mobile-first responsive design with JavaScript-enforced styles
 * - Dynamic statistics animation on scroll
 * - Professional gradient backgrounds
 * - Glass morphism effects
 * - Optimized for various screen sizes
 * 
 * The component includes mobile-specific style enforcement to ensure
 * consistent appearance across different devices and browsers.
 * 
 * Author: EVEMASK Team
 * Version: 1.0.0
 */

import React, { useEffect } from 'react';
import './HeroSection.css';
import './MobileFix.css';

const HeroSection: React.FC = () => {
  /**
   * Component initialization and mobile optimization
   * Applies responsive styles and ensures mobile compatibility
   */
  useEffect(() => {
    /**
     * Force mobile styles on load and resize
     * This function ensures consistent mobile styling across browsers
     * by programmatically applying CSS styles when viewport is mobile-sized
     */
    const applyMobileStyles = () => {
      if (window.innerWidth <= 768) {
        // Get hero section elements for mobile styling
        const heroElement = document.getElementById('hero');
        const statsElement = document.querySelector('.hero-stats') as HTMLElement;
        const actionsElement = document.querySelector('.hero-actions') as HTMLElement;
        const trustElement = document.querySelector('.hero-trust') as HTMLElement;
        
        // Apply mobile-optimized styles to hero container
        if (heroElement) {
          heroElement.style.cssText = `
            display: block !important;
            min-height: 60vh !important;
            width: calc(100% - 8px) !important;
            margin: 180px 4px 4px 4px !important;
            padding: 20px 8px 10px 8px !important;
            visibility: visible !important;
            opacity: 1 !important;
          `;
        }
        
        // Apply mobile-optimized styles to statistics section
        if (statsElement) {
          statsElement.style.cssText = `
            display: flex !important;
            justify-content: space-around !important;
            gap: 2px !important;
            margin: 5px 0 !important;
            padding: 3px !important;
            background: rgba(255, 255, 255, 0.1) !important;
            border-radius: 4px !important;
            visibility: visible !important;
            opacity: 1 !important;
          `;
        }
        
        // Apply mobile-optimized styles to action buttons
        if (actionsElement) {
          actionsElement.style.cssText = `
            display: flex !important;
            flex-direction: row !important;
            gap: 5px !important;
            margin: 5px 0 !important;
            justify-content: center !important;
            visibility: visible !important;
            opacity: 1 !important;
          `;
        }
        
        if (trustElement) {
          trustElement.style.cssText = `
            display: block !important;
            margin: 5px 0 !important;
            visibility: visible !important;
            opacity: 1 !important;
          `;
        }
        
        // Force trust indicators
        const trustIndicators = document.querySelector('.trust-indicators') as HTMLElement;
        if (trustIndicators) {
          trustIndicators.style.cssText = `
            display: flex !important;
            justify-content: center !important;
            gap: 5px !important;
            visibility: visible !important;
            opacity: 1 !important;
          `;
        }
      }
    };
    
    // Apply on load
    applyMobileStyles();
    
    // Apply on resize
    window.addEventListener('resize', applyMobileStyles);
    
    return () => {
      window.removeEventListener('resize', applyMobileStyles);
    };
  }, []);

  useEffect(() => {
    // Animation for stats counter
    const statNumbers = document.querySelectorAll('.stat-number');
    
    const animateStats = () => {
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

    // Trigger animation when section is in view
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
      if (heroSection) {
        observer.unobserve(heroSection);
      }
    };
  }, []);

  return (
    <section 
      id="hero" 
      className="section hero mobile-fixed-2024"
      style={{
        display: 'block',
        minHeight: window.innerWidth <= 768 ? 'auto' : 'auto',
        width: window.innerWidth <= 768 ? 'calc(100% - 8px)' : 'auto',
        margin: window.innerWidth <= 768 ? '180px 4px 4px 4px' : 'auto',
        padding: window.innerWidth <= 768 ? '20px 8px 10px 8px' : 'auto'
      }}
    >
      <div className="hero-background">
        <div className="hero-particles"></div>
        <div className="hero-gradient"></div>
      </div>
      <div className="hero-container mobile-container-2024">
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
            <div 
              className="hero-actions"
              style={{
                display: window.innerWidth <= 768 ? 'flex' : 'flex',
                flexDirection: window.innerWidth <= 768 ? 'row' : 'column',
                gap: window.innerWidth <= 768 ? '5px' : '8px',
                margin: window.innerWidth <= 768 ? '5px 0' : '10px 0',
                justifyContent: 'center',
                visibility: 'visible',
                opacity: 1
              }}
            >
              <a href="#demo" className="btn btn-primary btn-large hero-cta">
                <span>🎬 Demo</span>
                <div className="btn-shine"></div>
              </a>
              <a href={`${process.env.PUBLIC_URL}/assets/images/EVEMASK_Leaflet.pdf`} download="EVEMASK-Leaflet.pdf" className="btn btn-primary btn-large hero-cta-secondary">
                <span>📄 Download Leaflet</span>
              </a>
            </div>
            <div 
              className="hero-trust"
              style={{
                display: 'block',
                margin: window.innerWidth <= 768 ? '5px 0' : '10px 0',
                visibility: 'visible',
                opacity: 1
              }}
            >
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
              <img src={`${process.env.PUBLIC_URL}/assets/images/img_evenmask.png`} alt="EVEMASK Technology" className="hero-image" />
              <div className="floating-elements">
                <div className="floating-element element-1">🤖</div>
                <div className="floating-element element-2">⚡</div>
                <div className="floating-element element-3">🎯</div>
                <div className="floating-element element-4">🛡️</div>
              </div>
            </div>
            <div className="hero-image-container secondary">
              <div className="image-glow"></div>
              <img src={`${process.env.PUBLIC_URL}/assets/images/gilariver_evemask.png`} alt="Gilariver EVEMASK Demo" className="hero-image secondary-image" />
              <div className="floating-elements">
                <div className="floating-element element-5">🔍</div>
                <div className="floating-element element-6">🎥</div>
                <div className="floating-element element-7">💎</div>
                <div className="floating-element element-8">🚀</div>
              </div>
            </div>
          </div>
          <div 
            className="hero-stats"
            style={{
              display: window.innerWidth <= 768 ? 'flex' : 'flex',
              justifyContent: 'space-around',
              gap: window.innerWidth <= 768 ? '2px' : '5px',
              margin: window.innerWidth <= 768 ? '5px 0' : '10px 0',
              padding: window.innerWidth <= 768 ? '3px' : '5px',
              background: window.innerWidth <= 768 ? 'rgba(255, 255, 255, 0.1)' : 'transparent',
              borderRadius: window.innerWidth <= 768 ? '4px' : '0',
              visibility: 'visible',
              opacity: 1
            }}
          >
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
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
