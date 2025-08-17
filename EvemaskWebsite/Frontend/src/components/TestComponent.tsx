import React, { useEffect } from 'react';

const TestComponent: React.FC = () => {
  useEffect(() => {
    const linkElement = document.createElement('link');
    linkElement.rel = 'stylesheet';
    linkElement.href = '/assets/css/styles.css';
    linkElement.id = 'evemask-styles';
  
    if (!document.getElementById('evemask-styles')) {
      document.head.appendChild(linkElement);
    }

    const handleScroll = () => {
      const navbar = document.querySelector('.navbar');
      if (navbar) {
        if (window.scrollY > 20) {
          navbar.classList.add('navbar-shrink');
        } else {
          navbar.classList.remove('navbar-shrink');
        }
      }
    };

    window.addEventListener('scroll', handleScroll);

    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  return (
    <>
      <nav className="navbar">
        <div className="navbar-container">
          <div className="navbar-logo">
            <img
              src="assets/images/EveMask-logo.png"
              alt="EVEMASK Logo"
            />
          </div>
          <div className="nav-links">
            <a href="#challenges" className="btn btn-secondary">
              <span>Costs</span>
            </a>
            <a href="#solution" className="btn btn-secondary">
              <span>Solution</span>
            </a>
            <a href="#why-choose" className="btn btn-secondary">
              <span>Why Choose</span>
            </a>
            <a href="#team" className="btn btn-secondary">
              <span>About Us</span>
            </a>
            <a href="#demo" className="btn btn-secondary">
              <span>Demo</span>
            </a>
            <a href="#contact" className="btn btn-secondary">
              <span>Contact</span>
            </a>
          </div>
        </div>
      </nav>

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
              <div className="hero-actions">
                <a href="#demo" className="btn btn-primary btn-large hero-cta">
                  <span>🎬 Demo</span>
                  <div className="btn-shine"></div>
                </a>
                <a href="public/assets/images/EVEMASK_Leaflet.pdf" download="EVEMASK_Leaflet.pdf" className="btn btn-primary btn-large hero-cta-secondary">
                <span>📄 Download Leaflet</span>
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
            </div>
          </div>
        </div>
      </section>

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
            </div>
          </div>
          <div className="footer-credits">
            <div className="footer-divider"></div>
            <div className="footer-bottom">
              <div className="footer-legal">
                <span>© 2025 EVEMASK. All rights reserved. - React Version</span>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </>
  );
};

export default TestComponent;
