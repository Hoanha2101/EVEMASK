/**
 * Navbar Component - Main Navigation Interface
 * 
 * This component provides the main navigation bar for the EVEMASK website.
 * Features include:
 * - Responsive navigation menu with mobile hamburger menu
 * - Smooth scroll navigation to different sections
 * - Glass morphism design with blur effects
 * - Active section highlighting
 * - SVG icons for better performance and scalability
 * 
 * Icons are defined as React components for better performance and styling control.
 * The navbar automatically highlights the current section based on scroll position.
 * 
 * Author: EVEMASK Team
 * Version: 1.0.0
 */

import React, { useEffect, useState } from 'react';
import './Navbar.css';

const ChartIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M21 21H4.6C4.03995 21 3.75992 21 3.54601 20.891C3.35785 20.7951 3.20487 20.6422 3.10899 20.454C3 20.2401 3 19.9601 3 19.4V3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M20 8L16 12L12 8L7 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const LightbulbIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 7V9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M12 16.01L12.01 15.9989" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M16 16.5V16.5C16 15.12 17.12 14 18.5 14H19.1C20.93 14 22.1 12.4 21.5 10.67C21.1667 9.79892 21 8.90372 21 8C21 4.13401 17.866 1 14 1H10C6.13401 1 3 4.13401 3 8C3 8.90372 2.83329 9.79892 2.5 10.67C1.9 12.4 3.07 14 4.9 14H5.5C6.88 14 8 15.12 8 16.5V16.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M8 17H16C16 18.8856 16 19.8284 15.4142 20.4142C14.8284 21 13.8856 21 12 21C10.1144 21 9.17157 21 8.58579 20.4142C8 19.8284 8 18.8856 8 17Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const CheckmarkIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M8 12.5L10.5 15L16 9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="currentColor" strokeWidth="2"/>
  </svg>
);

const TeamIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 2L15.5 9H8.5L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M8.5 9L2 14L8.5 13V9Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M15.5 9L22 14L15.5 13V9Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M8.5 13V18C8.5 20.2091 10.2909 22 12.5 22H15.5V13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const PlayIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M5 4.99988L19 11.9999L5 18.9999V4.99988Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const ContactIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M2 12C2 17.5228 6.47715 22 12 22C14.1135 22 16.0681 21.3712 17.6693 20.292L20 21L19.293 18.6769C20.3782 17.0893 21 15.1538 21 13.0702C21 7.54736 16.5228 3.07021 11 3.07021C7.1358 3.07021 3.82428 5.28423 2.5 8.50023" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M8 14H13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M8 10H16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const HomeIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M3 9L12 2L21 9V20C21 20.5304 20.7893 21.0391 20.4142 21.4142C20.0391 21.7893 19.5304 22 19 22H5C4.46957 22 3.96086 21.7893 3.58579 21.4142C3.21071 21.0391 3 20.5304 3 20V9Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M9 22V12H15V22" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const MenuIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M3 12H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M3 6H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M3 18H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const CloseIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M18 6L6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M6 6L18 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

/**
 * Navbar Component - Main Navigation Interface
 * 
 * Manages navigation state and scroll-based interactions.
 * Features:
 * - Scroll-based navbar appearance changes
 * - Active section highlighting based on viewport position
 * - Mobile responsive hamburger menu
 * - Smooth scroll navigation
 * - Auto-collapse functionality for better UX
 */
const Navbar: React.FC = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [activeLink, setActiveLink] = useState<string>('hero'); 
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false); 

  useEffect(() => {
    const handleScroll = () => {
      // Update navbar styling on scroll
      if (window.scrollY > 20) {
        setIsScrolled(true);
      } else {
        setIsScrolled(false);
      }
      
      if (window.innerWidth <= 768) {
        if (window.scrollY > 150) {
          setIsCollapsed(true);
          setMobileMenuOpen(false); 
        } else {
          setIsCollapsed(false);
        }
      }
      
      const sections = document.querySelectorAll('section[id]');
      const scrollPosition = window.pageYOffset + 100; 
      
      if (window.pageYOffset < 50) {
        setActiveLink('hero');
      } else {
        sections.forEach(section => {
          const sectionTop = (section as HTMLElement).offsetTop;
          const sectionHeight = section.clientHeight;
          const sectionId = section.getAttribute('id') || '';
          
          if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
            setActiveLink(sectionId);
          }
        });
      }
    };
    
    const handleResize = () => {
      if (window.innerWidth > 768) {
        setMobileMenuOpen(false);
        setIsCollapsed(false); 
      }
    };

    window.addEventListener('scroll', handleScroll);
    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('scroll', handleScroll);
      window.removeEventListener('resize', handleResize);
    };
  }, [mobileMenuOpen]);

  const handleNavClick = (e: React.MouseEvent<HTMLAnchorElement>, id: string) => {
    e.preventDefault();
    setActiveLink(id);
    setMobileMenuOpen(false); 
    
    if (id === 'hero') {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } else {
      const element = document.getElementById(id);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
      }
    }
  };

  const toggleMobileMenu = () => {
    if (isCollapsed) {
      setIsCollapsed(false);
      setMobileMenuOpen(true);
    } else {
      setMobileMenuOpen(!mobileMenuOpen);
    }
  };



  return (
    <nav className={`navbar ${isScrolled ? 'navbar-shrink' : ''} ${isCollapsed ? 'collapsed' : ''}`}>
      <div className="navbar-container">
        {!isCollapsed && (
          <>
            <div className="navbar-logo">
              <img
                src={`${process.env.PUBLIC_URL}/assets/images/EveMask-logo.png`}
                alt="EVEMASK Logo"
              />
            </div>
            
            <button 
              className="mobile-menu-toggle" 
              onClick={toggleMobileMenu}
              aria-label="Toggle navigation menu"
            >
              {mobileMenuOpen ? <CloseIcon /> : <MenuIcon />}
            </button>
            
            <div className={`nav-links ${mobileMenuOpen ? 'mobile-open' : ''}`}>
              <a 
                href="#hero" 
                className={`nav-item ${activeLink === 'hero' ? 'active home-active' : ''}`}
                onClick={(e) => handleNavClick(e, 'hero')}
              >
                <HomeIcon />
                <span>Home</span>
              </a>
              <a 
                href="#challenges" 
                className={`nav-item ${activeLink === 'challenges' ? 'active' : ''}`}
                onClick={(e) => handleNavClick(e, 'challenges')}
              >
                <ChartIcon />
                <span>Costs</span>
              </a>
              <a 
                href="#solution" 
                className={`nav-item ${activeLink === 'solution' ? 'active' : ''}`}
                onClick={(e) => handleNavClick(e, 'solution')}
              >
                <LightbulbIcon />
                <span>Solution</span>
              </a>
              <a 
                href="#why-choose" 
                className={`nav-item ${activeLink === 'why-choose' ? 'active' : ''}`}
                onClick={(e) => handleNavClick(e, 'why-choose')}
              >
                <CheckmarkIcon />
                <span>Why Choose</span>
              </a>
              <a 
                href="#team" 
                className={`nav-item ${activeLink === 'team' ? 'active' : ''}`}
                onClick={(e) => handleNavClick(e, 'team')}
              >
                <TeamIcon />
                <span>About Us</span>
              </a>
              <a 
                href="#demo" 
                className={`nav-item ${activeLink === 'demo' ? 'active' : ''}`}
                onClick={(e) => handleNavClick(e, 'demo')}
              >
                <PlayIcon />
                <span>Demo</span>
              </a>
              <a 
                href="#contact" 
                className={`nav-item ${activeLink === 'contact' ? 'active' : ''}`}
                onClick={(e) => handleNavClick(e, 'contact')}
              >
                <ContactIcon />
                <span>Contact</span>
              </a>
            </div>
          </>
        )}

        {isCollapsed && (
          <button 
            className="mobile-menu-toggle" 
            onClick={toggleMobileMenu}
            aria-label="Toggle navigation menu"
          >
            <div className="hamburger">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </button>
        )}
      </div>

      {mobileMenuOpen && (
        <div className="nav-links mobile-expanded">
          <a 
            href="#hero" 
            className={`nav-item ${activeLink === 'hero' ? 'active home-active' : ''}`}
            onClick={(e) => handleNavClick(e, 'hero')}
          >
            <HomeIcon />
            <span>Home</span>
          </a>
          <a 
            href="#challenges" 
            className={`nav-item ${activeLink === 'challenges' ? 'active' : ''}`}
            onClick={(e) => handleNavClick(e, 'challenges')}
          >
            <ChartIcon />
            <span>Costs</span>
          </a>
          <a 
            href="#solution" 
            className={`nav-item ${activeLink === 'solution' ? 'active' : ''}`}
            onClick={(e) => handleNavClick(e, 'solution')}
          >
            <LightbulbIcon />
            <span>Solution</span>
          </a>
          <a 
            href="#why-choose" 
            className={`nav-item ${activeLink === 'why-choose' ? 'active' : ''}`}
            onClick={(e) => handleNavClick(e, 'why-choose')}
          >
            <CheckmarkIcon />
            <span>Why Choose</span>
          </a>
          <a 
            href="#team" 
            className={`nav-item ${activeLink === 'team' ? 'active' : ''}`}
            onClick={(e) => handleNavClick(e, 'team')}
          >
            <TeamIcon />
            <span>About Us</span>
          </a>
          <a 
            href="#demo" 
            className={`nav-item ${activeLink === 'demo' ? 'active' : ''}`}
            onClick={(e) => handleNavClick(e, 'demo')}
          >
            <PlayIcon />
            <span>Demo</span>
          </a>
          <a 
            href="#contact" 
            className={`nav-item ${activeLink === 'contact' ? 'active' : ''}`}
            onClick={(e) => handleNavClick(e, 'contact')}
          >
            <ContactIcon />
            <span>Contact</span>
          </a>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
