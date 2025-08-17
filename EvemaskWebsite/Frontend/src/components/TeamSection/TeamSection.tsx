/**
 * TeamSection Component - Team Member Showcase
 * 
 * This component displays the EVEMASK team members with their roles and photos.
 * It provides a professional presentation of the team behind the technology.
 * 
 * Features:
 * - Grid layout of team member cards
 * - Professional member photos with consistent styling
 * - Role and name information for each member
 * - Responsive design for various screen sizes
 * - Clean, modern card design with hover effects
 * 
 * Team roles include:
 * - AI Engineers responsible for core ML algorithms
 * - Software Engineers handling system development
 * - Product specialists ensuring quality and user experience
 * 
 * The component uses environment variables for image paths to ensure
 * compatibility across different deployment environments.
 * 
 * Author: EVEMASK Team
 * Version: 1.0.0
 */

import React from 'react';
import './TeamSection.css';

const TeamSection: React.FC = () => {
  return (
    <section className="section" id="team">
      <div className="container">
        {/* Section Header */}
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
        
        {/* Team Members Grid */}
        <div className="team-grid">
          <div className="team-member">
            <div className="team-member-image">
              <img
                src={`${process.env.PUBLIC_URL}/assets/images/chaubui.jpg`}
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
                src={`${process.env.PUBLIC_URL}/assets/images/khaihoan.jpg`}
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
                src={`${process.env.PUBLIC_URL}/assets/images/trongnghia.jpg`}
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
                src={`${process.env.PUBLIC_URL}/assets/images/thunguyen.jpg`}
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
  );
};

export default TeamSection;
