/**
 * DemoSection Component - Live Technology Demonstration
 * 
 * This component showcases the EVEMASK technology in action through a
 * demonstration video. It provides visitors with a visual proof of concept
 * showing the AI's real-time logo detection and blurring capabilities.
 * 
 * Features:
 * - Responsive video player with modern controls
 * - Auto-play functionality for immediate engagement
 * - Muted and looped playback for seamless presentation
 * - Fallback message for browsers without video support
 * - Professional video container with styling consistency
 * 
 * The demo video demonstrates:
 * - Real-time logo detection accuracy
 * - Smooth blurring effects without quality loss
 * - Processing speed and efficiency
 * - Integration capabilities with broadcast systems
 * 
 * Video specifications:
 * - Format: MP4 for broad browser compatibility
 * - Auto-play with mute for user experience compliance
 * - Loop functionality for continuous demonstration
 * 
 * Author: EVEMASK Team
 * Version: 1.0.0
 */

import React from 'react';
import './DemoSection.css';

const DemoSection: React.FC = () => {
  return (
    <section className="section" id="demo">
      <div className="container">
        {/* Section Header */}
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
        
        {/* Demo Video Container */}
        <div className="demo-video-container">
          <video
            src="assets/videos/video_evenmask.mp4"
            className="demo-video"
            controls
            autoPlay
            muted
            loop
          >
            {/* Fallback message for browsers without video support */}
            Your browser does not support the video tag.
          </video>
        </div>
      </div>
    </section>
  );
};

export default DemoSection;
