/**
 * WhyChooseSection Component - Key Benefits & Features
 * 
 * This component highlights the key advantages and selling points of EVEMASK.
 * It presents quantified benefits in an engaging bubble/card layout to showcase
 * the value proposition to potential customers.
 * 
 * Features:
 * - Bubble-style cards with key metrics and benefits
 * - Visual icons for each benefit category
 * - Quantified results (percentages, time savings, format support)
 * - Responsive grid layout
 * - Professional statistical presentation
 * 
 * Key benefits highlighted:
 * - AI Accuracy (90% precision in detection)
 * - Real-time Processing (< 2s latency)
 * - Time Saving (80% reduction in manual work)
 * - Format Support (10+ video formats)
 * - Comprehensive Coverage (image and video processing)
 * - Easy Integration (API and SDK options)
 * 
 * The layout uses a bubble design to make statistics more visually appealing
 * and easier to digest for potential customers.
 * 
 * Author: EVEMASK Team
 * Version: 1.0.0
 */

import React from 'react';
import './WhyChooseSection.css';

const WhyChooseSection: React.FC = () => {
  return (
    <section className="section" id="why-choose">
      <div className="container">
        {/* Section Header */}
        <div className="section-title section-title-center">
          <div className="tagline-wrapper tagline-wrapper-center">
            <span className="tagline">🌟 Why Choose</span>
          </div>
          <div className="section-content">
            <h2 className="section-heading">Why Choose EVEMASK?</h2>
            <p className="section-text">
              EVEMASK delivers a comprehensive solution with advanced AI technology, ensuring excellent broadcast quality and effective regulatory compliance.
            </p>
          </div>
        </div>

        {/* Benefits Bubble Grid */}
        <div className="why-choose-bubbles">
          {/* AI Accuracy Benefit */}
          <div className="bubble-section">
            <div className="bubble-section-icon">🤖</div>
            <div className="bubble-section-value">90%</div>
            <div className="bubble-section-title">AI Accuracy</div>
            <div className="bubble-section-description">Detects and blurs logos or violating content with high precision</div>
          </div>

          {/* Real-time Processing Benefit */}
          <div className="bubble-section">
            <div className="bubble-section-icon">🚀</div>
            <div className="bubble-section-value">&lt; 2s</div>
            <div className="bubble-section-title">Real-time Processing</div>
            <div className="bubble-section-description">Minimal latency during live broadcasts</div>
          </div>

          {/* Time Saving Benefit */}
          <div className="bubble-section">
            <div className="bubble-section-icon">💡</div>
            <div className="bubble-section-value">80%</div>
            <div className="bubble-section-title">Time Saving</div>
            <div className="bubble-section-description">Reduces manual moderation time compared to traditional methods</div>
          </div>

          {/* Video Format Support Benefit */}
          <div className="bubble-section">
            <div className="bubble-section-icon">🎬</div>
            <div className="bubble-section-value">10+</div>
            <div className="bubble-section-title">Video Formats</div>
            <div className="bubble-section-description">Supports various formats: MP4, AVI, MOV, WEBM, and more</div>
          </div>

          {/* Image Format Support Benefit */}
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
  );
};

export default WhyChooseSection;
