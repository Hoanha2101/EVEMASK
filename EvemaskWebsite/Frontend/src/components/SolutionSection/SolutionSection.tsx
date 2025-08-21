/**
 * SolutionSection Component - EVEMASK Solution Presentation
 * 
 * This component showcases the EVEMASK solution and its advantages over
 * traditional manual moderation workflows. It presents a visual comparison
 * between the old manual process and the new automated AI-powered solution.
 * 
 * Features:
 * - Side-by-side workflow comparison
 * - Visual workflow diagrams showing process differences
 * - Professional badge design to distinguish old vs new approaches
 * - Responsive image handling with proper fallbacks
 * - Clear value proposition messaging
 * 
 * The component emphasizes:
 * - Time efficiency gains through automation
 * - Accuracy improvements with AI detection
 * - Process simplification and reliability
 * - Real-time content protection capabilities
 * 
 * Author: EVEMASK Team
 * Version: 1.0.0
 */

import React from 'react';
import './SolutionSection.css';

const SolutionSection: React.FC = () => {
  return (
    <section className="section" id="solution">
      {/* Section Header */}
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

      {/* Workflow Comparison Grid */}
      <div className="workflow-grid workflow-comparison workflow-comparison-large">
          {/* Traditional Manual Workflow */}
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
              {/* Traditional workflow diagram */}
              <img src={`${process.env.PUBLIC_URL}/assets/images/Existed_Workflow.png`} alt="Quy trình kiểm duyệt truyền thống" className="workflow-image workflow-image-large" />
            </div>
          </div>

          {/* EVEMASK Automated Workflow */}
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
              {/* EVEMASK automated workflow diagram */}
              <img src={`${process.env.PUBLIC_URL}/assets/images/Evemask_Workflow.png`} alt="Quy trình EVEMASK tự động" className="workflow-image workflow-image-large" />
            </div>
          </div>
      </div>
    </section>
  );
};

export default SolutionSection;
