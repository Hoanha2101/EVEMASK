/**
 * ChallengesSection Component - Problem Statement Display
 * 
 * This component presents the major challenges faced by the broadcasting industry
 * that EVEMASK addresses. It highlights the problems that led to the development
 * of the solution.
 * 
 * Features:
 * - Grid layout of challenge cards with custom SVG icons
 * - Professional card design with hover effects
 * - Visual hierarchy with tagline, heading, and descriptions
 * - Responsive layout that adapts to different screen sizes
 * - Custom-designed icons for each challenge category
 * 
 * Challenges highlighted:
 * 1. Legal Risks & Financial Losses from gambling ads
 * 2. Inefficient Manual Moderation processes
 * 3. Brand Safety & Reputation Management issues
 * 4. Scalability Limitations of current solutions
 * 5. High Operational Costs of manual review
 * 6. Quality Control & Accuracy problems
 * 
 * Author: EVEMASK Team
 * Version: 1.0.0
 */

import React from 'react';
import './ChallengesSection.css';

const ChallengesSection: React.FC = () => {
  return (
    <section className="section" id="challenges">
      <div className="container">
        {/* Section Header */}
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
        
        {/* Challenges Grid Layout */}
        <div className="challenges-grid">
          {/* Challenge Card 1: Legal Risks */}
          <div className="challenge-card">
            <div className="challenge-icon">
              {/* Custom SVG icon for legal risks */}
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
          
          {/* Challenge Card 2: Manual Moderation */}
          <div className="challenge-card">
            <div className="challenge-icon">
              {/* Custom SVG icon for time/efficiency issues */}
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
  );
};

export default ChallengesSection;
