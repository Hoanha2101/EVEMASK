/**
 * API Configuration - EVEMASK Frontend Configuration
 * 
 * This file contains the centralized API configuration for the EVEMASK frontend.
 * It defines the base URLs and endpoints for all API communications, primarily
 * with the HuggingFace Space backend service.
 * 
 * Configuration includes:
 * - Base URL for HuggingFace Space integration
 * - API endpoints for various services
 * - Newsletter subscription endpoint
 * - Debug and testing endpoints
 * 
 * The configuration is exposed globally via window object for use across
 * different JavaScript files and modules in the application.
 * 
 * Usage:
 * - Newsletter subscription functionality
 * - Debug and monitoring services
 * - API endpoint centralization
 * - Environment-specific configurations
 * 
 * Author: EVEMASK Team
 * Version: 1.0.0
 */

// API Configuration Object
const API_CONFIG = {
  // Primary HuggingFace Space API base URL
  BASE_URL: 'https://nghiant20-evemask.hf.space',
  
  // API endpoint definitions
  ENDPOINTS: {
    NEWSLETTER_SIGNUP: '/api/newsletter/signup',  // Newsletter subscription endpoint
    DEBUG_GMAIL: '/api/debug/gmail-status'       // Gmail debug status endpoint
  }
};

// Export configuration globally for cross-file accessibility
window.API_CONFIG = API_CONFIG;
