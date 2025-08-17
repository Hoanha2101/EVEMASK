/**
 * Index Entry Point - React Application Bootstrap
 * 
 * This file serves as the main entry point for the React application.
 * It creates the root DOM element and renders the App component.
 * 
 * Author: EVEMASK Team
 * Version: 1.0.0
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

// TODO: Uncomment when navbar styles testing is complete
// const linkElement = document.createElement('link');
// linkElement.rel = 'stylesheet';
// linkElement.href = '/assets/css/styles.css';
// document.head.appendChild(linkElement);

/**
 * Create React root and render the application
 * Uses React.StrictMode for development warnings and checks
 */
const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  <React.StrictMode>
    {/* Main application component */}
    <App />
  </React.StrictMode>
);
