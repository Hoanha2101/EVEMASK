/**
 * App Component - Main Application Entry Point
 * 
 * This is the root component of the EVEMASK application.
 * It renders the main FullSiteComponent which contains all sections of the website.
 * 
 * Author: EVEMASK Team
 * Version: 1.0.0
 */

import React from 'react';
import './App.css';
import FullSiteComponent from './components/FullSiteComponent';

/**
 * Main App component that serves as the entry point for the application
 * @returns JSX element containing the complete EVEMASK website
 */
function App() {
  return (
    <div className="App">
      {/* Main site component containing all sections */}
      <FullSiteComponent />
    </div>
  );
}

export default App;
