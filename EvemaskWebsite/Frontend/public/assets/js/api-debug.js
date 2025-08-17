// API Debug utility for frontend
const checkAPIStatus = async () => {
  try {
    const apiUrl = "https://nghiant20-evemask.hf.space/api/debug/gmail-status";
    
    console.log('Checking API status at:', apiUrl);
    
    const response = await fetch(apiUrl, {
      method: "GET",
      mode: "cors",
      headers: {
        "Accept": "application/json"
      }
    });
    
    console.log('API Status Response:', response.status);
    
    if (response.ok) {
      const data = await response.json();
      console.log('API Status Data:', data);
      return data;
    } else {
      console.error('API Status Error:', response.status, response.statusText);
      return null;
    }
  } catch (error) {
    console.error('API Status Network Error:', error);
    return null;
  }
};

// Export for use in browser console or other files
window.checkAPIStatus = checkAPIStatus;

// Auto-check API status when page loads
document.addEventListener('DOMContentLoaded', () => {
  console.log('🚀 EVEMASK Frontend - API Debug Utility Loaded');
  console.log('📊 Use checkAPIStatus() in console to test API connectivity');
  
  // Auto-check after 2 seconds
  setTimeout(() => {
    checkAPIStatus();
  }, 2000);
});
