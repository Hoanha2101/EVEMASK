/**
 * EVEMASK Frontend JavaScript - Interactive Utilities
 * 
 * This file contains utility functions and interactive components for the
 * EVEMASK website. It provides toast notifications, smooth scrolling,
 * mobile menu functionality, and other user interface enhancements.
 * 
 * Main Features:
 * - Toast notification system with animations
 * - Smooth scrolling navigation
 * - Mobile menu toggle functionality
 * - Form validation and submission helpers
 * - Responsive design utilities
 * - Cross-browser compatibility functions
 * 
 * The script is designed to work alongside the React components and provides
 * additional functionality for enhanced user experience.
 * 
 * Dependencies:
 * - Modern browser API support
 * - CSS custom properties for theming
 * - DOM manipulation capabilities
 * 
 * Author: EVEMASK Team
 * Version: 1.0.0
 */

/**
 * Toast Notification Utility
 * Creates animated toast notifications that slide in from the top-right corner
 * 
 * @param {string} message - The message to display in the toast
 * @param {string} type - The type of toast ('success' or 'error')
 */
function showToast(message, type = "success") {
  // Create or get existing toast container
  let toastContainer = document.querySelector('.toast-container');
  if (!toastContainer) {
    toastContainer = document.createElement('div');
    toastContainer.className = 'toast-container';
    // Position container at top-right of screen
    toastContainer.style.position = 'fixed';
    toastContainer.style.top = '20px';
    toastContainer.style.right = '20px';
    toastContainer.style.zIndex = '9999';
    toastContainer.style.display = 'flex';
    toastContainer.style.flexDirection = 'column';
    toastContainer.style.gap = '12px';
    toastContainer.style.pointerEvents = 'none';
    document.body.appendChild(toastContainer);
  }

  // Create individual toast element
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  // Toast styling for professional appearance
  toast.style.minWidth = '280px';
  toast.style.maxWidth = '400px';
  toast.style.padding = '16px 20px';
  toast.style.borderRadius = '12px';
  toast.style.fontSize = '14px';
  toast.style.fontWeight = '500';
  toast.style.letterSpacing = '0.01em';
  toast.style.boxShadow = '0 10px 25px rgba(0,0,0,0.15), 0 4px 12px rgba(0,0,0,0.1)';
  toast.style.backdropFilter = 'blur(8px)';
  toast.style.color = '#fff';
  toast.style.display = 'flex';
  toast.style.alignItems = 'center';
  toast.style.gap = '12px';
  toast.style.position = 'relative';
  toast.style.cursor = 'pointer';
  toast.style.userSelect = 'none';
  // Dynamic background based on toast type
  toast.style.background = type === 'success'
    ? 'linear-gradient(135deg, #10b981 0%, #059669 100%)'
    : 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)';
  
  // Initial animation state - slide in from top-right
  toast.style.opacity = '0';
  toast.style.transform = 'translateX(100%) translateY(-20px) scale(0.9)';
  toast.style.transition = 'all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94)';
  toast.style.pointerEvents = 'auto';
  toast.style.overflow = 'hidden';

  // Create icon element for toast
  const icon = document.createElement('span');
  icon.style.fontSize = '18px';
  icon.style.flexShrink = '0';
  icon.style.display = 'flex';
  icon.style.alignItems = 'center';
  icon.style.justifyContent = 'center';
  icon.innerHTML = type === 'success'
    ? `<svg width="20" height="20" viewBox="0 0 20 20" fill="none">
         <circle cx="10" cy="10" r="10" fill="rgba(255,255,255,0.2)"/>
         <path d="M6 10.5L8.5 13L14 7.5" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
       </svg>`
    : `<svg width="20" height="20" viewBox="0 0 20 20" fill="none">
         <circle cx="10" cy="10" r="10" fill="rgba(255,255,255,0.2)"/>
         <path d="M7 7L13 13M13 7L7 13" stroke="#fff" stroke-width="2" stroke-linecap="round"/>
       </svg>`;

  // Message
  const msg = document.createElement('span');
  msg.textContent = message;
  msg.style.flex = '1';
  msg.style.lineHeight = '1.4';

  // Close button
  const closeBtn = document.createElement('button');
  closeBtn.innerHTML = '&times;';
  closeBtn.style.position = 'absolute';
  closeBtn.style.top = '8px';
  closeBtn.style.right = '12px';
  closeBtn.style.background = 'none';
  closeBtn.style.border = 'none';
  closeBtn.style.color = '#fff';
  closeBtn.style.fontSize = '18px';
  closeBtn.style.cursor = 'pointer';
  closeBtn.style.opacity = '0.7';
  closeBtn.style.transition = 'opacity 0.2s';
  closeBtn.style.width = '20px';
  closeBtn.style.height = '20px';
  closeBtn.style.display = 'flex';
  closeBtn.style.alignItems = 'center';
  closeBtn.style.justifyContent = 'center';
  
  closeBtn.addEventListener('mouseenter', () => closeBtn.style.opacity = '1');
  closeBtn.addEventListener('mouseleave', () => closeBtn.style.opacity = '0.7');
  closeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    // Slide out to top-right
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(100%) translateY(-20px) scale(0.9)';
    setTimeout(() => {
      if (toast.parentNode) {
        toast.remove();
      }
      if (toastContainer.childElementCount === 0) {
        toastContainer.remove();
      }
    }, 400);
  });

  toast.appendChild(icon);
  toast.appendChild(msg);
  toast.appendChild(closeBtn);

  toastContainer.appendChild(toast);

  // Animate in from top-right
  setTimeout(() => {
    toast.style.opacity = '1';
    toast.style.transform = 'translateX(0) translateY(0) scale(1)';
    
    // Add a subtle bounce effect for success
    if (type === 'success') {
      setTimeout(() => {
        toast.style.transform = 'translateX(0) translateY(0) scale(1.02)';
        setTimeout(() => {
          toast.style.transform = 'translateX(0) translateY(0) scale(1)';
        }, 150);
      }, 200);
    }
    
    // Shake effect for error
    if (type === 'error') {
      setTimeout(() => {
        toast.animate([
          { transform: 'translateX(0) translateY(0) scale(1)' },
          { transform: 'translateX(-4px) translateY(0) scale(1)' },
          { transform: 'translateX(4px) translateY(0) scale(1)' },
          { transform: 'translateX(-2px) translateY(0) scale(1)' },
          { transform: 'translateX(2px) translateY(0) scale(1)' },
          { transform: 'translateX(0) translateY(0) scale(1)' }
        ], { 
          duration: 400, 
          easing: 'cubic-bezier(.36,.07,.19,.97)' 
        });
      }, 100);
    }
  }, 50);

  // Auto-dismiss
  setTimeout(() => {
    if (document.body.contains(toast)) {
      closeBtn.click();
    }
  }, 4500);
}

// Animated counter for hero stats
function animateCounter(element, targetValue) {
  let start = 0;
  const duration = 2000; // 2 seconds
  const isDecimal = targetValue.toString().includes('.');
  const stepTime = 10; // update every 10ms
  const totalSteps = duration / stepTime;
  const increment = targetValue / totalSteps;

  const timer = setInterval(() => {
    start += increment;
    if (start >= targetValue) {
      clearInterval(timer);
      start = targetValue;
    }
    if (isDecimal) {
      element.textContent = start.toFixed(1);
    } else {
      element.textContent = Math.floor(start);
    }
  }, stepTime);
}

// Special handler for "24/7"
function handle247(element) {
  setTimeout(() => {
    element.style.opacity = 0;
    setTimeout(() => {
      element.textContent = "24/7";
      element.style.opacity = 1;
    }, 400);
  }, 1600);
}

document.addEventListener("DOMContentLoaded", () => {
  // Intersection Observer for hero stats
  const heroSection = document.querySelector('.hero');
  const statNumbers = document.querySelectorAll('.stat-number');
  let hasAnimated = false;

  const statObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting && !hasAnimated) {
        statNumbers.forEach(stat => {
          const targetValue = stat.getAttribute('data-value');
          if (targetValue === "24/7") {
            handle247(stat);
          } else {
            animateCounter(stat, parseFloat(targetValue));
          }
        });
        hasAnimated = true;
        statObserver.unobserve(heroSection);
      }
    });
  }, { threshold: 0.5 }); // Trigger when 50% of the section is visible

  if (heroSection) {
    statObserver.observe(heroSection);
  }

  // Smooth scrolling for navigation links
  const anchorLinks = document.querySelectorAll('a[href^="#"]');
  anchorLinks.forEach((link) => {
    link.addEventListener("click", function (e) {
      const targetId = this.getAttribute("href");
      if (targetId !== "#" && targetId.length > 1) {
        const targetElement = document.querySelector(targetId);
        if (targetElement) {
          e.preventDefault();
          targetElement.scrollIntoView({
            behavior: "smooth",
            block: "start",
          });
        }
      }
    });
  });

  // Newsletter form submission
  const newsletterForms = document.querySelectorAll(".newsletter-form");
  newsletterForms.forEach((form) => {
    form.addEventListener("submit", async function (e) {
      e.preventDefault();
      const emailInput = this.querySelector('input[type="email"]');
      const button = this.querySelector("button");
      const originalText = button.textContent;
      // Remove old feedback UI (now using toast)
      let feedback = this.querySelector('.newsletter-feedback');
      if (feedback) feedback.remove();
      if (emailInput && emailInput.value) {
        button.textContent = "Sending...";
        button.disabled = true;
        try {
          // Always use HuggingFace API - no more config confusion
          const apiUrl = "https://nghiant20-evemask.hf.space/api/newsletter/signup";
          
          // Debug: Log the URL being called
          console.log('Calling API URL:', apiUrl);
          console.log('Current hostname:', window.location.hostname);
          console.log('Current protocol:', window.location.protocol);
          console.log('Full current URL:', window.location.href);
            
          const response = await fetch(apiUrl, {
            method: "POST",
            mode: "cors", // Explicitly set CORS mode
            headers: {
              "Content-Type": "application/json",
              "Accept": "application/json"
            },
            body: JSON.stringify({ email: emailInput.value })
          });
          
          console.log('Response status:', response.status);
          console.log('Response headers:', response.headers);
          
          if (response.ok) {
            const data = await response.json();
            console.log('Success response:', data);
            button.textContent = "Subscribed!";
            button.style.backgroundColor = "var(--color-neutral-darkest)";
            button.style.color = "var(--color-white)";
            emailInput.value = "";
            showToast("Thank you for subscribing!", "success");
          } else {
            const data = await response.json();
            console.log('Error response:', data);
            button.textContent = data.detail || "Error!";
            button.style.backgroundColor = "#e74c3c";
            button.style.color = "#fff";
            showToast(data.detail || " Subscription failed. Please try again.", "error");
          }
        } catch (err) {
          console.error('Network error details:', err);
          console.log('Error name:', err.name);
          console.log('Error message:', err.message);
          
          button.textContent = "Network Error";
          button.style.backgroundColor = "#e74c3c";
          button.style.color = "#fff";
          
          // More specific error messages
          let errorMessage = "Network error. Please try again later.";
          if (err.name === 'TypeError' && err.message.includes('fetch')) {
            errorMessage = "Unable to connect to server. Please check your internet connection.";
          }
          
          showToast(errorMessage, "error");
        }
        setTimeout(() => {
          button.textContent = originalText;
          button.style.backgroundColor = "";
          button.style.color = "";
          button.disabled = false;
        }, 3000);
      }
    });
  });

  // Mobile menu toggle (if needed in future)
  const navbarContainer = document.querySelector(".navbar-container");
  if (window.innerWidth <= 991) {
    // Add mobile menu functionality here if needed
    console.log("Mobile view detected");
  }

  // Button hover effects enhancement
  const buttons = document.querySelectorAll(".btn");
  buttons.forEach((button) => {
    button.addEventListener("mouseenter", function () {
      this.style.transform = "translateY(-1px)";
      this.style.transition = "transform 0.2s ease";
    });

    button.addEventListener("mouseleave", function () {
      this.style.transform = "translateY(0)";
    });
  });

  // Feature card animations on scroll
  const observerOptions = {
    threshold: 0.1,
    rootMargin: "0px 0px -50px 0px",
  };

  const observer = new IntersectionObserver(function (entries) {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = "1";
        entry.target.style.transform = "translateY(0)";
      }
    });
  }, observerOptions);

  // Apply animation to feature cards and step cards
  const animatedElements = document.querySelectorAll(
    ".feature-card, .step-card",
  );
  animatedElements.forEach((element, index) => {
    // Initial state
    element.style.opacity = "0";
    element.style.transform = "translateY(30px)";
    element.style.transition = `opacity 0.6s ease ${index * 0.1}s, transform 0.6s ease ${index * 0.1}s`;

    // Observe for intersection
    observer.observe(element);
  });

  // Contact form validation and submission
  const contactForm = document.querySelector("form");
  if (contactForm) {
    contactForm.addEventListener("submit", function (e) {
      e.preventDefault();

      // Basic form validation
      const requiredFields = this.querySelectorAll("[required]");
      let isValid = true;

      requiredFields.forEach((field) => {
        if (!field.value.trim()) {
          isValid = false;
          field.style.borderColor = "#e74c3c";
        } else {
          field.style.borderColor = "var(--color-scheme-1-border)";
        }
      });

      if (isValid) {
        // Simulate form submission
        const submitButton = this.querySelector('button[type="submit"]');
        const originalText = submitButton.innerHTML;

        submitButton.innerHTML = "<span>Sending...</span>";
        submitButton.disabled = true;

        setTimeout(() => {
          // alert removed
          this.reset();
          submitButton.innerHTML = originalText;
          submitButton.disabled = false;
        }, 2000);
      } else {
        alert("Please fill in all required fields.");
      }
    });
  }

  // Demo button click tracking - DISABLED to allow smooth scrolling
  // const demoButtons = document.querySelectorAll('a[href="#demo"]');
  // demoButtons.forEach((button) => {
  //   button.addEventListener("click", function (e) {
  //     e.preventDefault();
  //     // Simulate demo request
  //     const modal = confirm(
  //       "Would you like to schedule a personalized demo of EVEMASK?",
  //     );
  //     if (modal) {
  //       window.location.href = "contact.html";
  //     }
  //   });
  // });

  // Scroll-to-top functionality
  let scrollTopButton = document.createElement("button");
  scrollTopButton.innerHTML = "↑";
  scrollTopButton.className = "scroll-to-top";
  scrollTopButton.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 50px;
        height: 50px;
        background: var(--color-neutral-darkest);
        color: var(--color-white);
        border: none;
        border-radius: 50%;
        font-size: 18px;
        cursor: pointer;
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s ease;
        z-index: 1000;
    `;

  document.body.appendChild(scrollTopButton);

  scrollTopButton.addEventListener("click", function () {
    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  });

  // Show/hide scroll-to-top button
// script.js
    window.addEventListener('scroll', function() {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 20) {
        navbar.classList.add('shrink');
    } else {
        navbar.classList.remove('shrink');
    }
    });

  console.log("EVEMASK website initialized successfully!");
});

// Window resize handler
window.addEventListener("resize", function () {
  // Handle responsive navigation changes
  const navbar = document.querySelector(".navbar");
  if (window.innerWidth <= 991) {
    navbar.classList.add("mobile");
  } else {
    navbar.classList.remove("mobile");
  }
});

// Performance optimization - lazy loading for images
const images = document.querySelectorAll("img");
const imageObserver = new IntersectionObserver((entries, observer) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      const img = entry.target;
      img.style.opacity = "0";
      img.style.transition = "opacity 0.3s";

      const loadImage = () => {
        img.style.opacity = "1";
      };

      if (img.complete) {
        loadImage();
      } else {
        img.addEventListener("load", loadImage);
      }

      observer.unobserve(img);
    }
  });
});

images.forEach((img) => {
  imageObserver.observe(img);
});
