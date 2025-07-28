// Toast notification utility
function showToast(message, type = "success") {
  let toastContainer = document.querySelector('.toast-container');
  if (!toastContainer) {
    toastContainer = document.createElement('div');
    toastContainer.className = 'toast-container';
    toastContainer.style.position = 'fixed';
    toastContainer.style.top = '32px';
    toastContainer.style.right = '32px';
    toastContainer.style.zIndex = '9999';
    toastContainer.style.display = 'flex';
    toastContainer.style.flexDirection = 'column';
    toastContainer.style.gap = '16px';
    document.body.appendChild(toastContainer);
  }

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.style.minWidth = '260px';
  toast.style.maxWidth = '380px';
  toast.style.padding = '18px 32px 18px 20px';
  toast.style.borderRadius = '16px';
  toast.style.fontSize = '1.08em';
  toast.style.fontWeight = '600';
  toast.style.letterSpacing = '0.01em';
  toast.style.boxShadow = '0 8px 32px 0 rgba(0,0,0,0.18), 0 1.5px 8px 0 rgba(0,0,0,0.10)';
  toast.style.backdropFilter = 'blur(6px)';
  toast.style.color = '#fff';
  toast.style.display = 'flex';
  toast.style.alignItems = 'center';
  toast.style.gap = '16px';
  toast.style.position = 'relative';
  toast.style.cursor = 'pointer';
  toast.style.userSelect = 'none';
  toast.style.background = type === 'success'
    ? 'linear-gradient(90deg, #1fd1a1 0%, #27ae60 60%, #2ecc71 100%)'
    : 'linear-gradient(90deg, #ff5858 0%, #e74c3c 60%, #ff7675 100%)';
  toast.style.opacity = '0';
  toast.style.transform = 'translateX(100px) scale(0.95)';
  toast.style.transition = 'opacity 0.45s cubic-bezier(.4,2,.6,1), transform 0.45s cubic-bezier(.4,2,.6,1)';
  toast.style.pointerEvents = 'auto';
  toast.style.overflow = 'hidden';

  // Icon
  const icon = document.createElement('span');
  icon.style.fontSize = '1.6em';
  icon.style.flexShrink = '0';
  icon.style.display = 'flex';
  icon.style.alignItems = 'center';
  icon.style.justifyContent = 'center';
  icon.innerHTML = type === 'success'
    ? `<svg width="28" height="28" viewBox="0 0 28 28" fill="none"><circle cx="14" cy="14" r="14" fill="#fff2"/><path d="M8 15.5L12.5 20L20 10" stroke="#fff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg>`
    : `<svg width="28" height="28" viewBox="0 0 28 28" fill="none"><circle cx="14" cy="14" r="14" fill="#fff2"/><path d="M9.5 9.5L18.5 18.5M18.5 9.5L9.5 18.5" stroke="#fff" stroke-width="2.5" stroke-linecap="round"/></svg>`;

  // Message
  const msg = document.createElement('span');
  msg.textContent = message;
  msg.style.flex = '1';
  msg.style.lineHeight = '1.5';

  // Close button
  const closeBtn = document.createElement('button');
  closeBtn.innerHTML = '&times;';
  closeBtn.style.position = 'absolute';
  closeBtn.style.top = '10px';
  closeBtn.style.right = '16px';
  closeBtn.style.background = 'none';
  closeBtn.style.border = 'none';
  closeBtn.style.color = '#fff';
  closeBtn.style.fontSize = '1.3em';
  closeBtn.style.cursor = 'pointer';
  closeBtn.style.opacity = '0.7';
  closeBtn.style.transition = 'opacity 0.2s';
  closeBtn.addEventListener('mouseenter', () => closeBtn.style.opacity = '1');
  closeBtn.addEventListener('mouseleave', () => closeBtn.style.opacity = '0.7');
  closeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(100px) scale(0.95)';
    setTimeout(() => {
      toast.remove();
      if (toastContainer.childElementCount === 0) toastContainer.remove();
    }, 400);
  });

  toast.appendChild(icon);
  toast.appendChild(msg);
  toast.appendChild(closeBtn);

  // Animate in
  setTimeout(() => {
    toast.style.opacity = '1';
    toast.style.transform = 'translateX(0) scale(1)';
    // Shake effect for error
    if (type === 'error') {
      toast.animate([
        { transform: 'translateX(0) scale(1)' },
        { transform: 'translateX(-8px) scale(1)' },
        { transform: 'translateX(8px) scale(1)' },
        { transform: 'translateX(-4px) scale(1)' },
        { transform: 'translateX(4px) scale(1)' },
        { transform: 'translateX(0) scale(1)' }
      ], { duration: 400, easing: 'cubic-bezier(.36,.07,.19,.97)' });
    }
  }, 50);

  // Auto-dismiss
  setTimeout(() => {
    if (document.body.contains(toast)) {
      closeBtn.click();
    }
  }, 5000);

  toastContainer.appendChild(toast);
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
          const response = await fetch("http://localhost:7860/api/newsletter/signup", {
            method: "POST",
            headers: {
              "Content-Type": "application/json"
            },
            body: JSON.stringify({ email: emailInput.value })
          });
          if (response.ok) {
            button.textContent = "Subscribed!";
            button.style.backgroundColor = "var(--color-neutral-darkest)";
            button.style.color = "var(--color-white)";
            emailInput.value = "";
            showToast("Thank you for subscribing!", "success");
          } else {
            const data = await response.json();
            button.textContent = data.detail || "Error!";
            button.style.backgroundColor = "#e74c3c";
            button.style.color = "#fff";
            showToast(data.detail || " Subscription failed. Please try again.", "error");
          }
        } catch (err) {
          button.textContent = "Network Error";
          button.style.backgroundColor = "#e74c3c";
          button.style.color = "#fff";
          showToast(" Network error. Please try again later.", "error");
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
