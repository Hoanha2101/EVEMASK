// EVEMASK Website Interactive Features

// Smooth scrolling for navigation links
document.addEventListener("DOMContentLoaded", function () {
  // Smooth scroll for anchor links
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
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      const emailInput = this.querySelector('input[type="email"]');
      if (emailInput && emailInput.value) {
        // Simulate successful subscription
        const button = this.querySelector("button");
        const originalText = button.textContent;
        button.textContent = "Subscribed!";
        button.style.backgroundColor = "var(--color-neutral-darkest)";
        button.style.color = "var(--color-white)";

        setTimeout(() => {
          button.textContent = originalText;
          button.style.backgroundColor = "";
          button.style.color = "";
          emailInput.value = "";
        }, 2000);
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
          alert(
            "Thank you for your message! We will get back to you within 24 hours.",
          );
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
  window.addEventListener("scroll", function () {
    if (window.pageYOffset > 300) {
      scrollTopButton.style.opacity = "1";
      scrollTopButton.style.visibility = "visible";
    } else {
      scrollTopButton.style.opacity = "0";
      scrollTopButton.style.visibility = "hidden";
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
