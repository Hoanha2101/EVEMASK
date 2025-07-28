# EVEMASK Frontend

A modern, responsive web application showcasing EVEMASK's innovative facial anonymization technology and research achievements.

## 🌟 Overview

The EVEMASK Frontend is a sophisticated, mobile-first web application designed to present EVEMASK's cutting-edge facial anonymization technology. Built with modern web technologies, it provides an engaging user experience while maintaining professional standards and accessibility.

## ✨ Key Features

- **Responsive Design** - Seamless experience across all devices and screen sizes
- **Interactive Newsletter** - Integrated subscription system with backend API
- **Modern UI/UX** - Clean, professional interface with smooth animations
- **Performance Optimized** - Fast loading times with optimized assets
- **SEO Ready** - Semantic HTML structure with meta tags optimization
- **Accessible** - WCAG 2.1 compliant design for all users
- **Cross-Browser Compatible** - Tested on Chrome, Firefox, Safari, and Edge

## 🛠️ Technology Stack

- **HTML5** - Semantic markup and modern web standards
- **CSS3** - Advanced styling with Flexbox/Grid and animations
- **Vanilla JavaScript** - ES6+ features for interactive functionality
- **Google Fonts** - Professional typography
- **Responsive Images** - Optimized media delivery
- **Modern CSS Features** - CSS Custom Properties, Grid, Flexbox

## 📁 Project Structure


```
Front-End/
├── index.html                # Main landing page
├── contact.html              # Contact page
├── css/
│   └── styles.css            # Main stylesheet
├── js/
│   └── script.js             # Interactive functionality
├── images/                   # Image assets
│   ├── EveMask-logo.png
│   ├── hero.png
│   ├── bgr.png
│   ├── Evemask_Workflow.png
│   ├── Existed_Workflow.png
│   ├── gilariver_evemask.png
│   ├── baibao1.png
│   ├── baibao2.png
│   ├── img_evenmask.png
│   ├── khaihoan.jpg
│   ├── thunguyen.jpg
│   ├── trongnghia.jpg
│   ├── chaubui.jpg
│   └── team/                 # Team member photos
│       ├── chaubui.jpg
│       ├── khaihoan.jpg
│       ├── thunguyen.jpg
│       └── trongnghia.jpg
├── videos/                   # Video content
│   ├── highlights.mp4
│   └── video_evenmask.mp4
├── .gitignore                # Git ignore rules
└── README.md                 # This documentation
```

## 🚀 Getting Started

### Prerequisites
- Modern web browser (Chrome 90+, Firefox 88+, Safari 14+, Edge 90+)
- Local web server (recommended for development)
- EVEMASK Backend API running (for newsletter functionality)

### Quick Setup

#### Option 1: Live Server (Recommended for Development)
```powershell
# Using VS Code Live Server extension
# 1. Install "Live Server" extension in VS Code
# 2. Right-click on index.html
# 3. Select "Open with Live Server"
```

#### Option 2: Python HTTP Server
```powershell

# Navigate to the Front-End directory
cd Front-End

# Python 3
python -m http.server 3000

# Access at http://localhost:3000
```

#### Option 3: Node.js HTTP Server
```powershell
# Install serve globally
npm install -g serve


# Navigate to the Front-End directory
cd Front-End


# Start server
serve -s . -l 3000

# Access at http://localhost:3000
```

## ⚙️ Configuration

### Backend API Integration

Update the API endpoint in `js/script.js`:

```javascript
// Development
const API_BASE_URL = "http://localhost:8000";

// Production
const API_BASE_URL = "https://your-api-domain.com";
```

### Environment-Specific Settings
Create environment-specific configurations:

```javascript
// config.js (create this file)
const CONFIG = {
  development: {
    API_BASE_URL: "http://localhost:8000",
    DEBUG: true
  },
  production: {
    API_BASE_URL: "https://api.evemask.com",
    DEBUG: false
  }
};
```

## 📱 Features Deep Dive

### Navigation System
- **Smooth Scrolling** - Seamless navigation between sections
- **Mobile Menu** - Responsive hamburger menu for mobile devices
- **Active States** - Visual feedback for current section
- **Keyboard Navigation** - Full keyboard accessibility support

### Newsletter Integration
- **Real-time Validation** - Client-side email validation
- **API Integration** - Seamless backend communication
- **Loading States** - User feedback during submission
- **Error Handling** - Comprehensive error message display
- **Success Confirmation** - Visual confirmation of successful subscription

### Interactive Elements
- **Hover Effects** - Subtle animations on interactive elements
- **Form Validation** - Real-time input validation
- **Smooth Transitions** - CSS-based animations for better UX
- **Lazy Loading** - Optimized image loading for performance

### Responsive Design
- **Mobile-First** - Designed primarily for mobile devices
- **Breakpoints**:
  - Mobile: < 768px
  - Tablet: 768px - 1024px
  - Desktop: > 1024px
- **Flexible Grid** - CSS Grid and Flexbox for layout
- **Scalable Typography** - Responsive font sizes using clamp()

## 🎨 Design System

### Color Palette
```css
:root {
  --primary-color: #007bff;
  --secondary-color: #6c757d;
  --success-color: #28a745;
  --danger-color: #dc3545;
  --warning-color: #ffc107;
  --info-color: #17a2b8;
  --light-color: #f8f9fa;
  --dark-color: #343a40;
}
```

### Typography
- **Primary Font**: Google Fonts integration
- **Font Weights**: 300, 400, 600, 700
- **Responsive Scaling**: Using CSS clamp() for fluid typography
- **Line Heights**: Optimized for readability

### Spacing System
- **Base Unit**: 8px
- **Scale**: 8px, 16px, 24px, 32px, 48px, 64px
- **Consistent Margins**: Using CSS custom properties

## 🔧 Development

### Code Standards
- **HTML**: Semantic HTML5 elements
- **CSS**: BEM methodology for class naming
- **JavaScript**: ES6+ features with backward compatibility
- **Comments**: Comprehensive code documentation
- **Indentation**: 2 spaces for HTML/CSS, 2 spaces for JavaScript

### Performance Optimization
- **Image Optimization**: WebP format with fallbacks
- **CSS Optimization**: Critical CSS inlined
- **JavaScript**: Minified for production
- **Lazy Loading**: Images loaded on demand
- **Caching**: Proper cache headers for static assets

### Browser Testing
Regular testing performed on:
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+
- ✅ Mobile Chrome (Android)
- ✅ Mobile Safari (iOS)

## 📊 Performance Metrics

### Lighthouse Scores (Target)
- **Performance**: 95+
- **Accessibility**: 95+
- **Best Practices**: 95+
- **SEO**: 95+

### Core Web Vitals
- **LCP (Largest Contentful Paint)**: < 2.5s
- **FID (First Input Delay)**: < 100ms
- **CLS (Cumulative Layout Shift)**: < 0.1

### Bundle Size
- **HTML**: ~15KB (gzipped)
- **CSS**: ~25KB (gzipped)
- **JavaScript**: ~10KB (gzipped)
- **Images**: Optimized per usage

## 🧪 Testing

### Manual Testing Checklist
- [ ] All navigation links work correctly
- [ ] Newsletter form submits successfully
- [ ] Responsive design works on all screen sizes
- [ ] Images load properly
- [ ] Videos play correctly
- [ ] All interactive elements provide feedback
- [ ] Form validation works as expected
- [ ] Error handling displays appropriate messages

### Cross-Browser Testing
```powershell
# Using BrowserStack or similar service
# Test on multiple browsers and devices
# Document any browser-specific issues
```

## 🚀 Deployment

### Static Hosting (Recommended)
#### Netlify
```powershell
# Install Netlify CLI
npm install -g netlify-cli

# Deploy
cd Front-End/Evemask
netlify deploy --prod --dir .
```

#### Vercel
```powershell
# Install Vercel CLI
npm install -g vercel

# Deploy
cd Front-End/Evemask
vercel --prod
```

#### GitHub Pages
```powershell
# 1. Push code to GitHub repository
# 2. Go to repository Settings
# 3. Enable GitHub Pages from source branch
# 4. Set custom domain if needed
```

### CDN Integration
```html
<!-- Add to <head> for production -->
<link rel="dns-prefetch" href="//fonts.googleapis.com">
<link rel="preconnect" href="//fonts.gstatic.com" crossorigin>
<link rel="preload" href="/css/styles.css" as="style">
```

### Production Checklist
- [ ] Minify CSS and JavaScript files
- [ ] Optimize all images (WebP format)
- [ ] Set up proper caching headers
- [ ] Configure CDN for static assets
- [ ] Add security headers
- [ ] Set up SSL certificate
- [ ] Configure custom domain
- [ ] Test all functionality in production environment
- [ ] Set up monitoring and analytics

## 📈 SEO Optimization

### Meta Tags
```html
<meta name="description" content="EVEMASK - Advanced facial anonymization technology for privacy protection">
<meta name="keywords" content="facial anonymization, privacy, AI, machine learning, EVEMASK">
<meta property="og:title" content="EVEMASK - Facial Anonymization Technology">
<meta property="og:description" content="Innovative facial anonymization solution protecting privacy while maintaining data utility">
<meta property="og:image" content="https://evemask.com/assets/images/og-image.png">
```

### Structured Data
```json
{
  "@context": "https://schema.org",
  "@type": "Organization",
  "name": "EVEMASK",
  "description": "Facial anonymization technology company",
  "url": "https://evemask.com"
}
```

## 🔒 Security

### Content Security Policy
```html
<meta http-equiv="Content-Security-Policy" 
      content="default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' fonts.googleapis.com; font-src fonts.gstatic.com;">
```

### Best Practices
- Input sanitization for all form fields
- HTTPS enforcement
- Secure headers implementation
- XSS protection measures

## 🤝 Contributing

### Getting Started
1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes** following code standards
4. **Test thoroughly** across browsers and devices
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open Pull Request**

### Development Workflow
```powershell
# 1. Clone repository
git clone https://github.com/Hoanha2101/EVEMASK.git
cd EVEMASK/Front-End

# 2. Create feature branch
git checkout -b feature/new-feature

# 3. Make changes and test
# 4. Commit and push
git add .
git commit -m "Add new feature"
git push origin feature/new-feature

# 5. Create Pull Request on GitHub
```

## 📞 Support & Contact

- **Email**: evemask.ai@gmail.com
- **Team**: EVEMASK Frontend Development Team
- **Repository**: https://github.com/Hoanha2101/EVEMASK
- **Issues**: https://github.com/Hoanha2101/EVEMASK/issues

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Version**: 2.0.0  
**Last Updated**: January 2025  
**Browser Compatibility**: Modern browsers (ES6+ support required)
