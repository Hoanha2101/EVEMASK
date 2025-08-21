# EVEMASK React App

This is the React version of the EVEMASK website, converted from pure HTML to React with TypeScript.

## Installation and Running the Project

### Step 1: Install Dependencies
```bash
cd Frontend
npm install
```

### Step 2: Run Development Server
```bash
npm start
```

The website will run at: http://localhost:3000

### Step 3: Build for Production
```bash
npm run build
```

## Project Structure

```
Frontend/
├── public/
│   ├── assets/          # CSS, images, videos from old HTML
│   └── index.html
├── src/
│   ├── components/      # React components
│   │   ├── Navbar/
│   │   ├── HeroSection/
│   │   ├── ChallengesSection/
│   │   ├── SolutionSection/
│   │   ├── WhyChooseSection/
│   │   ├── TeamSection/
│   │   ├── DemoSection/
│   │   ├── ContactSection/
│   │   └── Footer/
│   ├── App.tsx          # Main App component
│   ├── App.css
├── package.json
├── tsconfig.json
└── README.md
```

## Converted Features

- ✅ **Navbar with scroll effect**
- ✅ **Hero Section with animation counter**
- ✅ **Challenges Section with icons**
- ✅ **Solution Section with workflow comparison**
- ✅ **Why Choose Section with bubbles**
- ✅ **Team Section**
- ✅ **Demo Section with video**
- ✅ **Contact Section with map**
- ✅ **Footer with newsletter**
- ✅ **Responsive design maintained from original CSS**

## Improvements

1. **Component-based Architecture**: Each section separated into individual components
2. **TypeScript Support**: Type safety for entire codebase
3. **React Hooks**: Using useState, useEffect for interactivity
4. **Modern Development**: Hot reload, build optimization
5. **Maintainable Code**: Easier to maintain and extend

## Available Scripts

- `npm start` - Run development server
- `npm build` - Build for production
- `npm test` - Run tests
- `npm eject` - Eject configuration (not recommended)

## Notes

- All CSS styles from original HTML have been preserved
- Assets (images, videos) have been copied to public folder
- Interface will look identical to HTML version
- Responsive design still works as before
