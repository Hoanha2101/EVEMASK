# EVEMASK React App

Đây là phiên bản React của website EVEMASK, được chuyển đổi từ HTML thuần sang React với TypeScript.

## Cài đặt và Chạy Project

### Bước 1: Cài đặt Dependencies
```bash
cd Frontend
npm install
```

### Bước 2: Chạy Development Server
```bash
npm start
```

Website sẽ chạy tại: http://localhost:3000

### Bước 3: Build cho Production
```bash
npm run build
```

## Cấu Trúc Project

```
Frontend/
├── public/
│   ├── assets/          # CSS, images, videos từ HTML cũ
│   └── index.html
├── src/
│   ├── components/      # Các React components
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
│   ├── index.tsx        # Entry point
│   └── index.css
├── package.json
├── tsconfig.json
└── README.md
```

## Các Tính Năng Đã Chuyển Đổi

- ✅ **Navbar với scroll effect**
- ✅ **Hero Section với animation counter**
- ✅ **Challenges Section với icons**
- ✅ **Solution Section với workflow comparison**
- ✅ **Why Choose Section với bubbles**
- ✅ **Team Section**
- ✅ **Demo Section với video**
- ✅ **Contact Section với map**
- ✅ **Footer với newsletter**
- ✅ **Responsive design giữ nguyên từ CSS cũ**

## Các Cải Tiến

1. **Component-based Architecture**: Mỗi section được tách thành component riêng
2. **TypeScript Support**: Type safety cho toàn bộ codebase
3. **React Hooks**: Sử dụng useState, useEffect cho interactivity
4. **Modern Development**: Hot reload, build optimization
5. **Maintainable Code**: Easier to maintain và extend

## Scripts Có Sẵn

- `npm start` - Chạy development server
- `npm build` - Build cho production
- `npm test` - Chạy tests
- `npm eject` - Eject cấu hình (không khuyến khích)

## Lưu Ý

- Tất cả CSS styles từ HTML cũ đã được giữ nguyên
- Assets (images, videos) đã được copy sang thư mục public
- Giao diện sẽ giống hệt phiên bản HTML
- Responsive design vẫn hoạt động như cũ
