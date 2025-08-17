# Hướng dẫn Setup Supabase cho EVEMASK Backend

## 1. Tạo Project Supabase

1. Truy cập [Supabase](https://supabase.com/)
2. Đăng nhập và tạo project mới
3. Chọn region gần nhất (Singapore hoặc Tokyo cho VN)
4. Chờ project được khởi tạo

## 2. Tạo Table Subscribers

Truy cập SQL Editor trong Supabase Dashboard và chạy script sau:

```sql
-- Tạo table subscribers
CREATE TABLE subscribers (
  id BIGSERIAL PRIMARY KEY,
  email VARCHAR(255) UNIQUE NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  status VARCHAR(50) DEFAULT 'active',
  metadata JSONB DEFAULT '{}'::jsonb
);

-- Thêm index cho email để tìm kiếm nhanh hơn
CREATE INDEX idx_subscribers_email ON subscribers(email);

-- Thêm index cho created_at để sort theo thời gian
CREATE INDEX idx_subscribers_created_at ON subscribers(created_at);

-- Enable Row Level Security (RLS)
ALTER TABLE subscribers ENABLE ROW LEVEL SECURITY;

-- Tạo policy để cho phép insert và select (cho anonymous users)
CREATE POLICY "Allow anonymous insert" ON subscribers 
FOR INSERT TO anon WITH CHECK (true);

CREATE POLICY "Allow anonymous select" ON subscribers 
FOR SELECT TO anon USING (true);

-- Optional: Tạo view để thống kê
CREATE VIEW subscriber_stats AS
SELECT 
  COUNT(*) as total_subscribers,
  COUNT(*) FILTER (WHERE status = 'active') as active_subscribers,
  COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '1 day') as daily_signups,
  COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '7 days') as weekly_signups,
  COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '30 days') as monthly_signups
FROM subscribers;
```

## 3. Lấy API Keys

1. Trong Supabase Dashboard, đi tới **Settings** > **API**
2. Copy các giá trị sau:

### Project URL:
- Ở đầu trang API, copy **Project URL**
- Có dạng: `https://your-project-id.supabase.co`
- Đây là giá trị cho **SUPABASE_URL**

### Publishable Key (Anon Key):
- Trong phần **"Publishable key"**
- Copy key bắt đầu bằng `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`
- Đây là giá trị cho **SUPABASE_ANON_KEY**

### Ví dụ:
```env
SUPABASE_URL=https://abcdefghijklmnop.supabase.co
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFiY2RlZmdoaWprbG1ub3AiLCJyb2xlIjoiYW5vbiIsImlhdCI6MTY0MTc2NDgwMCwiZXhwIjoxOTU3MzQwODAwfQ.example-key-here
```

**Lưu ý:** 
- Publishable key (anon) an toàn để sử dụng ở frontend
- Secret key chỉ dùng cho backend và cần bảo mật

## 4. Cấu hình Environment Variables

Thêm các biến môi trường sau vào file `.env` hoặc cấu hình hosting:

```env
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your-anon-key-here
```

## 5. Test Connection

Sau khi setup xong, bạn có thể test connection bằng cách:

1. Chạy server: `python main.py`
2. Truy cập: `http://localhost:7860/api/debug/supabase-status`
3. Kiểm tra response để đảm bảo kết nối thành công

## 6. Migration từ JSON file

Nếu bạn có dữ liệu trong `subscribers.json`, bạn có thể migrate bằng script sau:

```python
import json
import os
from supabase import create_client

# Load existing subscribers
with open('subscribers.json', 'r') as f:
    subscribers = json.load(f)

# Initialize Supabase
supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_ANON_KEY')
)

# Migrate data
for subscriber in subscribers:
    try:
        # Convert timestamp format if needed
        created_at = subscriber.get('timestamp', subscriber.get('created_at'))
        
        supabase.table('subscribers').insert({
            'email': subscriber['email'],
            'created_at': created_at,
            'status': 'active'
        }).execute()
        
        print(f"Migrated: {subscriber['email']}")
    except Exception as e:
        print(f"Error migrating {subscriber['email']}: {e}")
```

## 7. Endpoints mới có sẵn

- `GET /api/debug/supabase-status` - Kiểm tra trạng thái Supabase
- `GET /api/subscribers/count` - Lấy số lượng subscribers
- `GET /api/subscribers/list?limit=50&offset=0` - Lấy danh sách subscribers với pagination

## 8. Fallback Mechanism

Hệ thống được thiết kế với fallback mechanism:
- Nếu Supabase không khả dụng, sẽ tự động fallback về JSON file
- Đảm bảo service luôn hoạt động ổn định

## 9. Monitoring và Dashboard

Trong Supabase Dashboard bạn có thể:
- Xem real-time subscribers data
- Monitor API usage
- Set up alerts
- Export data
