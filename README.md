# VisDrone Video Database with YOLO Detection

## Thông tin sinh viên thực hiện

**Sinh viên:** Nguyễn Thị Phương Thảo  
**MSSV:** N21DCCN078  
**Lớp:** D21CQCNHT01-N  
**Trường:** Học viện Công nghệ Bưu chính viễn thông TPHCM    
**Email:** ngphthao031028@gmail.com

## Thông tin đề bài

### Tên đề tài
**Xây dựng cơ sở dữ liệu video VisDrone với khả năng phát hiện đối tượng bằng YOLO**

### Mô tả đề bài
Xây dựng một hệ thống quản lý và phân tích video từ dataset VisDrone với các tính năng chính:
- Import và lưu trữ dataset VisDrone vào MongoDB
- Phát hiện đối tượng trong video sử dụng mô hình YOLO
- Xây dựng cấu trúc dữ liệu Segment Tree để tối ưu truy vấn
- Giao diện đồ họa để phát video và truy vấn dữ liệu
- Phát hiện đối tượng real-time trên video

### Yêu cầu kỹ thuật
- Sử dụng Python với các thư viện OpenCV, Tkinter, PyMongo
- Tích hợp mô hình YOLO (YOLOv5/YOLOv8)
- Cơ sở dữ liệu MongoDB
- Giao diện GUI thân thiện với người dùng

## Mô tả chi tiết bài đã làm

### 1. Kiến trúc hệ thống
Hệ thống được chia thành các module chính:

```
├── app/
│   ├── database_manager.py     # Quản lý cơ sở dữ liệu MongoDB
│   ├── video_processor.py      # Xử lý video với YOLO
│   ├── segment_tree.py        # Cấu trúc dữ liệu Segment Tree
│   ├── realtime_video_player.py # Phát video real-time
│   └── gui/                   # Giao diện người dùng
├── utils/
│   ├── config.py             # Cấu hình hệ thống
│   ├── visualization.py      # Hiển thị và vẽ bounding box
│   └── converters.py         # Chuyển đổi định dạng dữ liệu
└── requirements.txt          # Danh sách thư viện
```

### 2. Các tính năng đã triển khai

#### 2.1 Quản lý cơ sở dữ liệu
- **DatabaseManager**: Kết nối và quản lý MongoDB
- Import dataset VisDrone với cấu trúc:
  - `videos`: Thông tin video (tên, độ phân giải, FPS, số frame)
  - `frames`: Thông tin từng frame (đường dẫn ảnh, timestamp)  
  - `annotations`: Thông tin bounding box và class
  - `segment_trees`: Cấu trúc Segment Tree để tối ưu truy vấn

#### 2.2 Xử lý video với YOLO
- **VideoProcessor**: Xử lý video với mô hình YOLO
- Hỗ trợ các mô hình: YOLOv5n/s/m/l/x, YOLOv8n/s/m
- Mapping classes từ COCO (YOLO) sang VisDrone:
  - person → pedestrian
  - bicycle → bicycle  
  - car → car
  - motorcycle → motor
  - bus → bus
  - truck → truck
- Xử lý đa luồng để tăng tốc độ import

#### 2.3 Cấu trúc dữ liệu Segment Tree
- **FrameSegmentTree**: Tối ưu truy vấn theo khoảng frame
- Hỗ trợ truy vấn theo class đối tượng cụ thể
- Thời gian truy vấn O(log n) thay vì O(n)

#### 2.4 Giao diện người dùng
- **MainWindow**: Giao diện chính
- **VideoPlayer**: Phát video với annotations
- **QueryPanel**: Truy vấn frame theo điều kiện
- **RealTimeVideoPlayer**: Phát hiện đối tượng real-time
- **ImportDialog**: Import dataset VisDrone
- **VideoImportDialog**: Import video mới với YOLO

### 3. Các class và đối tượng chính

#### 3.1 VisDrone Classes
Hệ thống hỗ trợ 10 class đối tượng từ dataset VisDrone:
1. pedestrian (người đi bộ)
2. people (nhóm người)  
3. bicycle (xe đạp)
4. car (ô tô)
5. van (xe tải nhỏ)
6. truck (xe tải lớn)
7. tricycle (xe ba bánh)
8. awning-tricycle (xe ba bánh có mái)
9. bus (xe buýt)
10. motor (xe máy)

#### 3.2 YOLO Integration
- Tự động tải và cache các model YOLO
- Cấu hình confidence threshold và IoU threshold
- Frame skipping để tăng tốc xử lý
- Mapping tự động giữa COCO classes và VisDrone classes

### 4. Tối ưu hóa hiệu suất
- **Segment Tree**: Truy vấn nhanh theo khoảng frame
- **Indexing**: Tạo index cho MongoDB collections
- **Multi-threading**: Xử lý video song song
- **Frame caching**: Cache frame để hiển thị mượt mà

## Hướng dẫn cài đặt

### 1. Yêu cầu hệ thống
- Python 3.7+
- MongoDB 4.0+
- CUDA (tùy chọn, để tăng tốc YOLO)

### 2. Cài đặt dependencies
```bash
# Clone repository
git clone <repository-url>
cd visdrone-video-database

# Cài đặt Python packages
pip install -r requirements.txt

# Cài đặt YOLO (tùy chọn specific version)
pip install ultralytics
```

### 3. Cấu hình
Tạo file `.env` trong thư mục gốc:
```bash
mongodb+srv://thaolikesteris:peuBres9vqc0rC9l@videodb.41pmigo.mongodb.net/
```


## Hướng dẫn chạy chương trình

### 1. Khởi động MongoDB
Đảm bảo MongoDB đang chạy

### 2. Chạy ứng dụng chính
```bash
python run.py
```

### 3. Sử dụng giao diện

#### 3.1 Import Dataset VisDrone
1. Click menu **File → Import Dataset**
2. Chọn thư mục chứa dataset VisDrone (có folder `images` và `annotations`)
3. Nhập tên video và FPS
4. Click **Import** và đợi quá trình hoàn tất

#### 3.2 Import Video mới
1. Click menu **File → Import Video** 
2. Chọn file video (.mp4, .avi, .mov, .mkv)
3. Cấu hình YOLO settings:
   - Model: yolov8s, yolov8n, yolov8m, v.v.
   - Confidence threshold: 0.1-1.0
   - Frame skip: 1-30
4. Click **Import** để bắt đầu xử lý

#### 3.3 Phát video và truy vấn
1. Chọn video từ dropdown **Video Selection**
2. Sử dụng **Video Player**:
   - Play/Pause video
   - Điều chỉnh frame bằng slider
   - Thay đổi FPS phát
3. Sử dụng **Query Panel**:
   - Chọn Object Class (All hoặc class cụ thể)
   - Nhập khoảng frame (From - To)
   - Click **Run Query**
   - Double-click kết quả để jump tới frame

#### 3.4 Real-time Detection
1. Click button **Real-time Detection**
2. Chọn file video hoặc nhập đường dẫn
3. Điều chỉnh Confidence threshold
4. Click **Play** để xem detection real-time

## Ví dụ Demo

### Input 1: Import VisDrone Dataset
```
Dataset Structure:
VisDrone2019-DET-train/
├── images/
│   ├── 0000001_00000_d_0000001.jpg
│   ├── 0000001_00000_d_0000002.jpg
│   └── ...
└── annotations/
    ├── 0000001_00000_d_0000001.txt
    ├── 0000001_00000_d_0000002.txt
    └── ...

Annotation format (0000001_00000_d_0000001.txt):
641,397,67,63,1,1,0,0
729,413,69,72,1,1,0,0
912,468,103,67,1,5,0,0
```

**Output 1:** 
```
Connected to MongoDB at mongodb
Found 548 images in dataset
Found 1 videos in dataset
Processing frame 1 from 0000001_00000_d_0000001.jpg
Processing frame 2 from 0000001_00000_d_0000002.jpg
...
Processed 548 frames for video 0000001
Building segment trees...
Segment trees built and stored
Database indices created
Import completed successfully. Created 1 videos.
```

### Input 2: Import Video với YOLO
```
Video file: sample_traffic.mp4
YOLO Model: yolov8s
Confidence: 0.5
Frame Skip: 1
```

**Output 2:**
```
Loading YOLO model: yolov8s
YOLO model loaded successfully
Video properties: 1920x1080, 30 FPS, 900 frames, 30.00 seconds
Processing frame 1/900
Processing frame 2/900
...
Building segment trees for video 507f1f77bcf86cd799439011
Segment trees built successfully
Import completed successfully
```

### Input 3: Truy vấn Frame
```
Video: VisDrone Video 0000001 (548 frames)
Object Class: Car
Frame Range: 0 - 100
```

**Output 3:**
```
Query Results:
Found 23 frames with 67 objects
Frame 5: 2 Car, 1 Truck
Frame 12: 3 Car, 1 Bus  
Frame 28: 1 Car, 2 Motor
Frame 45: 4 Car, 1 Van
...
```

### Input 4: Real-time Detection
```
Video: traffic_video.mp4
Confidence: 0.6
```

**Output 4:**
Video player hiển thị:
- Bounding boxes màu sắc khác nhau cho từng class
- Labels hiển thị tên class và confidence score
- FPS counter: 25.3 FPS
- Frame counter: 156/900

### Performance Benchmarks

**Segment Tree Query Performance:**
- Linear search: 45ms (1000 frames)
- Segment Tree: 2ms (1000 frames)
- Speed improvement: ~22x faster

**YOLO Processing Speed:**
- YOLOv8n: ~35 FPS (GTX 1660)
- YOLOv8s: ~28 FPS (GTX 1660)  
- YOLOv8m: ~20 FPS (GTX 1660)

**Database Stats:**
- 10,000 frames imported: ~2 minutes
- MongoDB storage: ~500MB (with indexes)
- Query response time: <100ms average
