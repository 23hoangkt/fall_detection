# Hệ Thống Phát Hiện Ngã với Camera và Thông Báo SMS

## Tổng Quan
Dự án này triển khai hệ thống phát hiện ngã sử dụng thị giác máy tính, hỗ trợ nhiều tùy chọn đầu vào video (webcam/camera laptop, video file, hoặc camera IP qua RTSP) và gửi thông báo SMS khi phát hiện ngã thông qua API Android SMS Gateway ([capcom6/android-sms-gateway](https://github.com/capcom6/android-sms-gateway)). Hệ thống sử dụng các mô hình **YOLOv8** để phát hiện đối tượng, **SPPE (Single Person Pose Estimation)** để ước lượng tư thế, và **STGCN (Spatio-Temporal Graph Convolutional Networks)** để phân tích chuyển động nhằm phát hiện ngã chính xác.

Hệ thống phù hợp cho các ứng dụng y tế, chăm sóc người cao tuổi, và giám sát thông minh, đảm bảo thông báo kịp thời khi xảy ra sự cố ngã.

## Tính Năng
- **Phát Hiện Ngã Thời Gian Thực**: Kết hợp YOLOv8, SPPE, và STGCN để phát hiện ngã chính xác từ luồng video.
- **Hỗ Trợ Nhiều Nguồn Video**:
  - **Option 1**: Sử dụng webcam, camera laptop, hoặc video file (`main.py`).
  - **Option 2**: Sử dụng camera IP qua giao thức RTSP (`FallDetectWithIPcam.py`).
  - **Option 3**: Kết hợp camera IP và gửi thông báo SMS khi phát hiện ngã (`FallDetectWithSMS.py`).
- **Thông Báo SMS**: Gửi cảnh báo SMS đến các liên hệ khẩn cấp qua Android SMS Gateway API (option 3).
- **Khả Năng Mở Rộng**: Dễ dàng tích hợp thêm nguồn video hoặc phương thức thông báo khác.

## Yêu Cầu
Để chạy dự án, bạn cần đáp ứng các yêu cầu sau:

### Phần Cứng
- **Webcam/Camera Laptop** (cho Option 1): Webcam tích hợp hoặc gắn ngoài.
- **Camera IP** (cho Option 2 và 3): Camera hỗ trợ giao thức RTSP (camera hỗ trợ RTSP).
- **Máy Tính/Máy Chủ**: Hệ thống có đủ sức mạnh xử lý (khuyến nghị có GPU để tăng tốc độ suy luận).
- **Thiết Bị Android** (cho Option 3): Để chạy máy chủ Android SMS Gateway.

### Phần Mềm
- **Python 3.8+**: Cần thiết để chạy các tập lệnh phát hiện ngã.
- **Thư viện cần thiết**:
  - `opencv-python`: Để xử lý luồng video từ webcam, video file, hoặc RTSP.
  - `ultralytics`: Để sử dụng mô hình YOLOv8.
  - `requests` (cho Option 3): Để gửi yêu cầu HTTP đến API SMS Gateway.
  - Các thư viện bổ sung cho SPPE và STGCN (ví dụ: `torch`, `numpy`).
- **Android SMS Gateway** (cho Option 3): Cài đặt trên thiết bị Android để gửi SMS.

## Cài Đặt
Làm theo các bước sau để thiết lập dự án:

1. **Tải Repository**
   ```bash
   git clone https://github.com/23hoangkt/fall_detection.git
   cd fall_detection
   ```

2. **Cài Đặt Thư Viện Python**
   Tạo môi trường ảo và cài đặt các thư viện cần thiết:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Trên Windows: venv\Scripts\activate
   pip install opencv-python ultralytics requests torch numpy
   ```
   **Lưu ý**: Các thư viện bổ sung cho SPPE và STGCN có thể cần được cài đặt tùy theo triển khai cụ thể. Kiểm tra tài liệu của mô hình hoặc mã nguồn để biết thêm chi tiết.

3. **Cài Đặt Android SMS Gateway** (cho Option 3)
   - Làm theo hướng dẫn tại [capcom6/android-sms-gateway](https://github.com/capcom6/android-sms-gateway) để thiết lập máy chủ SMS Gateway trên thiết bị Android.
   - Ghi lại địa chỉ IP và cổng của máy chủ SMS Gateway (ví dụ: `http://<android-ip>:8080`).
   - Cấu hình khóa API và số điện thoại liên hệ khẩn cấp trong ứng dụng SMS Gateway.
   - Chạy `test_sms.py` để kiểm tra kết nối. Nếu trả về mã trạng thái `<202>`, kết nối hoạt động đúng.

4. **Cấu Hình Camera RTSP** (cho Option 2 và 3)
   - Lấy URL RTSP của camera IP (ví dụ: `rtsp://<camera-ip>:554/stream`).
   - Cập nhật URL RTSP trực tiếp trong mã nguồn của `FallDetectWithIPcam.py` hoặc `FallDetectWithSMS.py` (thường được hard-code trong tập lệnh).

5. **Tải Mô Hình**
   - Các mô hình đã được huấn luyện sẵn (YOLOv8, SPPE, STGCN) có thể được tải từ [Google Drive](https://drive.google.com/drive/u/0/folders/1lrTI56k9QiIfMJhG9kzNjBzJh98KCIIO).
   - Đặt các tệp mô hình (`yolov8m.pt` cho YOLOv8, tệp SPPE, và tệp STGCN) vào thư mục `models/` trong dự án.

## Cấu Hình
Do dự án không sử dụng tệp cấu hình, các tham số như URL RTSP, URL SMS Gateway, khóa API, và số liên hệ khẩn cấp được hard-code trực tiếp trong các tập lệnh (`FallDetectWithIPcam.py` hoặc `FallDetectWithSMS.py`). Để cấu hình:
- **Nguồn Video**:
  - Option 1: Chỉnh sửa `main.py` để sử dụng webcam (ID `0` hoặc `1`) hoặc đường dẫn video file (ví dụ: `path/to/video.mp4`).
  - Option 2 & 3: Chỉnh sửa `FallDetectWithIPcam.py` hoặc `FallDetectWithSMS.py` để cập nhật URL RTSP (ví dụ: `rtsp://<camera-ip>:554/stream`).
- **SMS Gateway API** (cho Option 3): Cập nhật URL điểm cuối (ví dụ: `http://<android-ip>:8080/send_sms`), khóa API, và số liên hệ khẩn cấp trong `FallDetectWithSMS.py`.
- **Ngưỡng Phát Hiện Ngã**: Độ tin cậy keypoint và ngưỡng góc được hard-code trong mã nguồn. Điều chỉnh các giá trị này trong tập lệnh nếu cần.

Ví dụ cấu hình trong `FallDetectWithSMS.py`:
```python
RTSP_URL = "rtsp://<camera-ip>:554/stream"
SMS_GATEWAY_URL = "http://<android-ip>:8080/send_sms"
SMS_API_KEY = "<your-api-key>"
EMERGENCY_CONTACTS = ["+84xxxxxxxxx", "+84xxxxxxxxx"]
FALL_CONFIDENCE_THRESHOLD = 0.7
FALL_ANGLE_THRESHOLD = 45.0
```

## Sử Dụng
Hệ thống hỗ trợ ba tùy chọn chạy với các tập lệnh riêng biệt:

1. **Option 1: Chạy với Webcam/Camera Laptop hoặc Video File**
   ```bash
   python main.py
   ```
   - Sử dụng webcam (ID `0` hoặc `1`) hoặc video file để phát hiện ngã.
   - Chỉnh sửa `main.py` để thay đổi nguồn video nếu cần (ví dụ: `cv2.VideoCapture(0)` hoặc `cv2.VideoCapture('path/to/video.mp4')`).
   - Không gửi thông báo SMS.

2. **Option 2: Chạy với Camera IP**
   ```bash
   python FallDetectWithIPcam.py
   ```
   - Sử dụng camera IP qua RTSP để phát hiện ngã.
   - Cập nhật URL RTSP trong `FallDetectWithIPcam.py` trước khi chạy.
   - Không gửi thông báo SMS.

3. **Option 3: Chạy với Camera IP và Gửi SMS**
   ```bash
   python FallDetectWithSMS.py
   ```
   - Sử dụng camera IP qua RTSP để phát hiện ngã.
   - Gửi thông báo SMS đến các liên hệ khẩn cấp khi phát hiện ngã.
   - Cập nhật URL RTSP, URL SMS Gateway, khóa API, và số liên hệ trong `FallDetectWithSMS.py` trước khi chạy.

### Quy Trình Hoạt Động
- Hệ thống sử dụng YOLOv8 để phát hiện người, SPPE để ước lượng tư thế, và STGCN để phân tích chuyển động.
- Khi phát hiện ngã (dựa trên ngưỡng độ tin cậy và góc), hệ thống (ở Option 3) gửi yêu cầu HTTP đến SMS Gateway để gửi tin nhắn SMS.

### Kiểm Tra Thông Báo (Option 3)
- Khi phát hiện ngã, tin nhắn SMS sẽ được gửi đến các số liên hệ khẩn cấp.
- Chạy `test_sms.py` để xác nhận kết nối SMS Gateway (mã trạng thái `<202>` là thành công).
- Kiểm tra thiết bị Android để đảm bảo tin nhắn được gửi.

## Khắc Phục Sự Cố
- **Lỗi `TimeoutError` khi không lấy được khung hình**:
  - Kiểm tra kết nối camera (webcam hoặc RTSP).
  - Đảm bảo URL RTSP đúng và camera hoạt động.
  - Tăng thời gian chờ trong mã nguồn (nếu có) hoặc thử dùng video file để kiểm tra.
- **Lỗi `TypeError` liên quan đến định dạng dữ liệu**:
  - Đảm bảo đầu ra của YOLOv8 (`detected`) được chuyển đổi thành `torch.Tensor` trước khi xử lý bởi SPPE hoặc STGCN.
  - Kiểm tra mã nguồn trong `main.py`, `FallDetectWithIPcam.py`, hoặc `FallDetectWithSMS.py` để sửa lỗi truy cập danh sách.
- **Không phát hiện được người**:
  - Kiểm tra tệp mô hình `yolov8m.pt` được tải đúng từ [Google Drive](https://drive.google.com/drive/u/0/folders/1lrTI56k9QiIfMJhG9kzNjBzJh98KCIIO).
  - Giảm ngưỡng độ tin cậy (`confidence`) trong mã nguồn (ví dụ: từ 0.7 xuống 0.5).
  - Tăng kích thước đầu vào video (nếu có) để cải thiện độ chính xác.

## Đóng Góp
- Để đóng góp, vui lòng fork dự án, thực hiện thay đổi và gửi pull request.
- Báo cáo lỗi hoặc đề xuất tính năng mới qua tab Issues trên GitHub.

## Giấy Phép
Dự án này được cấp phép theo [MIT License](LICENSE).

## Liên Hệ
Nếu có thắc mắc, liên hệ qua [email@example.com](mailto:email@example.com) hoặc mở issue trên GitHub.
