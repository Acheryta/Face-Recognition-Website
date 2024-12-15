Thư viện cần thiết: requirement.txt

Các folder/file gồm:
-static: chứa file css và folder face chứa ảnh được đăng ký và ảnh cần để tham chiếu khi login, folder unknown chứa ảnh người dung khi đăng ký Faceid và ảnh khi tìm kiếm khuôn mặt 
-templates: chứa page html
-application.py : mã nguồn chính
-helpers.py: thông tin tham khảo
-data.db: chứa database
-face_encodings.pkl: chứa đặc trưng khuôn mặt có trong folder face
-output.log: log của lần chạy trước đó 

Hướng dẫn sử dụng : có thể dùng cửa sổ cmd hoặc trình biên dịch và chạy dòng lệnh
py application.py hoặc
py application.py > output.log 2>&1 (nếu cần xuất file log)

lưu ý: nên chạy với quyền admin nếu cần thiết. Tạo folder static/face và static/face/unknown
