from flask import Flask, render_template, Response, jsonify
import cv2
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from models import db, VehicleRecord, TrafficSnapshot
import math
from werkzeug.utils import secure_filename
from flask import request
import os
import requests
import shutil

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': '没有选择视频文件'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': '没有选择视频文件'}), 400
    
    if file and allowed_file(file.filename):
        filename = 'traffic.mp4'  # 使用固定文件名
        file_path = os.path.join(os.path.dirname(__file__), filename)
        file.save(file_path)
        return jsonify({'message': '视频上传成功'}), 200
    
    return jsonify({'error': '不支持的文件类型'}), 400

app = Flask(__name__)

# 配置数据库
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(os.path.dirname(__file__), "traffic.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

def initialize_system():
    # 创建必要的目录
    os.makedirs('templates', exist_ok=True)
    
    # 检查模型文件是否存在
    model_path = os.path.join(os.path.dirname(__file__), 'yolov5s.pt')
    if not os.path.exists(model_path):
        print("错误：找不到YOLOv5模型文件！")
        print("请确保yolov5s.pt文件已放置在程序目录下。")
        exit(1)
    
    # 检查视频文件是否存在
    video_path = os.path.join(os.path.dirname(__file__), 'traffic.mp4')
    if not os.path.exists(video_path):
        print("提示：未找到traffic.mp4视频文件")
        print("请上传视频文件以开始检测。")
        # 创建一个空的视频文件作为占位符
        open(video_path, 'w').close()

# 初始化系统
initialize_system()

# 加载YOLOv5模型
from ultralytics import YOLO
model = YOLO('yolov5s.pt')
model.classes = [2, 3, 5, 7]  # 只检测车辆相关类别（car, motorcycle, bus, truck）

class VehicleDetector:
    def __init__(self):
        self.vehicle_counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
        self.last_count_time = datetime.now()
        self.last_positions = {}
        self.speed_threshold = 60  # km/h
        self.class_map = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

    def detect_vehicles(self, frame):
        # 执行目标检测
        results = model(frame)
        detections = results[0].boxes.data.cpu().numpy()
        
        # 清空当前计数
        current_counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
        violations = 0
        current_time = datetime.now()
        
        # 处理每个检测结果
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            vehicle_type = self.class_map[int(cls)]
            current_counts[vehicle_type] += 1
            
            # 计算车速
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            vehicle_id = f"{vehicle_type}_{int(x1)}_{int(y1)}"
            
            if vehicle_id in self.last_positions:
                last_pos, last_time = self.last_positions[vehicle_id]
                # 计算位移（像素）
                displacement = math.sqrt(
                    (center[0] - last_pos[0])**2 + 
                    (center[1] - last_pos[1])**2
                )
                # 计算时间差（秒）
                time_diff = (current_time - last_time).total_seconds()
                if time_diff > 0:
                    # 假设1像素=0.1米（需要根据实际情况调整）
                    speed = (displacement * 0.1 / time_diff) * 3.6  # 转换为km/h
                    if speed > self.speed_threshold:
                        violations += 1
                        cv2.putText(frame, f"Speed: {int(speed)}km/h", 
                                    (int(x1), int(y1)-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            self.last_positions[vehicle_id] = (center, current_time)
        
        # 更新统计数据
        if (current_time - self.last_count_time).seconds >= 3600:
            self.vehicle_counts = current_counts.copy()
            self.last_count_time = current_time
            
            # 保存到数据库
            snapshot = TrafficSnapshot(
                total_vehicles=sum(current_counts.values()),
                cars=current_counts['car'],
                motorcycles=current_counts['motorcycle'],
                buses=current_counts['bus'],
                trucks=current_counts['truck'],
                traffic_status='congested' if sum(current_counts.values()) > 20 else 'normal'
            )
            db.session.add(snapshot)
            db.session.commit()
        
        # 在帧上绘制统计信息
        y_offset = 30
        for vehicle_type, count in current_counts.items():
            cv2.putText(frame, f'{vehicle_type}: {count}', 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 0), 2)
            y_offset += 25
        
        cv2.putText(frame, f'Violations: {violations}', 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 0, 255), 2)
        
        return frame

detector = VehicleDetector()

def generate_frames():
    video_path = os.path.join(os.path.dirname(__file__), 'traffic.mp4')
    camera = cv2.VideoCapture(video_path)  # 使用视频文件作为输入源
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # 处理帧并进行车辆检测
        processed_frame = detector.detect_vehicles(frame)
        
        # 将帧转换为字节流
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/vehicle_stats')
def vehicle_stats():
    # 从数据库获取最新的车辆记录
    latest_snapshot = TrafficSnapshot.query.order_by(TrafficSnapshot.timestamp.desc()).first()
    if latest_snapshot:
        return jsonify({
            'car': latest_snapshot.cars,
            'motorcycle': latest_snapshot.motorcycles,
            'bus': latest_snapshot.buses,
            'truck': latest_snapshot.trucks,
            'violations': detector.violations if hasattr(detector, 'violations') else 0
        })
    return jsonify({
        'car': 0,
        'motorcycle': 0,
        'bus': 0,
        'truck': 0,
        'violations': 0
    })

@app.route('/api/traffic_status')
def traffic_status():
    # 从数据库获取最新的交通状况
    latest_snapshot = TrafficSnapshot.query.order_by(TrafficSnapshot.timestamp.desc()).first()
    if latest_snapshot:
        return jsonify({
            'status': latest_snapshot.traffic_status,
            'total_vehicles': latest_snapshot.total_vehicles
        })
    return jsonify({
        'status': 'normal',
        'total_vehicles': 0
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)