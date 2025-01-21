from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class VehicleRecord(db.Model):
    __tablename__ = 'vehicle_records'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    vehicle_type = db.Column(db.String(20))  # car, motorcycle, bus, truck
    count = db.Column(db.Integer)
    average_speed = db.Column(db.Float, nullable=True)
    violations = db.Column(db.Integer, default=0)
    
    def __repr__(self):
        return f'<VehicleRecord {self.vehicle_type} at {self.timestamp}>'

class TrafficSnapshot(db.Model):
    __tablename__ = 'traffic_snapshots'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    total_vehicles = db.Column(db.Integer)
    cars = db.Column(db.Integer)
    motorcycles = db.Column(db.Integer)
    buses = db.Column(db.Integer)
    trucks = db.Column(db.Integer)
    traffic_status = db.Column(db.String(20))  # normal, congested, heavy
    
    def __repr__(self):
        return f'<TrafficSnapshot {self.timestamp}>'