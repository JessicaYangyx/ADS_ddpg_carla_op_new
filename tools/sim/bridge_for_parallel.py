#!/usr/bin/env python3
import argparse
import carla
import math
import numpy as np
import time
import threading
from cereal import log
from multiprocessing import Process, Queue, Value, Array, Pipe
from typing import Any
import cereal.messaging as messaging
from common.params import Params
from common.numpy_fast import clip
from common.realtime import Ratekeeper, DT_DMON
from lib.can import can_function
from selfdrive.car.honda.values import CruiseButtons
from selfdrive.test.helpers import set_params_enabled
import sys
from sklearn.cluster import DBSCAN

W, H = 1164, 874
REPEAT_COUNTER = 5
STEER_RATIO = 15.
vEgo = 60
EXIT_SIGNAL = False

pm = messaging.PubMaster(['roadCameraState', 'sensorEvents', 'can', "gpsLocationExternal"])
sm = messaging.SubMaster(['carControl', 'controlsState', 'radarState', 'modelV2'])


class VehicleState:
  def __init__(self):
    self.speed = 0
    self.angle = 0
    self.bearing_deg = 0.0
    self.vel = carla.Vector3D()
    self.cruise_button = 0
    self.is_engaged = False


frame_id = 0
frameIdx = 0
def cam_callback(image):
  global frame_id
  img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
  img = np.reshape(img, (H, W, 4))
  img = img[:, :, [0, 1, 2]].copy()

  dat = messaging.new_message('roadCameraState')
  dat.roadCameraState = {
    "frameId": image.frame,
    "image": img.tobytes(),
    "transform": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
  }
  pm.send('roadCameraState', dat)
  frame_id += 1


def imu_callback(imu, vehicle_state):
  vehicle_state.bearing_deg = math.degrees(imu.compass)
  dat = messaging.new_message('sensorEvents', 2)
  dat.sensorEvents[0].sensor = 4
  dat.sensorEvents[0].type = 0x10
  dat.sensorEvents[0].init('acceleration')
  dat.sensorEvents[0].acceleration.v = [imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z]
  dat.sensorEvents[1].sensor = 5
  dat.sensorEvents[1].type = 0x10
  dat.sensorEvents[1].init('gyroUncalibrated')
  dat.sensorEvents[1].gyroUncalibrated.v = [imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z]
  pm.send('sensorEvents', dat)


def panda_state_function(exit_event: threading.Event):
  pm = messaging.PubMaster(['pandaState'])
  while not exit_event.is_set():
    dat = messaging.new_message('pandaState')
    dat.valid = True
    dat.pandaState = {
      'ignitionLine': True,
      'pandaType': "blackPanda",
      'controlsAllowed': True,
      'safetyModel': 'hondaNidec'
    }
    pm.send('pandaState', dat)
    time.sleep(0.5)


def gps_callback(gps, vehicle_state):
  dat = messaging.new_message('gpsLocationExternal')
  velNED = [-vehicle_state.vel.y, vehicle_state.vel.x, vehicle_state.vel.z]
  dat.gpsLocationExternal = {
    "timestamp": int(time.time() * 1000),
    "flags": 1,
    "accuracy": 1.0,
    "verticalAccuracy": 1.0,
    "speedAccuracy": 0.1,
    "bearingAccuracyDeg": 0.1,
    "vNED": velNED,
    "bearingDeg": vehicle_state.bearing_deg,
    "latitude": gps.latitude,
    "longitude": gps.longitude,
    "altitude": gps.altitude,
    "speed": vehicle_state.speed,
    "source": log.GpsLocationData.SensorSource.ublox,
  }
  pm.send('gpsLocationExternal', dat)


def radar_callback(radar_data, world, Radar_Point_Array, N_radar_points):
  global radar_points, frameIdx
  if N_radar_points.value == 0:
    radar_points = np.empty((0, 4), float)

  for detect in radar_data:
    detect_array = np.array([[detect.altitude, detect.azimuth, detect.depth, detect.velocity]])
    radar_points = np.vstack([radar_points, detect_array])

  if frameIdx > 999:
    n_points = min(radar_points.shape[0], 64)
    Radar_Point_Array[:n_points * 4] = radar_points.flatten()[:n_points * 4]
    N_radar_points.value = n_points


def fake_driver_monitoring(exit_event: threading.Event):
  pm = messaging.PubMaster(['driverState', 'driverMonitoringState'])
  while not exit_event.is_set():
    dat = messaging.new_message('driverState')
    dat.driverState.faceProb = 1.0
    pm.send('driverState', dat)

    dat = messaging.new_message('driverMonitoringState')
    dat.driverMonitoringState = {
      "faceDetected": True,
      "isDistracted": False,
      "awarenessStatus": 1.,
    }
    pm.send('driverMonitoringState', dat)
    time.sleep(DT_DMON)


def can_function_runner(vs: VehicleState, exit_event: threading.Event, Radar_Point_Array):
  i = 1
  while not exit_event.is_set():
    if i % 5 != 0 or Radar_Point_Array[-1] == 0 or frameIdx < 1000:
      can_function(pm, vs.speed, vs.angle, i, vs.cruise_button, vs.is_engaged, None)
    else:
      radar_can_message = np.copy(Radar_Point_Array[256:304]).reshape(16, 3)
      can_function(pm, vs.speed, vs.angle, i, vs.cruise_button, vs.is_engaged, radar_can_message)
      Radar_Point_Array[-1] = 0
    i += 1


def bridge(q: Queue, Radar_Point_Array, N_radar_points, vehicle_from_ddpg=None):
  global EXIT_SIGNAL

  client = carla.Client("127.0.0.1", 2000)
  client.set_timeout(10.0)
  world = client.get_world()

  # 简化地图加载
  world.unload_map_layer(carla.MapLayer.Foliage)
  world.unload_map_layer(carla.MapLayer.Buildings)
  world.unload_map_layer(carla.MapLayer.ParkedVehicles)

  blueprint_library = world.get_blueprint_library()
  player = None
  init_vehicle = world.get_actors().filter('vehicle.*')[0]
  possible_vehicles = world.get_actors().filter('vehicle.*')
  for candidate_vehicle in possible_vehicles:
    if candidate_vehicle.attributes['role_name'] == 'yanxi':
      print("Ego vehicle found")
      player = candidate_vehicle
      break

  vehicle = init_vehicle
  if player is not None:
    vehicle = player

  # 设置车辆物理参数
  max_steer_angle = vehicle.get_physics_control().wheels[0].max_steer_angle

  # 设置摄像头传感器
  blueprint = blueprint_library.find('sensor.camera.rgb')
  blueprint.set_attribute('image_size_x', str(W))
  blueprint.set_attribute('image_size_y', str(H))
  blueprint.set_attribute('fov', '70')
  blueprint.set_attribute('sensor_tick', '0.05')
  transform = carla.Transform(carla.Location(x=0.8, z=1.13))
  camera = world.spawn_actor(blueprint, transform, attach_to=vehicle)
  camera.listen(cam_callback)

  vehicle_state = VehicleState()

  # 设置IMU传感器
  imu_bp = blueprint_library.find('sensor.other.imu')
  imu = world.spawn_actor(imu_bp, transform, attach_to=vehicle)
  imu.listen(lambda imu: imu_callback(imu, vehicle_state))

  # 设置GPS传感器
  gps_bp = blueprint_library.find('sensor.other.gnss')
  gps = world.spawn_actor(gps_bp, transform, attach_to=vehicle)
  gps.listen(lambda gps: gps_callback(gps, vehicle_state))

  # 设置雷达传感器
  radar_bp = blueprint_library.find('sensor.other.radar')
  radar_bp.set_attribute('horizontal_fov', str(15))
  radar_bp.set_attribute('vertical_fov', str(15))
  radar_bp.set_attribute('range', str(256))
  radar_location = carla.Location(x=vehicle.bounding_box.extent.x, z=1.0)
  radar_rotation = carla.Rotation()
  radar_transform = carla.Transform(radar_location, radar_rotation)
  radar = world.spawn_actor(radar_bp, radar_transform, attach_to=vehicle)
  radar.listen(lambda radar_data: radar_callback(radar_data, world, Radar_Point_Array, N_radar_points))

  # 启动后台线程
  threads = []
  exit_event = threading.Event()
  threads.append(threading.Thread(target=panda_state_function, args=(exit_event,)))
  threads.append(threading.Thread(target=fake_driver_monitoring, args=(exit_event,)))
  threads.append(threading.Thread(target=can_function_runner, args=(vehicle_state, exit_event, Radar_Point_Array,)))

  for t in threads:
    t.start()

  rk = Ratekeeper(100, print_delay_threshold=0.05)

  # 主循环
  while not EXIT_SIGNAL:
    sm.update(0)

    # 获取车道线数据并发送到主进程
    md = sm["modelV2"]
    if len(md.position.y) > 0:
      lane_data = {
        'xLeftLine': list(md.laneLines[1].x),
        'yLeftLine': list(md.laneLines[1].y),
        'xRightLine': list(md.laneLines[2].x),
        'yRightLine': list(md.laneLines[2].y)
      }
      q.put(lane_data)

    rk.keep_time()
    time.sleep(0.01)

  # 清理资源
  exit_event.set()
  for t in reversed(threads):
    t.join()

  camera.destroy()
  imu.destroy()
  gps.destroy()
  radar.destroy()
  vehicle.destroy()


def bridge_keep_alive(q: Queue, Radar_Point_Array, N_radar_points):
  while True:
    try:
      bridge(q, Radar_Point_Array, N_radar_points)
      break
    except RuntimeError:
      print("Restarting bridge...")
      time.sleep(1)


if __name__ == "__main__":
  set_params_enabled()
  msg = messaging.new_message('liveCalibration')
  msg.liveCalibration.validBlocks = 20
  msg.liveCalibration.rpyCalib = [0.0, 0.0, 0.0]
  Params().put("CalibrationParams", msg.to_bytes())

  q = Queue()
  Radar_Point_Array = Array('d', np.array([0] * (64 * 4 + 16 * 3 + 1)).astype('float64'))
  N_radar_points = Value('i', 0)

  p = Process(target=bridge, args=(q, Radar_Point_Array, N_radar_points), daemon=True)
  p.start()
  p.join()
