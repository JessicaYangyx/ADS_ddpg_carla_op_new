#!/usr/bin/env python3
import argparse
from operator import truediv
import carla # pylint: disable=import-error
# import rospy
# from carla_msgs.msg import CarlaEgoVehicleControl

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

import sys,os,signal
# from sys import argv

from sklearn.cluster import DBSCAN
import pandas as pd

import subprocess

# parser = argparse.ArgumentParser(description='Bridge between CARLA and openpilot.')
# parser.add_argument('--joystick', action='store_true')
# parser.add_argument('--low_quality', action='store_true')
# parser.add_argument('--town', type=str, default='Town04_Opt')
# parser.add_argument('--spawn_point', dest='num_selected_spawn_point',
#         type=int, default=16)
#
# parser.add_argument('--cruise_lead', type=int, default=80) #(1 + 80%)V0 = 1.8V0
# parser.add_argument('--cruise_lead2', type=int, default=80) #(1 + 80%)V0 = 1.8V0 #change speed in the middle
# parser.add_argument('--init_dist', type=int, default=100) #meters; initial relative distance between vehicle and vehicle2
#
# parser.add_argument('--radar_eps', type=float, default=0.2)
#
#
# args = parser.parse_args()

W, H = 1164, 874
REPEAT_COUNTER = 5
PRINT_DECIMATION = 100
STEER_RATIO = 15.

vEgo = 60 #mph #set in selfdrive/controls/controlsd
Other_vehicles_Enable = False
reInitialize_bridge = False

FI_Enable = False #True #False: run the code in fault free mode; True: add fault inejction Engine
Panda_SafetyCheck_Enable = False
Driver_react_Enable = False
AEB_React_Enable = False
Mode_FI_duration = 1 # 0: FI lasts 2.5s after t_f; 1: FI whenever context is True between [t_f,t_f+2.5s]

Strategic_value_selection = False # Only set this to True for CAWT FI
Fixed_value_corruption = False # valid only when Strategic_value_selection=False
Supercomb_Output_Log_Enable = False

pm = messaging.PubMaster(['roadCameraState', 'sensorEvents', 'can', "gpsLocationExternal"])
sm = messaging.SubMaster(['carControl','controlsState','radarState','modelV2'])

class VehicleState:
  def __init__(self):
    self.speed = 0
    self.angle = 0
    self.bearing_deg = 0.0
    self.vel = carla.Vector3D()
    self.cruise_button= 0
    self.is_engaged=False

def steer_rate_limit(old, new):
  # Rate limiting to 0.5 degrees per step
  limit = 0.5
  if new > old + limit:
    return old + limit
  elif new < old - limit:
    return old - limit
  else:
    return new

##############
FI_flag = 0
FI_Type = 0
frameIdx = 0
frame_id = 0
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
  # copied these numbers from locationd
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
    # time.sleep(0.5)

def gps_callback(gps, vehicle_state):
  dat = messaging.new_message('gpsLocationExternal')

  # transform vel from carla to NED
  # north is -Y in CARLA
  velNED = [
    -vehicle_state.vel.y, # north/south component of NED is negative when moving south
    vehicle_state.vel.x, # positive when moving east, which is x in carla
    vehicle_state.vel.z,
  ]

  dat.gpsLocationExternal = {
    "timestamp": int(time.time() * 1000),
    "flags": 1, # valid fix
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

# resource: selfdrive/controls/radard.py search publish radarState
radar_points = np.empty((0, 4), float)
def radar_callback(radar_data,world,Radar_Point_Array,N_radar_points):
  global radar_points,frameIdx
  if N_radar_points.value == 0:
    radar_points = np.empty((0, 4), float) #reset radar_points

  velocity_range = 7.5 # m/s
  current_rot = radar_data.transform.rotation
  for detect in radar_data:
    detect_array = np.array([[detect.altitude, detect.azimuth, detect.depth, detect.velocity]])
    radar_points = np.vstack([radar_points, detect_array])

  if frameIdx>999:
    n_points = min(radar_points.shape[0],64) #only process 64 points
    Radar_Point_Array[:n_points*4] = radar_points.flatten()[:n_points*4]
    N_radar_points.value = n_points

detect_fcw = False
fcw_alert = False
fcw_start_dis = -1
fcw_start_rv = -1
fcw_start_egov = -1

aeb_alert = False
aeb_brake = 0
aeb_start_dis = -1
aeb_start_rv = -1
aeb_start_egov = -1

vehicle2_id = None
dRel = 0
vRel = 0

obstacle_count = 0
obstacle_drel = 0
obstacle_vrel = 0
obstacle_index = 0
def obstacle_callback(detection):
  global vehicle2_id, dRel, vRel
  global obstacle_count,obstacle_drel,obstacle_vrel,obstacle_index
  other_actor = detection.other_actor
  no_other = (False, carla.Vector3D(x = 0, y = 0, z = 0))
  if other_actor == None: # check to see if the other factor is not a car
    no_other[0] = True

  if other_actor.id != vehicle2_id:
    return detection #return when the other factor is not a car

  # print("-----------Obstacle Callback-----------")
  # print(detection)

  car = detection.actor.parent
  obstacle_drel =  detection.distance

  distance_to_obstacle = dRel#obstacle_drel

  ego_vel = car.get_velocity()
  other_vel = other_actor.get_velocity() if not no_other[0] else no_other[1]
  relative_vel = carla.Vector3D(x = ego_vel.x - other_vel.x, y = ego_vel.y - other_vel.y, z = ego_vel.z - other_vel.z)
  # print(relative_vel)

  # new control logic: use TTC and state transition
  proc_brake = False
  ego_vel_mag = math.sqrt(ego_vel.x**2 + ego_vel.y**2 + ego_vel.z**2)
  other_vel_mag = math.sqrt(other_vel.x**2 + other_vel.y**2 + other_vel.z**2)
  obstacle_vrel = math.sqrt(relative_vel.x**2 + relative_vel.y**2 + relative_vel.z**2)

  if ego_vel_mag<other_vel_mag:
    obstacle_vrel = -1*obstacle_vrel
  relative_vel_mag = obstacle_vrel

  ##fuse with vision data
  relative_vel_mag = -vRel


  if relative_vel_mag > 0:
    ttc = distance_to_obstacle / relative_vel_mag
  else:
    ttc = 10000


  # fcw time
  treact = 2.5
  tfcw = treact + ego_vel_mag / 5

  # 1st partial brake phase: decelerate at 4 m/s^2
  tpb1 = ego_vel_mag / 3.8
  # 2nd partial brake phase: decelerate at 6 m/s^2
  tpb2 = ego_vel_mag / 5.8
  # full brake phase: decelerate at 10 m/s^2
  tfb = ego_vel_mag / 9.8

  global aeb_alert, aeb_brake, fcw_alert
  if ttc > 0 and ttc < tfb:

    proc_brake = True
    # print("Activating full brake")
    # car.apply_control(carla.VehicleControl(brake=1))
    aeb_alert = True
    aeb_brake = 1
    # print("Disabling OpenPilot")

  elif ttc > 0 and ttc < tpb2:
    proc_brake = True
    # print("Activating second-phase partial brake")
    # car.apply_control(carla.VehicleControl(brake=0.95))
    aeb_alert = True
    aeb_brake =0.95
    # print("Disabling OpenPilot")

  elif ttc > 0 and ttc < tpb1:
    proc_brake = True
    # print("Activating first-phase partial brake")
    # car.apply_control(carla.VehicleControl(brake=0.9))
    aeb_alert = True
    aeb_brake = 0.9
    # print("Disabling OpenPilot")

  if ttc > tpb1 and ttc < tfcw:
    obstacle_count += 1
    if obstacle_count > 5:
      fcw_alert = True
      # print("fcw alert......",dRel,vRel)
  else:
    obstacle_count = 0
    fcw_alert = False

  global aeb_start_dis, aeb_start_rv, aeb_start_egov
  if aeb_alert:
    if aeb_start_dis ==-1:
      aeb_start_dis = distance_to_obstacle
      print("********************* the aeb start distance is:", aeb_start_dis)

    if aeb_start_rv == -1:
      aeb_start_rv = relative_vel_mag
      print("********************* the aeb start relative velocity is:", aeb_start_rv)

    if aeb_start_egov == -1:
      aeb_start_egov = ego_vel_mag
      print("********************* the aeb start ego velocity is:", aeb_start_egov)

  global fcw_start_dis, fcw_start_rv, fcw_start_egov
  if fcw_alert:
    if fcw_start_dis == -1:
      fcw_start_dis = distance_to_obstacle
      print("*********************== the fcw start distance is:", fcw_start_dis, dRel)

    if fcw_start_rv == -1:
      fcw_start_rv = relative_vel_mag
      print("********************* the fcw start relative velocity is:", fcw_start_rv, vRel)

    if fcw_start_egov == -1:
      fcw_start_egov = ego_vel_mag
      print("********************* the fcw start ego velocity is:", fcw_start_egov)

  obstacle_index += 1
  # print("---------------------------------------")
  return detection

collision_hist = []
def collision_callback(col_event):
  collision_hist.append(col_event)
  # print(col_event)

laneInvasion_hist = []
def laneInvasion_callback(LaneInvasionEvent):
  laneInvasion_hist.append(LaneInvasionEvent)



def fake_driver_monitoring(exit_event: threading.Event):
  pm = messaging.PubMaster(['driverState','driverMonitoringState'])
  while not exit_event.is_set():
    # dmonitoringmodeld output
    dat = messaging.new_message('driverState')
    dat.driverState.faceProb = 1.0
    pm.send('driverState', dat)

    # dmonitoringd output
    dat = messaging.new_message('driverMonitoringState')
    dat.driverMonitoringState = {
      "faceDetected": True,
      "isDistracted": False,
      "awarenessStatus": 1.,
    }
    pm.send('driverMonitoringState', dat)

    # time.sleep(DT_DMON)

RADAR_MODE = 0 #0: Diaable; 1: actual radar sensor; 2: obstacle sensor

def parse_radar(Radar_Point_Array,N_radar_points):
  i = 1
  while EXIT_SIGNAL==False:
    start_time = time.time()

    if i % 5 == 0 and N_radar_points.value > 0 :
      # print("N_radar_points=",N_radar_points.value,i % 5 != 0,N_radar_points.value == 0)
      radar_points = np.copy(Radar_Point_Array[:N_radar_points.value*4]).reshape(-1,4)
      # process radar points==========
      radar_points_clustering = DBSCAN(eps=0.2, min_samples=5).fit(radar_points / [math.radians(10), math.radians(17.5), 100.0, 35.0]) # normalize data points
      radar_points_clustering_centroids = np.zeros((16, 4), float)
      radar_points_clustering_label_counts = np.zeros((16, 1), int)
      # sum all tracks===========
      for idx, track_id in enumerate(radar_points_clustering.labels_):
        if track_id != -1 and track_id < 16:
          radar_points_clustering_centroids[track_id, :] += radar_points[idx, :]
          radar_points_clustering_label_counts[track_id] += 1
      # average all tracks to get centroids============
      for idx, radar_point in enumerate(radar_points_clustering_centroids):
        if radar_points_clustering_label_counts[idx] != 0:
          radar_points_clustering_centroids[idx] = radar_point / radar_points_clustering_label_counts[idx]
      # calculate longitudinal_dist, lateral_dist, and relative_velocity==============
      radar_can_message = np.zeros((16, 3), float)
      for idx, radar_point_centroid in enumerate(radar_points_clustering_centroids):
        if radar_points_clustering_label_counts[idx] == 0:
        # if radar_points_clustering_label_counts[idx] == 0 or radar_point_centroid<0.5:
          radar_can_message[idx, :] = np.array([[255.5, 0.0, 0.0]])
        else:
          radar_can_message[idx, 0] = math.cos(radar_point_centroid[0]) * math.cos(radar_point_centroid[1]) * radar_point_centroid[2] # radar_longitudinal_distance_offset # longitudinal distance
          radar_can_message[idx, 1] = math.cos(radar_point_centroid[0]) * math.sin(radar_point_centroid[1]) * radar_point_centroid[2] # lateral distance
          radar_can_message[idx, 2] = radar_point_centroid[3] # relative velocity
      # print(radar_points,radar_points_clustering,radar_points_clustering_centroids,radar_points_clustering_label_counts)
      # print(f"DBSCAN spend time = {time.time()-start_time}~~~~~")
      N_radar_points.value = 0
      Radar_Point_Array[256:304] = radar_can_message.flatten()
      Radar_Point_Array[-1] = 1
    # print(f"parse spend time = {time.time()-start_time}~~~~~")
    # time.sleep(0.01)
    i+=1


def can_function_runner(vs: VehicleState, exit_event: threading.Event, Radar_Point_Array):
  # global radar_points
  global frameIdx,obstacle_index,obstacle_drel,obstacle_vrel,dRel,vRel
  ob_process = 0
  i = 1
  while not exit_event.is_set():

    if i % 5 != 0 or Radar_Point_Array[-1] == 0 or frameIdx < 1000:
      can_function(pm, vs.speed, vs.angle, i, vs.cruise_button, vs.is_engaged, None)
    else:
      radar_can_message = np.copy(Radar_Point_Array[256:304]).reshape(16,3) #112-64=48=16*3
      # print(dRel,vRel,"=========can meassage",radar_can_message[:2,:])
      can_function(pm, vs.speed, vs.angle, i, vs.cruise_button, vs.is_engaged, radar_can_message)
      # radar_points = np.empty((0, 4), float)
      Radar_Point_Array[-1] = 0
    # time.sleep(0.01)
    i+=1


def bridge(q, Radar_Point_Array,N_radar_points, vehicle_from_ddpg=None):

  # setup CARLA
  client = carla.Client("127.0.0.1", 2000)
  client.set_timeout(10.0)
  world = client.get_world()

  print("test======================================================================")

  if 1:
    world.unload_map_layer(carla.MapLayer.Foliage)
    world.unload_map_layer(carla.MapLayer.Buildings)
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    world.unload_map_layer(carla.MapLayer.Particles)
    world.unload_map_layer(carla.MapLayer.Props)
    world.unload_map_layer(carla.MapLayer.StreetLights)

  blueprint_library = world.get_blueprint_library()

  # from scenario runner
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
  # vehicle = vehicle_from_ddpg

  max_steer_angle = vehicle.get_physics_control().wheels[0].max_steer_angle
  print('max_steer_angle',max_steer_angle) #70 degree

  # make tires less slippery
  # wheel_control = carla.WheelPhysicsControl(tire_friction=5)
  # physics_control = vehicle.get_physics_control()
  # physics_control.mass = 2326
  # # physics_control.wheels = [wheel_control]*4
  # physics_control.torque_curve = [[20.0, 500.0], [5000.0, 500.0]]
  # physics_control.gear_switch_time = 0.0
  # vehicle.apply_physics_control(physics_control)

  blueprint = blueprint_library.find('sensor.camera.rgb')
  blueprint.set_attribute('image_size_x', str(W))
  blueprint.set_attribute('image_size_y', str(H))
  blueprint.set_attribute('fov', '70')
  blueprint.set_attribute('sensor_tick', '0.05')
  transform = carla.Transform(carla.Location(x=0.8, z=1.13))
  camera = world.spawn_actor(blueprint, transform, attach_to=vehicle)
  camera.listen(cam_callback)

  vehicle_state = VehicleState()

  # reenable IMU
  imu_bp = blueprint_library.find('sensor.other.imu')
  imu = world.spawn_actor(imu_bp, transform, attach_to=vehicle)
  imu.listen(lambda imu: imu_callback(imu, vehicle_state))

  gps_bp = blueprint_library.find('sensor.other.gnss')
  gps = world.spawn_actor(gps_bp, transform, attach_to=vehicle)
  gps.listen(lambda gps: gps_callback(gps, vehicle_state))

  # add radar (reference: https://carla.readthedocs.io/en/latest/tuto_G_retrieve_data/#radar-sensor)
  # Get radar blueprint
  radar_bp = blueprint_library.find('sensor.other.radar')
  radar_bp.set_attribute('horizontal_fov', str(15))
  radar_bp.set_attribute('vertical_fov', str(15))
  radar_bp.set_attribute('range', str(256))
  radar_location = carla.Location(x=vehicle.bounding_box.extent.x, z=1.0)
  radar_rotation = carla.Rotation()
  radar_transform = carla.Transform(radar_location, radar_rotation)
  radar = world.spawn_actor(radar_bp, radar_transform, attach_to=vehicle)
  radar.listen(lambda radar_data: radar_callback(radar_data,world,Radar_Point_Array,N_radar_points))

  #AEBS
  obstacledet_bp = blueprint_library.find('sensor.other.obstacle')
  obstacledet_bp.set_attribute('distance', '40') # Distance for the Obstacle sensor to see objects within this range
  obstacledet_bp.set_attribute('hit_radius', '1') # Radius of the cylinder passed forward from the car to detect
  obstacledet_bp.set_attribute('debug_linetrace', 'True')

  obstacledet = world.spawn_actor(obstacledet_bp, transform, attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
  obstacledet.listen(lambda sensor_data: obstacle_callback(sensor_data))

  #collision sensor detector
  colsensor_bp = blueprint_library.find("sensor.other.collision")
  colsensor = world.spawn_actor(colsensor_bp, transform, attach_to=vehicle)
  colsensor.listen(lambda colevent: collision_callback(colevent))

  #lane invasion
  laneInvasion_bp = blueprint_library.find("sensor.other.lane_invasion")
  laneInvasion = world.spawn_actor(laneInvasion_bp, transform, attach_to=vehicle)
  laneInvasion.listen(lambda LaneInvasionEvent: laneInvasion_callback(LaneInvasionEvent))

  # launch fake car threads
  threads = []
  exit_event = threading.Event()
  threads.append(threading.Thread(target=panda_state_function, args=(exit_event,)))
  threads.append(threading.Thread(target=fake_driver_monitoring, args=(exit_event,)))
  threads.append(threading.Thread(target=can_function_runner, args=(vehicle_state, exit_event,Radar_Point_Array,)))

  for t in threads:
    t.start()

  # time.sleep(1)

  # can loop
  rk = Ratekeeper(100, print_delay_threshold=0.05) #rate =100, T=1/100s=10ms

  # init
  throttle_ease_out_counter = REPEAT_COUNTER
  brake_ease_out_counter = REPEAT_COUNTER
  steer_ease_out_counter = REPEAT_COUNTER


  # vc = carla.VehicleControl(throttle=0, steer=0, brake=0, reverse=False)

  is_openpilot_engaged = False

  old_steer = old_brake = old_throttle = 0
  throttle_manual_multiplier = 0.7 #keyboard signal is always 1
  brake_manual_multiplier = 0.7 #keyboard signal is always 1
  steer_manual_multiplier = 45 * STEER_RATIO  #keyboard signal is always 1
  # steer_manual_multiplier = 10.0

  # tm = client.get_trafficmanager(8008)

  speed = 0
  throttle_out_hist = 0
  FI_duration = 1000# set to be a larget value like 10 seconds so it won't be reached in the normal case with human driver engagement #250*10ms =2.5s
  Num_laneInvasion = 0
  t_laneInvasion = 0
  # pathleft = pathright = 0
  # roadEdgeLeft = roadEdgeRight = 0
  # laneLineleft=-1.85
  # laneLineright = 1.85
  Lead_vehicle_in_vision = False #lead vehicle is captured in the camera

  faulttime = -1
  alerttime = -1
  hazardtime = -1
  fault_duration = 0
  driver_alerted_time = -1
  H2_count = 0

  hazMsg = ""
  hazard = False
  hazType =0x0

  alertType_list =[]
  alertText1_list = []
  alertText2_list = []


  FI_Context_H3_combine_enable = 0
  global EXIT_SIGNAL
  global FI_Type, FI_flag, frameIdx
  global vRel,dRel
  global aeb_alert,fcw_alert
  global obstacle_drel #distance to front obstacle, used to trigger driver response

  # sys.path.append("/home/yanxi/carla/openpilot_0.8.9/openpilot0.8.9/tools/sim")
  # fp_radar = open("results/radarlog.csv","w")
  # fp_radar.write("frameIdx,dRel_truth,dRel,vRel")
  # for i in range(16):
  #   fp_radar.write(",rx{}".format(i))
  #   fp_radar.write(",ry{}".format(i))
  #   fp_radar.write(",rv{}".format(i))
  # for i in range(64):
  #   fp_radar.write(",al{}".format(i))
  #   fp_radar.write(",az{}".format(i))
  #   fp_radar.write(",dp{}".format(i))
  #   fp_radar.write(",ve{}".format(i))
  # fp_radar.write('\n')

  # fp_vrel = open("results/test_vrel.csv",'w')
  # fp_vrel.write("speed,speed2,vreltruth,vrel\n")

  # ros_subscriber = ROSSubscriber(vehicle)

  while world is not None:
    sm.update(0)
  # while frameIdx<5000:
  #   altMsg = ""
    # alert = False
    #
    # if is_openpilot_engaged:
    #   frameIdx += 1

    #simulate button Enable event
    # if rk.frame == 800:
    #   q.put("cruise_up")

    # cruise_button = 0
    # throttle_out = steer_out = brake_out = 0.0
    # throttle_op = steer_op = brake_op = 0
    # throttle_manual = steer_manual = brake_manual = 0.0
    # actuators_steeringAngleDeg = actuators_steer = actuators_accel = 0
    #
    # yRel = 2.5
    # vLead = 0
    # xRelPosOfLead = 0.0
    # yRelPosOfLead = 0.0
    # yPos = 0
    # ylaneLines = []
    # yroadEdges = []

    # --------------Step 1-------------------------------
    # while not q.empty():
    #   message = q.get()
    #   m = message.split('_')
    #   # print(message)
    #   if m[0] == "steer":
    #     steer_manual = float(m[1])
    #     is_openpilot_engaged = False
    #   elif m[0] == "throttle":
    #     throttle_manual = float(m[1])
    #     is_openpilot_engaged = False
    #   elif m[0] == "brake":
    #     brake_manual = float(m[1])
    #     is_openpilot_engaged = False
    #   elif m[0] == "reverse":
    #     #in_reverse = not in_reverse
    #     cruise_button = CruiseButtons.CANCEL
    #     is_openpilot_engaged = False
    #   elif m[0] == "cruise":
    #     # vehicle2.set_autopilot(True,8008)
    #     # if Other_vehicles_Enable:
    #     #   vehicle3.set_autopilot(True,8008)
    #     #   vehicle4.set_autopilot(True,8008)
    #     #   vehicle5.set_autopilot(True,8008)
    #
    #     if m[1] == "down":
    #       cruise_button = CruiseButtons.DECEL_SET
    #       is_openpilot_engaged = True
    #     elif m[1] == "up":
    #       cruise_button = CruiseButtons.RES_ACCEL
    #       is_openpilot_engaged = True
    #     elif m[1] == "cancel":
    #       cruise_button = CruiseButtons.CANCEL
    #       is_openpilot_engaged = False
    #   elif m[0] == "quit":
    #     # vehicle2.set_autopilot(False,8008)
    #     break
    #
    #   throttle_out = throttle_manual * throttle_manual_multiplier
    #   steer_out = steer_manual * steer_manual_multiplier
    #   brake_out = brake_manual * brake_manual_multiplier
    #
    #   #steer_out = steer_out
    #   # steer_out = steer_rate_limit(old_steer, steer_out)
    #   old_steer = steer_out
    #   old_throttle = throttle_out
    #   old_brake = brake_out

    # print('message',old_throttle, old_steer, old_brake)

    # is_openpilot_engaged = False
    # # if is_openpilot_engaged:
    # #   sm.update(0)
    # #   # TODO gas and brake is deprecated
    # #   throttle_op = clip(sm['carControl'].actuators.accel/4.0, 0.0, 1.0)
    # #   brake_op = clip(-sm['carControl'].actuators.accel/4.0, 0.0, 1.0)
    # #   steer_op = sm['carControl'].actuators.steeringAngleDeg
    # #   actuators = sm['carControl'].actuators
    # #   actuators_accel = actuators.accel
    # #   actuators_steer = actuators.steer
    # #   actuators_steeringAngleDeg = actuators.steeringAngleDeg
    # #
    # #   throttle_out = throttle_op
    # #   steer_out = steer_op
    # #   brake_out = brake_op
    # #
    # #   steer_out = steer_rate_limit(old_steer, steer_out)
    # #   old_steer = steer_out
    # #
    # #   # dRel = sm['radarState'].leadOne.dRel
    # #   # yRel = sm['radarState'].leadOne.yRel #y means lateral direction
    # #   # vRel = sm['radarState'].leadOne.vRel
    # #   # vLead = sm['radarState'].leadOne.vLead
    # #   # if not sm['radarState'].leadOne.status:
    # #   #   Lead_vehicle_in_vision = False
    # #   # else:
    # #   #   Lead_vehicle_in_vision = True
    # #   #
    # #   # #print or log supercomb predictions
    # #   # md = sm['modelV2']
    # #   # # mdsave=[0]*6472
    # #   # # mdsave = md.copy()
    # #   # # print(str(md.position.x))
    # #   #
    # #   # if len(md.position.y)>0:
    # #   #   yPos = round(md.position.y[0],2) # position
    # #   #   ylaneLines = [round(md.laneLines[0].y[0],2),round(md.laneLines[1].y[0],2),round(md.laneLines[2].y[0],2),round(md.laneLines[3].y[0],2)]
    # #   #   yroadEdges = [round(md.roadEdges[0].y[0],2), round(md.roadEdges[1].y[0],2)] #left and right roadedges
    # #   #   # print(ylaneLines[2] - yPos)
    # #   #   if len(ylaneLines)>2:
    # #   #     laneLineleft = ylaneLines[1]
    # #   #     laneLineright = ylaneLines[2]
    # #   #     pathleft = yPos- laneLineleft
    # #   #     pathright = laneLineright-yPos
    # #   #     roadEdgeLeft = yroadEdges[0]
    # #   #     roadEdgeRight = yroadEdges[1]
    # #   #
    # #   # #controlsState
    # #   # alertText1 = sm['controlsState'].alertText1
    # #   # alertText2 = sm['controlsState'].alertText2
    # #   # alertType  = sm['controlsState'].alertType
    # #   #
    # #   # if alertType and alertType not in alertType_list and alertText1 not in alertText1_list:
    # #   #   alertText1_list.append(alertText1)
    # #   #   alertType_list.append(alertType)
    # #   #   if(alerttime== -1 and 'startupMaster/permanent' not in alertType and 'buttonEnable/enable' not in alertType):
    # #   #     alerttime = frameIdx
    # #   #     alert = True
    # #   #
    # #   #   print("=================Alert============================")
    # #   #   print(alertType,":",alertText1,alertText2)
    #
    # # else:
    # #   sm.update(0)
    #   # if throttle_out==0 and old_throttle>0:
    #   #   if throttle_ease_out_counter>0:
    #   #     throttle_out = old_throttle
    #   #     throttle_ease_out_counter += -1
    #   #   else:
    #   #     throttle_ease_out_counter = REPEAT_COUNTER
    #   #     old_throttle = 0
    #   #
    #   # if brake_out==0 and old_brake>0:
    #   #   if brake_ease_out_counter>0:
    #   #     brake_out = old_brake
    #   #     brake_ease_out_counter += -1
    #   #   else:
    #   #     brake_ease_out_counter = REPEAT_COUNTER
    #   #     old_brake = 0
    #   #
    #   # if steer_out==0 and old_steer!=0:
    #   #   if steer_ease_out_counter>0:
    #   #     steer_out = old_steer
    #   #     steer_ease_out_counter += -1
    #   #   else:
    #   #     steer_ease_out_counter = REPEAT_COUNTER
    #   #     old_steer = 0
    #
    #
    # # --------------Step 2-------------------------------
    #
    # steer_carla = steer_out / (max_steer_angle * STEER_RATIO * -1)
    #
    # steer_carla = np.clip(steer_carla, -1,1)
    # steer_out = steer_carla * (max_steer_angle * STEER_RATIO * -1)
    # old_steer = steer_carla * (max_steer_angle * STEER_RATIO * -1)

    # vc.throttle = throttle_out/0.8
    # vc.steer = steer_carla
    # vc.brake = brake_out
    # vehicle.apply_control(vc)

    # --------------Step 3-------------------------------
    # vel = vehicle.get_velocity()
    # speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) # in m/s
    # acc = vehicle.get_acceleration()
    # acceleration = math.sqrt(acc.x**2 + acc.y**2 + acc.z**2) # in m/s^2
    # # if speed==acceleration==0:
    # #   acceleration =1
    # vehicle_state.speed = speed
    # vehicle_state.vel = vel
    # vehicle_state.angle = steer_out
    # vehicle_state.cruise_button = cruise_button
    # vehicle_state.is_engaged = is_openpilot_engaged
    #
    # veh_pos = vehicle.get_transform().location

    #Lane position
    ylaneLines = []
    yroadEdges = []
    pathleft = pathright = 0
    roadEdgeLeft = roadEdgeRight = 0
    laneLineleft = -1.85
    laneLineright = 1.85
    xLeftLine = []
    xRightLine = []
    yLeftLine = []
    yRightLine = []
    obsVels = []
    obsRelDists = []
    obsRelPosX = []
    obsRelPosY = []
    obsLeadProb = []
    obsLeadProbTime = []
    md = sm["modelV2"]
    if len(md.position.y) > 0:
      yPos = round(md.position.y[0], 2)  # position
      xLeftLine = list(md.laneLines[1].x)
      xRightLine = list(md.laneLines[2].x)
      yLeftLine = list(md.laneLines[1].y)
      yRightLine = list(md.laneLines[2].y)
      # print("yyx debug op --- ", [xLeftLine, yLeftLine, xRightLine, yRightLine])
      q.put([xLeftLine, yLeftLine, xRightLine, yRightLine])
      # time.sleep(0.01)

    if len(md.leadsV3) > 0:
      # vLead = md.leadsV3[2].v[0]
      # dRel = math.sqrt(md.leadsV3[2].x[0] ** 2 + md.leadsV3[2].y[0] ** 2)
      # xRelPosOfLead = md.leadsV3[2].x[0]
      # yRelPosOfLead = md.leadsV3[2].y[0]

      # ----------
      for obs in md.leadsV3:
        # print(obs)
        # print('obs v size: ', len(obs.v), ", x size: ", len(obs.x), ", y size: ", len(obs.y))
        if obsVels.count(obs.v[0]) == 0:
          obsVels.append(obs.v[0])
          obsRelDists.append(math.sqrt(obs.x[0] ** 2 + obs.y[0] ** 2))
          obsRelPosX.append(obs.x[0])
          obsRelPosY.append(obs.y[0])
          obsLeadProb.append(obs.prob)
          obsLeadProbTime.append(obs.probTime)
          # vLead = obs.v[0]
          # dRel = math.sqrt(obs.x[0] ** 2 + obs.y[0] ** 2)
          # xRelPosOfLead = obs.x[0]
          # yRelPosOfLead = obs.y[0]

    # else:
    #   vLead = 0
    #   dRel = 0
    # log_data = []
    # log_data.append(
    #   [rk.frame, speed, round(vc.steer, 3), round(vc.throttle, 3), round(vc.brake, 3), xLeftLine, yLeftLine, xRightLine,
    #    yRightLine, obsVels, obsRelPosX, obsRelPosY, veh_pos.x, veh_pos.y, obsRelDists, obsLeadProb, obsLeadProbTime])
    # df = pd.DataFrame(log_data, columns=['Frame', 'Speed', 'Steer', 'Throttle', 'Brake', 'LeftLineX', 'LeftLineY',
    #                                      'RightLineX', 'RightLineY', 'ObsSpeed', 'ObsRelX', 'ObsRelY',
    #                                      'EgoPosX', 'EgoPosY', 'RelativeDistance', 'ObsLeadProb', 'ObsLeadProbTime'])
    # file_path = f'/home/yanxi/data_collection/OppositeVehicleRunningRedLight_5.csv'
    # header = not pd.io.common.file_exists(file_path)
    # df.to_csv(file_path, mode='a', header=header, index=False)


    #-----------------------------------------------------
    # if frameIdx == 1000:
    #   if speed <0.02 and throttle_out <0.02 and brake_out <0.02: #fail to start
    #     reInitialize_bridge = True
    #     print("reInitialize bridge.py...\n")
    #     break

    #------------------------------------------------------
    # if driver_alerted_time == -1 and fault_duration>0 and (fcw_alert or alert or throttle_out>= 0.6 or speed>1.1*vEgo*0.4407 or brake_out>0.95 or obstacle_drel<5): # or abs(patch.mean()>=0.15) #max gas//max brake//exceed speed limit/unsafe following distance<5m
    #   driver_alerted_time =frameIdx #driver is alerted

    #Accident: collision
    # if len(collision_hist):
    #   print(collision_hist[0],collision_hist[0].other_actor)
    #   # print(vehicle2)
    #   if collision_hist[0].other_actor.id == vehicle2.id: #collide with vehicle2:
    #     dRel = -0.1
    #     if "lead" not in hazMsg:
    #       hazMsg +="||collide with lead vihecle||"
    #   else:
    #     if "curb" not in hazMsg:
    #       hazMsg +="||collide with curb||"
    #
    #     if hazType&0x04 == 0:
    #       hazard = True
    #       hazardtime =frameIdx
    #       hazMsg +="||H3"
    #       hazType |= 0x04 #0b 100



    #if laneInvasion
    laneInvasion_Flag = False
    # if len(laneInvasion_hist)>Num_laneInvasion:
    #   # hazard = True
    #   laneInvasion_Flag =True
    #   Num_laneInvasion = len(laneInvasion_hist)
    #   t_laneInvasion = frameIdx
    #   print(Num_laneInvasion,laneInvasion_hist[-1],laneInvasion_hist[-1].crossed_lane_markings)
    #   # del(laneInvasion_hist[0])
    #
    # #lable hazard
    # if dRel <0.5 and Lead_vehicle_in_vision and 'curb' not in hazMsg: # unsafe distance # collide with curb is not H1
    #   if hazType&0x01 == 0:
    #     hazard = True
    #     hazardtime =frameIdx
    #
    #     hazMsg +="H1"
    #     hazType |= 0x01 #0b 001
    #
    # if speed<0.02 and (dRel >50 or Lead_vehicle_in_vision==False) and fault_duration>0: #decrease the speed to full stop without a lead vehicle
    #   if hazType&0x02 == 0:
    #     H2_count += 1
    #     if H2_count>100: #last for 1 second
    #       hazard = True
    #       hazardtime =frameIdx
    #       hazMsg +="H2"
    #       hazType |= 0x02 #0b 100
    # else:
    #   H2_count = 0
    #
    # if Num_laneInvasion > 0 and (roadEdgeRight <3.7 and (pathright <1.15) or roadEdgeRight>7.4): #lane width = 3.7m vehicle width =2.3m or(ylaneLines[3] -ylaneLines[2] <1.15)
    #   if hazType&0x04 == 0:
    #     hazard = True
    #     hazardtime =frameIdx
    #     hazMsg +="H3"
    #     hazType |= 0x04 #0b 100


    rk.keep_time()
    time.sleep(1.0)
    # throttle_out_hist = vc.throttle

    #brake with hazard
    # if hazard:# or FI_flag ==-1 and speed<0.01:
    #   if 'collide' in hazMsg or frameIdx - hazardtime >250: #terminate the simulation right after any collision or wait 2 seconds after any hazard
    #     break

  #store alert,hazard message to a file, which will be stored in a summary file
  # Alert_flag = len(alertType_list)>0 and 'startupMaster/permanent' not in alertType_list and 'buttonEnable/enable' not in alertType_list
  # fp_temp = open("temp.txt",'w')
  # fp_temp.write("{},{},{},{},{},{},{},{},{}".format("||".join(alertType_list),hazMsg,faulttime,alerttime,hazardtime, Alert_flag,hazard,fault_duration,Num_laneInvasion  ))
  # fp_temp.close()


  # Clean up resources in the opposite order they were created.
  exit_event.set()
  # for t in reversed(threads):
  #   t.join()
    # t.stop()
  # gps.destroy()
  # imu.destroy()
  # camera.destroy()
  # radar.destroy()
  # obstacledet.destroy()
  # vehicle.destroy()
  # colsensor.destroy()

  # fp_radar.close()
  # fp_vrel.close()

  # print(f"overhead time ={np.mean(overhead_time[0])}")
  # os.killpg(os.getpgid(os.getpid()), signal.SIGINT) #kill the remaining threads
  # EXIT_SIGNAL = True
  # sys.exit(0)
  # exit()


def bridge_keep_alive(q: Any, Radar_Point_Array,N_radar_points):
  while 1:
    try:
      bridge(q, Radar_Point_Array,N_radar_points)
      break
    except RuntimeError:
      print("Restarting bridge...")
    # time.sleep(0.01)

# def list_devices_fast():
#   """Faster device listing using os.scandir"""
#   print('Available devices:')
#   try:
#     with os.scandir('/dev/input') as it:
#       for entry in it:
#         if entry.name.startswith('js'):
#           print(f'  {entry.path}')
#   except OSError:
#     pass

if __name__ == "__main__":
  # print(os.getcwd())
  # os.system('rm ./results/*')
  EXIT_SIGNAL = False
  # make sure params are in a good state
  set_params_enabled()

  msg = messaging.new_message('liveCalibration')
  msg.liveCalibration.validBlocks = 20
  msg.liveCalibration.rpyCalib = [0.0, 0.0, 0.0]
  Params().put("CalibrationParams", msg.to_bytes())

  q: Any = Queue()

  Radar_Point_Array = Array('d', np.array([0] * (64 * 4 + 16 * 3 + 1)).astype(
    'float64'))  # [0:64*4] radar points, [256:304]: can message 16*3, [304]: can message ready
  N_radar_points = Value('i', 0)

  #=================================================
  p = Process(target=bridge, args=(q,Radar_Point_Array,N_radar_points), daemon=True)
  p.start()

  # list_devices_fast()

  # from lib.manual_ctrl_new import wheel_poll_thread
  # wheel_poll_thread(q)
  # p.join()

