# # !/usr/bin/env python3
# import array
# import os
# import select
# import struct
# import time
# from fcntl import ioctl
# from multiprocessing import Process, Queue
# from typing import NoReturn
#
# NORMALIZE_FACTOR = 1.0 / 32767.0
#
# # pre-defined templates, improve efficiency
# COMMAND_TEMPLATES = {
#   "throttle": "throttle_{:.2f}",
#   "brake": "brake_{:.2f}",
#   "steer": "steer_{:.2f}",
#   "cruise_down": "cruise_down",
#   "cruise_up": "cruise_up",
#   "cruise_cancel": "cruise_cancel",
#   "reverse_switch": "reverse_switch"
# }
#
#
# def wheel_poll_thread(q: Queue) -> NoReturn:
#   # Open the joystick device.
#   fn = '/dev/input/js0'
#   print('Opening %s...' % fn)
#   jsdev = open(fn, 'rb')
#
#   # Get the device name.
#   buf = array.array('B', [0] * 64)
#   ioctl(jsdev, 0x80006a13 + (0x10000 * len(buf)), buf)  # JSIOCGNAME(len)
#   js_name = buf.tobytes().rstrip(b'\x00').decode('utf-8')
#   print('Device name: %s' % js_name)
#
#   # Get number of axes and buttons.
#   buf = array.array('B', [0])
#   ioctl(jsdev, 0x80016a11, buf)  # JSIOCGAXES
#   num_axes = buf[0]
#
#   buf = array.array('B', [0])
#   ioctl(jsdev, 0x80016a12, buf)  # JSIOCGBUTTONS
#   num_buttons = buf[0]
#
#   # init
#   axis_states = [0.0] * num_axes
#   button_states = [0] * num_buttons
#
#   # Enable FF
#   import evdev  # pylint: disable=import-error
#   from evdev import ecodes, InputDevice  # pylint: disable=import-error
#   device = evdev.list_devices()[0]
#   evtdev = InputDevice(device)
#   val = 24000
#   evtdev.write(ecodes.EV_FF, ecodes.FF_AUTOCENTER, val)
#
#   last_axis_states = {}  # throttle(z-2), brake(rx-3), steer(x-0)
#
#   import fcntl
#   flags = fcntl.fcntl(jsdev, fcntl.F_GETFL)
#   fcntl.fcntl(jsdev, fcntl.F_SETFL, flags | os.O_NONBLOCK)
#
#   # loop
#   while True:
#     # r, _, _ = select.select([jsdev], [], [], 0.0)
#     # if jsdev in r:
#       evbuf = jsdev.read(8)
#       if evbuf:
#         value, mtype, number = struct.unpack('4xhBB', evbuf)
#
#         if mtype & 0x02:  # axis event
#           fvalue = value * NORMALIZE_FACTOR
#           axis_states[number] = fvalue
#
#           if number == 2:  # z (throttle)
#             normalized = (1 - fvalue) * 50
#             q.put(COMMAND_TEMPLATES["throttle"].format(normalized))
#             last_axis_states[number] = normalized
#           if number == 3:  # rx (brake)
#             normalized = (1 - fvalue) * 50
#             q.put(COMMAND_TEMPLATES["brake"].format(normalized))
#             last_axis_states[number] = normalized
#           if number == 0:  # x (steer)
#             normalized = -fvalue * 40
#             q.put(COMMAND_TEMPLATES["steer"].format(normalized))
#             last_axis_states[number] = normalized
#
#         elif mtype & 0x01 and value == 1:  # button event
#           if number in (0, 19):  # X
#             q.put(COMMAND_TEMPLATES["cruise_down"])
#           elif number in (3, 18):  # triangle
#             q.put(COMMAND_TEMPLATES["cruise_up"])
#           elif number in (1, 6):  # square
#             q.put(COMMAND_TEMPLATES["cruise_cancel"])
#           elif number in (10, 21):  # R3
#             q.put(COMMAND_TEMPLATES["reverse_switch"])
#       else:
#         # send last value if there is no new event
#         # print(last_axis_states)
#         if last_axis_states.get(2) > 0.0:
#           q.put(COMMAND_TEMPLATES["throttle"].format(last_axis_states.get(2)))
#         if last_axis_states.get(3) > 0.0:
#           q.put(COMMAND_TEMPLATES["brake"].format(last_axis_states.get(3)))
#         if last_axis_states.get(0) < 0.0 or last_axis_states.get(0) > 0.0:
#           q.put(COMMAND_TEMPLATES["steer"].format(last_axis_states.get(0)))
#         time.sleep(0.01)
#
#
# if __name__ == '__main__':
#   q: Queue[str] = Queue()
#   p = Process(target=wheel_poll_thread, args=(q,))
#   p.start()




# !/usr/bin/env python3
import pygame
from typing import NoReturn
import time
from multiprocessing import Process, Queue

COMMAND_TEMPLATES = {
  "throttle": "throttle_{:.2f}",
  "brake": "brake_{:.2f}",
  "steer": "steer_{:.2f}",
  "cruise_down": "cruise_down",
  "cruise_up": "cruise_up",
  "cruise_cancel": "cruise_cancel",
  "reverse_switch": "reverse_switch"
}


def wheel_poll_thread(q: 'Queue[str]') -> NoReturn:
  # init pygame
  pygame.init()

  # check for joystick cont
  if pygame.joystick.get_count() == 0:
    print("No joystick detected!")
    return

  # init the first joystick
  js = pygame.joystick.Joystick(0)
  js.init()
  print(f'Device name: {js.get_name()}')
  print(f'Number of axes: {js.get_numaxes()}')
  print(f'Number of buttons: {js.get_numbuttons()}')

  # Initialize axis and button states
  axis_states = [0.0] * js.get_numaxes()
  button_states = [0] * js.get_numbuttons()

  # Enable FF
  import evdev  # pylint: disable=import-error
  from evdev import ecodes, InputDevice  # pylint: disable=import-error
  device = evdev.list_devices()[0]
  evtdev = InputDevice(device)
  val = 24000
  evtdev.write(ecodes.EV_FF, ecodes.FF_AUTOCENTER, val)

  while True:
    pygame.event.pump()

    # for event in events:
    for i in range(js.get_numaxes()):
      axis_states[i] = js.get_axis(i)
      if i == 2:
        normalized = (1 - js.get_axis(i)) * 0.5 #[-1.0, 1.0]
        q.put(COMMAND_TEMPLATES["throttle"].format(normalized))
      elif i == 3:
        normalized = (1 - js.get_axis(i)) * 0.5 #[-1.0, 1.0]
        q.put(COMMAND_TEMPLATES["brake"].format(normalized))
      elif i == 0:
        normalized = -js.get_axis(i) #[-1.0, 1.0]
        q.put(COMMAND_TEMPLATES["steer"].format(normalized))

    for i in range(js.get_numbuttons()):
      button_states[i] = js.get_button(i)
      if i in (0, 19):
        q.put("cruise_down")
      elif i in (3, 18):
        q.put("cruise_up")
      elif i in (1, 6):
        q.put("cruise_cancel")
      elif i in (10, 21):
        q.put("reverse_switch")

    time.sleep(0.01)

if __name__ == '__main__':
  q: Queue[str] = Queue()
  p = Process(target=wheel_poll_thread, args=(q,))
  p.start()
