import gym
import numpy as np
from gym import spaces
import argparse
from dataclasses import dataclass
from peaceful_pie.unity_comms import UnityComms
import requests

@dataclass
class SubPos:
    x: float
    y: float
    z: float

@dataclass
class SubRot:
    roll: float
    pitch: float
    yaw: float

@dataclass
class SubVel:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float

class SubmarineEnv(gym.env):
  def __init__(self, unity_port: str = 9999, inputs_url: str = '127.0.0.1', inputs_port: int = 9999):
    self.unity_comms = UnityComms(port=unity_port)
    self.url = f'http://{inputs_url}:{inputs_port}/inputs'
    self.pos_url = f'http://{inputs_url}:{inputs_port}/position'
    self.rot_url = f'http://{inputs_url}:{inputs_port}/rotation'
    self.vel_url = f'http://{inputs_url}:{inputs_port}/velocity'


    # Observation = [x, y, z, roll, pitch, yaw, vx, vy, vz, vroll, vpitch, vyaw]
    self.observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(12,),
        dtype=np.float32
    )

  # Action = [X, Y, Z, Roll, Pitch, Yaw, S1, S2, S3, Arm]
  self.action_space = spaces.Box(
      low=np.array([-128]*9 + [0], dtype=np.float32),
      high=np.array([128]*9 + [1], dtype=np.float32),
      shape=(10,),
      dtype=np.float32
  )

  def reset(self):
    self.unity_comms.restartPosition()

  def step(self, action):
    # step through action
    requests.post(self.url, json=action.tolist())

  def getSubPosition(self):
    # get sub position
    pos_data = requests.get(self.pos_url).json()
    return SubPos(pos_data['x'], pos_data['y'], pos_data['z'])

  def getSubRotation(self):
    # get sub rotation
    rot_data = requests.get(self.rot_url).json()
    return SubRot(rot_data['roll'], rot_data['pitch'], rot_data['yaw'])

  def getSubVelocity(self):
    # get sub velocity
    vel_data = requests.get(self.vel_url).json()
    return SubVel(vel_data['x'], vel_data['y'], vel_data['z'], vel_data['roll'], vel_data['pitch'], vel_data['yaw'])

  def getObservation(self):
    # Get data from Unity
    pos = self.unity_comms.get_pos()        # [x, y, z]
    rot = self.unity_comms.get_rot()        # [roll, pitch, yaw]
    vel = self.unity_comms.get_vel()        # [[vx, vy, vz], [vroll, vpitch, vyaw]]

    linear_vel = vel[0]                     # [vx, vy, vz]
    angular_vel = vel[1]                    # [vroll, vpitch, vyaw]

    # Concatenate into one 12D observation vector
    observation = np.array(
        pos + rot + linear_vel + angular_vel,
        dtype=np.float32
    )

    return observation