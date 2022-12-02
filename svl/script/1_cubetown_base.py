#!/usr/bin/python3
import socket
import lgsvl
from lgsvl.geometry import Transform
from time import sleep

ASSET_ID = "3c5d44b4-8629-4888-a338-2e9e37604a97"
SPAWN_X = 0
SPAWN_Y = 0
SPAWN_Z = 18.3
SPAWN_R = 90

class Exp(object):
	def __init__(self):
		self.sim = lgsvl.Simulator(
		    address="localhost",
		    port=8181)

		# reset scene
		self.sim.load("CubeTown")
		# calc position
		spawns = self.sim.get_spawn()
		# calc offset
		self.origin = Transform(
			spawns[0].position,
			spawns[0].rotation)
		self.origin.position.x += SPAWN_X
		self.origin.position.y += SPAWN_Y
		self.origin.position.z += SPAWN_Z
		self.origin.rotation.y += SPAWN_R

		self.u_forward = lgsvl.utils.transform_to_forward(self.origin)
		self.u_right = lgsvl.utils.transform_to_right(self.origin)
		self.u_up = lgsvl.utils.transform_to_up(self.origin)

	def create_ego(self, sim):
		# ego (main car)
		ego_state = lgsvl.AgentState()
		ego_state.transform = \
			Transform(self.origin.position, self.origin.rotation)
		ego = sim.add_agent(ASSET_ID,
			lgsvl.AgentType.EGO, ego_state)
		
		ego.connect_bridge("localhost", 9090)

		return

	def wait_for_bridge(self):
		bridge_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		location = ("localhost", 9090)
		
		socket_open = bridge_socket.connect_ex(location)

		while socket_open:
			print('[System] Wait for rosbridge')
			sleep(2)
			socket_open = bridge_socket.connect_ex(location)

		print('[System] Bridge connected')

		bridge_socket.close()

	def setup_sim(self):
		self.wait_for_bridge()
		self.create_ego(self.sim)

	def run(self):
		self.sim.reset()
		self.setup_sim()
		print('[System] Simulation Starts')
		self.sim.run()

if __name__ == '__main__':
	e = Exp()
	e.run()
	exit()
