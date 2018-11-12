"""
    Ensimag 2018 TP Perf.
"""

import simpy
import math
from random import expovariate
import matplotlib.pyplot as plt

class Packet(object):
	""" Packet structure

    Attributes:
    	id (int):
    		Packet identifier
        size (int): 
        	Packet size in Bytes.
		generation_timestamp (float): 
			Timestamp (simulated time) of packet generation
		output_timestamp (float): 
			Timestamp (simulated time) when packet leaves a system
	"""
	def __init__(self, id, size, generation_timestamp):
		self.id=id
		self.size=size
		self.generation_timestamp=generation_timestamp
		self.output_timestamp=0

class Source(object):
	""" Packet generator

	Attributes:
        env (simpy.Environment): 
        	Simulation environment
        name (str): 
        	Name of the source
        gen_distribution (callable): 
        	Function that returns the successive inter-arrival times of the packets
        size_distribution (callable): 
        	Function that returns the successive sizes of the packets
        init_delay (int): 
        	Starts generation after an initial delay. Default = 0
    	destination (object):
    		Entity that receives the packets from the generator
        debug (bool): 
        	Set to true to activate verbose debug 
	"""
	def __init__(self, env, name, init_delay=0, gen_distribution=lambda:1, size_distribution=lambda:1000,debug=False):
		self.env=env
		self.name=name
		self.init_delay=init_delay
		self.gen_distribution=gen_distribution
		self.size_distribution=size_distribution
		self.packet_count=0
		self.debug=debug
		self.destination= None
		self.action= env.process(self.run())

	def run(self):
		""" Packet generation loop

		"""
		# Initial waiting time
		yield self.env.timeout(self.init_delay)
		while True:
			#### TODO: add here packet generation event ####
			# yield ...
			#### TODO: define here packet size
			# packet_size=
			generated_packet= Packet(id=self.packet_count,size=packet_size,generation_timestamp=env.now)
			if self.debug:
				print("Packet (id=%d,size=%d) generated by %s at %f" %(self.packet_count, packet_size, self.name,
					generated_packet.generation_timestamp))
			if self.destination is not None:
				if self.debug:
					print("%s => %s" % (self.name, self.destination.name))
				self.destination.put(generated_packet)
			self.packet_count+=1

	def attach(self, destination):
		""" Method to set a destination for the generated packets
		
		Args:
			destination (QueuedServer || XMonitor): 
		"""		
		self.destination= destination



class QueuedServer(object):
	""" Represents a waiting queue and an associated server.
	
	Attributes:
		env (simpy.Environment): 
        	Simulation environment
    	name (str):
    		Name of the source
		buffer (simpy.Store):
			Simpy FIFO queue 
		buffer_max_size (int):
			Maximum buffer size in bytes
		buffer_size (int): 
			Current size of the buffer in bytes
		service_rate (float):
			Server service rate in byte/sec
    	destination (object):
    		Entity that receives the packets from the server 
		debug (bool): 
			Set to true to activate verbose debug 
		busy (bool):
			Is set if packet is currently processed by the server
		packet_count (int):
			Number of packet received
		packet_drop (int):
			Number of packets dropped


	"""

	def __init__(self, env,name, buffer_max_size=None, service_rate=1000,debug=False):
		self.env= env
		self.name= name
		self.buffer= simpy.Store(self.env,capacity=math.inf) # buffer size is limited by put method
		self.buffer_max_size= buffer_max_size
		self.buffer_size=0  
		self.service_rate= service_rate
		self.destination=None
		self.debug=debug
		self.busy=False
		self.packet_count=0
		self.packets_drop=0
		self.action= env.process(self.run())

	def run(self):
		""" Packet waiting & service loop

		"""
		while True:
			#### TODO: add event to get packet from buffer
			#...
			self.busy=True
			#### TODO: add event to process packet
			#...
			packet.output_timestamp= env.now
			if self.destination is not None:
				destination.put(packet)
			self.busy=False

	def put(self, packet):
		self.packet_count += 1
		buffer_futur_size = self.buffer_size + packet.size

		if self.buffer_max_size is None or buffer_futur_size <= self.buffer_max_size:
			self.buffer_size = buffer_futur_size
			# TODO: add packet put event in the buffer
			# ...
			if self.debug:
				print("Packet %d added to queue %s." % (packet.id, self.name))
		else:
			self.packets_drop += 1
			if self.debug:
				print("Packet %d is discarded by queue %s. Reason: Buffer overflow." % (packet.id, self.name))

	def attach(self, destination):
		""" Method to set a destination for the serviced packets
		
		Args:
			destination (QueuedServer || XMonitor): 
		"""		
		self.destination=destination

class QueuedServerMonitor(object):
    """ A monitor for a QueuedServer. Observes the packets in service and in 
    	the queue and records that info in the sizes[] list. The monitor looks at the queued server 
    	at time intervals given by the sampling dist.

        
        Attributes:
        env (simpy.Environment):
        	Simulation environment
        queued_server (QueuedServer):
        	QueuedServer to monitor
        sample_distribution (callable):
        	Function that returns the successive inter-sampling times
    	sizes (list[int]):
    		List of the successive number of elements in queue. Elements can be packet or bytes 
    		depending on the attribute count_bytes 
		count_bytes (bool):
			If set counts number of bytes instead of number of packets 
    """
    def __init__(self, env, queued_server, sample_distribution=lambda:1, count_bytes=False):
        self.env= env
        self.queued_server= queued_server
        self.sample_distribution= sample_distribution
        self.count_bytes= count_bytes
        self.sizes= []
        self.time_count=0
        self.action= env.process(self.run())
        

    def run(self):
        while True:
            yield self.env.timeout(self.sample_distribution())
            self.time_count+=1
            if self.count_bytes:
                total= self.queued_server.buffer_size
            else:
                total= len(self.queued_server.buffer.items) + self.queued_server.busy
            self.sizes.append(total)
			


if __name__=="__main__":
	# Link capacity 64kbps
	process_rate= 64000/8 # => 8kbps
	# Packet length exponentially distributed with average 400 bytes
	dist_size= lambda:expovariate(1/400) 
	# Packet inter-arrival time exponentially distributed 
	gen_dist= lambda:expovariate(15) # 15 packets per second
	env= simpy.Environment()
	src1= Source(env, "Source 1",gen_distribution=gen_dist,size_distribution=dist_size,debug=True)
	qs1= QueuedServer(env,"Router 1", buffer_max_size=math.inf, service_rate=process_rate,debug=True)
	# Link Source 1 to Router 1
	src1.attach(qs1)
	# Associate a monitor to Router 1
	qs1_monitor=QueuedServerMonitor(env,qs1,sample_distribution=lambda:1,count_bytes=False)
	env.run(until=1000)