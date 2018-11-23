"""
    Ensimag 2018 TP Perf.
"""

import simpy
import math
from random import expovariate
import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.stats as st

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
        self.id = id
        self.size = size
        self.generation_timestamp = generation_timestamp
        self.output_timestamp = 0


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

    def __init__(self, env, name, init_delay=0, gen_distribution=lambda: 1, size_distribution=lambda: 1000, debug=False):
        self.env = env
        self.name = name
        self.init_delay = init_delay
        self.gen_distribution = gen_distribution
        self.size_distribution = size_distribution
        self.packet_count = 0
        self.debug = debug
        self.destination = None
        self.action = env.process(self.run())

    def run(self):
        """ Packet generation loop

        """
        # Initial waiting time
        yield self.env.timeout(self.init_delay)
        while True:
            #### TODO: add here packet generation event ####
            yield env.timeout(self.gen_distribution())
            # TODO: define here packet size
            packet_size = self.size_distribution()
            generated_packet = Packet(id=self.packet_count, size=packet_size, generation_timestamp=env.now)
            if self.debug:
                print("Packet (id=%d,size=%d) generated by %s at %f" % (
                    self.packet_count, packet_size, self.name, generated_packet.generation_timestamp))
            if self.destination is not None:
                if self.debug:
                    print("%s => %s" % (self.name, self.destination.name))
                self.destination.put(generated_packet)
            self.packet_count += 1

    def attach(self, destination):
        """ Method to set a destination for the generated packets

        Args:
                destination (QueuedServer || XMonitor):
        """
        self.destination = destination


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

    def __init__(self, env, name, buffer_max_size=None, service_rate=1000, debug=False):
        self.env = env
        self.name = name
        # buffer size is limited by put method
        self.buffer = simpy.Store(self.env, capacity=math.inf)
        self.buffer_max_size = buffer_max_size
        self.buffer_size = 0
        self.service_rate = service_rate
        self.destination = None
        self.debug = debug
        self.busy = False
        self.packet_count = 0
        self.packets_drop = 0
        self.action = env.process(self.run())

    def run(self):
        """ Packet waiting & service loop

        """
        while True:
            # TODO: add event to get packet from buffer
            packet = yield self.buffer.get()
            self.busy = True
            # TODO: add event to process packet
            yield env.timeout(packet.size/self.service_rate)
            packet.output_timestamp = env.now
            if self.destination is not None:
                if self.destination.busy is False:
                    self.destination.put(packet)
                else:
                    self.packets_drop += 1
                    self.destination.put(packet)
            self.busy = False
	
    # def put(self, packet):
    def put(self, packet):
        self.packet_count += 1
        buffer_futur_size = self.buffer_size + packet.size

        if self.buffer_max_size is None or buffer_futur_size <= self.buffer_max_size:
            self.buffer_size = buffer_futur_size
            # TODO: add packet put event in the buffer
            self.buffer.put(packet)
                
            if self.debug:
                print("Packet %d added to queue %s." % (packet.id, self.name))
        
            # Remove duplicates
        #     self.catch_collision()

        else:
            self.packets_drop += 1
            if self.debug:
                print("Packet %d is discarded by queue %s. Reason: Buffer overflow." % (
                packet.id, self.name))

    def catch_collision(self):
        if len(self.buffer.items) > 1:
                # Precision to centi seconds
            if int(100*self.buffer.items[0].output_timestamp) == int(100*self.buffer.items[1].output_timestamp):
                self.buffer.get()
                self.packets_drop += 1
                if self.debug:
                    print("Packet %d is discarded by queue %s. Reason: Collision." % (
                        self.buffer.items[0].id, self.name))

    def attach(self, destination):
        """ Method to set a destination for the serviced packets

        Args:
                destination (QueuedServer || XMonitor):
        """
        self.destination = destination


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
                
                ADDED:
        debug_average_number (bool):
                If set, displays the average number of packets/bytes depending on count_bytes
        debug_latency (bool):
                If set, displays the latency of the current queued_server
        debug_dropped (bool):
                If set, displays the number of packet dropped by each queued_server


    """

    def __init__(self, env, queued_server, sample_distribution=lambda: 1, count_bytes=False, debug_average_number = False, debug_latency = False, debug_dropped = False):
        self.env = env
        self.queued_server = queued_server
        self.sample_distribution = sample_distribution
        self.count_bytes = count_bytes
        self.sizes = []
        self.time_count = 0
        self.action = env.process(self.run())
        self.latencies = [] # represents the latencies at each sample distribution time
        self.debug_average_number = debug_average_number
        self.debug_latency = debug_latency
        self.debug_dropped = debug_dropped

    def run(self):
        while True:
            yield self.env.timeout(self.sample_distribution())
            self.time_count += 1
            if self.count_bytes:
                total = self.queued_server.buffer_size
            else:
                total = len(self.queued_server.buffer.items) + self.queued_server.busy
            self.sizes.append(total)
            
            # Compute the current latency
            if len(self.queued_server.buffer.items) > 0:
                current_latency = self.queued_server.buffer.items[len(self.queued_server.buffer.items)-1].output_timestamp - self.queued_server.buffer.items[len(self.queued_server.buffer.items)-1].generation_timestamp
                self.latencies.append(current_latency)
            
            # Print average bytes/number of packets and latency according to the debug parameters
            if self.debug_latency:
                average_latency = np.mean(self.latencies)
                # print("Average latency: " + str(average_latency))

                latency_interval = st.t.interval(0.99, len(self.latencies), loc=np.mean(self.latencies), scale=st.sem(self.latencies))
                print('Mean of latency %f, interval(99): %s' %(average_latency, latency_interval))
                
        
            if self.debug_average_number:
                average_packet_nb = np.mean(self.sizes)
                # print("Average packet number: " + str(average_packet_nb) + " | at time : " + str(self.time_count))

                nb_packet_interval = st.t.interval(0.99, len(self.sizes)-1, loc=np.mean(self.sizes), scale=st.sem(self.sizes))
                print('Average number of packets %f, interval(99): %s' %(average_packet_nb, nb_packet_interval))

            # Print packet dropped by queued_server
            if self.debug_dropped:
                print("Packets counted by " + str(self.queued_server.name) + ": " + str(self.queued_server.packet_count))
                print("Packets dropped by " + str(self.queued_server.name) + ": " + str(self.queued_server.packets_drop))
                print("Ratio transmit/total: " + str(int(100*(self.queued_server.packet_count-self.queued_server.packets_drop)/self.queued_server.packet_count)) + "%")


if __name__ == "__main__":
        arg = sys.argv[1] # either with or without 

        if arg != "with" or arg != "without":
                print("unknow parameter, please choose either 'with' or 'without'")

        # # SIMULATION TWO SOURCES WITHOUT COLLISION
        if arg == "without":
                # Link capacity 64kbps
                process_rate = 64000/8  # => 8 kBytes per second
                # Packet length exponentially distributed with average 400 bytes
                dist_size= lambda:expovariate(1/400)
                # Packet inter-arrival time exponentially distributed
                gen_dist1= lambda:expovariate(7.5)  # 7.5 packets per second
                gen_dist2= lambda:expovariate(7.5)  # 7.5 packets per second
                env = simpy.Environment()
                src1 = Source(env, "Source 1", gen_distribution=gen_dist1,
                                size_distribution=dist_size, debug=False)
                src2 = Source(env, "Source 2", gen_distribution=gen_dist2,
                                size_distribution=dist_size, debug=False)
                qs1 = QueuedServer(env, "Router 1", buffer_max_size=math.inf,
                                        service_rate=process_rate, debug=False)
                # Link Source 1 to Router 1
                src1.attach(qs1)
                # Link Source 2 to Router 1
                src2.attach(qs1)
                # Associate a monitor to Router 1
                qs1_monitor = QueuedServerMonitor(
                        env, qs1, sample_distribution=lambda: 1, count_bytes=False, debug_average_number=True, debug_latency=False)

                # Create another QueuedServer to "catch" packet output_timestamp
                qs2 = QueuedServer(env, "Router 1 output", buffer_max_size=math.inf,
                                        service_rate=process_rate, debug=False)
                # Attaching qs1 to qs2 so we catch the packets
                qs1.attach(qs2)
                # Create another monitor that will display the latency of each packet received by qs2 (given by qs1)
                qs2_monitor = QueuedServerMonitor(
                        env, qs2, sample_distribution=lambda: 1, count_bytes=False, debug_average_number=False, debug_latency=True)

                env.run(until=1000)

        if arg == "with":
        # # SIMULATION TWO ROUTER WITH COLLISION
                process_rate = 64000/8  # => 8 kBytes per second
                # Packet length exponentially distributed with average 400 bytes
                dist_size= lambda:expovariate(1/400)
                # Packet inter-arrival time exponentially distributed
                gen_dist1= lambda:expovariate(7.5)  # 7.5 packets per second
                gen_dist2= lambda:expovariate(7.5)  # 7.5 packets per second
                env = simpy.Environment()
                src1 = Source(env, "Source 1", gen_distribution=gen_dist1,
                                size_distribution=dist_size, debug=False)
                src2 = Source(env, "Source 2", gen_distribution=gen_dist2,
                                size_distribution=dist_size, debug=False)
                qs1 = QueuedServer(env, "Router 1", buffer_max_size=math.inf,
                                        service_rate=process_rate, debug=False)
                qs2 = QueuedServer(env, "Router 2", buffer_max_size=math.inf,
                                        service_rate=process_rate, debug=False)
                # Link Source 1 to Router 1
                src1.attach(qs1)
                # Link Source 2 to Router 2
                src2.attach(qs2)
                
                qs3 = QueuedServer(env, "Router 3", buffer_max_size=math.inf,
                                        service_rate=process_rate, debug=False)
                qs4 = QueuedServer(env, "Router 3 output", buffer_max_size=math.inf,
                                        service_rate=process_rate, debug=False)
                # Link Router 1 and 2 to Router 3
                qs1.attach(qs3)
                qs2.attach(qs3)
                qs3.attach(qs4)
                # Associate a monitor to Router 1
                qs1_monitor = QueuedServerMonitor(
                        env, qs1, sample_distribution=lambda: 1, count_bytes=False, debug_average_number=False, debug_latency=False, debug_dropped=True)
                # Associate a monitor to Router 2
                qs2_monitor = QueuedServerMonitor(
                        env, qs2, sample_distribution=lambda: 1, count_bytes=False, debug_average_number=False, debug_latency=False, debug_dropped=True)
                # Associate a monitor to Router 3
                qs3_monitor = QueuedServerMonitor(
                        env, qs3, sample_distribution=lambda: 1, count_bytes=False, debug_average_number=True, debug_latency=False, debug_dropped=False)
                # Associate a monitor to Router 3 output
                qs4_monitor = QueuedServerMonitor(
                        env, qs4, sample_distribution=lambda: 1, count_bytes=False, debug_average_number=False, debug_latency=True, debug_dropped=False)

                env.run(until=1000)
        