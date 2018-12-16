"""
    Ensimag 2018 TP Perf.
"""

import simpy
import math
from random import expovariate
import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.stats as st
from enum import Enum

# class State(Enum):
#     IDLE = "IDLE"
#     BUSY = "BUSY"

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

    def __init__(self, env, queued_server, sample_distribution=lambda: 1, count_bytes=False, debug_average_number = False, debug_latency = False, debug_dropped = False, d = 1, debug_throughput = False):
        self.env = env
        self.queued_server = queued_server
        self.sample_distribution = sample_distribution
        self.count_bytes = count_bytes
        self.sizes = []
        self.time_count = 0
        self.debug_average_number = debug_average_number
        self.debug_latency = debug_latency
        self.debug_dropped = debug_dropped
        self.debug_throughput = debug_throughput
        self.action = env.process(self.run())
        self.latenciesMonitor = []
        self.average_packet_nb = 0
        self.d = d
        self.throughput = []
        self.rho = []

    def run(self):
        while True:
            yield self.env.timeout(self.sample_distribution())
            self.time_count += 1
            if self.count_bytes:
                total = self.queued_server.buffer_size
            else:
                total = len(self.queued_server.buffer.items) + self.queued_server.busy
            self.sizes.append(total)
            
            # Print average bytes/number of packets and latency according to the debug parameters
            if self.debug_latency:
                latencies = self.queued_server.latencies
                # print("Average latency: " + str(average_latency))

                self.latenciesMonitor.append(np.mean(latencies))

                latency_interval = st.t.interval(0.99, len(latencies), loc=np.mean(latencies), scale=st.sem(latencies))
                print('Mean of latency %f, interval(99): %s' %(np.average(latencies), latency_interval))
                
        
            if self.debug_average_number:
                self.average_packet_nb = np.mean(self.sizes)
                # print("Average packet number: " + str(average_packet_nb) + " | at time : " + str(self.time_count))

                nb_packet_interval = st.t.interval(0.99, len(self.sizes)-1, loc=np.mean(self.sizes), scale=st.sem(self.sizes))
                print('Average number of packets %f, interval(99): %s' %(self.average_packet_nb, nb_packet_interval))

            # Print packet dropped by queued_server
            if self.debug_dropped:
                print("Packets counted by " + str(self.queued_server.name) + ": " + str(self.queued_server.packet_count))
                print("Packets dropped by " + str(self.queued_server.name) + ": " + str(self.queued_server.packets_drop))
                print("Ratio transmit/total: " + str(int(100*(self.queued_server.packet_count-self.queued_server.packets_drop)/self.queued_server.packet_count)) + "%")
            
            if self.debug_throughput:
                self.throughput.append(np.mean(self.queued_server.throughput))
                print("Average throughput " + str(np.mean(self.throughput)) + " b/s")
                self.rho.append(np.mean(self.queued_server.rho))
                print("Average rho " + str(np.mean(self.rho)))
            
        #     if self.time_count == 299*self.d:
                # plt.plot(self.latenciesMonitor, "-", color="blue", linewidth=2.5, label="average latency")
                # plt.xlabel("Time")
                # plt.ylabel("Latency")
                # plt.legend(loc="lower right", frameon=False)
                # plt.title("Latency evolution over time")
                # plt.show()

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
            channel (Channel):
                Channel linked with
    """

    def __init__(self, env, name, channel, buffer_max_size=None, service_rate=1000, debug=False, d=1):
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
        self.channel = channel
        self.latencies = []
        self.d = d
        self.throughput = []
        self.rho = []
        self.action = env.process(self.run())

    def run(self):
        """ Packet waiting & service loop

        """
        while True:
            packet = yield self.buffer.get()
            self.busy = True
            synchroTime = self.channel.synchronize(self.env.now, packet.size)
            yield env.timeout(synchroTime)
            self.channel.add_sender(self)
            # Aloha implementation
            if self.channel.state == "IDLE":
                yield env.timeout(packet.size/self.channel.service_rate)
            else:
                randPeriod = random.randint(1, 5) #number of slotTime
                self.channel.remove_sender(self)
                yield env.timeout(randPeriod * self.channel.timeSlot+0.001) #timeSlot of the channel (usually 0.05 but strange behavior with this value)
            packet.output_timestamp = env.now
            latency = packet.output_timestamp - packet.generation_timestamp

            self.throughput.append(8*packet.size/latency)
            self.rho.append(7.5 * packet.size/self.channel.service_rate)

            if self.destination is not None:
                if self.channel.state == "IDLE" and self.destination.busy is False:
                    self.destination.put(packet)
                else:
                    self.packets_drop += 1
            else:
                if self.channel.state == "BUSY":
                    self.packets_drop += 1
                else:
                    self.latencies.append(packet.output_timestamp - packet.generation_timestamp)
                #     print(np.average(self.latencies))
                    
            self.busy = False
            self.channel.remove_sender(self)

    def put(self, packet):
        self.packet_count += 1
        buffer_futur_size = self.buffer_size + packet.size

        if self.buffer_max_size is None or buffer_futur_size <= self.buffer_max_size:
            self.buffer_size = buffer_futur_size
            self.buffer.put(packet)
                
            if self.debug:
                print("Packet %d added to queue %s." % (packet.id, self.name))

        else:
            self.packets_drop += 1
            if self.debug:
                print("Packet %d is discarded by queue %s. Reason: Buffer overflow." % (
                packet.id, self.name))

    def attach(self, destination):
        """ Method to set a destination for the serviced packets

        Args:
                destination (QueuedServer || XMonitor):
        """
        self.destination = destination

class Channel(object):
    """ A channel that aims to manage the packet transmission.
    It has a state, busy or not and broadcasts this information to its sources
    
    Attributes:
    env (simpy.Environment):
            Simulation environment
    senders_list (List[QueuedServer]):
            List of QueuedServers linked
    service_rate (int):
            Service rate of the total simulation
    collision (bool):
            If true, simulates with collisions else without collisions
    state (State):
            The current state of the channel, either "BUSY" or "IDLE"
    debug (bool):
            If set, display debug informations
    """

    def __init__(self, env, name, service_rate, collision, state="IDLE", senders_list=[], debug=False, sample_distribution= lambda:1):
        self.env = env
        self.name = name
        self.senders_list = senders_list
        self.service_rate = service_rate
        self.collision = collision
        self.state = state
        self.debug = debug
        self.sample_distribution = sample_distribution
        self.action = env.process(self.broadcast_collision())
        self.timeSlot = 400/self.service_rate

    def add_sender(self, sender):
        # print("call add_sender")
        self.senders_list.append(sender)

    def remove_sender(self, sender):
        # print("call remove_sender")
        if sender in self.senders_list:
            self.senders_list.remove(sender)

    def broadcast_collision(self):
        print("Channel created: " + self.name)
        while True:
            yield self.env.timeout(self.sample_distribution())
            if self.collision and len(self.senders_list) > 1:
                self.state = "BUSY"
                if self.debug:
                    print('Collision detected between [%s]' % ', '.join(map(str, self.senders_list)))
            else:
                self.state = "IDLE"

    def synchronize(self, envnow, packetSize):
        if envnow > self.timeSlot: # When the first packet arrive
            slotsCounter = 1
            copyEnvNow = envnow
            slotsCounter += int((packetSize/self.service_rate)/self.timeSlot)
            while copyEnvNow > self.timeSlot:
                copyEnvNow -= self.timeSlot
                slotsCounter += 1

        #     print("fff: " + str(( slotsCounter) * self.timeSlot))
        #     print("envnow: " + str(envnow))
            return ( slotsCounter) * self.timeSlot - envnow
        else: # If the router want to send packet before the first timeSlot, it is fine
            return 0
        
        
def alohaPure(process_rate, dist_size, gen_dist1, gen_dist2, env, d, l=1):
        
        src1 = Source(env, "Source 1", gen_distribution=gen_dist1,
                        size_distribution=dist_size, debug=False)
        src2 = Source(env, "Source 2", gen_distribution=gen_dist2,
                        size_distribution=dist_size, debug=False)
        # Create channel
        ch = Channel(env, "Disney", process_rate, collision=True, state="IDLE", senders_list=[], debug=False)
        qs1 = QueuedServer(env, "Router 1", ch, buffer_max_size=math.inf,
                                service_rate=process_rate, debug=False, d=d)
        qs2 = QueuedServer(env, "Router 2", ch, buffer_max_size=math.inf,
                           service_rate=process_rate, debug=False, d=d)
        # Link Source 1 to Router 1
        src1.attach(qs1)
        # Link Source 2 to Router 2
        src2.attach(qs2)

        
        # Associate a monitor to Router 1
        qs1_monitor = QueuedServerMonitor(
                env, qs1, sample_distribution=lambda: 1, count_bytes=False, debug_average_number=True, debug_latency=True, debug_dropped=False, d=d, debug_throughput=True)
        # Create another monitor that will display the latency of each packet received by qs2 (given by qs1)
        qs2_monitor = QueuedServerMonitor(
                env, qs2, sample_distribution=lambda: 1, count_bytes=False, debug_average_number=True, debug_latency=True, debug_dropped=False, d=d, debug_throughput=True)

        # env.run(until=300*d)
        # env.run(until=300*l)
        env.run(until=1000)

        # latenciesMonitor1 = qs1_monitor.latenciesMonitor[len(qs1_monitor.latenciesMonitor) - 1] #final latency of qs1_monitor
        # latenciesMonitor2 = qs2_monitor.latenciesMonitor[len(qs2_monitor.latenciesMonitor) - 1] #final latency of qs2_monitor
        # latenciesMonitor = (latenciesMonitor1+latenciesMonitor2)/2
        
        # averagePacketsMonitor = (qs1_monitor.average_packet_nb+qs2_monitor.average_packet_nb)/2
        # return averagePacketsMonitor

        throughput1 = qs1_monitor.throughput
        rho1 = qs1_monitor.rho
        return [throughput1, rho1]


def alohaSlotted(env, process_rate, dist_size, gen_dist1, gen_dist2):
        src1 = Source(env, "Source 1", gen_distribution=gen_dist1,
                        size_distribution=dist_size, debug=False)
        src2 = Source(env, "Source 2", gen_distribution=gen_dist2,
                        size_distribution=dist_size, debug=False)
        # Create channel
        ch = Channel(env, "Disney", process_rate, collision=True, state="IDLE", senders_list=[], debug=False)
        qs1 = QueuedServer(env, "Router 1", ch, buffer_max_size=math.inf,
                                service_rate=process_rate, debug=False)
        qs2 = QueuedServer(env, "Router 2", ch, buffer_max_size=math.inf,
                           service_rate=process_rate, debug=False)
        # Link Source 1 to Router 1
        src1.attach(qs1)
        # Link Source 2 to Router 2
        src2.attach(qs2)

        
        # Associate a monitor to Router 1
        qs1_monitor = QueuedServerMonitor(
                env, qs1, sample_distribution=lambda: 1, count_bytes=False, debug_average_number=True, debug_latency=True, debug_dropped=True, d=1, debug_throughput=True)
        # Create another monitor that will display the latency of each packet received by qs2 (given by qs1)
        qs2_monitor = QueuedServerMonitor(
                env, qs2, sample_distribution=lambda: 1, count_bytes=False, debug_average_number=True, debug_latency=True, debug_dropped=True, d=1, debug_throughput=True)

        env.run(until=300)
        throughput1 = qs1_monitor.throughput
        rho1 = qs1_monitor.rho
        return [throughput1, rho1]
        
if __name__ == "__main__":
        ###########################
        #### Init of variables ####
        ###########################
        # Link capacity 64kbps
        process_rate = 64000/8  # => 8 kBytes per second
        # Packet length exponentially distributed with average 400 bytes
        dist_size= lambda:expovariate(1/400)
        # Packet inter-arrival time exponentially distributed
        gen_dist1= lambda:expovariate(7.5)  # 7.5 packets per second
        gen_dist2= lambda:expovariate(7.5)  # 7.5 packets per second
        env = simpy.Environment()


        #################################
        #### Tests for slotted Aloha ####
        #################################
        ans = alohaSlotted(env, process_rate, dist_size, gen_dist1, gen_dist2)

        ##############################
        #### Tests for pure Aloha ####
        ##############################
        # testOfDLatencies = []
        # for d in np.linspace(0.5, 3.5, 11):
        #         latencies = alohaPure(process_rate, dist_size, gen_dist1, gen_dist2, env, d)
        #         testOfDLatencies.append(latencies)
        
        # plt.plot(np.linspace(0.5, 3.5, 11), testOfDLatencies, '-o', color="blue", linewidth=2.5, label="average latency")
        # plt.xlabel("d (Upper bound of random interval)")
        # plt.ylabel("Latency (s)")
        # plt.legend(loc="upper left", frameon=False)
        # plt.title("Evolution of latency according to upper bound of random interval")
        # plt.show()

        # testOfDLatencies = []
        # averagePacketsNumber = []
        # for l in np.linspace(3, 8, 5):
        #         avg = alohaPure(process_rate, dist_size, gen_dist1, gen_dist2, env, 1.5)
        #         averagePacketsNumber.append(avg)
        
        # plt.plot(np.linspace(3, 8, 5), averagePacketsNumber, '-o', color="blue", linewidth=2.5, label="average packet's number")
        # plt.xlabel("d (Upper bound of random interval)")
        # plt.ylabel("Average packet's number")
        # plt.legend(loc="upper left", frameon=False)
        # plt.title("Packet's number according to upper bound of random interval")
        # plt.show()

        # ans = alohaPure(process_rate, dist_size, gen_dist1, gen_dist2, env, 1.5)

        # Display plot for throughput according to rho:
        # plt.plot(ans[1], ans[0], '+', color="red", linewidth=2.5)
        # plt.xlabel("p (rho)")
        # plt.ylabel("Throughput (b/s)")
        # plt.legend(loc="upper left", frameon=False)
        # plt.title("Throughput according to p")
        # plt.show()

