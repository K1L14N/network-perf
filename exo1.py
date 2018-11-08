import simpy
import random
import numpy as np

# Producer
def producer(env, store, lamb):
        # packet = 0
        while True:
                time = random.expovariate(lamb)
                yield env.timeout(time)
                yield store.put(env.now)
                print('Arrive at %f' % env.now)
                # packet += 1

# Consumer
def consumer(env, store, mu):
        nb_packet_tab = []
        latency_tab = []
        while True:
                time = random.expovariate(mu)
                yield env.timeout(time)
                isStoreEmpty = True
                if len(store.items) != 0:
                        time_arriving_of_packet_leaving = store.items[0]
                        isStoreEmpty = False
                yield store.get()
                if not isStoreEmpty:
                        print('Leave at %f' % env.now)
                        print('it took ' + str(env.now-time_arriving_of_packet_leaving))
                        latency_tab.append(float(env.now-time_arriving_of_packet_leaving))
                        nb_packet_tab.append(len(store.items))
                        print('Mean of latency %f' % np.mean(latency_tab))
                        print('Average number of packets %f' %np.mean(nb_packet_tab))
# Setup
lamb = 15
mu = 20
env = simpy.Environment()

# Store create and process init
store = simpy.Store(env)
env.process(consumer(env, store, mu))
env.process(producer(env, store, lamb))

# Run experiment
env.run(until=100)
