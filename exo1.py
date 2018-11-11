import simpy
import random
import numpy as np
import scipy.stats as st

# Producer


def producer(env, store, lamb):
        while True:
                time = random.expovariate(lamb)
                yield env.timeout(time)
                # we put the time arrival for their name
                yield store.put(env.now)
                # print('Packet arrived at %f' % env.now)

# Consumer


def consumer(env, store, mu):
        nb_packet_tab = []
        latency_tab = []
        while True:
                time = random.expovariate(mu)
                yield env.timeout(time)
                if len(store.items) != 0:
                        # we retreive the element wihdrawed, its name is the date of its arrival
                        time_arriving_of_packet_leaving = yield store.get()
                        # print('Left at %f it took %s' %(env.now, env.now-time_arriving_of_packet_leaving))

                        latency_tab.append(float(env.now-time_arriving_of_packet_leaving))
                        latency_interval = st.t.interval(0.95, len(latency_tab), loc=np.mean(latency_tab), scale=st.sem(latency_tab))
                        print('Mean of latency %f, interval(95): %s' %(np.mean(latency_tab), latency_interval))
                        # print('Confidence interval of latency (95%): ' + str(latency_interval))

                        nb_packet_tab.append(len(store.items))
                        nb_packet_interval = st.t.interval(0.95, len(nb_packet_tab)-1, loc=np.mean(nb_packet_tab), scale=st.sem(nb_packet_tab))
                        print('Average number of packets %f, interval(95): %s' % (np.mean(nb_packet_tab), nb_packet_interval))
                        # print('Confidence interval of average of packets (95%): ' + str(nb_packet_interval))

# Setup
lamb=15
mu=20
duration=300
env=simpy.Environment()

# Store create and process init
store=simpy.Store(env)
env.process(consumer(env, store, mu))
env.process(producer(env, store, lamb))

# Run experiment
env.run(until=duration)
