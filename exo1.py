import simpy
import random
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

# Producer
def producer(env, store, lamb, duration):
	print("Process is running, it might take time...\n") # for development, don't mind the warning message...
	while env.now < duration*0.9:
		time = random.expovariate(lamb)
		yield env.timeout(time)
		# we put the time arrival for their name
		yield store.put(env.now)
		# print('Packet arrived at %f' % env.now)
	env.exit()

# Consumer
def consumer(env, store, mu, duration):
	latency_tab = []
	nb_packet_tab = []

	latency_plot = []
	latency_confidence_plot_bottom = []
	latency_confidence_plot_top = []
	packet_plot = []
	packet_confidence_plot_bottom = []
	packet_confidence_plot_top = []

	while env.now < duration*0.9:
		time = random.expovariate(mu)
		yield env.timeout(time)
		if len(store.items) != 0:
			# we retreive the element wihdrawed, its name is the date of its arrival
			time_arriving_of_packet_leaving = yield store.get()
			# print('Left at %f it took %s' %(env.now, env.now-time_arriving_of_packet_leaving))

			# LATENCY RELATED
			latency_tab.append(float(env.now-time_arriving_of_packet_leaving))
			latency_interval = st.t.interval(0.95, len(latency_tab), loc=np.mean(latency_tab), scale=st.sem(latency_tab))
			# print('Mean of latency %f, interval(95): %s' %(np.mean(latency_tab), latency_interval))
			# print('Confidence interval of latency (95%): ' + str(latency_interval))
			latency_plot.append(np.mean(latency_tab))
			latency_confidence_plot_bottom.append(latency_interval[0])
			latency_confidence_plot_top.append(latency_interval[1])

			# PACKETS RELATED
			nb_packet_tab.append(len(store.items))
			nb_packet_interval = st.t.interval(0.95, len(nb_packet_tab)-1, loc=np.mean(nb_packet_tab), scale=st.sem(nb_packet_tab))
			# print('Average number of packets %f, interval(95): %s' %(np.mean(nb_packet_tab), nb_packet_interval))
			# print('Confidence interval of average of packets (95%): ' + str(nb_packet_interval))
			packet_plot.append(np.mean(nb_packet_tab))
			packet_confidence_plot_bottom.append(nb_packet_interval[0])
			packet_confidence_plot_top.append(nb_packet_interval[1])

	# PLOT
	plt.subplot(211)
	plt.plot(latency_plot, "-", color="blue", linewidth=2.5, label="average latency")
	plt.plot(latency_confidence_plot_bottom, "-", color="green", linewidth=1, label="confidence interval")
	plt.plot(latency_confidence_plot_top, "-", color="green", linewidth=1)
	plt.xlabel("Event")
	plt.ylabel("Latency")
	plt.legend(loc="lower right", frameon=False)
	plt.title("Latency evolution (lambda %d, mu %d)" % (lamb, mu))

	plt.subplot(212)
	plt.plot(packet_plot, "-", color="blue", linewidth=2.5, label="average packet number")
	plt.plot(packet_confidence_plot_bottom, "-", color="green", linewidth=1, label="confidence interval")
	plt.plot(packet_confidence_plot_top, "-", color="green", linewidth=1)
	plt.xlabel("Event")
	plt.ylabel("Number of packets")
	plt.legend(loc="lower right", frameon=False)
	plt.title("Evolution of number of packets in queue (lambda %d, mu %d)" % (lamb, mu))

	print("")
	print('Mean of latency %f, interval(95): %s' %(np.mean(latency_tab), latency_interval))
	print('Average number of packets %f, interval(95): %s' %(np.mean(nb_packet_tab), nb_packet_interval))

	plt.show()

	env.exit()


# Setup
lamb = 15
mu = 20
duration = 250
env = simpy.Environment()

# Store create and process init
store = simpy.Store(env)
env.process(consumer(env, store, mu, duration))
env.process(producer(env, store, lamb, duration))

# Run experiment
env.run(until=duration)

# #############################
# ### FOR COMPARISON LAMBD > MU
# #############################
# # Setup
# lamb = 20
# mu = 15
# duration = 250
# env = simpy.Environment()

# # Store create and process init
# store = simpy.Store(env)
# env.process(consumer(env, store, mu, duration))
# env.process(producer(env, store, lamb, duration))

# # Run experiment
# env.run(until=duration)

# #############################
# ### FOR COMPARISON LAMBD == MU
# #############################
# # Setup
# lamb = 15
# mu = 15
# duration = 250
# env = simpy.Environment()

# # Store create and process init
# store = simpy.Store(env)
# env.process(consumer(env, store, mu, duration))
# env.process(producer(env, store, lamb, duration))

# # Run experiment
# env.run(until=duration)

