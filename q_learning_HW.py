
import tensorflow as tf
import cv2
import pong_game as game
import random
import numpy as np
from collections import deque


# Game name.
GAME = 'Pong'

# Number of valid actions.
ACTIONS = 3

# Decay rate of past observations.
GAMMA = 0.99 

# Timesteps to observe before training.
OBSERVE = 5000. 

# Frames over which to anneal epsilon.
EXPLORE = 250000. 

# Final value of epsilon.
FINAL_EPSILON = 0.05 

# Starting value of epsilon.
INITIAL_EPSILON = 1.0 

# Number of previous transitions to remember in the replay memory.
REPLAY_MEMORY = 300000 

# Size of minibatch.
BATCH = 32 

# Only select an action every Kth frame, repeat the same action for other frames.
K = 5

# Learning Rate.
Lr = 1e-6


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.01)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.01, shape = shape)
	return tf.Variable(initial)

def conv2d(x, W, stride):
	return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
	
	# Initialize the network weights and biases.
	W_conv1 = weight_variable([8, 8, 4, 32])

	b_conv1 = bias_variable([32])

	W_conv2 = weight_variable([4, 4, 32, 64])

	b_conv2 = bias_variable([64])

	W_conv3 = weight_variable([3, 3, 64, 64])

	b_conv3 = bias_variable([64])

	W_fc1 = weight_variable([1600, 512])

	b_fc1 = bias_variable([512])

	W_fc2 = weight_variable([512, ACTIONS])

	b_fc2 = bias_variable([ACTIONS])

	# Input layer.
	s = tf.placeholder("float", [None, 80, 80, 4])

	# Hidden layers.
	h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)	
	h_pool1 = max_pool_2x2(h_conv1)
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
	h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
	h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
	h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

	# Output layer
	readout = tf.matmul(h_fc1, W_fc2) + b_fc2

	return s, readout


def get_action_index(readout_t,epsilon,t):
	rand_num = np.random.random()
	if OBSERVE >t or epsilon > rand_num:
		action_index = random.randint(0,2)
	else:
		action_index = np.argmax(readout_t)
	return action_index



def scale_down_epsilon(epsilon,t):
    if epsilon > FINAL_EPSILON :
        if t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
    return epsilon




def run_selected_action(a_t,s_t,game_state):
	x,r_t,terminal=game_state.frame_step(a_t)
	x_t = cv2.cvtColor(cv2.resize(x, (80, 80)), cv2.COLOR_BGR2GRAY)
	ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
	s_t1=np.stack((s_t[:,:,1], s_t[:,:,2], s_t[:,:,3], x_t), axis = 2)
	return s_t1,r_t,terminal 


def compute_cost(target_q,a_t,q_value):
	
	target_y = tf.reduce_sum(q_value * a_t, axis=1)
	cost = tf.reduce_sum(tf.square(target_y - target_q))
	return cost


def compute_target_q(next_state_batch,r_batch,readout_j1_batch,minibatch):
	terminal= [d[4] for d in minibatch]
	target_q_batch=[]
	for i in range(len(readout_j1_batch)):
		if(terminal[i]==True):
			target_q_batch.append(r_batch[i])
		else:
			target_q_batch.append(r_batch[i]+GAMMA*max(readout_j1_batch[i]))
	return target_q_batch





def trainNetwork(s, readout, sess):
	# Placeholder for the action.
	a = tf.placeholder("float", [None, ACTIONS])

	# Placeholder for the target Q value.
	y = tf.placeholder("float", [None])

	# Compute the loss.
	cost = compute_cost(y,a,readout)

	# Training operation.
	train_step = tf.train.AdamOptimizer(Lr).minimize(cost)

	# Open up a game state to communicate with emulator.
	game_state = game.GameState()

	# Initialize the replay memory.
	D = deque()

	# Initialize the action vector.
	do_nothing = np.zeros(ACTIONS)
	do_nothing[0] = 1

	# Initialize the state of the game.
	x_t, r_0, terminal = game_state.frame_step(do_nothing)
	x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
	ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
	s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)

	# Save and load model checkpoints.
	saver = tf.train.Saver(max_to_keep = 1000000)
	sess.run(tf.initialize_all_variables())
	checkpoint = tf.train.get_checkpoint_state("saved_networks_q_learning")
	if checkpoint and checkpoint.model_checkpoint_path:
		saver.restore(sess, checkpoint.model_checkpoint_path)
		print("Successfully loaded:", checkpoint.model_checkpoint_path)
	else:
		print("Could not find old network weights")

	# Initialize the epsilon value for the exploration phase.
	epsilon = INITIAL_EPSILON

	# Initialize the iteration counter.
	t = 0
    

    

	while True:

		# Choose an action epsilon-greedily.
		readout_t = readout.eval(feed_dict = {s : [s_t]})[0]

		action_index = get_action_index(readout_t,epsilon,t)

		a_t = np.zeros([ACTIONS])

		a_t[action_index] = 1

		# Scale down epsilon during the exploitation phase.
		epsilon = scale_down_epsilon(epsilon,t)

		#run the selected action and update the replay memeory

		for i in range(0, K):
			# Run the selected action and observe next state and reward.
			s_t1,r_t,terminal = run_selected_action(a_t,s_t,game_state)

			# Store the transition in the replay memory D.
			D.append((s_t, a_t, r_t, s_t1, terminal))
			if len(D) > REPLAY_MEMORY:
				D.popleft()


		# Start training once the observation phase is over.
		if (t > OBSERVE):

			# Sample a minibatch to train on.
			minibatch = random.sample(D, BATCH)

			# Get the batch variables.
			s_j_batch = [d[0] for d in minibatch]
			a_batch = [d[1] for d in minibatch]
			r_batch = [d[2] for d in minibatch]
			s_j1_batch = [d[3] for d in minibatch]

			# Compute the target Q-Value
			readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})

			target_q_batch = compute_target_q(s_j1_batch,r_batch,readout_j1_batch,minibatch)

			# Perform gradient step.
			train_step.run(feed_dict = {
				y : target_q_batch,
				a : a_batch,
				s : s_j_batch})


		# Update the state.
		s_t = s_t1

		# Update the number of iterations.
		t += 1

		# Save a checkpoint every 10000 iterations.
		if t % 10000 == 0:
			saver.save(sess, 'saved_networks_q_learning/' + GAME + '-dqn', global_step = t)

		# Print info.
		state = ""
		if t <= OBSERVE:
			state = "observe"
		elif t > OBSERVE and t <= OBSERVE + EXPLORE:
			state = "explore"
		else:
			state = "train"
		if t % 1000 == 0:
			print("TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t))
		        



def playGame():

	# Start an active session.
	sess = tf.InteractiveSession()

	# Create the network.
	s, readout = createNetwork()

	# Choose between Q-Learning and Policy Gradient.
	s, readout = trainNetwork(s, readout, sess)
	


def main():
	""" Main function """
	playGame()

if __name__ == "__main__":
	main()


