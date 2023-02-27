import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from generate_channel import main_generate_channel
import scipy.io as sio
from tensorflow.keras.layers import BatchNormalization,Dense
from util_fun import get_cascaded_channel

'System information'
frames = 1
tau = 10
N_w_using = 30
total_w_number = 15
total_blocks = N_w_using * total_w_number # this is changeable with the proposed method

num_ris_elements = 64
num_antenna_bs = 1
num_user = 1
irs_Nh = 8
Rician_factor = 10
# change this scale factor will never cause effect on computing the rate
scale_factor = 135

'Channel Information'
# uplink transmitting power and noise
# transmitting power and noise power in dBm
Pt_up = 15 #dBm
# we want to enlarge this term to make the network hard to estimate
noise_power_db_up = -105 #dBm
noise_power_linear_up = 10 ** ((noise_power_db_up - Pt_up + scale_factor) / 10)
noise_sqrt_up = np.sqrt(noise_power_linear_up)
SNR_test_indB_up = Pt_up - noise_power_db_up - scale_factor
print('raw SNR in uplink (dB):', SNR_test_indB_up)

# downlink transmitting power and noise
Pt_down = 15 #dBm
# modify this noise power will cause effect on the optimal rate and the worst rate
noise_power_db_down = -115 #dBm
noise_power_linear_down = 10 ** ((noise_power_db_down - Pt_down + scale_factor) / 10)
noise_sqrt_down = np.sqrt(noise_power_linear_down)
SNR_test_indB_down = Pt_down - noise_power_db_down - scale_factor
print('raw SNR in downlink (dB):', SNR_test_indB_down)

#####################################################
'Learning Parameters'
initial_run = 0  # 0: Continue training; 1: Starts from the scratch
n_epochs = 10000  # Num of epochs
learning_rate = 0.0001 # 0.0001  # Learning rate
batch_per_epoch = 100  # Number of mini batches per epoch
batch_size_test_1frame = 3000
batch_size_train_1frame = batch_size_test_1frame
######################################################
tf.reset_default_graph()  # Reseting the graph
he_init = tf.variance_scaling_initializer()  # Define initialization method
######################################## Place Holders
channel_cascaded_input = tf.placeholder(tf.complex64, shape=(None, num_ris_elements), name="channel_cascaded_input")
batch_size = tf.shape(channel_cascaded_input)[0]

Downlink_PhaseShifts_stack = []
with tf.name_scope("channel_sensing"):
    hidden_size = 512
    '(A,b), (C,d) are all hidden state (function) in DNN'
    '(A,b) for calculating w'
    A1 = tf.get_variable("A1", shape=[tau * 2, 1024], dtype=tf.float32, initializer=he_init)
    A2 = tf.get_variable("A2", shape=[1024, 1024], dtype=tf.float32, initializer=he_init)
    A3 = tf.get_variable("A3", shape=[1024, 1024], dtype=tf.float32, initializer=he_init)
    A4 = tf.get_variable("A4", shape=[1024, 2 * num_ris_elements], dtype=tf.float32, initializer=he_init)

    b1 = tf.get_variable("b1", shape=[1024], dtype=tf.float32, initializer=he_init)
    b2 = tf.get_variable("b2", shape=[1024], dtype=tf.float32, initializer=he_init)
    b3 = tf.get_variable("b3", shape=[1024], dtype=tf.float32, initializer=he_init)
    b4 = tf.get_variable("b4", shape=[2 * num_ris_elements], dtype=tf.float32, initializer=he_init)

    '#generate all the received pilots at this frame'
    for t in range(tau):
        real_v0 = tf.get_variable('real_v0' + str(t), shape=(num_ris_elements, 1))
        imag_v0 = tf.get_variable('imag_v0' + str(t), shape=(num_ris_elements, 1))
        real_v = real_v0 / tf.sqrt(tf.square(real_v0) + tf.square(imag_v0))
        imag_v = imag_v0 / tf.sqrt(tf.square(real_v0) + tf.square(imag_v0))
        v_her_complex = tf.complex(real_v, imag_v)
        v_her_complex = tf.reshape(tf.tile(v_her_complex, (batch_size, 1)), [batch_size, num_ris_elements])

        # Note that here, y is the pilots received by the base station in the uplink
        y_noiseless = tf.reduce_sum(tf.multiply(v_her_complex, channel_cascaded_input), 1, keepdims=True)
        noise = tf.complex(tf.random_normal(tf.shape(y_noiseless), mean=0.0, stddev=0.5), tf.random_normal(tf.shape(y_noiseless), mean=0.0, stddev=0.5))
        y_complex = y_noiseless + noise_sqrt_up * noise
        y_real = tf.concat([tf.real(y_complex), tf.imag(y_complex)], axis=1)

        if t == 0:
            y_real_all = y_real
        else:
            y_real_all = tf.concat([y_real_all, y_real], axis=1)

    'after receive all the pilots, design the w'
    x1 = tf.nn.relu(y_real_all @ A1 + b1)
    x1 = BatchNormalization()(x1)
    x2 = tf.nn.relu(x1 @ A2 + b2)
    x2 = BatchNormalization()(x2)
    x3 = tf.nn.relu(x2 @ A3 + b3)
    x3 = BatchNormalization()(x3)
    w_her = x3 @ A4 + b4

    real_w0 = tf.reshape(w_her[:, 0:num_ris_elements], [batch_size, num_ris_elements])
    imag_w0 = tf.reshape(w_her[:, num_ris_elements:2 * num_ris_elements], [batch_size, num_ris_elements])
    real_w = real_w0 / tf.sqrt(tf.square(real_w0) + tf.square(imag_w0))
    imag_w = imag_w0 / tf.sqrt(tf.square(real_w0) + tf.square(imag_w0))
    w_her_complex = tf.complex(real_w, imag_w)
    w_her_complex = tf.reshape(w_her_complex, [batch_size, num_ris_elements])
    'record the designed downlink IRS phase shifts'
    Downlink_PhaseShifts_stack.append(w_her_complex)

    'compute the rate'
    bf_gain_onebatch = tf.reduce_sum(tf.multiply(w_her_complex, channel_cascaded_input), 1, keepdims=True)
    received_SNR_onebatch = tf.abs(bf_gain_onebatch) ** 2 / noise_power_linear_down
    rate = tf.reduce_mean(tf.log1p(received_SNR_onebatch) / np.log(2))

####################################################################################
####### Loss Function
loss = -rate
####### Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss, name="training_op")
init = tf.global_variables_initializer()
saver = tf.train.Saver()

###########################################################################################################################
###########  testing set (validation)
'generate all the channels'
shape_h_input_val_total = (batch_size_test_1frame * total_blocks, num_user, num_ris_elements, num_user)
h_input_val = np.zeros(shape_h_input_val_total, dtype=complex)

channel_tx_test, channel_rx_test = main_generate_channel(num_antenna_bs, num_ris_elements, num_user, irs_Nh,
                                                         batch_size_test_1frame, Rician_factor, total_blocks, scale_factor,
                                                         rho_ht=0.995, rho_hr=0.99, location_bs=np.array([200, -200, 0]),
                                                         location_irs=np.array([0, 0, 0]), isTesting=False)
# here, we want to couple the ht and hr in the same time frame! # computing the cascaded channel
for block in range(total_blocks):
    h_input_val[(block * batch_size_test_1frame) : ((block + 1) * batch_size_test_1frame), :, :, :] = get_cascaded_channel(channel_tx_test[:, :, :, block], channel_rx_test[:, :, :, block])

h_input_val = np.squeeze(h_input_val) # shape_h = (batch_size_test_1frame * total_blocks, num_ris_elements)

###########################################################################################################################
'generate the w at the certain blocks'
# channel_tx_test: [num_samples, num_elements_irs, num_user, total_blocks]
shape_h_tr_train = (batch_size_test_1frame, num_ris_elements, num_user, total_w_number)
channel_tx_test_train = np.zeros(shape_h_tr_train, dtype=complex)
channel_rx_test_train = np.zeros(shape_h_tr_train, dtype=complex)

shape_hc_train = (batch_size_test_1frame * total_w_number, num_user, num_ris_elements, num_user)
h_input_val_train_all = np.zeros(shape_hc_train, dtype=complex)

# computing the cascaded channel [here, we want to couple the ht and hr in the same time frame!]
for frame in range(total_w_number):
    channel_tx_test_train[:, :, :, frame] = channel_tx_test[:, :, :, frame * N_w_using]
    channel_rx_test_train[:, :, :, frame] = channel_rx_test[:, :, :, frame * N_w_using]
    h_input_val_train_all[(frame * batch_size_test_1frame): ((frame + 1) * batch_size_test_1frame), :, :, :] = get_cascaded_channel(channel_tx_test_train[:, :, :, frame], channel_rx_test_train[:, :, :, frame])

h_input_val_train_all = np.squeeze(h_input_val_train_all)
#########################################################################
Downlink_PhaseShifts_all = []
shape_h_input_val_train = (batch_size_test_1frame, num_ris_elements)
h_input_val_train = np.zeros(shape_h_input_val_train, dtype=complex)
for ii in range(total_w_number):
    h_input_val_train[:, :] = h_input_val_train_all[ii * batch_size_test_1frame: (ii + 1) * batch_size_test_1frame, :]

    feed_dict_val = {channel_cascaded_input: h_input_val_train}

    ###########  Training:
    snr_temp = SNR_test_indB_up
    with tf.Session() as sess:
        if initial_run == 1:
            init.run()
        else:
            saver.restore(sess, './params/addaptive_reflection_snr_'+str(int(snr_temp))+'_'+str(tau) + '_' + str(frames))

        best_loss, Downlink_PhaseShift_per_frame = sess.run([loss, Downlink_PhaseShifts_stack], feed_dict=feed_dict_val)

        Downlink_PhaseShifts_all.append(Downlink_PhaseShift_per_frame)

Downlink_PhaseShifts_all = np.array(Downlink_PhaseShifts_all).reshape([total_w_number * batch_size_test_1frame, num_ris_elements])

###########################################################################################################################
'Then, compute the trained rate for each frame'
trained_rate_perblock = []
for w_num in range(total_w_number):
    # N_w_using defines the blocks number that a w could be used in is N_w_using
    for block in range(N_w_using):
        y_tmp_val = np.sum(np.multiply(Downlink_PhaseShifts_all[w_num * batch_size_test_1frame: (w_num + 1) * batch_size_test_1frame, :], h_input_val[(block + w_num * N_w_using) * batch_size_test_1frame: (block + 1 + w_num * N_w_using) * batch_size_test_1frame, :]), axis=1)
        rate_this_frame = np.mean(np.log2(1 + (np.abs(y_tmp_val) ** 2) / noise_power_linear_down))
        trained_rate_perblock.append(rate_this_frame)



###########################################################################################################################
# save the data
trained_rate_perframe_DNN_random_trainable = np.array(trained_rate_perblock).reshape([total_blocks, 1])
print('\n trained_rate_perframe_DNN_random_trainable',trained_rate_perframe_DNN_random_trainable)

# save the recorded data
file_path_trained_rate_perframe_DNN_random_trainable = 'trained_rate_perframe_DNN_random_trainable.mat'

sio.savemat(file_path_trained_rate_perframe_DNN_random_trainable, {'trained_rate_perframe_DNN_random_trainable': trained_rate_perframe_DNN_random_trainable})

