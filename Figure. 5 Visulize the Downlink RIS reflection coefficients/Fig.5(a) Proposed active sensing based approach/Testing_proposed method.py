import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from generate_channel_visualization import main_generate_channel
import scipy.io as sio
from tensorflow.keras.layers import BatchNormalization, Dense
from util_fun import get_cascaded_channel

sample_num_x = 25
sample_num_y = 25

'System information'
frames = 36
frames_train = 10 # for the consistency of the restore file
tau = 10
# for each designed w, it has the potential to be used in multiple frames due to the slow change of the channel
# define the frames number that a w could be used in is N_w_using
N_w_using = 30
total_blocks = N_w_using * frames

num_ris_elements = 64
num_antenna_bs = 1
num_user = 1
irs_Nh = 8
Rician_factor = 10
# enlarge this scale factor will never cause effect on the optimal rate
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
# SNR_test_indB_up = 10 * np.log10(10 ** (Pt_up / 10) / noise_power_linear_up)
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
batch_size_test_1block = 1 #'notice, in visualization, sample number should be 1'
batch_size_train_1block = batch_size_test_1block
######################################################
tf.reset_default_graph()  # Reseting the graph
he_init = tf.variance_scaling_initializer()  # Define initialization method
######################################## Place Holders
'notice, this is the only place holder!'
channel_cascaded_input = tf.placeholder(tf.complex64, shape=(None, num_ris_elements), name="channel_cascaded_input")
batch_size = tf.shape(channel_cascaded_input[0: batch_size_test_1block, :])[0]

hidden_size = 512
previous_hidden_state = tf.placeholder(tf.float32, shape=(None, hidden_size), name="previous_hidden_state")
previous_cell_state = tf.placeholder(tf.float32, shape=(None, hidden_size), name="previous_cell_state")

with tf.name_scope("channel_sensing"):
    'DNN layers'
    '(C,d) for calculating the DL BF vector w for each frame'
    C1 = tf.get_variable("C1", shape=[hidden_size, 1024], dtype=tf.float32, initializer=he_init)
    C2 = tf.get_variable("C2", shape=[1024, 1024], dtype=tf.float32, initializer=he_init)
    C3 = tf.get_variable("C3", shape=[1024, 1024], dtype=tf.float32, initializer=he_init)
    C4 = tf.get_variable("C4", shape=[1024, 2 * num_ris_elements], dtype=tf.float32, initializer=he_init)

    d1 = tf.get_variable("d1", shape=[1024], dtype=tf.float32, initializer=he_init)
    d2 = tf.get_variable("d2", shape=[1024], dtype=tf.float32, initializer=he_init)
    d3 = tf.get_variable("d3", shape=[1024], dtype=tf.float32, initializer=he_init)
    d4 = tf.get_variable("d4", shape=[2 * num_ris_elements], dtype=tf.float32, initializer=he_init)

    '(A,b) for calculating the UL sensing vector v for each frame'
    A1 = tf.get_variable("A1", shape=[hidden_size, 1024], dtype=tf.float32, initializer=he_init)
    A2 = tf.get_variable("A2", shape=[1024, 1024], dtype=tf.float32, initializer=he_init)
    A3 = tf.get_variable("A3", shape=[1024, 1024], dtype=tf.float32, initializer=he_init)
    A4 = tf.get_variable("A4", shape=[1024, 2 * num_ris_elements * tau], dtype=tf.float32, initializer=he_init)

    b1 = tf.get_variable("b1", shape=[1024], dtype=tf.float32, initializer=he_init)
    b2 = tf.get_variable("b2", shape=[1024], dtype=tf.float32, initializer=he_init)
    b3 = tf.get_variable("b3", shape=[1024], dtype=tf.float32, initializer=he_init)
    b4 = tf.get_variable("b4", shape=[2 * num_ris_elements * tau], dtype=tf.float32, initializer=he_init)

    'RNN layers'
    # upper Active sensing LSTM RNN structure
    layer_Ui_upper = Dense(units=hidden_size, activation='linear')
    layer_Wi_upper = Dense(units=hidden_size, activation='linear')
    layer_Uf_upper = Dense(units=hidden_size, activation='linear')
    layer_Wf_upper = Dense(units=hidden_size, activation='linear')
    layer_Uo_upper = Dense(units=hidden_size, activation='linear')
    layer_Wo_upper = Dense(units=hidden_size, activation='linear')
    layer_Uc_upper = Dense(units=hidden_size, activation='linear')
    layer_Wc_upper = Dense(units=hidden_size, activation='linear')

    def RNN_upper(input_x, h_old, c_old):
        i_t_upper = tf.sigmoid(layer_Ui_upper(input_x) + layer_Wi_upper(h_old))
        f_t_upper = tf.sigmoid(layer_Uf_upper(input_x) + layer_Wf_upper(h_old))
        o_t_upper = tf.sigmoid(layer_Uo_upper(input_x) + layer_Wo_upper(h_old))
        c_t_upper = tf.tanh(layer_Uc_upper(input_x) + layer_Wc_upper(h_old))
        c_upper = i_t_upper * c_t_upper + f_t_upper * c_old
        h_new_upper = o_t_upper * tf.tanh(c_upper)
        return h_new_upper, c_upper


    'initialization c_0, h_0'
    # don't forget we need to update the initialized (c,h) (only) at the first frame
    y_real_init = tf.ones([batch_size, 2 * tau])
    h_old_init = previous_hidden_state
    c_old_init = previous_cell_state
    h_old_0, c_old_0 = RNN_upper(y_real_init, h_old_init, c_old_init)
############################################################################################ the proposed testing methods
    'notice we are using the same pretrained LSTM and DNN blocks for different time frames'
    # Assign value with placeholder
    h_old = previous_hidden_state
    c_old = previous_cell_state

    'design the uplink sensing vectors (in total tau 个)'
    g1 = tf.nn.relu(c_old @ A1 + b1)
    g1 = BatchNormalization()(g1)
    g2 = tf.nn.relu(g1 @ A2 + b2)
    g2 = BatchNormalization()(g2)
    g3 = tf.nn.relu(g2 @ A3 + b3)
    g3 = BatchNormalization()(g3)
    # remember, here, v_her: [batch_size, 2 * num_ris_elements * tau]
    v_her = g3 @ A4 + b4

    'generate UL received pilots at this frame'
    for t in range(tau):
        'assign the right sensing vector to each of the pilots'
        # for every t (every pilot), we should design a unique sensing vector
        real_v0 = tf.reshape(v_her[:, (0 + t * 2 * num_ris_elements):(num_ris_elements + t * 2 * num_ris_elements)], [batch_size, num_ris_elements])
        imag_v0 = tf.reshape(v_her[:, (num_ris_elements + t * 2 * num_ris_elements):(2 * num_ris_elements + t * 2 * num_ris_elements)], [batch_size, num_ris_elements])
        # satisfy the unit modulus constraints
        real_v = real_v0 / tf.sqrt(tf.square(real_v0) + tf.square(imag_v0))
        imag_v = imag_v0 / tf.sqrt(tf.square(real_v0) + tf.square(imag_v0))

        v_her_complex = tf.complex(real_v, imag_v)
        v_her_complex = tf.reshape(v_her_complex, [batch_size, num_ris_elements])

        'receiving the pilots'
        # 'BS observes uplink pilot [notice, for every frame, the channel is changing! but for different t, it is assumed to be fixed]'
        y_noiseless = tf.reduce_sum(tf.multiply(v_her_complex, channel_cascaded_input), 1, keepdims=True)
        noise = tf.complex(tf.random_normal(tf.shape(y_noiseless), mean=0.0, stddev=0.5),tf.random_normal(tf.shape(y_noiseless), mean=0.0, stddev=0.5))
        # Note that here, y is the pilots received by the base station in the uplink
        y_complex = y_noiseless + noise_sqrt_up * noise
        y_real = tf.concat([tf.real(y_complex), tf.imag(y_complex)], axis=1)

        if t == 0:
            y_real_all = y_real
        else:
            y_real_all = tf.concat([y_real_all, y_real], axis=1)

    h_old, c_old = RNN_upper(y_real_all, h_old, c_old)

    'use another DNN to come up with downlink RIS reflection coefficients w [input is the cell state of the upper LSTM RNN]'
    z1 = tf.nn.relu(c_old @ C1 + d1)
    z1 = BatchNormalization()(z1)
    z2 = tf.nn.relu(z1 @ C2 + d2)
    z2 = BatchNormalization()(z2)
    z3 = tf.nn.relu(z2 @ C3 + d3)
    z3 = BatchNormalization()(z3)
    w_her = z3 @ C4 + d4
    real_w0 = tf.reshape(w_her[:, 0:num_ris_elements], [batch_size, num_ris_elements])
    imag_w0 = tf.reshape(w_her[:, num_ris_elements:2 * num_ris_elements], [batch_size, num_ris_elements])
    real_w = real_w0 / tf.sqrt(tf.square(real_w0) + tf.square(imag_w0))
    imag_w = imag_w0 / tf.sqrt(tf.square(real_w0) + tf.square(imag_w0))
    w_her_complex = tf.complex(real_w, imag_w)
    w_her_complex = tf.reshape(w_her_complex, [batch_size, num_ris_elements])

    # notice, the channel is changing btw blocks!
    # compute the received_SNR for this block
    bf_gain_oneframe = tf.reduce_sum(tf.multiply(w_her_complex, channel_cascaded_input), 1, keepdims=True)
    received_SNR_oneframe = tf.abs(bf_gain_oneframe) ** 2 / noise_power_linear_down
    rate = tf.reduce_mean(tf.log1p(received_SNR_oneframe) / np.log(2))

####### Loss Function
loss = -rate
####### Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss, name="training_op")
init = tf.global_variables_initializer()
saver = tf.train.Saver()

####################################################################################################################################################################################################################################################################################################
# remember, here the batch size is only 1!
# 'testing the pretrained model and get all the w and user coordinate changes'
######################################################################### generate all the channels
'here, we want to use this fixed length structure into a infinite long transmission frame scenario'
'generate all the channels'
channel_tx_test_all, channel_rx_test_all, user_corrdinate_all, range_coordinate_all_without_z, channel_hc_range_all, channel_user_hc_all = main_generate_channel(sample_num_x, sample_num_y, num_antenna_bs, num_ris_elements, num_user, irs_Nh,
                                                         batch_size_test_1block, Rician_factor, total_blocks, scale_factor,
                                                         rho_ht=0.995, rho_hr=0.99, location_bs=np.array([200, -200, 0]),
                                                         location_irs=np.array([0, 0, 0]), isTesting=True)

#########################################################################  here, we want to obtained the designed w for the certain frames
'generate the channels at certain blocks'
# channel_tx_test_all: [num_samples, num_elements_irs, num_user, total_blocks]
shape_h_tr_train = (batch_size_test_1block, num_ris_elements, num_user, frames)
channel_tx_test_train = np.zeros(shape_h_tr_train, dtype=complex)
channel_rx_test_train = np.zeros(shape_h_tr_train, dtype=complex)

shape_hc_train = (frames, num_user, num_ris_elements, num_user)
h_input_val_train_all = np.zeros(shape_hc_train, dtype=complex)

'computing the cascaded channel [here, we want to couple the ht and hr in the same block!]'
for frame in range(frames):
    channel_tx_test_train[:, :, :, frame] = channel_tx_test_all[:, :, :, frame * N_w_using]
    channel_rx_test_train[:, :, :, frame] = channel_rx_test_all[:, :, :, frame * N_w_using]
    h_input_val_train_all[frame, :, :, :] = get_cascaded_channel(channel_tx_test_train[:, :, :, frame], channel_rx_test_train[:, :, :, frame])

h_input_val_train_all = np.squeeze(h_input_val_train_all)

'feed the channels into the network and get the designed w'
Downlink_PhaseShifts_all = []
shape_h_input_val_train = (batch_size_test_1block, num_ris_elements)
h_input_val_train = np.zeros(shape_h_input_val_train, dtype=complex)

for frame in range(frames):
    h_input_val_train[:, :] = h_input_val_train_all[frame, :]

    'initialize h_0, c_0'
    if frame == 0:
        h_old_test_init = np.zeros([batch_size_test_1block, hidden_size], dtype=float)
        c_old_test_init = np.zeros([batch_size_test_1block, hidden_size], dtype=float)
        feed_dict_val = {channel_cascaded_input: h_input_val_train, previous_hidden_state: h_old_test_init, previous_cell_state: c_old_test_init}

        snr_temp = SNR_test_indB_up
        with tf.Session() as sess:
            if initial_run == 1:
                init.run()
            else:
                saver.restore(sess,'./params/addaptive_reflection_snr_' + str(int(snr_temp)) + '_' + str(tau) + '_' + str(frames_train))

            h_old_test, c_old_test = sess.run([h_old_0, c_old_0], feed_dict=feed_dict_val)

    feed_dict_val = {channel_cascaded_input: h_input_val_train, previous_hidden_state: h_old_test, previous_cell_state: c_old_test}

    #########################################################################
    ###########  Training:
    snr_temp = SNR_test_indB_up
    with tf.Session() as sess:
        if initial_run == 1:
            init.run()
        else:
            saver.restore(sess, './params/addaptive_reflection_snr_'+str(int(snr_temp))+'_'+str(tau) + '_' + str(frames_train))

        best_loss, Downlink_PhaseShifts_per_frame, h_old_test, c_old_test = sess.run([loss, w_her_complex, h_old, c_old], feed_dict=feed_dict_val)

        Downlink_PhaseShifts_all.append(Downlink_PhaseShifts_per_frame)

Downlink_PhaseShifts = np.array(Downlink_PhaseShifts_all).reshape([frames, num_ris_elements])

# Downlink_PhaseShifts: [frames, num_ris_elements]
#########################################################################  here, we compute the trained rate for each block on the testing area
# remember, here the batch_size is only 1!
'after we got the optimal DL IRS phase shift w w.r.t. the user coordinate, we want to visualize the rate in the defined range w.r.t. those certain picked frames'
# compute the rate in the whole range
rate_range_all = np.zeros([sample_num_x * sample_num_y, 1, total_blocks])

# channel_hc_range_all = np.zeros([sample_num_x * sample_num_y, 1, num_elements_irs, total_blocks], dtype=complex)
for w_num in range(frames):
    # N_w_using defines the total block number that a w could be used in
    for block in range(N_w_using):
        # for x coordinate on the moving range
        for jj in range(sample_num_x):
            # for y coordinate on the moving range
            for kk in range(sample_num_y):
                y_tmp_val = np.sum(np.multiply(Downlink_PhaseShifts[w_num, :], channel_hc_range_all[sample_num_y * jj + kk, :, :, block + w_num * N_w_using]), axis=1)
                rate_range_all[sample_num_y * jj + kk, :, block + w_num * N_w_using] = np.mean(np.log2(1 + (np.abs(y_tmp_val) ** 2) / noise_power_linear_down))

path_rate_range_all = 'rate_range_all.mat'
sio.savemat(path_rate_range_all, {'rate_range_all': rate_range_all})

#########################################################################
'get the x and y axis of the user coordinate for all the blocks'
user_coor_x = np.zeros([total_blocks, 1])
user_coor_y = np.zeros([total_blocks, 1])
# user_corrdinate_all: [total_blocks, 1, 3]
for ii in range(total_blocks):
    user_coor_x[ii, 0] = user_corrdinate_all[ii, 0, 0]
    user_coor_y[ii, 0] = user_corrdinate_all[ii, 0, 1]

# save the user coordinate change (this will be used as the center (reference point) to the rate visualization graph)
path_user_coor_x = 'user_coor_x.mat'
path_user_coor_y = 'user_coor_y.mat'
sio.savemat(path_user_coor_x, {'user_coor_x': user_coor_x})
sio.savemat(path_user_coor_y, {'user_coor_y': user_coor_y})

#########################################################################
'get the x and y axis of the testing range [same for all the blocks]'
range_coordinate_x_all = np.zeros([sample_num_x * sample_num_y, 1])
range_coordinate_y_all = np.zeros([sample_num_x * sample_num_y, 1])

range_coordinate_x = np.zeros([sample_num_x, 1])

for ii in range(sample_num_x * sample_num_y):
    range_coordinate_x_all[ii, 0] = range_coordinate_all_without_z[ii, 0, 0]
    range_coordinate_y_all[ii, 0] = range_coordinate_all_without_z[ii, 0, 1]

range_coordinate_y = range_coordinate_y_all[0:sample_num_y, :]
for kk in range(sample_num_x):
  range_coordinate_x[kk,0] = range_coordinate_x_all[kk*sample_num_y, 0]


path_range_coordinate_x = 'range_coordinate_x.mat'
path_range_coordinate_y = 'range_coordinate_y.mat'
sio.savemat(path_range_coordinate_x, {'range_coordinate_x': range_coordinate_x})
sio.savemat(path_range_coordinate_y, {'range_coordinate_y': range_coordinate_y})










