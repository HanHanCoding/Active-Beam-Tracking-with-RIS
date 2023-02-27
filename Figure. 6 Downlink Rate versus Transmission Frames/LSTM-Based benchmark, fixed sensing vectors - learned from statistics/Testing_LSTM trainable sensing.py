import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from generate_channel import main_generate_channel
import scipy.io as sio
from tensorflow.keras.layers import BatchNormalization,Dense
from util_fun import get_cascaded_channel

'System information'
frames = 15
frames_train = 10 # for the consistency of the restore file
tau = 10
# for each designed w, it has the potential to be used in multiple frames due to the slow change of the channel
# define the blocks number that a w could be used in is N_w_using
N_w_using = 30

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
batch_size_test_1block = 3000
batch_size_train_1block = batch_size_test_1block
######################################################
tf.reset_default_graph()  # Reseting the graph
he_init = tf.variance_scaling_initializer()  # Define initialization method
######################################## Place Holders
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
    # 用placeholder进行赋值
    h_old = previous_hidden_state
    c_old = previous_cell_state
##############################################
    'Generate the shared random uplink IRS coefficients btw different frames'
    real_v0 = tf.get_variable('real_v0', shape=(num_ris_elements, tau))
    imag_v0 = tf.get_variable('imag_v0', shape=(num_ris_elements, tau))
    real_v = real_v0 / tf.sqrt(tf.square(real_v0) + tf.square(imag_v0))
    imag_v = imag_v0 / tf.sqrt(tf.square(real_v0) + tf.square(imag_v0))
    v_her_complex = tf.complex(real_v, imag_v)
    # duplicate the v_her_complex for batch size
    v_her_complex = tf.expand_dims(v_her_complex, 0)
    v_her_complex = tf.tile(v_her_complex, (batch_size, 1, 1))

    'generate UL received pilots at this frame'
    for t in range(tau):
        'BS observes uplink pilot [notice, for every block, the channel is changing! but for different t, it is assumed to be fixed]'
        y_noiseless = tf.reduce_sum(tf.multiply(v_her_complex[:, :, t], channel_cascaded_input), 1, keepdims=True)
        noise = tf.complex(tf.random_normal(tf.shape(y_noiseless), mean=0.0, stddev=0.5), tf.random_normal(tf.shape(y_noiseless), mean=0.0, stddev=0.5))
        # Note that here, y is the pilots received by the base station in the uplink
        y_complex = y_noiseless + noise_sqrt_up * noise
        y_real = tf.concat([tf.real(y_complex), tf.imag(y_complex)], axis=1)

        if t == 0:
            y_real_all = y_real
        else:
            y_real_all = tf.concat([y_real_all, y_real], axis=1)

    h_old, c_old = RNN_upper(y_real_all, h_old, c_old)

    'use another DNN to come up with downlink beamforming vector w [input is the cell state of the upper LSTM RNN]'
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
#loss = -1 * tf.reduce_mean(received_power)
####### Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss, name="training_op")
init = tf.global_variables_initializer()
saver = tf.train.Saver()

######################################################################### testing
'here, we want to use this fixed length structure into a infinite long transmission frame scenario'
'generate all the channels'
total_blocks = N_w_using * frames
channel_tx_test_all, channel_rx_test_all = main_generate_channel(num_antenna_bs, num_ris_elements, num_user, irs_Nh,
                                                                 batch_size_test_1block, Rician_factor, total_blocks, scale_factor,
                                                                 rho_ht=0.995, rho_hr=0.99, location_bs=np.array([200, -200, 0]),
                                                                 location_irs=np.array([0, 0, 0]), isTesting=False)

shape_hc_total = (batch_size_test_1block * total_blocks, num_user, num_ris_elements, num_user)
hc_total = np.zeros(shape_hc_total, dtype=complex)
# 'here, we want to couple the ht and hr in the same time frame!' # computing all the cascaded channel
for block in range(total_blocks):
    hc_total[(block * batch_size_test_1block): ((block + 1) * batch_size_test_1block), :, :, :] = get_cascaded_channel(channel_tx_test_all[:, :, :, block], channel_rx_test_all[:, :, :, block])

# shape_hc_total = (batch_size_test_1block * total_blocks, num_ris_elements)
hc_total = np.squeeze(hc_total)

###########################################################################################################################
'testing the trained performance'
# channel_tx_test_all = np.zeros([num_samples, num_elements_irs, num_user, total_blocks]) + 1j * np.zeros([num_samples, num_elements_irs, num_user, total_blocks])
shape_h_tr_train = (batch_size_test_1block, num_ris_elements, num_user, frames)
channel_tx_test_train = np.zeros(shape_h_tr_train, dtype=complex)
channel_rx_test_train = np.zeros(shape_h_tr_train, dtype=complex)

shape_hc_train = (batch_size_test_1block * frames, num_user, num_ris_elements, num_user)
h_input_val_train_all = np.zeros(shape_hc_train, dtype=complex)

# computing the cascaded channel [here, we want to couple the ht and hr in the same time frame!]
for frame in range(frames):
    channel_tx_test_train[:, :, :, frame] = channel_tx_test_all[:, :, :, frame * N_w_using]
    channel_rx_test_train[:, :, :, frame] = channel_rx_test_all[:, :, :, frame * N_w_using]
    h_input_val_train_all[(frame * batch_size_test_1block): ((frame + 1) * batch_size_test_1block), :, :, :] = get_cascaded_channel(channel_tx_test_train[:, :, :, frame], channel_rx_test_train[:, :, :, frame])

h_input_val_train_all = np.squeeze(h_input_val_train_all)

'here, we select the channels at certain block (beginning at each frame) for testing'
Downlink_PhaseShifts_all = []
shape_h_input_val_train = (batch_size_test_1block, num_ris_elements)
h_input_val_train = np.zeros(shape_h_input_val_train, dtype=complex)

for frame in range(frames):
    h_input_val_train[:, :] = h_input_val_train_all[frame * batch_size_test_1block: (frame + 1) * batch_size_test_1block, :]

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
    # 记录上行的SNR
    snr_temp = SNR_test_indB_up
    with tf.Session() as sess:
        if initial_run == 1:
            init.run()
        else:
            saver.restore(sess, './params/addaptive_reflection_snr_'+str(int(snr_temp))+'_'+str(tau) + '_' + str(frames_train))

        best_loss, Downlink_PhaseShifts_per_frame, h_old_test, c_old_test = sess.run([loss, w_her_complex, h_old, c_old], feed_dict=feed_dict_val)

        Downlink_PhaseShifts_all.append(Downlink_PhaseShifts_per_frame)

Downlink_PhaseShifts = np.array(Downlink_PhaseShifts_all).reshape([frames * batch_size_test_1block, num_ris_elements])

'Then, compute the trained rate for each frame'
trained_rate_perblock = []
for w_num in range(frames):
    # N_w_using defines the frames number that each generated w could be used
    for block in range(N_w_using):
        y_tmp_val = np.sum(np.multiply(Downlink_PhaseShifts[w_num * batch_size_test_1block: (w_num + 1) * batch_size_test_1block, :], hc_total[(block + w_num * N_w_using) * batch_size_test_1block: (block + 1 + w_num * N_w_using) * batch_size_test_1block, :]), axis=1)
        rate_this_block = np.mean(np.log2(1 + (np.abs(y_tmp_val) ** 2) / noise_power_linear_down))
        trained_rate_perblock.append(rate_this_block)


###########################################################################################################################
# save the data
trained_rate_perframe_LSTM_random_trainable = np.array(trained_rate_perblock).reshape([total_blocks, 1])

print('\n trained_rate_perframe_LSTM_random_trainable',trained_rate_perframe_LSTM_random_trainable)

# save the recorded data
file_path_trained_rate_perframe_LSTM_random_trainable = 'trained_rate_perframe_LSTM_random_trainable.mat'

sio.savemat(file_path_trained_rate_perframe_LSTM_random_trainable, {'trained_rate_perframe_LSTM_random_trainable': trained_rate_perframe_LSTM_random_trainable})




























