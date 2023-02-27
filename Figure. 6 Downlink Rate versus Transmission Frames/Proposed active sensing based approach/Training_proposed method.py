import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from generate_channel import main_generate_channel
from tensorflow.keras.layers import BatchNormalization,Dense
from util_fun import get_cascaded_channel

'System information'
frames = 10
tau = 10
# for each designed w, it has the potential to be used in multiple frames due to the slow change of the channel
# define the blocks number that a w could be used in is N_w_using
N_w_using = 30 # notice, here, we assume the channel reciprocity!
total_blocks = N_w_using * frames

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
initial_run = 1  # 0: Continue training; 1: Starts from the scratch
n_epochs = 10000  # Num of epochs
learning_rate = 0.0001 # 0.0001  # Learning rate
batch_per_epoch = 100  # Number of mini batches per epoch
batch_size_test_1block = 32
batch_size_train_1block = batch_size_test_1block
######################################################
tf.reset_default_graph()  # Reseting the graph
he_init = tf.variance_scaling_initializer()  # Define initialization method
######################################## Place Holders
'notice, this is the only place holder!'
channel_cascaded_input = tf.placeholder(tf.complex64, shape=(None, num_ris_elements), name="channel_cascaded_input")
batch_size = tf.shape(channel_cascaded_input[0: batch_size_test_1block, :])[0]

with tf.name_scope("channel_sensing"):
    hidden_size = 512
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
    # 1. units are the dimensionality of the output space!
    #    for a 2D input with shape (batch_size, input_dim), the output would have shape (batch_size, units).
    #    so we only need to ensure that the batch size are consistant, means we need to use tf.concat([a,b], axis=1)! key: [axis = 1]
    # 2. Keras Dense Layer already includes weights and bias

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
#######################################################################################################################
    'LSTM RNN consecutive blocks: here, notice we are using the same pretrained LSTM blocks for different time frames'
    rate = tf.constant(0, dtype=tf.float32)
    width_input_y = 2 * tau # here, 2 means y has the real and imag part

    'the proposed structure'
    for frame in range(frames):
        'Initialization'
        if frame == 0:
            y_real_init = tf.ones([batch_size, width_input_y])
            h_old_init = tf.zeros([batch_size, hidden_size])
            c_old_init = tf.zeros([batch_size, hidden_size])
            # initialize c_0, s_0
            h_old, c_old = RNN_upper(y_real_init, h_old_init, c_old_init)

        'design the uplink sensing vectors'
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
            # channel_cascaded_input = shape_h_train = (batch_size_train_1block * total_blocks, num_ris_elements)
            y_noiseless = tf.reduce_sum(tf.multiply(v_her_complex, channel_cascaded_input[((frame*N_w_using) * batch_size_test_1block): ((frame*N_w_using+1) * batch_size_test_1block), :]), 1, keepdims=True)
            noise = tf.complex(tf.random_normal(tf.shape(y_noiseless), mean=0.0, stddev=0.5), tf.random_normal(tf.shape(y_noiseless), mean=0.0, stddev=0.5))
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
        rate_perframe = tf.constant(0, dtype=tf.float32)
        for block in range((frame*N_w_using), ((frame+1)*N_w_using)):
            bf_gain_oneblock = tf.reduce_sum(tf.multiply(w_her_complex, channel_cascaded_input[(block * batch_size_test_1block): ((block+1) * batch_size_test_1block), :]), 1, keepdims=True)
            received_SNR_oneblock = tf.abs(bf_gain_oneblock) ** 2 / noise_power_linear_down
            rate_perblock = tf.log1p(received_SNR_oneblock) / np.log(2)
            rate_perframe += rate_perblock

        # AVERAGE OVER THE BATCH
        rate_perframe = tf.reduce_mean(rate_perframe)
        rate += rate_perframe
####################################################################################
####### Loss Function
loss = -rate / frames
####### Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss, name="training_op")
init = tf.global_variables_initializer()
saver = tf.train.Saver()
#########################################################################
###########  testing set (validation)
'we can use this part to generate the testing set!'
shape_h = (batch_size_test_1block * total_blocks, num_user, num_ris_elements, num_user)
h_input_val = np.zeros(shape_h, dtype=complex)

channel_tx_test, channel_rx_test = main_generate_channel(num_antenna_bs, num_ris_elements, num_user, irs_Nh,
                                                         batch_size_test_1block, Rician_factor, total_blocks, scale_factor,
                                                         rho_ht=0.995, rho_hr=0.99, location_bs=np.array([200, -200, 0]),
                                                         location_irs=np.array([0, 0, 0]), isTesting=False)
'here, we want to couple the ht and hr in the same time block!'
for block in range(total_blocks):
    h_input_val[(block * batch_size_test_1block): ((block + 1) * batch_size_test_1block), :, :, :] = get_cascaded_channel(channel_tx_test[:, :, :, block], channel_rx_test[:, :, :, block])

h_input_val = np.squeeze(h_input_val)
# here, we directly feed h_val as a whole
feed_dict_val = {channel_cascaded_input: h_input_val}

#########################################################################
###########  Training:
snr_temp = SNR_test_indB_up
with tf.Session() as sess:
    if initial_run == 1:
        init.run()
    else:
        saver.restore(sess, './params/addaptive_reflection_snr_'+str(int(snr_temp))+'_'+str(tau) + '_' + str(frames))

    best_loss = sess.run(loss, feed_dict=feed_dict_val)

    print('the trained loss (avg rate) for testing:', -best_loss)
    print(tf.test.is_gpu_available()) #Prints whether or not GPU is on

    # training session:
    no_increase = 0
    for epoch in range(n_epochs):
        batch_iter = 0

        for rnd_indices in range(batch_per_epoch):
            'since for each epoch, we need to re-train the NNet again'
            shape_h_train = (batch_size_train_1block * total_blocks, num_user, num_ris_elements, num_user)
            h_input_batch = np.zeros(shape_h_train, dtype=complex)

            channel_tx_batch, channel_rx_batch = main_generate_channel(num_antenna_bs, num_ris_elements, num_user,
                                                                       irs_Nh, batch_size_train_1block, Rician_factor,
                                                                       total_blocks, scale_factor, rho_ht=0.995, rho_hr=0.99,
                                                                       location_bs=np.array([200, -200, 0]),
                                                                       location_irs=np.array([0, 0, 0]), isTesting=False)

            for block in range(total_blocks):
                h_input_batch[(block * batch_size_train_1block): ((block + 1) * batch_size_train_1block), :, :, :] = get_cascaded_channel(channel_tx_batch[:, :, :, block], channel_rx_batch[:, :, :, block])

            h_input_batch = np.squeeze(h_input_batch)
            feed_dict_batch = {channel_cascaded_input: h_input_batch}
            sess.run(training_op, feed_dict=feed_dict_batch)
            batch_iter += 1

        loss_val = sess.run(loss, feed_dict=feed_dict_val)

        print('epoch', epoch, '  loss_test:%2.5f' % -loss_val, '  best_test:%2.5f  ' % -best_loss, 'no_increase:', no_increase)

        if loss_val < best_loss:
            save_path = saver.save(sess, './params/addaptive_reflection_snr_' + str(int(snr_temp)) + '_' + str(tau) + '_' + str(frames))
            'this step explains why best_loss is changing!'
            best_loss = loss_val
            no_increase = 0
        else:
            no_increase = no_increase + 1
        if no_increase > 20:
            break





























