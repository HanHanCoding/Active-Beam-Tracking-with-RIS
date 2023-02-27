import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from generate_channel import main_generate_channel
from tensorflow.keras.layers import BatchNormalization,Dense
from util_fun import get_cascaded_channel

'in this program, we want to mix all the channel stages into one batch, since we only want to train a fixed Active sensing structure'

'System information'
frames = 1
tau = 10
N_w_using = 30
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
channel_cascaded_input = tf.placeholder(tf.complex64, shape=(None, num_ris_elements), name="channel_cascaded_input")
batch_size = tf.shape(channel_cascaded_input[0: batch_size_test_1block, :])[0]

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
        real_v0 = tf.get_variable('real_v0' + str(t), shape=(num_ris_elements, 1), trainable=False)
        imag_v0 = tf.get_variable('imag_v0' + str(t), shape=(num_ris_elements, 1), trainable=False)
        real_v = real_v0 / tf.sqrt(tf.square(real_v0) + tf.square(imag_v0))
        imag_v = imag_v0 / tf.sqrt(tf.square(real_v0) + tf.square(imag_v0))
        v_her_complex = tf.complex(real_v, imag_v)
        v_her_complex = tf.reshape(tf.tile(v_her_complex, (batch_size, 1)), [batch_size, num_ris_elements])

        # Note that here, y is the pilots received by the base station in the uplink
        y_noiseless = tf.reduce_sum(tf.multiply(v_her_complex, channel_cascaded_input[(0 * batch_size_test_1block): (1 * batch_size_test_1block), :]), 1, keepdims=True)
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

    rate_perframe = tf.constant(0, dtype=tf.float32)
    for block in range(0, N_w_using):
        bf_gain_oneblock = tf.reduce_sum(tf.multiply(w_her_complex, channel_cascaded_input[(block * batch_size_test_1block): ((block + 1) * batch_size_test_1block),:]), 1, keepdims=True)
        received_SNR_oneblock = tf.abs(bf_gain_oneblock) ** 2 / noise_power_linear_down
        rate_perblock = tf.log1p(received_SNR_oneblock) / np.log(2)
        rate_perframe += rate_perblock

    # AVERAGE OVER THE BATCH
    rate_perframe = tf.reduce_mean(rate_perframe)


####################################################################################
####### Loss Function
loss = -rate_perframe
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
                                                         location_irs=np.array([0, 0, 0]), isTesting=True)
'here, we want to couple the ht and hr in the same time block!' # computing the cascaded channel
for block in range(total_blocks):
    h_input_val[(block * batch_size_test_1block): ((block + 1) * batch_size_test_1block), :, :, :] = get_cascaded_channel(channel_tx_test[:, :, :, block], channel_rx_test[:, :, :, block])

h_input_val = np.squeeze(h_input_val)
# here, we won't feed h_val frame per frame, but directly feed it as a whole
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




























