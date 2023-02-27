import numpy as np
import scipy.io as sio
import os

'This code is for generating the trajectory of the user moving instance'
'This code is for generating channels according to the Rician Model. '
'for generate_channel_{Tx, Rx}, we directly generate channels in total_blocks'
def generate_location(curr_direction_state, curr_volocity_state, coordinate_UE_prev):
    coordinate_UE_curr = np.empty([1, 3])
    coordinate_UE_prev = coordinate_UE_prev.reshape(3)
    [x, y, z] = coordinate_UE_prev.tolist()
    speed_1 = 0.043
    speed_2 = 0.06
    # define the state based on the previous position
    # here, should be in total 5 states including the current state named '0'
    if curr_volocity_state == 0:
        state = {
            0: [x, y, z],
            1: [x, y+speed_1, z],
            2: [x-speed_1, y, z],
            3: [x, y-speed_1, z],
            4: [x+speed_1, y, z],
        }
    else:
        state = {
            0: [x, y, z],
            1: [x, y + speed_2, z],
            2: [x - speed_2, y, z],
            3: [x, y - speed_2, z],
            4: [x + speed_2, y, z],
        }

    coordinate_k = np.array(state[curr_direction_state])
    coordinate_UE_curr[0, :] = coordinate_k #Because we want to give coordinate_UE_curr a shape of (1, 3)
    return coordinate_UE_curr


def path_loss_r(d):
    loss = 30 + 22.0 * np.log10(d)
    return loss


def path_loss_d(d):
    loss = 32.6 + 36.7 * np.log10(d)
    return loss


def generate_pathloss_aoa_aod(location_user, location_bs, location_irs):
    """
    :param location_user: array (num_user,2)
    :param location_bs: array (2,)
    :param location_irs: array (2,)
    :return: pathloss = (pathloss_irs_bs, pathloss_irs_user, pathloss_bs_user)
            cos_phi = (cos_phi_1, cos_phi_2, cos_phi_3)
    """
    num_user = location_user.shape[0]
    # ========bs-irs==============
    d0 = np.linalg.norm(location_bs - location_irs)
    pathloss_irs_bs = path_loss_r(d0)
    aoa_bs = ( location_irs[0] - location_bs[0]) / d0
    aod_irs_y = (location_bs[1]-location_irs[1]) / d0
    aod_irs_z = (location_bs[2]-location_irs[2]) / d0
    # =========irs-user=============
    pathloss_irs_user = []
    aoa_irs_y = []
    aoa_irs_z = []
    for k in range(num_user):
        d_k = np.linalg.norm(location_user[k] - location_irs)
        pathloss_irs_user.append(path_loss_r(d_k))
        aoa_irs_y_k = (location_user[k][1] - location_irs[1]) / d_k
        aoa_irs_z_k = (location_user[k][2] - location_irs[2]) / d_k
        aoa_irs_y.append(aoa_irs_y_k)
        aoa_irs_z.append(aoa_irs_z_k)
    aoa_irs_y = np.array(aoa_irs_y)
    aoa_irs_z = np.array(aoa_irs_z)

    # =========bs-user=============
    'actually this part is not used in the paper.[we do not count the direct link!]'
    pathloss_bs_user = np.zeros([num_user, 1])
    for k in range(num_user):
        d_k = np.linalg.norm(location_user[k] - location_bs)
        pathloss_bs_user_k = path_loss_d(d_k)
        pathloss_bs_user[k, :] = pathloss_bs_user_k

    pathloss = (pathloss_irs_bs, np.array(pathloss_irs_user), np.array(pathloss_bs_user))
    aoa_aod = (aoa_bs, aod_irs_y, aod_irs_z, aoa_irs_y, aoa_irs_z)
    return pathloss, aoa_aod


# channel between IRS and BS
def generate_ht_LOS(params_system, irs_Nh, location_bs, location_irs, scale_factor, location_user):
    (num_antenna_bs, num_elements_irs, num_user) = params_system

    # find the path losses
    pathloss, aoa_aod = generate_pathloss_aoa_aod(location_user, location_bs, location_irs)
    (pathloss_irs_bs, pathloss_irs_user, pathloss_bs_user) = pathloss
    (aoa_bs, aod_irs_y, aod_irs_z, aoa_irs_y, aoa_irs_z) = aoa_aod

    pathloss_irs_bs = pathloss_irs_bs - scale_factor / 2
    pathloss_irs_bs = np.sqrt(10 ** ((-pathloss_irs_bs) / 10))

    a_bs = np.exp(1j * np.pi * aoa_bs * np.arange(num_antenna_bs))
    a_bs = np.reshape(a_bs, [num_antenna_bs, 1])

    i1 = np.mod(np.arange(num_elements_irs), irs_Nh)
    i2 = np.floor(np.arange(num_elements_irs) / irs_Nh)
    a_irs_bs = np.exp(1j * np.pi * (i1 * aod_irs_y + i2 * aod_irs_z))
    a_irs_bs = np.reshape(a_irs_bs, [num_elements_irs, 1])
    los_irs_bs = a_bs @ np.transpose(a_irs_bs.conjugate())

    return los_irs_bs, pathloss_irs_bs


# channel between IRS and user
def generate_hr_LOS(params_system, irs_Nh, location_bs, location_irs, scale_factor, location_user):
    (num_antenna_bs, num_elements_irs, num_user) = params_system

    # find the path losses
    pathloss, aoa_aod = generate_pathloss_aoa_aod(location_user, location_bs, location_irs)
    (pathloss_irs_bs, pathloss_irs_users, pathloss_bs_user) = pathloss
    (aoa_bs, aod_irs_y, aod_irs_z, aoa_irs_y, aoa_irs_z) = aoa_aod

    pathloss_irs_users = pathloss_irs_users - scale_factor / 2
    pathloss_irs_users = np.sqrt(10 ** ((-pathloss_irs_users) / 10))

    i1 = np.mod(np.arange(num_elements_irs), irs_Nh)
    i2 = np.floor(np.arange(num_elements_irs) / irs_Nh)

    a_irs_user = np.exp(1j * np.pi * (i1 * aoa_irs_y[0] + i2 * aoa_irs_z[0]))
    pathloss_irs_user = pathloss_irs_users[0]
    a_irs_user = np.transpose(a_irs_user)

    return a_irs_user, pathloss_irs_user


'All channels corresponding to the number of total_blocks have been generated!'
def generate_channel(params_system, irs_Nh, frames, location_bs, location_irs, Rician_factor, scale_factor, num_samples, rho_ht, rho_hr):
    (num_antenna_bs, num_elements_irs, num_user) = params_system

    channel_Tx = np.zeros([num_samples, num_elements_irs, num_user, frames]) + 1j * np.zeros([num_samples, num_elements_irs, num_user, frames])
    channel_Rx = np.zeros([num_samples, num_elements_irs, num_user, frames]) + 1j * np.zeros([num_samples, num_elements_irs, num_user, frames])

    'this is for visualization [since in this generate channel func, only take one sample at a time!]'
    channel_Tx_NLOS_part = np.zeros([1, num_elements_irs, frames]) + 1j * np.zeros([1, num_elements_irs, frames])
    channel_Rx_NLOS_part = np.zeros([1, num_elements_irs, frames]) + 1j * np.zeros([1, num_elements_irs, frames])

    # define the trasition matrix for the direction [since current position state will always be 0, so we will only need to define the transition matrix w.r.t. state 0]
    direction_trasition_matrix_init = np.array([0, 0.25, 0.25, 0.25, 0.25])
    direction_trasition_matrix_1 = np.array([0, 0.995, 0.002, 0.001, 0.002])
    direction_trasition_matrix_2 = np.array([0, 0.002, 0.995, 0.002, 0.001])
    direction_trasition_matrix_3 = np.array([0, 0.001, 0.002, 0.995, 0.002])
    direction_trasition_matrix_4 = np.array([0, 0.002, 0.001, 0.002, 0.995])

    # define the trasition matrix for velocity
    v_trasition_matrix = np.array([0.9, 0.1])

    'record the coordinate change of the user'
    user_corrdinate_all = []

    # generate channel
    'In this program, num_samples will always be 1! (since we just want to visualize the trajectory for one user)'
    for ii in range(num_samples):
        for frame in range(frames):
            # MARKOV model for each moving instance
            if frame == 0:
                # generate initial location
                # x = np.random.uniform(-20, 80)
                # y = np.random.uniform(-50, 100)
                # For better visualization, limit the initial range of the user to be smaller
                x = np.random.uniform(-10, 40)
                y = np.random.uniform(-25, 50)
                z = -10
                coordinate_UE_initial = [x, y, z]  # define coordinate_UE_initial as a list object
                coordinate_UE_curr = np.array(coordinate_UE_initial).reshape([1,3])
                coordinate_UE_prev = coordinate_UE_curr
                # calculate the next direction [here is the initial direction deciding]
                curr_direction_state = np.random.choice([0, 1, 2, 3, 4], p=direction_trasition_matrix_init)

                # deal with the Gauss-Marov evolution of the NLOS part ['for 0-th frame, ht 和 hr 的NLOS part是一样的']
                NLOS_part_oneSample_ht_real = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_elements_irs])
                NLOS_part_oneSample_ht_imag = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_elements_irs])

                NLOS_part_oneSample_hr_real = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_elements_irs])
                NLOS_part_oneSample_hr_imag = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_elements_irs])
            else:
                # calculate the velocity of this time
                curr_volocity_state = np.random.choice([0, 1], p=v_trasition_matrix)

                # since here the UE can only move in the states [0, 1, 2, 3, 4]
                # calculate the current user location
                coordinate_UE_curr = generate_location(curr_direction_state, curr_volocity_state, coordinate_UE_prev)
                coordinate_UE_prev = coordinate_UE_curr

                # 'since every frame, the prev_state state should always be considered as 0'
                # calculate the next direction
                if curr_direction_state == 1:
                    curr_direction_state = np.random.choice([0, 1, 2, 3, 4], p=direction_trasition_matrix_1)
                if curr_direction_state == 2:
                    curr_direction_state = np.random.choice([0, 1, 2, 3, 4], p=direction_trasition_matrix_2)
                if curr_direction_state == 3:
                    curr_direction_state = np.random.choice([0, 1, 2, 3, 4], p=direction_trasition_matrix_3)
                if curr_direction_state == 4:
                    curr_direction_state = np.random.choice([0, 1, 2, 3, 4], p=direction_trasition_matrix_4)

                # innovation part Gauss-Markov evolution model [for NLOS part of the channel]
                xi_r = np.random.normal(loc=0, scale=np.sqrt((1 - rho_hr ** 2) / 2), size=[num_antenna_bs, num_elements_irs])
                xi_t = np.random.normal(loc=0, scale=np.sqrt((1 - rho_ht ** 2) / 2), size=[num_antenna_bs, num_elements_irs])

                # deal with the Gauss-Marov evolution of the NLOS part
                NLOS_part_oneSample_ht_real = rho_ht * NLOS_part_oneSample_ht_real + xi_t
                NLOS_part_oneSample_ht_imag = rho_ht * NLOS_part_oneSample_ht_imag + xi_t

                NLOS_part_oneSample_hr_real = rho_hr * NLOS_part_oneSample_hr_real + xi_r
                NLOS_part_oneSample_hr_imag = rho_hr * NLOS_part_oneSample_hr_imag + xi_r

            'h_t: ndarray, (num_antenna_bs,num_elements_irs) channel between IRS and BS'
            # for LOS part of ht
            ht_LOS, pathloss_irs_bs = generate_ht_LOS(params_system, irs_Nh, location_bs, location_irs, scale_factor, coordinate_UE_curr)
            # for NLOS part of ht
            NLOS_part_oneSample_ht = NLOS_part_oneSample_ht_real + 1j * NLOS_part_oneSample_ht_imag
            h_t = (np.sqrt(Rician_factor / (1 + Rician_factor)) * ht_LOS + np.sqrt(1 / (1 + Rician_factor)) * NLOS_part_oneSample_ht) * pathloss_irs_bs
            channel_Tx[ii, :, :, frame] = np.reshape(h_t, [num_elements_irs, num_antenna_bs])

            'h_r: ndarray, (num_user, num_elements_irs) channel between IRS and user'
            # for LOS part of hr
            hr_LOS, pathloss_irs_user = generate_hr_LOS(params_system, irs_Nh, location_bs, location_irs, scale_factor, coordinate_UE_curr)
            # for NLOS part of hr
            NLOS_part_oneSample_hr = NLOS_part_oneSample_hr_real + 1j * NLOS_part_oneSample_hr_imag
            h_r = (np.sqrt(Rician_factor / (1 + Rician_factor)) * hr_LOS + np.sqrt(1 / (1 + Rician_factor)) * NLOS_part_oneSample_hr) * pathloss_irs_user
            channel_Rx[ii, :, :, frame] = np.reshape(h_r, [num_elements_irs, num_antenna_bs])

            'record the user location'
            # this is valid when sample number is 1!
            user_corrdinate_all.append(coordinate_UE_curr)

            'record the NLOS part of the user channel'
            channel_Tx_NLOS_part[:, :, frame] = NLOS_part_oneSample_ht
            channel_Rx_NLOS_part[:, :, frame] = NLOS_part_oneSample_hr

    'user_corrdinate_all: [total_frames, 1, 3]'
    user_corrdinate_all = np.array(user_corrdinate_all)

    return channel_Tx, channel_Rx, user_corrdinate_all, channel_Tx_NLOS_part, channel_Rx_NLOS_part



def main_generate_channel(sample_num_x, sample_num_y, num_antenna_bs, num_elements_irs, num_user, irs_Nh, num_samples, Rician_factor, frames, scale_factor,
                          rho_ht = 0.995, rho_hr = 0.99, location_bs=np.array([50, -30, 0]), location_irs=np.array([0, 0, 0]), isTesting = False):
    params_system = (num_antenna_bs, num_elements_irs, num_user)
    if isTesting:
        if not os.path.exists('Rician_channel'):
            os.makedirs('Rician_channel')
        params_all = (num_elements_irs, num_user, irs_Nh, num_samples, Rician_factor,scale_factor, frames)
        file_test_channel = './Rician_channel/channel' + str(params_all) +'.mat'
        if os.path.exists(file_test_channel):
            data_test = sio.loadmat(file_test_channel)
            channel_Tx, channel_Rx, user_corrdinate_all, channel_Tx_NLOS_part, channel_Rx_NLOS_part = data_test['channel_Tx'], data_test['channel_Rx'], data_test['user_corrdinate_all'], data_test['channel_Tx_NLOS_part'], data_test['channel_Rx_NLOS_part']
        else:
            channel_Tx, channel_Rx, user_corrdinate_all, channel_Tx_NLOS_part, channel_Rx_NLOS_part = generate_channel(params_system, irs_Nh=irs_Nh, frames=frames,
                                                                         location_bs=location_bs,
                                                                         location_irs=location_irs,
                                                                         Rician_factor=Rician_factor,
                                                                         scale_factor=scale_factor,
                                                                         num_samples=num_samples, rho_ht=rho_ht,
                                                                         rho_hr=rho_hr)

            sio.savemat(file_test_channel, {'channel_Tx': channel_Tx, 'channel_Rx': channel_Rx, 'user_corrdinate_all': user_corrdinate_all, 'channel_Tx_NLOS_part': channel_Tx_NLOS_part, 'channel_Rx_NLOS_part': channel_Rx_NLOS_part})
    else:
        channel_Tx, channel_Rx, user_corrdinate_all, channel_Tx_NLOS_part, channel_Rx_NLOS_part = generate_channel(params_system, irs_Nh=irs_Nh, frames=frames,
                                                                     location_bs=location_bs,
                                                                     location_irs=location_irs,
                                                                     Rician_factor=Rician_factor,
                                                                     scale_factor=scale_factor,
                                                                     num_samples=num_samples, rho_ht=rho_ht,
                                                                     rho_hr=rho_hr)

    'Generate the channel for the whole moving range!'
    ###############################################################################################################################
    # we want to sample the range in [sample_num_x * sample_num_y]
    range_coordinate_all_without_z = np.zeros([sample_num_x * sample_num_y, 1, 2])
    channel_ht_range_all = np.zeros([sample_num_x * sample_num_y, 1, num_elements_irs, frames], dtype=complex)
    channel_hr_range_all = np.zeros([sample_num_x * sample_num_y, 1, num_elements_irs, frames], dtype=complex)
    channel_hc_range_all = np.zeros([sample_num_x * sample_num_y, 1, num_elements_irs, frames], dtype=complex)

    channel_user_ht_all = np.zeros([1, num_elements_irs, frames], dtype=complex)
    channel_user_hr_all = np.zeros([1, num_elements_irs, frames], dtype=complex)
    channel_user_hc_all = np.zeros([1, num_elements_irs, frames], dtype=complex)

    'generate all the channels in the user moving range'
    for frame in range(frames):
        # test, generate the channel for the user here
        user_cur = user_corrdinate_all[frame, :, :]
        ht_LOS_user, pathloss_irs_bs_user = generate_ht_LOS(params_system, irs_Nh, location_bs, location_irs,scale_factor, user_cur)
        # for NLOS part of ht
        'since the NLOS part is the same in this whole range!'
        NLOS_part_oneSample_ht_user = channel_Tx_NLOS_part[:, :, frame]
        h_t_user = (np.sqrt(Rician_factor / (1 + Rician_factor)) * ht_LOS_user + np.sqrt(1 / (1 + Rician_factor)) * NLOS_part_oneSample_ht_user) * pathloss_irs_bs_user
        channel_user_ht_all[:, :, frame] = h_t_user

        'h_r: ndarray, (num_user, num_elements_irs) channel between IRS and user'
        # for LOS part of hr
        hr_LOS_user, pathloss_irs_user_user = generate_hr_LOS(params_system, irs_Nh, location_bs, location_irs,scale_factor, user_cur)
        # for NLOS part of hr
        NLOS_part_oneSample_hr_user = channel_Rx_NLOS_part[:, :, frame]
        h_r_user = (np.sqrt(Rician_factor / (1 + Rician_factor)) * hr_LOS_user + np.sqrt(1 / (1 + Rician_factor)) * NLOS_part_oneSample_hr_user) * pathloss_irs_user_user
        channel_user_hr_all[:, :, frame] = h_r_user

        channel_user_hc_all[:, :, frame] = channel_user_ht_all[:, :, frame] * channel_user_hr_all[:, :, frame]


        # generate LOS part of the channel according to the points on the range
        # for x
        for jj in range(sample_num_x):
            # for y
            for kk in range(sample_num_y):
                range_coordinate_x = -20 + jj * (100 / sample_num_x)
                range_coordinate_y = -50 + kk * (150 / sample_num_y)
                range_coordinate_z = -10
                range_coordinate = [range_coordinate_x, range_coordinate_y, range_coordinate_z]
                range_coordinate = np.array(range_coordinate).reshape([1, 3])
                # record the x-axis for range_coordinate_all_without_z
                range_coordinate_all_without_z[sample_num_y * jj + kk, :, 0] = range_coordinate_x
                range_coordinate_all_without_z[sample_num_y * jj + kk, :, 1] = range_coordinate_y

                'h_t: ndarray, (num_antenna_bs,num_elements_irs) channel between IRS and BS'
                # for LOS part of ht
                ht_LOS, pathloss_irs_bs = generate_ht_LOS(params_system, irs_Nh, location_bs, location_irs, scale_factor, range_coordinate)
                # for NLOS part of ht
                'since the NLOS part is the same in this whole range!'
                NLOS_part_oneSample_ht = channel_Tx_NLOS_part[:, :, frame]
                h_t = (np.sqrt(Rician_factor / (1 + Rician_factor)) * ht_LOS + np.sqrt(1 / (1 + Rician_factor)) * NLOS_part_oneSample_ht) * pathloss_irs_bs
                channel_ht_range_all[sample_num_y * jj + kk, :, :, frame] = h_t


                'h_r: ndarray, (num_user, num_elements_irs) channel between IRS and user'
                # for LOS part of hr
                hr_LOS, pathloss_irs_user = generate_hr_LOS(params_system, irs_Nh, location_bs, location_irs, scale_factor, range_coordinate)
                # for NLOS part of hr
                NLOS_part_oneSample_hr = channel_Rx_NLOS_part[:, :, frame]
                h_r = (np.sqrt(Rician_factor / (1 + Rician_factor)) * hr_LOS + np.sqrt(1 / (1 + Rician_factor)) * NLOS_part_oneSample_hr) * pathloss_irs_user
                channel_hr_range_all[sample_num_y * jj + kk, :, :, frame] = h_r

                # compute the cascaded channel hc
                channel_hc_range_all[sample_num_y * jj + kk, :, :, frame] = channel_ht_range_all[sample_num_y * jj + kk, :, :, frame] * channel_hr_range_all[sample_num_y * jj + kk, :, :, frame]

    return channel_Tx, channel_Rx, user_corrdinate_all, range_coordinate_all_without_z, channel_hc_range_all, channel_user_hc_all

























