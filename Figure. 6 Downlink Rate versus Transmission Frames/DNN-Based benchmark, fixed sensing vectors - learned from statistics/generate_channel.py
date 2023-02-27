import numpy as np
import scipy.io as sio
import os

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
def generate_channel(params_system, irs_Nh, total_blocks, location_bs, location_irs, Rician_factor, scale_factor, num_samples, rho_ht, rho_hr):
    (num_antenna_bs, num_elements_irs, num_user) = params_system

    channel_Tx = np.zeros([num_samples, num_elements_irs, num_user, total_blocks]) + 1j * np.zeros([num_samples, num_elements_irs, num_user, total_blocks])
    channel_Rx = np.zeros([num_samples, num_elements_irs, num_user, total_blocks]) + 1j * np.zeros([num_samples, num_elements_irs, num_user, total_blocks])

    # define the trasition matrix for the direction [since current position state will always be 0, so we will only need to define the transition matrix w.r.t. state 0]
    direction_trasition_matrix_init = np.array([0, 0.25, 0.25, 0.25, 0.25])
    direction_trasition_matrix_1 = np.array([0, 0.995, 0.002, 0.001, 0.002])
    direction_trasition_matrix_2 = np.array([0, 0.002, 0.995, 0.002, 0.001])
    direction_trasition_matrix_3 = np.array([0, 0.001, 0.002, 0.995, 0.002])
    direction_trasition_matrix_4 = np.array([0, 0.002, 0.001, 0.002, 0.995])

    # define the trasition matrix for velocity
    v_trasition_matrix = np.array([0.9, 0.1])

    # generate channel
    for ii in range(num_samples):
        for block in range(total_blocks):
            # MARKOV model for each moving instance
            if block == 0:
                # generate initial location
                x = np.random.uniform(-20, 80)
                y = np.random.uniform(-50, 100)
                z = -10
                coordinate_UE_initial = [x, y, z]  # define coordinate_UE_initial as a list object
                coordinate_UE_curr = np.array(coordinate_UE_initial).reshape([1,3])
                coordinate_UE_prev = coordinate_UE_curr
                # calculate the next direction [here is the initial direction deciding]
                curr_direction_state = np.random.choice([0, 1, 2, 3, 4], p=direction_trasition_matrix_init)

                # deal with the Gauss-Marov evolution of the NLOS part ['for 0-th block, ht 和 hr 的NLOS part是一样的']
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

                # 'since every block, the prev_state state should always be considered as 0'
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
            channel_Tx[ii, :, :, block] = np.reshape(h_t, [num_elements_irs, num_antenna_bs])

            'h_r: ndarray, (num_user, num_elements_irs) channel between IRS and user'
            # for LOS part of hr
            hr_LOS, pathloss_irs_user = generate_hr_LOS(params_system, irs_Nh, location_bs, location_irs, scale_factor, coordinate_UE_curr)
            # for NLOS part of hr
            NLOS_part_oneSample_hr = NLOS_part_oneSample_hr_real + 1j * NLOS_part_oneSample_hr_imag
            h_r = (np.sqrt(Rician_factor / (1 + Rician_factor)) * hr_LOS + np.sqrt(1 / (1 + Rician_factor)) * NLOS_part_oneSample_hr) * pathloss_irs_user
            channel_Rx[ii, :, :, block] = np.reshape(h_r, [num_elements_irs, num_antenna_bs])

    return channel_Tx, channel_Rx


def main_generate_channel(num_antenna_bs, num_elements_irs, num_user, irs_Nh, num_samples, Rician_factor, total_blocks, scale_factor,
                          rho_ht = 0.995, rho_hr = 0.99, location_bs=np.array([50, -30, 0]), location_irs=np.array([0, 0, 0]), isTesting = False):
    params_system = (num_antenna_bs, num_elements_irs, num_user)
    if isTesting:
        if not os.path.exists('Rician_channel'):
            os.makedirs('Rician_channel')
        params_all = (num_elements_irs, num_user, irs_Nh, num_samples, Rician_factor,scale_factor, total_blocks)
        file_test_channel = './Rician_channel/channel' + str(params_all) +'.mat'
        if os.path.exists(file_test_channel):
            data_test = sio.loadmat(file_test_channel)
            channel_Tx, channel_Rx = data_test['channel_Tx'], data_test['channel_Rx']
        else:
            channel_Tx, channel_Rx = generate_channel(params_system, irs_Nh=irs_Nh, total_blocks=total_blocks,
                                                                         location_bs=location_bs,
                                                                         location_irs=location_irs,
                                                                         Rician_factor=Rician_factor,
                                                                         scale_factor=scale_factor,
                                                                         num_samples=num_samples, rho_ht=rho_ht,
                                                                         rho_hr=rho_hr)

            sio.savemat(file_test_channel, {'channel_Tx': channel_Tx, 'channel_Rx': channel_Rx})
    else:
        channel_Tx, channel_Rx = generate_channel(params_system, irs_Nh=irs_Nh, total_blocks=total_blocks,
                                                                     location_bs=location_bs,
                                                                     location_irs=location_irs,
                                                                     Rician_factor=Rician_factor,
                                                                     scale_factor=scale_factor,
                                                                     num_samples=num_samples, rho_ht=rho_ht,
                                                                     rho_hr=rho_hr)
    return channel_Tx, channel_Rx




