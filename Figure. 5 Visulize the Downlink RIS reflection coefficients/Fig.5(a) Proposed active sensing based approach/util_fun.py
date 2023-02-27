import numpy as np

def get_cascaded_channel(channel_tx,channel_rx):
    [num_samples,num_elements_irs,num_user] = channel_tx.shape
    channel_a = np.zeros([num_samples,num_user, num_elements_irs,num_user],dtype=complex)
    for kk in range(num_user):
        for jj in range(num_user):
            channel_a[:,jj,:,kk] = channel_tx[:,:,jj]*channel_rx[:,:,kk]
    return channel_a
