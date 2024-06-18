import global_val
from highway_env import utils
import numpy as np
import yaml

from highway_env.vehicle.behavior import NDDVehicle  # Background vehicle





def select_controlled_bv_and_action(obs, env):

    bv_action_idx_list=[]
    for i in range(len(env.controlled_bvs)):  # Loop over all candidate controlled BVs.
        possi_array = get_NDD_possi(env.controlled_bvs[i], obs[i])
        # give driving action to bvs

        bv_action_idx = np.random.choice(len(global_val.BV_ACTIONS), 1, replace=False, p=possi_array)

        bv_action_idx_list.append(bv_action_idx)

    return bv_action_idx_list



def get_NDD_possi(bv, obs_bv):
    """
    Obtain the BV naturalistic action probability distribution.

    Args:
        bv: the specific BV.
        obs_bv: BV's observation.

    Returns:
        list(float): the probability mass function for different actions. The first element is the left lane change probability, the second element is the right lane change probability, the followed by
        longitudinal acceleration probability.

    """

    possi_array = np.zeros((len(global_val.BV_ACTIONS)), dtype=float)
    _, _, lat_possi_array = bv.Lateral_NDD(obs_bv, modify_flag=False)
    _, long_possi_array = bv.Longitudinal_NDD(obs_bv, modify_flag=False)
    possi_array[0], possi_array[1] = lat_possi_array[0], lat_possi_array[2]
    possi_array[2:] = lat_possi_array[1] * long_possi_array
    return possi_array
