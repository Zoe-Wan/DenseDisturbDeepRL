import sys
import argparse
import os
from timeit import default_timer as timer

from NADE_core import select_controlled_bv_and_action  # Import NADE algorithm
import global_val
from highway_env.envs.highway_env_NDD import *  # Environment
from CAV_agent.agent import AV_RL_agent  # AV agent
from DisturbRL_ori import Disturber
import torch
import matplotlib.pyplot as plt
start = timer()
# settings
parser = argparse.ArgumentParser()
parser.add_argument('--experiment-name', type=str, required=True,
                    help='The name of the experiment folder where the data will be stored')
parser.add_argument("--episode",type=int, default='5000',
                    help='Number of test to run in one worker')
parser.add_argument("--epsilon",type=float,default='0.1',
                    help='Epsilon value of the NADE algorithm')
parser.add_argument("--load-epi",type=int,default='0')
parser.add_argument("--render-flag",action='store_true', default=False)
parser.add_argument("--test-flag",action='store_true', default=False)

parser.add_argument("--save-freq",type=int,default='100')

parser.add_argument("--save-dir",type=str,default='./ckptRL' )





args = parser.parse_args()


def output_whole_dict(resdict):
    """
    Print out test results. E.g., total number of episodes have accident, percentage of episodes have accident.

    Args:
        resdict: test result dictionary.

    """
    print(resdict)
    whole_num = sum(resdict.values())
    for key in resdict:
        rate = resdict[key] / whole_num
        print(key, "rate", rate)

def transfer_to_state_input(env, original_obs_df):
    """
    Transfer observations to network state input.

    Args:
        original_obs_df: AV observation.

    Returns:
        network input.
    """
    # Normalize
    observed_BV_num = env.cav_observation_num
    state_df =  normalize_state(env,original_obs_df)
    # Fill with dummy vehicles
    if state_df.shape[0] <  observed_BV_num + 1:
        fake_vehicle_row = [-1, -1, -1]  # at the back of observation box, with minimum speed and at the top lane
        fake_vehicle_rows = [fake_vehicle_row for _ in range( observed_BV_num + 1 - state_df.shape[0])]
        rows = np.array(fake_vehicle_rows)
        # rows = -np.ones(( observed_BV_num + 1 - state_df.shape[0], len(observation.FEATURES_acc_training)))
        for row in rows:
            state_df.loc[len(state_df)] = row
        # state_df = state_df.append(pd.DataFrame(data=rows, columns=observation.FEATURES_acc_training), ignore_index=True)

    return state_df.values.flatten()

def normalize_state(env, df):
    """
    Transfer the observation to relative states.

    Args:
        df: AV observation.

    Returns:
        normalized observations.
    """
    df_copy = df.copy()

    # Get relative BV data first
    df_copy.loc[1:, 'x'] = df_copy['x'][1:] - df_copy['x'][0]

    # Normalize BV
    x_position_range =  env.cav_observation_range
    side_lanes =  env.road.network.all_side_lanes( env.vehicle.lane_index)
    lane_num = len(side_lanes)
    lane_range = lane_num - 1

    df_copy.loc[1:, 'x'] = utils.remap(df_copy.loc[1:, 'x'], [-x_position_range, x_position_range], [-1, 1])
    df_copy.loc[1:, 'lane_id'] = utils.remap(df_copy.loc[1:, 'lane_id'], [0, lane_range], [-1, 1])
    df_copy.loc[1:, 'v'] = utils.remap(df_copy.loc[1:, 'v'], [ env.min_velocity,  env.max_velocity], [-1, 1])

    # Normalize CAV
    df_copy.loc[0, 'x'] = 0
    df_copy.loc[0, 'lane_id'] = utils.remap(df_copy.loc[0, 'lane_id'], [0, lane_range], [-1, 1])
    df_copy.loc[0, 'v'] = utils.remap(df_copy.loc[0, 'v'], [ env.min_velocity,  env.max_velocity], [-1, 1])

    assert ((-1.1 <= df_copy.x).all() and (df_copy.x <= 1.1).all())
    assert ((-1.1 <= df_copy.v).all() and (df_copy.v <= 1.1).all())
    assert ((-1.1 <= df_copy.lane_id).all() and (df_copy.lane_id <= 1.1).all())

    return df_copy

# 

if __name__ == '__main__':

    savepath = os.path.join(args.save_dir, args.experiment_name)
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    return_list=[]
    disrupt_count_list=[]
    if args.load_epi!=0:
        # 注意，这里没有对return_list和disrupt_count_list和loss进行初始化
        logfile = open(os.path.join(savepath,'return.txt'),'a+')
        lossfile = open(os.path.join(savepath,'loss.txt'),'a+')
        
    else:    
        logfile = open(os.path.join(savepath,'return.txt'),'w')
        lossfile = open(os.path.join(savepath,'loss.txt'),'w')

    CAV_model = "AV2"  # AV2: using RL AV model (AV2 in the paper).
    env_config = {"min_distance": 0, "max_distance": 1200, "min_velocity": 20, "max_velocity": 40, "min_lane": 0, "max_lane": 2, "cav_observation_range": global_val.cav_obs_range,
                  "bv_observation_range": global_val.bv_obs_range, "candidate_controlled_bv_num": 8, "cav_observation_num": 10, "bv_observation_num": 10, "generate_vehicle_mode": "NDD",
                  "delete_BV_position": 1200, "CAV_model": CAV_model, "policy_frequency": 1}


    whole_dict = {"AV-Finish-Test": 0, "AV-Crash": 0}
    
    env = HighwayEnvNDD(env_config)  # Initialize simulation environment.
    if CAV_model == "AV2":
        print("Using AV2 (RL agent) as the CAV model!")
        CAV_agent = AV_RL_agent(env)
    else:
        raise NotImplementedError('{0} does not supported..., set to AV2'.format(CAV_model))

    disturber = Disturber(
        state_dim = (1+env_config["cav_observation_num"])*3, 
        hidden_dim=128, 
        action_dim=(1+env_config["cav_observation_num"]), 
        actor_lr=5e-3, 
        critic_lr=1e-2, 
        lmbda=0.95, 
        epochs=10, 
        eps=0.2, 
        gamma=0.98, 
        device=torch.device("cuda"))
    # disturber = Disturber(
    #     state_dim=(1+env_config["cav_observation_num"])*3, 
    #     action_dim=(1+env_config["cav_observation_num"]), 
    #     lr_actor=5e-3, 
    #     lr_critic=1e-2, 
    #     gamma=0.98, 
    #     K_epochs=10, 
    #     eps_clip=0.2, 
    #     has_continuous_action_space=False, 
    #     action_std_init=0.6)


    if args.load_epi!=0:
        disturber.load(os.path.join(savepath, str(args.load_epi)))

    for test_item in range(args.load_epi,args.episode):  # Main loop for each simulation episode.
        global_val.episode = test_item

        obs_and_indicator, _ = env.reset()  # Initialize and reset the simulation environment.
        obs, action_indicator = obs_and_indicator[0], obs_and_indicator[1]  # observation of the AV and the action indicator array (1 means the action is safe and 0 means dangerous)
        done = False  # The flag of the end of each episode.
        if args.render_flag:
            env.render()

        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        episode_return = 0
        disrupt_count = 0
        aloss_buffer, closs_buffer = [], []

        while not done:
            
            bv_action_idx_list = select_controlled_bv_and_action(obs.bv_observation, env)
            BV_action = bv_action_idx_list


            n = len(obs.cav_observation)
            state = transfer_to_state_input(env, obs.cav_observation)
            disturb_action = disturber.take_action(state,n)
            # print(disturber.critic(torch.tensor([state],dtype=torch.float).to(torch.device("cuda"))))

            action_indicator_after_lane_conflict = CAV_agent.lane_conflict_safety_check(obs.cav_observation, action_indicator.cav_indicator)
            CAV_action_ori = CAV_agent.decision(obs.cav_observation, action_indicator_after_lane_conflict)

            disturbed_obs = disturber.disturb(obs.cav_observation, disturb_action)

            for vehicle in env.road.vehicles:
                vehicle.disturbed = False
                
            if disturb_action!=0:
                env.cav_obs_vehs_list[disturb_action-1].disturbed = True
            else:
                pass


            action_indicator_after_lane_conflict = CAV_agent.lane_conflict_safety_check(disturbed_obs, action_indicator.cav_indicator)
            CAV_action = CAV_agent.decision(disturbed_obs, action_indicator_after_lane_conflict)

            if CAV_action==CAV_action_ori and not args.test_flag: # 在非训练场景下是否可以事先得知agent的动作？
                env.cav_obs_vehs_list[disturb_action-1].disturbed = False
                disturb_action = 0

                
            if args.render_flag:
                env.render()
   
            action = Action(cav_action=CAV_action, bv_action=BV_action)


            obs_and_indicator, done, info, weight = env.step(action)  # Simulate one step.
            next_obs, next_action_indicator = obs_and_indicator[0], obs_and_indicator[1]

            next_state = transfer_to_state_input(env,next_obs.cav_observation)


            if info["scene_type"] == "AV-Crash":
                reward = 100
            elif disturb_action!=0:
                disrupt_count+=1
                reward =  -1
            else:
                reward = 0


            # 这里考虑是否要使用reward sharping


            # 评估的方法：
            # 1. 发生碰撞事件时，驾驶过程中插入的扰动次数均值
            # 2. 发生碰撞事件的概率

            episode_return += reward

            # 这里是不是可以shuffle一下state
            transition_dict['states'].append(state)
            transition_dict['actions'].append(disturb_action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)

            obs, action_indicator = next_obs, next_action_indicator
            
        if info["scene_type"] == "AV-Crash" and not args.test_flag:
            aloss, closs = disturber.update(transition_dict)

            # lossfile.write(str(aloss)+" "+str(closs)+"\n")
            disrupt_count_list.append(disrupt_count)
            return_list.append(episode_return)
        logfile.write(str(episode_return)+" "+str(disrupt_count)+"\n")
        whole_dict[info["scene_type"]] += 1
        output_whole_dict(whole_dict)


        if test_item % args.save_freq==0 and not args.test_flag:
            lossfile.write(str(aloss)+" "+str(closs)+"\n")
            d = os.path.join(savepath,str(test_item))
            if not os.path.exists(d):
                os.mkdir(d)
            disturber.save(d)

    logfile.close()
    lossfile.close()
    end = timer()
    print("Time:", end - start)
    # episodes_list = list(range(len(return_list)))
    # plt.plot(episodes_list, return_list)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title('PPO on highway')
    # plt.show()
