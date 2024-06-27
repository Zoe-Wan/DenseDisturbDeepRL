import sys
import argparse
import os
from timeit import default_timer as timer
import pandas as pd
from NADE_core import select_controlled_bv_and_action  # Import NADE algorithm
import global_val
from highway_env.envs.highway_env_NDD import *  # Environment
from CAV_agent.agent import AV_RL_agent  # AV agent
from DisturbRL import Disturber
import torch
import matplotlib.pyplot as plt
import math
from scenario import Filter
import random
start = timer()
# settings
parser = argparse.ArgumentParser()
parser.add_argument('--experiment-name', type=str, required=True,
                    help='The name of the experiment folder where the data will be stored')
parser.add_argument("--episode",type=int, default='15000',
                    help='Number of test to run in one worker')
parser.add_argument("--reward",type=int, default='1')
parser.add_argument("--epsilon",type=float,default='0.1',
                    help='Epsilon value of the NADE algorithm')
parser.add_argument("--load-epi",type=int,default='0')
parser.add_argument("--render-flag",action='store_true', default=False)
parser.add_argument("--given-ini",action='store_true', default=False)
parser.add_argument("--fuzz",action='store_true', default=False)
parser.add_argument("--nomask",action='store_true', default=False)
parser.add_argument("--nature",action='store_true', default=False)
parser.add_argument("--test-flag",action='store_true', default=False)

parser.add_argument("--save-freq",type=int,default='100')
parser.add_argument("--d",type=int,default='30')

parser.add_argument("--save-dir",type=str,default='./ckptRL' )


args = parser.parse_args()

def get_nddprossi(state,action):
    matrix = [
        [0.0031,0.0082,0.0161,0.0262],
        [0.0031,0.0082,0.0161,0.0262],
        [0.0031,0.0082,0.0161,0.0262],
        [0.0031,0.0082,0.0161,0.0262],
        [0.0002,0.0015,0.0053,0.0101],
        ]
    
    if action==36 or action is None:
        return 1
    else:
        action_type = action%6
        if action_type!=5:
            dobj= action//6
            dis = min(3,abs(state[dobj*2]*120))

            nddprossi = matrix[action_type][int(dis//15)]
        else:
            nddprossi =  0.0015 
    return nddprossi

def get_nddaction(state):
    matrix = [
        [0.0031,0.0082,0.0161,0.0262],
        [0.0031,0.0082,0.0161,0.0262],
        [0.0031,0.0082,0.0161,0.0262],
        [0.0031,0.0082,0.0161,0.0262],
        [0.0002,0.0015,0.0053,0.0101],
        [0.0010,0.0010,0.0010,0.0010],
        [0.9669,0.9532,0.9108,0.8528]
        ]
    matrix = np.array(matrix)
    ndd_actions = []
    for i in range(6):
        objdis = state[i*2]*120
        arr = matrix[:,min(3,abs(int(objdis//15)))]
        act = random.choices(list(range(7)),arr)[0]
        ndd_actions.append(act)
    return ndd_actions


def transfer_to_state_input(env, df):
    """
    Transfer observations to network state input.

    Args:
        df: AV observation.

    Returns:
        network input.
    """
    # Normalize
    # observed_BV_num = env.cav_observation_num
    ego = df.loc[0]
    indexes = [None,None,None,None,None,None]

    # 初始化存储最小正 x 和最大负 x 的字典
    min_positive_x = {0: np.inf, 1: np.inf, 2: np.inf}
    max_negative_x = {0: -np.inf, 1: -np.inf, 2: -np.inf}

    # 遍历 DataFrame，寻找最小正 x 和最大负 x，记录对应的 index
    for idx, row in df[1:].iterrows():
        laneid = int(row['lane_id'])-int(ego['lane_id'])+1
        x = row['x']-ego['x']
        if laneid>2 or laneid<0:
            continue
        else:
            if x > 0 and x < min_positive_x[laneid]:
                min_positive_x[laneid] = x
                indexes[laneid*2]=idx
                # 这里不用+1，还原的时候可以直接放，不用考虑ego
            elif x < 0 and x > max_negative_x[laneid]:
                max_negative_x[laneid] = x
                indexes[laneid*2+1]=idx

    state_df =  normalize_state(env,df,indexes)
    


    return state_df.flatten(),indexes

def normalize_state(env, df, indexes):
    """
    Transfer the observation to relative states.

    Args:
        df: AV observation.

    Returns:
        normalized observations.
    """
    df_copy = []

    for i in indexes:
        if i is not None:
            df_copy.append([df.loc[i,'x']-df.loc[0,'x'],df.loc[i,'v']-df.loc[0,'v']])
        else:
            df_copy.append([global_val.INVALID_STATE,global_val.INVALID_STATE])

    # Normalize BV
    x_position_range =  env.cav_observation_range
    side_lanes =  env.road.network.all_side_lanes(env.vehicle.lane_index)

    df_copy = np.array(df_copy)
    
    df_copy[:, 0] = utils.remap(df_copy[:, 0], [-x_position_range, x_position_range], [-1, 1])
    df_copy[:, 1] = utils.remap(df_copy[:, 1], [ -20,  20], [-1, 1])


    df_copy[df_copy>1.1]=1
    df_copy[df_copy<-1.1]=-1

    return df_copy


def s_to_l(s):
    lane = s[:-2].split(',')
    l = []
    for car in lane:
        # print(car)

        x,v = car.split(' ')
        x = int(x)
        v = int(v)
        l.append([x,v])
    return l

def load_ini():
    
    ini_data=[]
    with open("test.txt",'r') as f:
        while True:
            l1 = f.readline()
            if not l1:
                break
            env = []
            env.append(s_to_l(l1))
            l2 = f.readline()
            env.append(s_to_l(l2))
            l3 = f.readline()
            env.append(s_to_l(l3))
            ini_data.append(env)
            f.readline()

    return ini_data
                

if __name__ == '__main__':

    savepath = os.path.join(args.save_dir, args.experiment_name)
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    return_list=[]
    disturb_prob_list=[]
    weighted_collision_rate_list=[]


    CAV_model = "AV2"  # AV2: using RL AV model (AV2 in the paper).
    env_config = {"min_distance": 0, "max_distance": 400, "min_velocity": 20, "max_velocity": 60, "min_lane": 0, "max_lane": 2, "cav_observation_range": global_val.cav_obs_range,
                  "bv_observation_range": global_val.bv_obs_range, "candidate_controlled_bv_num": 8, "cav_observation_num": 10, "bv_observation_num": 10, "generate_vehicle_mode": "NDD",
                  "delete_BV_position": 1200, "CAV_model": CAV_model, "policy_frequency":1}

    ini_data = None
    if args.given_ini:
        env_config["generate_vehicle_mode"] = "given_ini"
        ini_data = load_ini()


    whole_dict = {
        "AV-Finish-Test": {'count':0, 'disrupt_rate':0},
        "AV-Crash": {'count':0, 'disrupt_rate':0}
        }
    
    env = HighwayEnvNDD(env_config)  # Initialize simulation environment.
    if CAV_model == "AV2":
        print("Using AV2 (RL agent) as the CAV model!")
        CAV_agent = AV_RL_agent(env)
    else:
        raise NotImplementedError('{0} does not supported..., set to AV2'.format(CAV_model))

    disturber = Disturber(
        state_dim=(6)*2, 
        action_dim=(6)*6+1, 
        lr_actor=1e-4, 
        lr_critic=1e-4, 
        gamma=1.0, 
        K_epochs=10, 
        eps_clip=0.2, 
        has_continuous_action_space=False, 
        action_std_init=0.6)
    
    
    scenario_filter = Filter(d = args.d)
    start_epi = 0
    if args.load_epi!=0:
        disturber.load(os.path.join(savepath, str(args.load_epi)))
        if not args.test_flag:
            start_epi = args.load_epi
    if args.test_flag:
        logfile = open(os.path.join(savepath,'test-'+str(args.load_epi)+'.txt'),'w')
        Truenessfile = open(os.path.join(savepath,'test-Trueness-'+str(args.load_epi)+'.txt'),'w')
        WCRfile = open(os.path.join(savepath,'test-WCR-'+str(args.load_epi)+'.txt'),'w')
        
        
        
    elif args.load_epi!=0:
        # 注意，这里没有对return_list和disrupt_count_list和loss进行初始化
        logfile = open(os.path.join(savepath,'return.txt'),'a+')
        lossfile = open(os.path.join(savepath,'loss.txt'),'a+')
        Truenessfile = open(os.path.join(savepath,'Trueness.txt'),'a+')
        WCRfile = open(os.path.join(savepath,'WCR.txt'),'a+')
        # 速度差和crushfile
        
    else:    
        logfile = open(os.path.join(savepath,'return.txt'),'w')
        lossfile = open(os.path.join(savepath,'loss.txt'),'w')
        Truenessfile = open(os.path.join(savepath,'Trueness.txt'),'w')
        WCRfile = open(os.path.join(savepath,'WCR.txt'),'w')
        
        

    for test_item in range(start_epi,args.episode):  # Main loop for each simulation episode.
        
        global_val.episode = test_item
        if args.given_ini:
            obs_and_indicator, _ = env.reset(ini_data[test_item])  # Initialize and reset the simulation environment.
        else:
            obs_and_indicator, _ = env.reset() 
        # obs_and_indicator, _ = env.reset() 
        obs, action_indicator = obs_and_indicator[0], obs_and_indicator[1]  # observation of the AV and the action indicator array (1 means the action is safe and 0 means dangerous)
        done = False  # The flag of the end of each episode.
        if args.render_flag:
            env.render()

        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        episode_return = 0
        disrupt_count = 0
        frame_count = 0
        Trueness = 1
        sum_weight=1
        
        while not done:
            for vehicle in env.road.vehicles:
                vehicle.selected = False
                vehicle.disturbed = 0
            bv_action_idx_list = select_controlled_bv_and_action(obs.bv_observation, env)
            # give driving action to bvs
            BV_action = bv_action_idx_list

            state,indexes = transfer_to_state_input(env, obs.cav_observation)

            if args.nomask:
                action_mask = list(range(36))
            else:
                action_mask = scenario_filter.get_action_mask(obs.cav_observation,indexes)
            new_mask = [36]
            for idx in action_mask:
                if indexes[idx//6]is not None:
                    env.cav_obs_vehs_list[indexes[idx//6]-1].selected=True 
                new_mask.append(idx)

            action_mask = new_mask
            
            disturb_action=None
            disturbed_obs = obs.cav_observation

            ndd_prossi=1
            action_logprob=0
            
            if len(action_mask)>0:
                ndd_actions =  get_nddaction(state)
                if not args.nature:
                    if args.test_flag:
                        disturb_action, state_val, action_logprob = disturber.take_action_test(state,action_mask)
                    else:
                        disturb_action, state_val, action_logprob = disturber.take_action(state,action_mask)
                
                    disturb_obji = disturb_action//6
                    if disturb_obji<6:
                        disturb_type = disturb_action%6
                        ndd_actions[disturb_obji] = disturb_type
                        ndd_prossi = get_nddprossi(state,disturb_action)
                    else:
                        pass
            
                disturbed_obs,disturb_actions = disturber.disturb(obs.cav_observation, ndd_actions, indexes)
            else:
                pass
            


            # ori_action_indicator_after_lane_conflict = CAV_agent.lane_conflict_safety_check(obs.cav_observation, action_indicator.cav_indicator)
            # ori_PA, ori_CAV_action = CAV_agent.decision(obs.cav_observation, ori_action_indicator_after_lane_conflict)


            action_indicator_after_lane_conflict = CAV_agent.lane_conflict_safety_check(disturbed_obs, action_indicator.cav_indicator)
            PA, CAV_action = CAV_agent.decision(disturbed_obs, action_indicator_after_lane_conflict)

            for i in range(6):
                if disturb_actions[i]!=6:
                    if disturb_actions[i]!=5:
                        vid = indexes[i]
                        if vid is not None:
                            v = env.cav_obs_vehs_list[vid-1]
                            v.disturbed = disturb_actions[i]+1
                    else:
                        # print("幽灵目标: "+str(i))
                        pass
                
            if args.render_flag:
                env.render()
   
            action = Action(cav_action=CAV_action, bv_action=BV_action)


            obs_and_indicator, done, info, _ = env.step(action)  # Simulate one step.

            next_obs, next_action_indicator = obs_and_indicator[0], obs_and_indicator[1]

            next_state = transfer_to_state_input(env,next_obs.cav_observation)


            reward = 0
           
           
            Trueness*=ndd_prossi
            sum_weight*=(ndd_prossi/math.exp(action_logprob))
            if (ndd_prossi/math.exp(action_logprob))<0.999:
                sum_weight*=2
            # breakpoint()

            
            if info["scene_type"] == "AV-Crash":
                # sum_weight = 1
                # for s,log_prob in zip(disturber.buffer.ndd_prossi_list,disturber.buffer.logprobs):
                    # sum_weight*=s/math.exp(log_prob)    
                reward = args.reward*(1-sum_weight*100)
                if reward<-args.reward:
                    reward = -args.reward

                
            frame_count+=1
                
            episode_return += reward
            if  disturb_action is not None and disturb_action != 36:
                disturber.buffer.rewards.append(reward)
                disturber.buffer.ndd_prossi_list.append(ndd_prossi)
                disturber.buffer.is_terminals.append(done)
            # else:
            #     if not done and args.test_flag:
            #         disturber.buffer.clear()
            obs, action_indicator = next_obs, next_action_indicator
            
        if not args.test_flag and not args.nature:
            # aloss, closs = disturber.update(transition_dict)
            if len(disturber.buffer.ndd_prossi_list)>0:
                aloss, closs = disturber.update()
                lossfile.write(str(aloss)+" "+str(closs)+"\n")
        if info["scene_type"] == "AV-Crash" :
            return_list.append(episode_return)
            disturb_prob_list.append(Trueness)
            weighted_collision_rate_list.append(sum_weight)
            WCRfile.write(str(sum_weight)+"\n")
        else:
            weighted_collision_rate_list.append(0)
            WCRfile.write("0\n")
            
        Truenessfile.write(str(Trueness)+"\n")
        if not args.test_flag:
            logfile.write(str(episode_return)+" "+str(disrupt_count)+"\n")
        else:
            logfile.write(str(episode_return)+" "+str(disrupt_count)+"\n")
        whole_dict[info["scene_type"]]['count'] += 1
        whole_dict[info["scene_type"]]['disrupt_rate'] += disrupt_count/frame_count

        s = whole_dict['AV-Crash']['count'] + whole_dict['AV-Finish-Test']['count']
        if len(weighted_collision_rate_list)!=0:
            WC_array = np.array(weighted_collision_rate_list)
            WCR = WC_array.mean()
            CIHW = 1.96*WC_array.std()/math.sqrt(s)
        else:
            WCR = -1
            CIHW = -1
        
        
        print(f"总仿真次数为：{s}\n"+
        f"发生碰撞的比率为{whole_dict['AV-Crash']['count']/s}\n"+
        f"本轮仿真中，感知扰动在真实情况下的发生概率为{Trueness}\n"+
        f"待测系统自然情况下的碰撞概率为{WCR}\n"+
        f"估计的95%置信区间半宽为{CIHW}\n\n")
        # f"碰撞双方速度差为{}\n"+
        # f"碰撞类型属于{}\n\n")
        disturber.buffer.clear()


        if test_item % args.save_freq==0 and not args.test_flag:
            d = os.path.join(savepath,str(test_item))
            if not os.path.exists(d):
                os.mkdir(d)
            disturber.save(d)
    WCRfile.close()
    Truenessfile.close()
    logfile.close()
    lossfile.close()
    end = timer()
    print("Time:", end - start)

'''
15
总仿真次数为：2344
发生碰撞的比率为0.29180887372013653
本轮仿真中，感知扰动在真实情况下的发生概率为6.560496746496001e-16
待测系统自然情况下的碰撞概率为1.0906681156108353e-05
估计的95%置信区间半宽为9.405741221789734e-06
;
30
总仿真次数为：2072
发生碰撞的比率为0.2833011583011583
本轮仿真中，感知扰动在真实情况下的发生概率为0.00261121
待测系统自然情况下的碰撞概率为9.232830402870328e-06
估计的95%置信区间半宽为3.9323335745987045e-06
'''