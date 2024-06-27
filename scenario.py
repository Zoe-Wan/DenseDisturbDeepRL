import math
import copy
class Filter:
    def __init__(self,d=30):
        self.scenarios = []
        self.scenarios.append(BackScenario(d=d))
        self.scenarios.append(FrontScenario(d=d))
        self.scenarios.append(LeftChangeFrontScenario(d=d))
        self.scenarios.append(LeftChangeBackScenario(d=d))
        self.scenarios.append(RightChangeFrontScenario(d=d))
        self.scenarios.append(RightChangeBackScenario(d=d))
      
    def get_action_mask(self,obs,indexes):
        
        # 由于距离较远的bv的状态变动不会对ego车辆驾驶行为造成影响，这里critic的目的是排除那些无意义目标
        # 30表示什么都不做
        action_masks = []
        for s in self.scenarios:
            action_masks.extend(s.get_action_mask(obs,indexes))
        action_masks = set(action_masks)
        return list(action_masks)
        
        
    

    
class Scenario:
    def __init__(self):

        pass
    
    
class BackScenario(Scenario):
    # 被追尾
    def __init__(self,d):
        # 条件放粗一点，这样子才能方便DQN发挥
        self.t = 0 # 反应时间一秒钟
        self.d = d # 安全车距30m
        self.v = 5 # 最大速度差 10m/s
        
    def get_action_mask(self,obs,indexes):
        action_mask=[]
        if indexes[2] is not None:
            action_mask.append(2*6+1)
            action_mask.append(2*6+3)
        if indexes[3] is not None:
            action_mask.append(3*6+1)
            action_mask.append(3*6+3)
        action_mask.append(3*6+5)
        if indexes[3] is not None:
            v = obs.loc[indexes[3],'v']-obs.loc[0,'v']
            x = obs.loc[indexes[3],'x']-obs.loc[0,'x']
            if (v>0 and -self.d-x<v**2/4) :
                # print("可触发被追尾场景")
                return action_mask
            else:
                return []
        return []

        

class FrontScenario(Scenario):
    #追尾
    def __init__(self,d):
        self.t = 0
        self.d = d
        self.v = 5 # 最大速度差 10m/s

        
    def get_action_mask(self,obs,indexes):
        # print(bv_list[2].lanelet_pose.s)
        
        action_mask=[]
        if indexes[2] is not None:
            action_mask.append(2*6)
            action_mask.append(2*6+2)
            action_mask.append(2*6+4)
        
        if indexes[2] is not None:
            v = obs.loc[0,'v']-obs.loc[indexes[2],'v']
            x = obs.loc[indexes[2],'x']-obs.loc[0,'x']
            # print("前方有车，"+str(v*self.t)+"<"+str(-x+self.d))
            if (v>0 and x-v**2/2<self.d):
                # print("可触发追尾场景")
                return action_mask
                # return []
            else:
                return []
        else:
            return []
    
class LeftChangeFrontScenario(Scenario):
    # 左变道时追尾前车
    # 变道这件事情比较麻烦，分为变道前的确认（是否）变道，变道时，位姿回正
    # 由于变道时车辆位姿/lanelet位置难以计算，因此这里只考虑变道前的确认部分扰动
    def __init__(self,d):
        self.t = 0
        self.d = d
        self.lanechange_v = 0
        self.lanechange_t = 2

        
    
    def get_action_mask(self,obs,indexes):
        action_mask=[]
        # 错误认为目标车道前车在前/消失
        if indexes[0] is not None:
            action_mask.append(0*6)
            action_mask.append(0*6+2)
            action_mask.append(0*6+4)
        action_mask.append(2*6+5)
        



        if indexes[0] is not None:
            # 如何判断驾驶意图？ 不管?
            v = obs.loc[0,'v']-obs.loc[indexes[0],'v']
            x = obs.loc[indexes[0],'x']-obs.loc[0,'x']
            # print("左前方有车，"+str(v*(self.lanechange_t))+"<"+str(-bv_list[0].lanelet_pose.s+self.d))

            # 这里应该算变道速度和变道时间的
            if v>0 and x-v*(self.lanechange_t)<self.d:
                # print("可触发左变道追尾场景")
                return action_mask
            else:
                return []
        else:
            return []

class LeftChangeBackScenario(Scenario):
    # 左变道时被追尾
    # 变道这件事情比较麻烦，分为变道前的确认（是否）变道，变道时，位姿回正
    # 由于变道时车辆位姿/lanelet位置难以计算，因此这里只考虑变道前的确认部分扰动
    def __init__(self,d):
        self.t = 0
        self.d = d
        self.lanechange_v = 0
        self.lanechange_t = 2

        
    
    def get_action_mask(self,obs,indexes):
        action_mask=[]
        # 错误认为目标车道后车在后/消失
        if indexes[1] is not None:
            action_mask.append(1*6+1)
            action_mask.append(1*6+3)
            action_mask.append(1*6+4)
        # 错误认为目标车道前车在后/出现
        if indexes[0] is not None:
            action_mask.append(0*6+1)
            action_mask.append(0*6+3)
        action_mask.append(2*6+5)
        


        if indexes[1] is not None:
            # 如何判断驾驶意图？ 不管?
            v = obs.loc[indexes[1],'v']-obs.loc[0,'v']
            x = obs.loc[0,'x']-obs.loc[indexes[1],'x']
            # print("左后方有车，"+str(v*(self.lanechange_t))+"<"+str(-bv_list[1].lanelet_pose.s+self.d))

            # 这里应该算变道速度和变道时间的
            if v>0 and x-v*self.lanechange_t<self.d:
                # print("可触发左变道被追尾场景")
                return action_mask
                # return []
            else:
                return []
        else:
            return []


    
class RightChangeFrontScenario(Scenario):
    # 右变道时追尾前车
    # 变道这件事情比较麻烦，分为变道前的确认（是否）变道，变道时，位姿回正
    # 由于变道时车辆位姿/lanelet位置难以计算，因此这里只考虑变道前的确认部分扰动
    def __init__(self,d):
        self.t = 0
        self.d = d
        self.lanechange_v = 0
        self.lanechange_t = 2

        
    
    def get_action_mask(self,obs,indexes):
        action_mask=[]
        # 错误认为目标车道前车在前/消失
        if indexes[4] is not None:
            action_mask.append(4*6)
            action_mask.append(4*6+2)
            action_mask.append(4*6+4)
        action_mask.append(2*6+5)
        

        # 对自车道前车的扰动，逼迫ego采取变道？ 暂且不要考虑那么复杂


        if indexes[4] is not None:
            # 如何判断驾驶意图？ 不管?
            v = obs.loc[indexes[4],'v']-obs.loc[0,'v']
            x = obs.loc[indexes[4],'x']-obs.loc[0,'x']

            # 这里应该算变道速度和变道时间的
            if v<0 and v*(self.lanechange_t)<-x+self.d:
                # print("可触发右变道追尾场景")
                return action_mask
            else:
                return []
        else:
            return []

class RightChangeBackScenario(Scenario):
    # 右变道时被追尾
    # 变道这件事情比较麻烦，分为变道前的确认（是否）变道，变道时，位姿回正
    # 由于变道时车辆位姿/lanelet位置难以计算，因此这里只考虑变道前的确认部分扰动
    def __init__(self,d):
        self.t = 0
        self.d = d
        self.lanechange_v = 0
        self.lanechange_t = 2

        
    
    def get_action_mask(self,obs,indexes):
        action_mask=[]
        # 错误认为目标车道后车在后/消失
        if indexes[5] is not None:
            action_mask.append(5*6+1)
            action_mask.append(5*6+3)
            action_mask.append(5*6+4)
        # 错误认为目标车道前车在后
        if indexes[5] is not None:
            action_mask.append(4*6+1)
            action_mask.append(4*6+3)
        action_mask.append(2*6+5)
        


        if indexes[5] is not None:
            v = obs.loc[indexes[5],'v']-obs.loc[0,'v']
            x = obs.loc[indexes[5],'x']-obs.loc[0,'x']
            # print("后方有车，"+str(v*(self.lanechange_t))+"<"+str(-bv_list[1].lanelet_pose.s+self.d))

            # 这里应该算变道速度和变道时间的
            if v>0 and -self.d-x<  v*self.lanechange_t:
                # print("可触发右变道被追尾场景")
                return action_mask
                # return []
            else:
                return []
        else:
            return []


    