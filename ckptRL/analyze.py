from matplotlib import pyplot as plt
import numpy as np
return_list = [0]
crash_list=[0]
episode = 0
crush = 0
episodes_list = [0]
rl = []
path = 'final400m15s'
with open(path+"/return.txt") as f:
    while True:
        episode+=1
        # rl=[]
        # for i in range(1):
        t = f.readline()
        if not t:
            break
        a,c = t.split(" ")
        a = eval(a)
        c = eval(c)
        rl.append(a)

        # if a<0: # 未触发危险事件时的平均扰动次数 
        #     # crash_list.append(c)
        #     # episodes_list.append(episode)   
        #     # crash_list.append((episode-1)/(episode)*crash_list[-1]+c/episode)
        #     pass
        # if a>0:
        #     crush+=1
        #     episodes_list.append(episode*3)
        #     crash_list.append(crush*2/3/episode)

            # crash_list.append(c)       

        # episodes_list.append(episode)
        # crash_list.append(crush/episode)
        # if episode%500==0:
         
        # a+=3
        
        # return_list.append(a)
        # m = a
        # n=0
        # for i in range(max(0,episode-300),episode):
        #     m+=return_list[i]
        #     n+1
        
        # m/=301

        # if not t:
        #     break
        # if episode == 40000:
        #     break
        # r = np.array(rl).mean()
        # episodes_list.append(episode)    
        # return_list.append(r)
        # if r > 0:    
        #     episodes_list.append(episode)    
        #     crush+=1                                      
        #     return_list.append(r) 

# print(episode) # ori:260

# print(crush) # ori:260
# print(np.array(return_list[:-1]).reshape(500,100).mean(1))
# print(np.array(crash_list).mean())

# ratio = np.array(return_list)
# print(ratio.mean()) # ori:95.907
srl = []
for i in range(len(rl)-50):
    mean = 0
    for j in range(50):
        mean+=rl[i+j]
    mean/=50
    srl.append(mean)
    
plt.plot(range(len(srl)),srl)


plt.xlabel('Episodes')
plt.ylabel('Crash Rate')
plt.title('PPO on highway')
plt.savefig('./'+path+'.svg')

plt.show()

