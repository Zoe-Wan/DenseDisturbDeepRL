from matplotlib import pyplot as plt
import numpy as np
return_list = []
crash_list=[]
episode = 0
crush = 0
episodes_list = []
path = 'test'
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
        # rl.append(a)
        if a>0:
            crush+=1
            a=a-30
            crash_list.append(c)
        
        episodes_list.append(episode)    
        return_list.append(a)
        if not t:
            break
        # if episode == 5000:
        #     break
        # r = np.array(rl).mean()
        # episodes_list.append(episode)    
        # return_list.append(r)
        # if r > 0:    
        #     episodes_list.append(episode)    
        #     crush+=1                                      
        #     return_list.append(r) 


print(crush) # ori:260
print(np.array(return_list).mean())
print(np.array(crash_list).mean())

# ratio = np.array(return_list)
# print(ratio.mean()) # ori:95.907
plt.scatter(episodes_list, return_list,s=1)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on highway')
plt.savefig('./'+path+'.svg')

plt.show()

