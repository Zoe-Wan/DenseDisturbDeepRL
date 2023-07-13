from matplotlib import pyplot as plt
import numpy as np
return_list = []

episode = 0
crush = 0
episodes_list = []

with open("new_param_penal/loss.txt") as f:
    while True:
        t = f.readline()
        if not t:
            break
        episode+=1
        a,c = t.split(" ")
        a = eval(a)
        c = eval(c)
        episodes_list.append(episode)    
        return_list.append((a,c))
        # if r > 0:    
        #     episodes_list.append(episode)    
        #     crush+=1                                      
        #     return_list.append(r) 


# print(crush) # ori:260
# ratio = np.array(return_list)
# print(ratio.mean()) # ori:95.907
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on highway')

plt.show()
