import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import seaborn as sns
colors = sns.color_palette("tab10",10)

def vis_weight_decay(vis_path="HalfCheetah-v2"):
    vis_path = os.path.join("logger",vis_path)
    vis_list = os.listdir(vis_path)
    loss_curve = []
    reward_curve = []
    series_names = []
    min_min = [0.]*2
    max_max = [0.]*2
    min_len = 3000
    for log in vis_list:
        if log.split('.')[-1] != "txt":
            continue
        series_name = log.split(".txt")[0].split('-')[1] if len(log.split('.txt')[0].split('-')) > 1 else None
        series_names.append(series_name if series_name!="" else None)
        f = open(os.path.join(vis_path,log),"rt")
        data = f.read()
        data = np.array([[int(line.split(' ')[0]), float(line.split(' ')[1]), float(line.split(' ')[3])] for line in data.split('\n') if line != ""])
        loss_curve.append(data[:,0:2])
        reward_curve.append(np.concatenate([data[:,0].reshape(-1,1),data[:,2].reshape(-1,1)],axis=-1))
        min_len = min_len if loss_curve[-1][-1,0]>=min_len else loss_curve[-1][-1,0]
        min_min[0] = min_min[0] if np.min(loss_curve[-1][:,1]) >= min_min[0] else np.min(loss_curve[-1][:,1])
        min_min[1] = min_min[1] if np.min(reward_curve[-1][:,1]) >= min_min[1] else np.min(reward_curve[-1][:,1])
        max_max[0] = max_max[0] if np.max(loss_curve[-1][:,1]) <= max_max[0] else np.max(loss_curve[-1][:,1])
        max_max[1] = max_max[1] if np.max(reward_curve[-1][:,1]) <= max_max[1] else np.max(reward_curve[-1][:,1])
    
    min_len = 30
    print(min_len)
    print(min_min)
    print(max_max)
    # loss curve
    plt.close()
    plt.xlim(0,min_len)
    plt.ylim(min_min[0],max_max[0])
    for j,data in enumerate(loss_curve):
        plt.plot(data[:min_len if len(data) > min_len else len(data),0],data[:min_len if len(data) > min_len else len(data),1],label=series_names[j],c=colors[j])
    plt.legend()
    plt.savefig(os.path.join(vis_path,"loss.png"))
    plt.close()

    # reward curve
    plt.close()
    plt.xlim(0,min_len)
    plt.ylim(min_min[1],max_max[1])
    for j,data in enumerate(reward_curve):
        plt.plot(data[:min_len if len(data) > min_len else len(data),0],data[:min_len if len(data) > min_len else len(data),1],label=series_names[j],c=colors[j])
    plt.legend()
    plt.savefig(os.path.join(vis_path,"reward.png"))
    plt.close()
    

if __name__=="__main__":
    vis_weight_decay("HalfCheetah-v2")