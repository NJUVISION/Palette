import numpy as np
import matplotlib
import matplotlib.pyplot as plt

LOG_PATH = './results/log_central'

START = 0
END = 128600


epoch = []
td_loss = []
avg_reward = []
avg_entropy = []

matplotlib.rcParams['axes.unicode_minus']=False

def smooth(data_array, weight=0.99):
    """
    A smooth function, just like tensorbard
    """
    last = data_array[0]
    smoothed = []
    for new in data_array:
        smoothed_val = last * weight + (1 - weight) * new
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def main():
    with open(LOG_PATH, 'rb') as f:
        for line in f:
            if (('INFO:root:').encode() in line):
                parse = line[10:-1].split(('\t').encode())
                if(len(parse) == 5):
                    epoch.append(int((parse[0].split((':').encode()))[-1]))
                    td_loss.append(float((parse[1].split((':').encode()))[-1]))
                    avg_reward.append(float((parse[2].split((':').encode()))[-1]))
                    avg_entropy.append(float((parse[4].split((':').encode()))[-1]))

    f, ax = plt.subplots(3,1)

    ax[0].set_title(LOG_PATH)
    ax[0].plot(epoch[START:END], avg_reward[START:END],alpha=0.3,color='#0072BD')
    ax[0].plot(epoch[START:END], smooth(avg_reward,weight=0.997)[START:END],alpha=1,color='#0072BD')
    ax[0].set_ylabel('Average Reward',fontsize=12)

    ax[1].plot(epoch[START:END], avg_entropy[START:END],alpha=0.3,color='#0072BD')
    ax[1].plot(epoch[START:END], smooth(avg_entropy,weight=0.997)[START:END],alpha=1,color='#0072BD')
    ax[1].set_ylabel('Policy Entropy',fontsize=12)

    ax[2].plot(epoch[START:END], td_loss[START:END],"k",alpha=0.3,color='#0072BD')
    ax[2].plot(epoch[START:END], smooth(td_loss)[START:END],"k",alpha=1,color='#0072BD')
    ax[2].set_ylabel('TD Loss',fontsize=12)
    ax[2].set_xlabel('Epoch',fontsize=12)

    f.subplots_adjust(hspace=0)

    plt.show()

if __name__ == '__main__':
    main()
