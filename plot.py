import re
import torch
import matplotlib.pyplot as plt


def process_nohup(file_name):
    with open(file_name, 'r', encoding='UTF-8') as f:
        file = f.readlines()
    res = []
    for i in range(len(file)):
        temp = re.match('train loss.*', file[i])
        if temp:
            s = temp.group()
            if len(s) > 60:
                tmp = [float(re.search("cont=.{7}", s).group()[5:-1]), float(re.search("loss=.{7}", s).group()[5:]),
                       float(re.search("loss=.{7}", s).group()[5:])]
                res.append(tmp)
    return res


def plot_loss():
    result = torch.tensor(process_nohup("nohup_ICEWS14.out"))

    plt.style.use('ggplot')
    plt.figure(figsize=(18, 6), dpi=150)
    plt.xticks([])

    for i in range(3):
        plt.subplot(131+i)
        plt.plot(result[:, i])
    plt.show()


plot_loss()