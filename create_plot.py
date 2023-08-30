import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import date

def create_plot(init_train_acc, init_dev_acc, train_acc, dev_acc, n_data, id, threshold, epoch, eval):

    plt.style.use('ggplot')

    init_train_array = np.array(list(init_train_acc.values()))
    init_mean_train_acc = np.average(init_train_array, axis=0)
    init_std_train_acc = np.std(init_train_array, axis=0)
    init_dev_array = np.array(list(init_dev_acc.values()))
    init_mean_dev_acc = np.average(init_dev_array, axis=0)
    init_std_dev_acc = np.std(init_dev_array, axis=0)

    train_array = np.array(list(train_acc.values()))
    mean_train_acc = np.average(train_array, axis=0)
    std_train_acc = np.std(train_array, axis=0)
    dev_array = np.array(list(dev_acc.values()))
    mean_dev_acc = np.average(dev_array, axis=0)
    std_dev_acc = np.std(dev_array, axis=0)

    fig, ax = plt.subplots()

    ax.plot(n_data, mean_train_acc, linestyle='-', color='green', label='Entrainnement sur la machine')
    ax.plot(n_data, init_mean_train_acc, linestyle='--', lw=2, color='green', label='Entrainnement sur l\'arbre')
    ax.plot(n_data, mean_dev_acc, linestyle='-', color='red', label='Test sur la machine')
    ax.plot(n_data, init_mean_dev_acc, linestyle='--', lw=2, color='red', label='Test sur l\'arbre')

    ax.fill_between(n_data, mean_train_acc - std_train_acc, mean_train_acc + std_train_acc, alpha = 0.3)
    ax.fill_between(n_data, mean_dev_acc - std_dev_acc, mean_dev_acc + std_dev_acc, color='red', alpha = 0.3)
    ax.fill_between(n_data, init_mean_train_acc - init_std_train_acc, init_mean_train_acc + init_std_train_acc, color='green', alpha = 0.3)
    ax.fill_between(n_data, init_mean_dev_acc - init_std_dev_acc, init_mean_dev_acc + init_std_dev_acc, color='red', alpha = 0.3)
    ax.set_xlabel("#data")
    ax.set_ylabel("Accuracy")
    names_dict = {
        0: "Machines avec 2 états",
        1: "Machines avec 3 états",
        2: "Machines avec 4 états",
        3: "Machines avec 5 états",
        4: "Machines avec 6 états",
        5: "Machines avec 7 états",
        6: "Machines avec 8 états",
        7: "Machines avec 9 états",
        8: "Machines avec 10 états",
        9: "Machines avec 11 états"
    }
    if (epoch == "best"):
        # Do not mention epoch in the title
        title = f"{names_dict[id]}"
    else:
        title = f"{names_dict[id]}, {epoch}"
    plt.title(title)
    plt.tight_layout()
    ax.legend()

    day = date.today()
    if not os.path.exists("./images_b4_ICML"):
        os.makedirs("./images_b4_ICML")
    os.makedirs(f"./images_b4_ICML/{day}",exist_ok=True)
    plotname = f"./images_b4_ICML/{day}/acc-{id}-{str(threshold)}-{epoch}-{eval}.png"
    print(f"Saved {plotname}")
    plt.savefig(plotname)
    # plt.show()
