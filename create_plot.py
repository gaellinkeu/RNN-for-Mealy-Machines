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

    ax.plot(n_data, mean_train_acc, linestyle='-', color='green', label='train')
    ax.plot(n_data, init_mean_train_acc, linestyle='--', lw=2, color='green', label='initial train')
    ax.plot(n_data, mean_dev_acc, linestyle='-', color='red', label='dev')
    ax.plot(n_data, init_mean_dev_acc, linestyle='--', lw=2, color='red', label='initial dev')

    ax.fill_between(n_data, mean_train_acc - std_train_acc, mean_train_acc + std_train_acc, alpha = 0.3)
    ax.fill_between(n_data, mean_dev_acc - std_dev_acc, mean_dev_acc + std_dev_acc, color='red', alpha = 0.3)
    ax.fill_between(n_data, init_mean_train_acc - init_std_train_acc, init_mean_train_acc + init_std_train_acc, color='green', alpha = 0.3)
    ax.fill_between(n_data, init_mean_dev_acc - init_std_dev_acc, init_mean_dev_acc + init_std_dev_acc, color='red', alpha = 0.3)
    ax.set_xlabel("#data")
    ax.set_ylabel("Accuracy")
    names_dict = {
        0: "Machines with 2 states",
        1: "Machines with 3 states",
        2: "Machines with 4 states",
        3: "Machines with 5 states",
        4: "Machines with 6 states",
        5: "Machines with 7 states",
        6: "Machines with 8 states",
        7: "Machines with 9 states",
        8: "Machines with 10 states",
        9: "Machines with 11 states"
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
    plt.show()
