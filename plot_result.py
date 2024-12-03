import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    csv_files = []
    path = "./result_data"
    for file in os.listdir(path):
        if (file.endswith(".csv")):
            csv_files.append(file)
    
    plt.figure(figsize=(10, 6))
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    markers = ['o', 's', '^', 'D', 'v', 'P']
    count = 0
    for csv_file in csv_files:
        file_path = os.path.join(path, csv_file)
        data = pd.read_csv(file_path, header=None)
        x = data[0]
        y = data[1]
        label = csv_file.split("_")[1].split(".")[0]
        plt.plot(x, y, marker=markers[count], color=colors[count], label=label)
        count += 1
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("test acc")
    plt.title("Model Comparison")
    plt.grid()
    
    plt.savefig("./result_data/comparison.png")
    plt.show()
    
if (__name__ == "__main__"):
    main()