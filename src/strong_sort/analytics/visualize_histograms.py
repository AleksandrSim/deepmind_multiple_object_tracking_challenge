import matplotlib.pyplot as plt
import json

def plot_data_skip_first_n_then_top_m(data, skip_n, top_m):
    # Sort the data by counts in descending order
    sorted_data = sorted(data.items(), key=lambda item: item[1], reverse=True)
    
    # Skip the first 'skip_n' items, then take the top 'top_m' items
    data_after_n = sorted_data[skip_n: skip_n + top_m]

    # Extract keys and values
    keys = [item[0] for item in data_after_n]
    values = [item[1] for item in data_after_n]

    # Create figure and plot
    plt.figure(figsize=(20, 10))
    plt.bar(keys, values)
    plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability
    plt.title(f'Data after first {skip_n} items, showing next {top_m} items')
    plt.xlabel('Keys')
    plt.ylabel('Values')
    plt.show()

if __name__=='__main__':
    with open('/Users/aleksandrsimonyan/Desktop/deepmind/1_comb.json', 'r') as f:
        data = json.load(f)

    plot_data_skip_first_n_then_top_m(data, 40, 100)