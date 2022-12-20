import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def main():

    serial_fname = 'log_serial.txt'
    serial_f = open(serial_fname, 'r')
    serial_lines = serial_f.readlines()
    serial_data = {}
    
    i = 0
    methods = []
    resolutions = []
    kernel_sizes = []
    while i < len(serial_lines):
        if '|' in serial_lines[i]:

            assert i + 1 < len(serial_lines), f'no time info in row {i}'

            attr_line = serial_lines[i]
            time_line = serial_lines[i + 1]

            # head = line.split('|')[0].split('=')[1].strip()
            method = attr_line.split('|')[1].split('=')[1].strip()[:-4] # ignore ".out"
            resolution = attr_line.split('|')[2].split('=')[1].strip()
            kernel_size = attr_line.split('|')[3].split('=')[1].strip()
            time = float(time_line.split(' ')[2]) # example : Elapsed time: 0.071317 sec

            if method not in serial_data.keys():
                serial_data[method] = {}

            if resolution not in serial_data[method].keys():
                serial_data[method][resolution] = {}
                
            if kernel_size not in kernel_sizes:
                kernel_sizes.append(kernel_size)
            if method not in methods:
                methods.append(method)
            if resolution not in resolutions:
                resolutions.append(resolution)
            
            serial_data[method][resolution][kernel_size] = time
            
            i += 1
        
        i += 1

    # for k1 in serial_data.keys():
    #     for k2 in serial_data[k1].keys():
    #         for k3 in serial_data[k1][k2]:
    #             print(f'serial_data[{k1}][{k2}][{k3}] = {serial_data[k1][k2][k3]}')

    # print(methods)
    # print(resolutions)
    # print(kernel_sizes)

    # ax.plot(plot_data[kernel_size][method], label=method)
    # ax.scatter(x, plot_data[kernel_size][method])
    # for xval, yval in zip(x, plot_data[kernel_size][method]):
    #     ax.text(xval, yval + 1, str(yval), fontsize=8)
    
    # line colors
    colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange']

    # for same kernels different resulution
    plot_data = {}
    for kernel_size in kernel_sizes:
        plot_data[kernel_size] = {}
        for method in methods:
            plot_data[kernel_size][method] = []
            for resolution in resolutions:
                plot_data[kernel_size][method].append(np.round(serial_data[method][resolution][kernel_size] * 1000) / 1000)
    
    font = {'size'   : 9}
    matplotlib.rc('font', **font)

    for kernel_size in kernel_sizes:
        
        x = np.arange(len(resolutions))
        width = 0.25
        padding = 0.05

        max_y = 0.0
        fig, ax = plt.subplots(figsize=(1.5 * len(resolutions), 5))
        for i, method in enumerate(methods):
            w = x - (len(methods) - 1) / 2 * (width + padding) + i * (width + padding)
            max_y = max(max_y, np.max(plot_data[kernel_size][method]))
            rect = ax.bar(w, 
                          plot_data[kernel_size][method], 
                          width, 
                          label=method, zorder=3)
            ax.bar_label(rect, padding=3)

        max_y = max_y * 1.2
        ax.set_xticks(x, resolutions)
        ax.set_ylim(0.0, max_y)
        ax.grid(axis = 'y', zorder=0)
        ax.legend(loc='best', fontsize=8)
        ax.set_title(f'Serial Results (Kernel Size = {kernel_size})', fontsize=16, pad=12.0)
        ax.set_xlabel('Resolution', fontsize=12)
        ax.set_ylabel('Time (sec)', fontsize=12)
        fig.tight_layout()
        plt.savefig(f'plot_result/serial_result/serial_resulution_{kernel_size}.png')
        plt.close()


    # for same kernels different resulution
    plot_data = {}
    for resolution in resolutions:
        plot_data[resolution] = {}
        for method in methods:
            plot_data[resolution][method] = []
            for kernel_size in kernel_sizes:
                plot_data[resolution][method].append(np.round(serial_data[method][resolution][kernel_size] * 1000) / 1000)
    
    font = {'size'   : 9}
    matplotlib.rc('font', **font)

    for resolution in resolutions:
        
        x = np.arange(len(kernel_sizes))
        width = 0.25
        padding = 0.05

        max_y = 0.0
        fig, ax = plt.subplots(figsize=(1.5 * len(kernel_sizes), 5))
        for i, method in enumerate(methods):
            w = x - (len(methods) - 1) / 2 * (width + padding) + i * (width + padding)
            max_y = max(max_y, np.max(plot_data[resolution][method]))
            rect = ax.bar(w, 
                          plot_data[resolution][method], 
                          width, 
                          label=method, zorder=3)
            ax.bar_label(rect, padding=3)

        max_y = max_y * 1.2
        ax.set_xticks(x, kernel_sizes)
        ax.set_ylim(0.0, max_y)
        ax.grid(axis = 'y', zorder=0)
        ax.legend(loc='best', fontsize=8)
        ax.set_title(f'Serial Results (Resolution = {resolution})', fontsize=16, pad=12.0)
        ax.set_xlabel('Kernel Size', fontsize=12)
        ax.set_ylabel('Time (sec)', fontsize=12)
        fig.tight_layout()
        plt.savefig(f'plot_result/serial_result/serial_kernel_{resolution}.png')
        plt.close()


if __name__=="__main__":
    main()