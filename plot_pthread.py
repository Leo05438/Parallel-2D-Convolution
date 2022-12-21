import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def main():

    # for reference
    serial_fname = 'log_serial.txt'
    serial_f = open(serial_fname, 'r')
    serial_lines = serial_f.readlines()
    serial_data = {}
    
    i = 0
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
            
            serial_data[method][resolution][kernel_size] = time
            
            i += 1
        
        i += 1

    # for evaluation
    pthread_fname = 'log_pthread.txt'
    pthread_f = open(pthread_fname, 'r')
    pthread_lines = pthread_f.readlines()
    pthread_data = {}
    
    i = 0
    methods = []
    thread_nums = []
    resolutions = []
    kernel_sizes = []
    while i < len(pthread_lines):
        if '|' in pthread_lines[i]:

            assert i + 1 < len(pthread_lines), f'no time info in row {i}'

            attr_line = pthread_lines[i]
            time_line = pthread_lines[i + 1]
            # ignore conv_tp.out
            # if 'conv_tp.out' in attr_line:
            #     i += 1
            #     continue

            # head = line.split('|')[0].split('=')[1].strip()
            method = attr_line.split('|')[1].split('=')[1].strip()[:-4] # ignore ".out"
            thread_num = int(attr_line.split('|')[2].split('=')[1].strip())
            resolution = attr_line.split('|')[3].split('=')[1].strip()
            kernel_size = attr_line.split('|')[4].split('=')[1].strip()
            time = float(time_line.split(' ')[2]) # example : Elapsed time: 0.071317 sec

            if thread_num == 1:
                i += 1
                continue

            if method not in pthread_data.keys():
                pthread_data[method] = {}

            if thread_num not in pthread_data[method].keys():
                pthread_data[method][thread_num] = {}

            if resolution not in pthread_data[method][thread_num].keys():
                pthread_data[method][thread_num][resolution] = {}
                
            if method not in methods:
                methods.append(method)
            if thread_num not in thread_nums:
                thread_nums.append(thread_num)
            if resolution not in resolutions:
                resolutions.append(resolution)
            if kernel_size not in kernel_sizes:
                kernel_sizes.append(kernel_size)
            
            pthread_data[method][thread_num][resolution][kernel_size] = time
            
            i += 1
        
        i += 1

    # for k1 in pthread_data.keys():
    #     for k2 in pthread_data[k1].keys():
    #         for k3 in pthread_data[k1][k2].keys():
    #             for k4 in pthread_data[k1][k2][k3].keys():
    #                 print(f'pthread_data[{k1}][{k2}][{k3}][{k4}] = {pthread_data[k1][k2][k3][k4]}')

    # print(methods)
    # print(thread_nums)
    # print(resolutions)
    # print(kernel_sizes)
    
    # line colors
    colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange']

    # for same kernels different resulution
    plot_data = {}
    for resolution in resolutions:
        plot_data[resolution] = {}
        for kernel_size in kernel_sizes:
            plot_data[resolution][kernel_size] = {}
            for method in methods:
                plot_data[resolution][kernel_size][method] = []
                for thread_num in thread_nums:
                    val = pthread_data[method][thread_num][resolution][kernel_size]
                    if 'sk' in method:
                        val = serial_data['conv_sk'][resolution][kernel_size] / val
                    else:
                        val = serial_data['conv'][resolution][kernel_size] / val
                    val = np.round(val * 1000) / 1000
                    plot_data[resolution][kernel_size][method].append(val)
    
    font = {'size': 8}
    matplotlib.rc('font', **font)

    list_length = len(thread_nums)
    for resolution in resolutions:
        for kernel_size in kernel_sizes:
        
            x = np.arange(list_length)
            width = 0.18
            padding = 0.0

            max_y = 0.0
            fig, ax = plt.subplots(figsize=(2.5 * list_length, 5))
            for i, method in enumerate(methods):
                w = x - (len(methods) - 1) / 2 * (width + padding) + i * (width + padding)
                max_y = max(max_y, np.max(plot_data[resolution][kernel_size][method]))
                rect = ax.bar(w, 
                            plot_data[resolution][kernel_size][method], 
                            width, 
                            label=method, zorder=3)
                ax.bar_label(rect, padding=3)

            max_y = max_y * 1.2
            ax.set_xticks(x, thread_nums)
            ax.set_ylim(0.0, max_y)
            ax.grid(axis = 'y', zorder=0)
            ax.legend(loc='best', fontsize=8)
            ax.set_title(f'Pthread Results (Resolution = {resolution} Kernel Size = {kernel_size})', fontsize=16, pad=12.0)
            ax.set_xlabel('Thread Number', fontsize=12)
            ax.set_ylabel('Speedup', fontsize=12)
            fig.tight_layout()
            plt.savefig(f'plot_result/pthread_result/pthread_thread_num_{resolution}_{kernel_size}.png')
            plt.close()

    # for same kernels different resulution
    plot_data = {}
    for thread_num in thread_nums:
        plot_data[thread_num] = {}
        for kernel_size in kernel_sizes:
            plot_data[thread_num][kernel_size] = {}
            for method in methods:
                plot_data[thread_num][kernel_size][method] = []
                for resolution in resolutions:
                    val = pthread_data[method][thread_num][resolution][kernel_size]
                    if 'sk' in method:
                        val = serial_data['conv_sk'][resolution][kernel_size] / val
                    else:
                        val = serial_data['conv'][resolution][kernel_size] / val
                    val = np.round(val * 1000) / 1000
                    plot_data[thread_num][kernel_size][method].append(val)

    list_length = len(resolutions)
    for thread_num in thread_nums:
        for kernel_size in kernel_sizes:
        
            x = np.arange(list_length)
            width = 0.2
            padding = 0.0

            max_y = 0.0
            fig, ax = plt.subplots(figsize=(3.5 * list_length, 5))
            for i, method in enumerate(methods):
                w = x - (len(methods) - 1) / 2 * (width + padding) + i * (width + padding)
                max_y = max(max_y, np.max(plot_data[thread_num][kernel_size][method]))
                rect = ax.bar(w, 
                            plot_data[thread_num][kernel_size][method], 
                            width, 
                            label=method, zorder=3)
                ax.bar_label(rect, padding=3)

            max_y = max_y * 1.2
            ax.set_xticks(x, resolutions)
            ax.set_ylim(0.0, max_y)
            ax.grid(axis = 'y', zorder=0)
            ax.legend(loc='best', fontsize=8)
            ax.set_title(f'Pthread Results (Thread Number = {thread_num} Kernel Size = {kernel_size})', fontsize=16, pad=12.0)
            ax.set_xlabel('Resolution', fontsize=12)
            ax.set_ylabel('Speedup', fontsize=12)
            fig.tight_layout()
            plt.savefig(f'plot_result/pthread_result/pthread_resolution_{thread_num}_{kernel_size}.png')
            plt.close()

    
    # for same kernels different resulution
    plot_data = {}
    for resolution in resolutions:
        plot_data[resolution] = {}
        for thread_num in thread_nums:
            plot_data[resolution][thread_num] = {}
            for method in methods:
                plot_data[resolution][thread_num][method] = []
                for kernel_size in kernel_sizes:
                    val = pthread_data[method][thread_num][resolution][kernel_size]
                    if 'sk' in method:
                        val = serial_data['conv_sk'][resolution][kernel_size] / val
                    else:
                        val = serial_data['conv'][resolution][kernel_size] / val
                    val = np.round(val * 1000) / 1000
                    plot_data[resolution][thread_num][method].append(val)

    list_length = len(kernel_sizes)
    for thread_num in thread_nums:
        for resolution in resolutions:
        
            x = np.arange(list_length)
            width = 0.18
            padding = 0.0

            max_y = 0.0
            fig, ax = plt.subplots(figsize=(2.5 * list_length, 5))
            for i, method in enumerate(methods):
                w = x - (len(methods) - 1) / 2 * (width + padding) + i * (width + padding)
                max_y = max(max_y, np.max(plot_data[resolution][thread_num][method]))
                rect = ax.bar(w, 
                            plot_data[resolution][thread_num][method], 
                            width, 
                            label=method, zorder=3)
                ax.bar_label(rect, padding=3)

            max_y = max_y * 1.2
            ax.set_xticks(x, kernel_sizes)
            ax.set_ylim(0.0, max_y)
            ax.grid(axis = 'y', zorder=0)
            ax.legend(loc='best', fontsize=8)
            ax.set_title(f'Pthread Results (Thread Number = {thread_num} Resolution = {resolution})', fontsize=16, pad=12.0)
            ax.set_xlabel('Kernel Size', fontsize=12)
            ax.set_ylabel('Speedup', fontsize=12)
            fig.tight_layout()
            plt.savefig(f'plot_result/pthread_result/pthread_kernel_size_{thread_num}_{resolution}.png')
            plt.close()


if __name__=="__main__":
    main()