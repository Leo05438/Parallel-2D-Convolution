import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

def main(post_fix : str):

    serial_fname = f'log_serial{post_fix}.txt'
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


    cuda_fname = f'log_cuda{post_fix}.txt'
    cuda_f = open(cuda_fname, 'r')
    cuda_lines = cuda_f.readlines()
    cuda_data = {}
    
    i = 0
    methods = []
    general_methods = []
    seperable_methods = []
    resolutions = []
    kernel_sizes = []
    while i < len(cuda_lines):
        if '|' in cuda_lines[i]:

            assert i + 1 < len(cuda_lines), f'no time info in row {i}'

            # ignore any implementations using pitch
            if 'pitch' in cuda_lines[i]:
                i += 1
                continue

            attr_line = cuda_lines[i]
            time_line = cuda_lines[i + 1]

            # head = line.split('|')[0].split('=')[1].strip()
            method = attr_line.split('|')[1].split('=')[1].strip()[:-4] # ignore ".out"
            resolution = attr_line.split('|')[2].split('=')[1].strip()
            kernel_size = attr_line.split('|')[3].split('=')[1].strip()
            time = float(time_line.split(' ')[2]) # example : Elapsed time: 0.071317 sec

            if method not in cuda_data.keys():
                cuda_data[method] = {}

            if resolution not in cuda_data[method].keys():
                cuda_data[method][resolution] = {}
                
            if kernel_size not in kernel_sizes:
                kernel_sizes.append(kernel_size)

            if method not in seperable_methods:
                if 'conv_sk' in method:
                    seperable_methods.append(method)
            
            if method not in general_methods:
                if 'conv_sk' not  in method:
                    general_methods.append(method)
            
            if method not in methods:
                methods.append(method)

            if resolution not in resolutions:
                resolutions.append(resolution)
            
            cuda_data[method][resolution][kernel_size] = time
            
            i += 1
        
        i += 1
    
    font = {'size': 9}
    matplotlib.rc('font', **font)

    # for same kernels different resulution
    plot_general_data = {}
    plot_seperable_data = {}
    plot_all_data = {}
    for resolution in resolutions:
        plot_general_data[resolution] = {}
        plot_seperable_data[resolution] = {}
        plot_all_data[resolution] = {}
        for method in methods:
            plot_general_data[resolution][method] = []
            plot_seperable_data[resolution][method] = []
            plot_all_data[resolution][method] = []
            for kernel_size in kernel_sizes:
                val = cuda_data[method][resolution][kernel_size]
                if 'conv_sk' in method:
                    val = serial_data['conv_sk'][resolution][kernel_size] / val
                    val = np.round(val * 1000) / 1000
                    plot_seperable_data[resolution][method].append(val)
                else:
                    val = serial_data['conv'][resolution][kernel_size] / val
                    val = np.round(val * 1000) / 1000
                    plot_general_data[resolution][method].append(val)
                plot_all_data[resolution][method].append(val)


    # for general
    list_length = len(kernel_sizes)
    for resolution in resolutions:
        
        x = np.arange(list_length)
        width = 0.2
        padding = 0.0

        max_y = 0.0
        fig, ax = plt.subplots(figsize=(2.5 * list_length, 5))
        for i, method in enumerate(general_methods):
            w = x - (len(general_methods) - 1) / 2 * (width + padding) + i * (width + padding)
            max_y = max(max_y, np.max(plot_general_data[resolution][method]))
            rect = ax.bar(w, 
                        plot_general_data[resolution][method], 
                        width, 
                        label=method, zorder=3)
            ax.bar_label(rect, padding=3)

        max_y = max_y * 1.4
        ax.set_xticks(x, kernel_sizes)
        ax.set_ylim(0.0, max_y)
        ax.grid(axis = 'y', zorder=0)
        ax.legend(loc='best', fontsize=8)
        ax.set_title(f'CUDA General Kernel Results (Resolution = {resolution})', fontsize=16, pad=12.0)
        ax.set_xlabel('Kernel Size', fontsize=12)
        ax.set_ylabel('Speedup', fontsize=12)
        fig.tight_layout()
        plt.savefig(f'plot_result/cuda_result/cuda_general_kernel_size_{resolution}.png')
        plt.close()

    # for seperable
    for resolution in resolutions:
        
        x = np.arange(list_length)
        width = 0.2
        padding = 0.0

        max_y = 0.0
        fig, ax = plt.subplots(figsize=(2.5 * list_length, 5))
        for i, method in enumerate(seperable_methods):
            w = x - (len(seperable_methods) - 1) / 2 * (width + padding) + i * (width + padding)
            max_y = max(max_y, np.max(plot_seperable_data[resolution][method]))
            rect = ax.bar(w, 
                        plot_seperable_data[resolution][method], 
                        width, 
                        label=method, zorder=3)
            ax.bar_label(rect, padding=3)

        max_y = max_y * 1.4
        ax.set_xticks(x, kernel_sizes)
        ax.set_ylim(0.0, max_y)
        ax.grid(axis = 'y', zorder=0)
        ax.legend(loc='best', fontsize=8)
        ax.set_title(f'CUDA Seperable Kernel Results (Resolution = {resolution})', fontsize=16, pad=12.0)
        ax.set_xlabel('Kernel Size', fontsize=12)
        ax.set_ylabel('Speedup', fontsize=12)
        fig.tight_layout()
        plt.savefig(f'plot_result/cuda_result/cuda_seperable_kernel_size_{resolution}.png')
        plt.close()

    # for all
    for resolution in resolutions:
        
        x = np.arange(list_length)
        width = 0.14
        padding = 0.0

        max_y = 0.0
        fig, ax = plt.subplots(figsize=(5 * list_length, 5))
        for i, method in enumerate(methods):
            w = x - (len(methods) - 1) / 2 * (width + padding) + i * (width + padding)
            max_y = max(max_y, np.max(plot_all_data[resolution][method]))
            rect = ax.bar(w, 
                        plot_all_data[resolution][method], 
                        width, 
                        label=method, zorder=3)
            ax.bar_label(rect, padding=3)

        max_y = max_y * 1.4
        ax.set_xticks(x, kernel_sizes)
        ax.set_ylim(0.0, max_y)
        ax.grid(axis = 'y', zorder=0)
        ax.legend(loc='best', fontsize=8)
        ax.set_title(f'CUDA Results (Resolution = {resolution})', fontsize=16, pad=12.0)
        ax.set_xlabel('Kernel Size', fontsize=12)
        ax.set_ylabel('Speedup', fontsize=12)
        fig.tight_layout()
        plt.savefig(f'plot_result/cuda_result/cuda_all_kernel_size_{resolution}.png')
        plt.close()

    # for same resulution different kernels
    plot_general_data = {}
    plot_seperable_data = {}
    plot_all_data = {}
    for kernel_size in kernel_sizes:
        plot_general_data[kernel_size] = {}
        plot_seperable_data[kernel_size] = {}
        plot_all_data[kernel_size] = {}
        for method in methods:
            plot_general_data[kernel_size][method] = []
            plot_seperable_data[kernel_size][method] = []
            plot_all_data[kernel_size][method] = []
            for resolution in resolutions:
                val = cuda_data[method][resolution][kernel_size]
                if 'conv_sk' in method:
                    val = serial_data['conv_sk'][resolution][kernel_size] / val
                    val = np.round(val * 1000) / 1000
                    plot_seperable_data[kernel_size][method].append(val)
                else:
                    val = serial_data['conv'][resolution][kernel_size] / val
                    val = np.round(val * 1000) / 1000
                    plot_general_data[kernel_size][method].append(val)
                plot_all_data[kernel_size][method].append(val)


    # for general
    list_length = len(resolutions)
    for kernel_size in kernel_sizes:
        
        x = np.arange(list_length)
        width = 0.2
        padding = 0.0

        max_y = 0.0
        fig, ax = plt.subplots(figsize=(2.5 * list_length, 5))
        for i, method in enumerate(general_methods):
            w = x - (len(general_methods) - 1) / 2 * (width + padding) + i * (width + padding)
            max_y = max(max_y, np.max(plot_general_data[kernel_size][method]))
            rect = ax.bar(w, 
                        plot_general_data[kernel_size][method], 
                        width, 
                        label=method, zorder=3)
            ax.bar_label(rect, padding=3)

        max_y = max_y * 1.4
        ax.set_xticks(x, resolutions)
        ax.set_ylim(0.0, max_y)
        ax.grid(axis = 'y', zorder=0)
        ax.legend(loc='best', fontsize=8)
        ax.set_title(f'CUDA General Kernel Results (Kernel Size = {kernel_size})', fontsize=16, pad=12.0)
        ax.set_xlabel('Resolution', fontsize=12)
        ax.set_ylabel('Speedup', fontsize=12)
        fig.tight_layout()
        plt.savefig(f'plot_result/cuda_result/cuda_general_resolutions_{kernel_size}.png')
        plt.close()

    # for seperable
    for kernel_size in kernel_sizes:
        
        x = np.arange(list_length)
        width = 0.2
        padding = 0.0

        max_y = 0.0
        fig, ax = plt.subplots(figsize=(2.5 * list_length, 5))
        for i, method in enumerate(seperable_methods):
            w = x - (len(seperable_methods) - 1) / 2 * (width + padding) + i * (width + padding)
            max_y = max(max_y, np.max(plot_seperable_data[kernel_size][method]))
            rect = ax.bar(w, 
                        plot_seperable_data[kernel_size][method], 
                        width, 
                        label=method, zorder=3)
            ax.bar_label(rect, padding=3)

        max_y = max_y * 1.4
        ax.set_xticks(x, resolutions)
        ax.set_ylim(0.0, max_y)
        ax.grid(axis = 'y', zorder=0)
        ax.legend(loc='best', fontsize=8)
        ax.set_title(f'CUDA Seperable Kernel Results (Kernel Size = {kernel_size})', fontsize=16, pad=12.0)
        ax.set_xlabel('Resolutions', fontsize=12)
        ax.set_ylabel('Speedup', fontsize=12)
        fig.tight_layout()
        plt.savefig(f'plot_result/cuda_result/cuda_seperable_resolutions_{kernel_size}.png')
        plt.close()

    # for all
    for kernel_size in kernel_sizes:
        
        x = np.arange(list_length)
        width = 0.14
        padding = 0.0

        max_y = 0.0
        fig, ax = plt.subplots(figsize=(5 * list_length, 5))
        for i, method in enumerate(methods):
            w = x - (len(methods) - 1) / 2 * (width + padding) + i * (width + padding)
            max_y = max(max_y, np.max(plot_all_data[kernel_size][method]))
            rect = ax.bar(w, 
                        plot_all_data[kernel_size][method], 
                        width, 
                        label=method, zorder=3)
            ax.bar_label(rect, padding=3)

        max_y = max_y * 1.4
        ax.set_xticks(x, resolutions)
        ax.set_ylim(0.0, max_y)
        ax.grid(axis = 'y', zorder=0)
        ax.legend(loc='best', fontsize=8)
        ax.set_title(f'CUDA Results (Kernel Size = {kernel_size})', fontsize=16, pad=12.0)
        ax.set_xlabel('Resolutions', fontsize=12)
        ax.set_ylabel('Speedup', fontsize=12)
        fig.tight_layout()
        plt.savefig(f'plot_result/cuda_result/cuda_all_resolutions_{kernel_size}.png')
        plt.close()

if __name__=="__main__":
    post_fix = ''
    if len(sys.argv) > 1:
        post_fix = sys.argv[1]
    main(post_fix)