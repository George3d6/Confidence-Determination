import json

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
# https://towardsdatascience.com/a-new-plot-theme-for-matplotlib-gadfly-2cffc745ff84

dataset_arr = ['a','b','c']
model_arr = ['M+C','MC','Mprim','M']
mode_arr = ['polynomial_w_coef','polynomial','linear']

model_perf_map = {}
model_perf_cnt_map = {}
for dataset in dataset_arr:
    model_perf_map[dataset] = {}
    model_perf_cnt_map[dataset] = {}

for degree in ['3','4','5','6']:
    data = json.load(open(f'results/Results_{degree}_batch2.json','r'))
    metric_arr = list(list(data.values())[0]['polynomial_w_coef']['M']['a'].keys())
    for mode in mode_arr:
        for dataset in dataset_arr:
            for metric in metric_arr:

                if metric not in model_perf_map[dataset]:
                    model_perf_map[dataset][metric] = {}
                    model_perf_cnt_map[dataset][metric] = {}

                model_val = {}
                for model in model_arr:
                    if data[degree][mode][model][dataset][metric] is None:
                        continue
                    val = data[degree][mode][model][dataset][metric]
                    model_val[model] = val

                for model, val in model_val.items():
                    if model not in model_perf_map[dataset][metric]:
                        model_perf_map[dataset][metric][model] = 0
                        model_perf_cnt_map[dataset][metric][model] = 0
                    if str(val) == 'nan':
                        val = np.min([x for x in model_val.values() if str(x) != 'nan'])

                    model_perf_map[dataset][metric][model] += val/1
                    model_perf_cnt_map[dataset][metric][model] += 1

for dataset in model_perf_map:
    print(f'\n\n Results on dataset: {dataset}\n\n')
    for k, v in model_perf_map[dataset].items():

        print(f'\n\nMeasuring metric: {k}')
        for model in v:
            val = v[model]/model_perf_cnt_map[dataset][k][model]
            print(f'{model} - {val}')
#exit()

data = json.load(open('results/Results_6_batch2.json','r'))

dataset_arr = ['a','b','c']
model_arr = ['M+C','MC','Mprim','M']
mode = 'relative'
#mode = 'absolute'

plt.rcParams["figure.figsize"] = (10,10)
for metric in data['6']['polynomial_w_coef']['M']['a'].keys():

    N = 3
    ind = np.arange(N)
    width = 0.15
    fig = plt.figure()
    ax = fig.add_subplot(111)


    model_arr_mod = model_arr
    minimum = pow(10,12)
    maximum = 0
    data_arr = []
    for i, model in enumerate(model_arr):
        X = []
        for dataset in dataset_arr:
            val = data['6']['polynomial_w_coef'][model][dataset][metric]
            if val is not None:
                if 'acc/conf' in metric:
                    val = abs(1 - val)
                if val < minimum:
                    minimum = val
                if val > maximum:
                    maximum = val
            X.append(val)
        if X[0] is None:
            model_arr_mod = ['M+C','MC','Mprim']
            continue
        data_arr.append(X)

    plot_arr = [ax.bar(ind+width*i*1.2, X, width) for i, X in enumerate(data_arr)]

    if mode == 'relative':
        mm = maximum - minimum
        ax.set(ylim=[minimum - mm*0.01,maximum + mm*0.05])

    ax.set_title(metric)
    ax.set_xticks(ind+width)
    ax.set_xticklabels(dataset_arr)

    samples = []
    for plot in plot_arr:
        samples.append(plot[0])

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend( samples, model_arr_mod, bbox_to_anchor=(1, 0.8), fontsize='xx-large')

    for plot in plot_arr:
        for ele in plot:
            h = ele.get_height()
            #ax.text(ele.get_x()+ele.get_width()/2., 1.5*h, '%d'%int(h*1.5), ha='center', va='bottom')
    #plt.show()
    fn = 'img/' + metric.replace(' ', '_').replace('/','_over_') + f'_{mode}_' + '.png'
    print(fn)
    plt.savefig(fn)
