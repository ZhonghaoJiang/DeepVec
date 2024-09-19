import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def line_chart(ax, data_path, colors, labels, show_ylabel):
    data = pd.read_csv(f"rq2/rq2_{data_path}.csv")
    original_order = ['Random', 'DeepState', 'RNNTest-HSCov(CAM)', 'DeepStellar-BSCov(CTM)', 'DeepStellar-BTCov(CTM)',
                      'testRNN-SC(CTM)', 'testRNN-SC(CAM)', 'NC(CTM)', 'NC(CAM)', "DeepVec","MMD-Critic","K-Medoids", "DeepGini", "ATS"]
    color_mapping = {label: color for label, color in zip(original_order, colors)}

    datas = [data['Vec'] * 100, data['state'] * 100,
             data["mmdcritic"] * 100, data["kmedoids"] * 100, data["deepgini"] * 100, data["ats"] * 100,
             data['random'] * 100,
             data['RNNTestcov'] * 100, data['Stellarbscov'] * 100,
             data['Stellarbtcov'] * 100, data['testRNNsc'] * 100,
             data['testRNNsc_cam'] * 100, data['nc_ctm'] * 100, data['nc_cam'] * 100]

    x = np.arange(1, 40)
    for i, label in enumerate(labels):
        ax.plot(x, datas[i][1:41], label=label, color=color_mapping[label], linewidth=3 if i == 0 or i == 1 or i == 2 else 1.5)

    formatted_title = convert_title(data_path)
    ax.set_title(formatted_title, fontsize=35)
    ax.set_xlabel('Selection Rate', fontsize=28)
    if show_ylabel:
        ax.set_ylabel('Inclusiveness', fontsize=28)
    ax.tick_params(axis='x', labelsize=28)
    ax.tick_params(axis='y', labelsize=28)
    ax.set_ylim(0, 100)

def convert_title(file_name):
    parts = file_name.split('_')
    model = parts[0].capitalize()
    structure = parts[1].upper() if parts[1] != "blstm" else "BiLSTM"
    return f"{model}-{structure}"

def main_plot():
    # file_names = ['mnist_lstm', 'mnist_gru' , 'mnist_blstm',
    #               'fashion_lstm', 'fashion_gru', 'fashion_blstm',
    #               'snips_lstm', 'snips_gru','snips_blstm',
    #               'agnews_lstm', 'agnews_gru','agnews_blstm',
    #               'svhn_lstm', 'svhn_gru', 'svhn_blstm']
    file_names = ['mnist_lstm', 'fashion_lstm', 'snips_lstm', 'agnews_lstm', 'svhn_lstm',
                  'mnist_gru', 'fashion_gru', 'snips_gru', 'agnews_gru', 'svhn_gru',
                  'mnist_blstm', 'fashion_blstm', 'snips_blstm', 'agnews_blstm', 'svhn_blstm',
                  ]
    colors = ['#000000', '#FFCD00', '#3016B0', '#2DD700', '#9F0013', '#A68500', '#190773', '#1D8C00', '#FA7080', '#F5001D',
              '#FFE373','#8170D8', '#85EB6A', '#F7AA58', '#FFD06F']

    labels = ["DeepVec", 'DeepState', "MMD-Critic", "K-Medoids", "DeepGini", "ATS", 'Random', 'RNNTest-HSCov(CAM)',
              'DeepStellar-BSCov(CTM)', 'DeepStellar-BTCov(CTM)', 'testRNN-SC(CTM)', 'testRNN-SC(CAM)', 'NC(CTM)', 'NC(CAM)']

    fig, axes = plt.subplots(3, 5, figsize=(45,20))
    plt.rcParams['pdf.use14corefonts'] = True

    for i, ax in enumerate(axes.ravel()):
        show_ylabel = (i % 5 == 0)  # Only for the first column of each row
        line_chart(ax, file_names[i], colors, labels, show_ylabel)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize=28, ncol=7)  # Add a global legend

    plt.tight_layout(rect=[0, 0, 0.92, 0.92])  # Adjust layout to make room for the legend
    plt.subplots_adjust(wspace=0.1)  # Adjust spacing to prevent overlap
    plt.savefig(f"./rq2-result-fig/Fig5.pdf", dpi=200, bbox_inches='tight')

if __name__ == '__main__':
    main_plot()
