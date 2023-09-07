import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def line_chart(ax, data_path, colors):
    data = pd.read_csv(f"rq2/rq2_{data_path}.csv")

    # Create a mapping from label to its original color
    original_order = ['Random', 'DeepState', 'RNNTest-HSCov(CAM)', 'DeepStellar-BSCov(CTM)', 'DeepStellar-BTCov(CTM)',
                      'testRNN-SC(CTM)', 'testRNN-SC(CAM)', 'NC(CTM)', 'NC(CAM)', "DeepVec"]
    color_mapping = {label: color for label, color in zip(original_order, colors)}

    # New order of data and labels
    datas = [data['my'] * 100, data['state'] * 100, data['random'] * 100,
             data['RNNTestcov'] * 100, data['Stellarbscov'] * 100,
             data['Stellarbtcov'] * 100, data['testRNNsc'] * 100,
             data['testRNNsc_cam'] * 100, data['nc_ctm'] * 100, data['nc_cam'] * 100]

    labels = ["DeepVec", 'DeepState', 'Random', 'RNNTest-HSCov(CAM)',
              'DeepStellar-BSCov(CTM)', 'DeepStellar-BTCov(CTM)',
              'testRNN-SC(CTM)', 'testRNN-SC(CAM)', 'NC(CTM)', 'NC(CAM)']

    x = np.arange(1, 41)

    for i, label in enumerate(labels):
        ax.plot(x, datas[i][1:41], label=label, color=color_mapping[label], linewidth=3 if i == 0 or i == 1 or i == 2 else 1.5)


    formatted_title = convert_title(data_path)
    ax.set_title(formatted_title, fontsize=28)
    ax.legend(loc='upper left', fontsize=15)  # Adjust the location to be upper left and remove bbox_to_anchor
    ax.set_xlabel('Selection Rate', fontsize=22)
    ax.set_ylabel('Inclusiveness', fontsize=22)
    ax.tick_params(axis='x', labelsize=22)  # Set x-axis tick label font size
    ax.tick_params(axis='y', labelsize=22)  # Set y-axis tick label font size


def convert_title(file_name):
    parts = file_name.split('_')
    model = parts[0].capitalize()
    structure = parts[1].upper() if parts[1] != "blstm" else "BiLSTM"
    return f"{model}-{structure}"


def main_plot():
    file_names = ['mnist_lstm', 'mnist_blstm', 'fashion_lstm', 'fashion_gru', 'snips_blstm', 'snips_gru',
                  'agnews_lstm', 'agnews_blstm']

    colors = ['#E76254', '#EF8A47', '#F7AA58', '#FFD06F', '#FFE6B7', '#AADCE0', '#72BCD5', '#528FAD', '#376795',
              '#000000']
    colors.reverse()

    fig, axes = plt.subplots(2, 4, figsize=(30,12))
    plt.rcParams['pdf.use14corefonts'] = True

    for ax, file_name in zip(axes.ravel(), file_names):
        line_chart(ax, file_name, colors)
        ax.set_xlabel('Selection Rate', fontsize=22)
        ax.set_ylabel('Inclusiveness', fontsize=22)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)  # Adjust spacing to prevent overlap
    plt.savefig(f"./rq2-result-fig/combined_figure.pdf", dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main_plot()

