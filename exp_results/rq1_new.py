import pandas as pd
import matplotlib.pyplot as plt


def convert_title(file_name):
    # 将给定的file_name转换为你希望的格式
    parts = file_name.split('_')
    model = parts[0].capitalize()  # 将首字母大写
    if parts[1] == 'lstm':
        structure = 'LSTM'
    elif parts[1] == 'blstm':
        structure = 'BiLSTM'
    elif parts[1] == 'gru':
        structure = 'GRU'
    else:
        structure = parts[1].upper()

    return f"{model}-{structure}"
def plot_single_box(ax, data, labels, r):
    boxes = [data[f'Stellarbtcov{r}'] * 100, data[f'Stellarbscov{r}'] * 100, data[f'nc_ctm{r}'] * 100,
             data[f'nc_cam{r}'] * 100, data[f'testRNNsc{r}'] * 100, data[f'testRNNsc_cam{r}'] * 100,
             data[f'RNNTestcov{r}'] * 100, data[f'random{r}'] * 100, data[f'state_w{r}'] * 100, data[f'dis_w{r}'] * 100]

    f = ax.boxplot(boxes, labels=labels, vert=False)
    ax.set_yticks(range(1, len(labels) + 1))
    ax.set_yticklabels(labels, fontsize=22)
    ax.grid(axis='y')
    ax.tick_params(axis='x', labelsize=22)  # Adjusted for space reasons

    for box in f['boxes']:
        box.set(linewidth=1.1)
    for whisker in f['whiskers']:
        whisker.set(linewidth=1.1)
    for cap in f['caps']:
        cap.set(linewidth=1.1)
    for median in f['medians']:
        median.set(linewidth=1.1, color='#E76254')

def main_plot():
    file_names = ['mnist_lstm', 'mnist_blstm', 'fashion_lstm', 'fashion_gru', 'snips_blstm', 'snips_gru',
                  'agnews_lstm', 'agnews_blstm']
    labels = ['BTCov(CTM)', 'BSCov(CTM)', 'NC(CTM)', 'NC(CAM)', 'SC(CTM)', 'SC(CAM)',
              'HSCov(CAM)', 'Random', 'DeepState', 'DeepVec']  # 图例
    ratio = 10

    fig, axes = plt.subplots(2, 4, sharey=True, figsize=(30, 10))
    plt.rcParams['pdf.use14corefonts'] = True

    for ax, file_name in zip(axes.ravel(), file_names):
        data = pd.read_csv(f'rq1/rq1_{file_name}.csv')
        plot_single_box(ax, data, labels, ratio)
        formatted_title = convert_title(file_name)
        ax.set_title(formatted_title, fontsize=28)

    plt.tight_layout()  # 调整整体空白
    plt.subplots_adjust(wspace=0.05)  # 调整子图间距
    plt.savefig('./rq1-result-fig/combined_figure.pdf', dpi=200)

if __name__ == '__main__':
    main_plot()
