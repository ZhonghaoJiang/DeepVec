import pandas as pd

# 文件名列表
file_names = ['mnist_lstm', 'mnist_gru', 'mnist_blstm',
              'fashion_lstm', 'fashion_gru', 'fashion_blstm',
              'snips_lstm', 'snips_gru', 'snips_blstm',
              'agnews_lstm', 'agnews_gru', 'agnews_blstm',
              'svhn_lstm', 'svhn_gru', 'svhn_blstm']

# 用于存储所有结果的DataFrame
all_results = pd.DataFrame()

# 遍历每个文件
for file_name in file_names:
    # 构造文件路径
    file_path = f'./rq1/rq1_{file_name}.csv'

    # 读取数据
    data = pd.read_csv(file_path)

    # 提取以10和20结尾的列
    columns_10 = [col for col in data.columns if col.endswith('10')]
    columns_20 = [col for col in data.columns if col.endswith('20')]

    # 计算均值
    mean_10 = data[columns_10].mean()
    mean_20 = data[columns_20].mean()

    # 创建一个DataFrame用于存储当前文件的结果
    result = pd.DataFrame({
        '10%': mean_10.values,
        '20%': mean_20.values
    }, index=[col[:-2] for col in columns_10])

    # 在文件名前加上前缀以标识
    result.index = [f"{file_name}_{index}" for index in result.index]

    # 将结果添加到汇总DataFrame中
    all_results = pd.concat([all_results, result])

# 将所有结果写入一个CSV文件
all_results.to_csv('./rq1/all_methods_mean_values.csv', sep=',', header=True)
