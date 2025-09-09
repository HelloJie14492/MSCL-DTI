import os


def get_kfold_data(i, datasets, k=5):
    fold_size = len(datasets) // k

    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:]
        trainset = datasets[0:val_start]

    return trainset, validset


def split_and_save_specific_fold(input_file, output_dir, fold_i, k=5):
    """
    分割并保存特定折数的数据

    参数:
        input_file: 输入文件路径
        output_dir: 输出目录路径
        fold_i: 要处理的折数(0-based索引)
        k: 总折数(默认为5)
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取原始数据
    with open(input_file, 'r') as f:
        data = f.readlines()

    # 检查fold_i是否有效
    if fold_i < 0 or fold_i >= k:
        raise ValueError(f"fold_i 必须在0到{k - 1}之间")

    # 获取指定折的训练集和测试集
    train_data, test_data = get_kfold_data(fold_i, data, k)

    # 定义输出文件路径
    train_file = os.path.join(output_dir, 'train.txt')
    test_file = os.path.join(output_dir, 'test.txt')

    # 保存训练集和测试集
    with open(train_file, 'w') as f:
        f.writelines(train_data)

    with open(test_file, 'w') as f:
        f.writelines(test_data)

    print(f'Fold {fold_i + 1}:')
    print(f'  Train samples: {len(train_data)} (保存到: {train_file})')
    print(f'  Test samples: {len(test_data)} (保存到: {test_file})')


# 使用示例
input_file = '/stu-3031/MMDG_Data/datasets/datasets/Davis/original/Davis_DTI.txt'
output_dir = '/stu-3031/MMDG_Data/datasets/datasets/Davis/data_split'

# 选择要处理的折数(0-based索引，0表示第1折，4表示第5折)
selected_fold = 0  # 这里选择处理第1折

split_and_save_specific_fold(
    input_file=input_file,
    output_dir=output_dir,
    fold_i=selected_fold,
    k=5
)