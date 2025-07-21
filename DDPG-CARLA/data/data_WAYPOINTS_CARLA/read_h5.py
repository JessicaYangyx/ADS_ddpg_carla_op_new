import h5py

# 打开HDF5文件
with h5py.File('RANDOM_150_actor.h5', 'r') as file:
    # 列出所有的组
    keys = list(file.keys())
    print("Keys: %s" % keys)
    # dense1_name = 'dense_1'
    # dense1 = file[dense1_name][()]
    # print("dense1: ", dense1)

    # 遍历所有键并打印它们的形状和部分内容
    for key in keys:
        dataset = file[key]
        print(dataset.name)
        # print(dataset.shape)
        print(dataset.value)
        # print(dataset)
        # print(f"Shape of {key}: {dataset.shape}")
        # print(f"Data of {key}: {dataset[:10]}")  # 打印前10个数据



