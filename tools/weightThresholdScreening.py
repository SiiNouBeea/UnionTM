import numpy as np
# \UnionTM\tools\weightThresholdScreening.py
# WTS-Mechanism
def weightThresholdScreening(weights):
    print('\n\n>>>>>>>Select Model Based on Weights Distribution<<<<<<<<<<<<<<<<<<<')
    weight=[0,0,0,0]
    for i in range(len(weights)):
        weight[i] = weights[i]
    weight = weight / np.sum(weight)
    run_flag = [1, 1, 1, 1]
    dia_models = {0: 'SimpleTM', 1: 'PatchTST', 2: 'TimeMixer', 3: 'PathFormer'}
    single_model = False
    single_model_name = ''
    # 判断可否跳过某模型
    for i in range(0, 4):
        if weight[i] < 0.11:
            run_flag[i] = 0
            weight[i] = 0
            weight = weight / np.sum(weight)
            print(f"{dia_models[i]} OUT >>> Weights updated to {weight}")
    # 判断是否启用PBSM模式（单模型训练模式）
    for i in range(0, 4):
        if weight[i] == 1:
            single_model = True
            single_model_name = dia_models[i]
            print(f"Single model {single_model_name} selected")
            break
    return run_flag, single_model, single_model_name, weight
