import numpy as np
# \UnionTM\tools\optimize_weights.py
# AWO-EMP
def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def optimize_weights(model_preds, y_true, initial_learning_rate=0.001, epochs=20, loss_function='mse', max_patience=10, delta=1e-7, output_epochs=10):
    # model_preds: list of numpy arrays [pred1, pred2, pred3]
    # y_true: numpy array of true values

    # Initialize weights
    weights = np.random.rand(len(model_preds))
    #weights = [1,1,1]
    weights = weights / np.sum(weights)  # Normalize

    print(f"weight:{weights}")

    current_lr = initial_learning_rate
    min_lr = 1e-4  # 最小学习率
    patience = 0  # 早停计数器
    best_loss = float('inf')  # 最佳损失
    best_weights = weights.copy()  # 最佳权值

    last_loss = None  # 上一个损失值

    print(f'>>>>>>>>>START OPTIMIZE WEIGHTS<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    for epoch in range(epochs):
        # Compute ensemble prediction
        ensemble_pred = sum(w * pred for w, pred in zip(weights, model_preds))

        # Compute loss
        if loss_function == 'mse':
            loss = calculate_mse(y_true, ensemble_pred)
        elif loss_function == 'mae':
            loss = calculate_mae(y_true, ensemble_pred)
        else:
            raise ValueError("Unsupported loss function")

        # Check for improvement
        if loss < best_loss - delta:
            best_loss = loss
            best_weights = weights.copy()
            patience = 0  # 重置耐心计数器
        else:
            patience += 1

        # Early stopping
        if patience >= max_patience:
            print(f"Epoch {epoch + 1}, Loss: {loss:.8f}, Weight: {weights}, Learning Rate: {current_lr}, Patience: {patience}, Best Loss: {best_loss:.8f}\nEarly stopping at epoch {epoch + 1} due to no improvement for {max_patience} epochs")
            break

        # Adjust learning rate dynamically
        if last_loss is not None and loss >= last_loss * 0.9999:  # 如果损失没有显著下降
            current_lr *= 0.8  # 学习率降低
            current_lr = max(current_lr, min_lr)  # 限制学习率不低于最小值
            if (epoch + 1) % output_epochs == 0:
                print(f"Learning rate adjusted to: {current_lr}")

        # Update last_loss for next epoch
        last_loss = loss

        # Compute gradients (using approximate gradient for simplicity)
        gradients = []
        for i in range(len(model_preds)):
            # Perturb weight i slightly
            weights_perturbed = weights.copy()
            weights_perturbed[i] += 1e-4
            # print(f"Perturbed weight {i}: {weights_perturbed}")
            weights_perturbed = weights_perturbed / np.sum(weights_perturbed)
            # print(f"Perturbed weight {i} after: {weights_perturbed}")

            ensemble_pred_perturbed = sum(w * pred for w, pred in zip(weights_perturbed, model_preds))
            if loss_function == 'mse':
                perturbed_loss = calculate_mse(y_true, ensemble_pred_perturbed)
            else:
                perturbed_loss = calculate_mae(y_true, ensemble_pred_perturbed)

            gradient = (perturbed_loss - loss) / 1e-4  # 计算梯度
            # print(f"Gradient for weight {i}: {gradient}\n")
            gradients.append(gradient)

        gradients = np.array(gradients)

        # Update weights
        new_weights = weights - current_lr * gradients

        # Apply constraints: non-negative and sum to 1
        new_weights = np.maximum(new_weights, 0)
        new_weights /= np.sum(new_weights)

        weights = new_weights

        # Print progress
        if (epoch+1) % output_epochs == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss:.8f}, Weight: {weights}, Learning Rate: {current_lr}, Patience: {patience}, Best Loss: {best_loss:.8f}")

    print(f'>>>>>>>>>>>END OF OPTIMIZE WEIGHTS<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    return best_weights if best_loss < float('inf') else weights

'''
import pandas as pd

f = pd.read_csv("../dataset/ETT-small/ETTh1.csv")



y_true = f['OT'].values.reshape(-1, 1, 1)  # 转换为3D数组（样本数，序列长度，特征数）
sequence_length = 95
n_features = 7


y_true = np.broadcast_to(y_true, (len(y_true), sequence_length, n_features))


noise_scale = 1  # 控制噪声大小

model1_preds = y_true + np.random.randn(*y_true.shape) * noise_scale
model2_preds = y_true + np.random.randn(*y_true.shape) * noise_scale * 0.8
model3_preds = y_true + np.random.randn(*y_true.shape) * noise_scale * 1.2
model4_preds = y_true + np.random.randn(*y_true.shape) * noise_scale * 1.5

# 优化权值
optimal_weights = optimize_weights(
    [model1_preds, model2_preds, model3_preds, model4_preds],
    y_true,
    epochs=300,
    loss_function='mse',
    output_epochs=1
)
print("Optimal Weights:", optimal_weights)
'''