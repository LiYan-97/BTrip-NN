import warnings
# warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore")
# 忽略警告
import pandas as pd
import tensorflow as tf
import numpy as np
import time
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from tensorflow import keras
# import tensorflow_probability as tfp
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import math
import h5py
import csv
from scipy.interpolate import griddata
from spektral.layers import GCNConv
import matplotlib as mpl
import math
import matplotlib.pyplot as plt
import shap
import tensorflow.keras.backend as K
from matplotlib import font_manager
from scipy.stats import linregress
from matplotlib.colors import LogNorm

# 设置有效的 mathtext fontset
mpl.rcParams['mathtext.fontset'] = 'stix'  # 或者选择其他支持的字体集

# print('tf.__version__', tf.__version__)   #  tf.__version__ 2.15.1
# print(h5py.__version__)
# print(h5py.version.hdf5_version)

# 预热warm up：减少第一次迭代时的训练时间
# 早停法early stopping：当验证集的损失值在一定数量的 epoch 内没有显著改善时，训练会被停止
# 网格搜索grid search：找到最佳batch_size和epoch的数值组合
# 10次交叉验证：获取最佳的超参数

# 手动管理权重和计算,较为底层.相比于使用高级API，代码更加冗长且复杂，不易维护.
# 优点：提供了对模型内部操作的完全控制，适合需要高度定制化的场景；可以更灵活地定义复杂的网络结构或执行特定的数学运算。
# 需要对模型进行深度定制时，才考虑采用此种方式
# 已知出发交通小区


# class GoalNetworkMain(tf.keras.Model):
#     def __init__(self, num_features=9, hidden_units=10, num_outputs=5):
#         # input = 5 x, output = 5 goals_main
#         super(GoalNetworkMain, self).__init__()
#         self.num_outputs = num_outputs  # 添加 num_outputs 属性
#         # 初始化输入层的权重和偏置
#         self.input_weights = self.add_weight(shape=(num_features, hidden_units), initializer='random_normal', trainable=True, name="input_weights" )
#         self.input_bias = self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name="input_bias")
#         # input_weights即为贝塔，input_bias：asc常数项
#
#         # 初始化隐藏层的权重和偏置
#         self.hidden_weights = [self.add_weight(shape=(hidden_units, hidden_units), initializer='random_normal', trainable=True, name=f"hidden_weights_{i}") for i in range(3)]
#         self.hidden_biases = [self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name=f"hidden_bias_{i}") for i in range(3)]
#
#         # 初始化输出层的权重和偏置
#         self.output_weights = self.add_weight(shape=(hidden_units, num_outputs), initializer='random_normal', trainable=True, name="output_weights")
#         self.output_bias = self.add_weight(shape=(num_outputs,), initializer='zeros', trainable=True, name="output_bias")
#
#         # 效用函数参数：缩放因子 & 偏移量
#         self.utility_a = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(1.0), trainable=True, name="utility_a")
#         self.utility_b = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(2.0), trainable=True, name="utility_b")
#
#     def call(self, inputs):
#         # 自定义输入层计算
#         hidden_input = tf.nn.relu(tf.matmul(inputs, self.input_weights) + self.input_bias)
#         # 多个隐藏层的计算
#         hidden_output = hidden_input
#         for i in range(3):  # 假设有3个隐藏层
#             # 自定义隐藏层计算
#             hidden_output = tf.nn.relu(tf.matmul(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])
#         # 自定义输出层计算
#         logits = tf.matmul(hidden_output, self.output_weights) + self.output_bias
#         # 应用效用函数
#         utility = self.utility_a * logits + self.utility_b
#         # 应用 Sigmoid 函数
#         outputs = tf.sigmoid(utility)
#
#         return outputs



class GoalNetworkMain(tf.keras.Model):
    def __init__(self, num_features=9, hidden_units=10, num_outputs=5):
        # num_features原来=8，后加入了ZoneDepartMain，所以现在num_features=9
        super(GoalNetworkMain, self).__init__()

        # 初始化输入层的权重和偏置
        self.input_weights = self.add_weight(shape=(num_features, hidden_units), initializer='random_normal',
                                             trainable=True, name="input_weights")
        self.input_bias = self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name="input_bias")

        # 初始化隐藏层的权重和偏置
        self.hidden_weights = [
            self.add_weight(shape=(hidden_units, hidden_units), initializer='random_normal', trainable=True,
                            name=f"hidden_weights_{i}") for i in range(3)]
        self.hidden_biases = [
            self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name=f"hidden_bias_{i}") for i
            in range(3)]

        # 初始化输出层的权重和偏置
        self.output_weights = self.add_weight(shape=(hidden_units, num_outputs), initializer='random_normal',
                                              trainable=True, name="output_weights")
        self.output_bias = self.add_weight(shape=(num_outputs,), initializer='zeros', trainable=True,
                                           name="output_bias")
        # print('output_bias', self.output_bias.shape)  # (5,)

        # # 上下文感知模块，用于生成效用函数参数的调整量
        # self.context_net = tf.keras.Sequential([
        #     layers.Dense(32, activation='relu', input_shape=(num_features,)),
        #     layers.Dense(num_outputs * 2)  # 假设需要调整num_outputs个权重和偏置, num_outputs * 2的设计是为了生成两组调整量：一组用于调整输出层的权重，另一组用于调整输出层的偏置
        # ])

        # 上下文感知模块，用于生成效用函数参数的调整量
        self.context_net = tf.keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(num_features,)),
            layers.Dense(num_outputs * (hidden_units + 1))  # 假设需要调整num_outputs个权重和偏置
        ])

        # 用于存储动态效用参数
        self.dynamic_params_history = []

    def call(self, inputs):
        # 上下文感知模块生成的效用函数参数调整量
        # context_adjustments = self.context_net(inputs)
        # print("Shape of context_adjustments:", context_adjustments.shape)  # 调试信息
        # print("Shape of output_weights:", self.output_weights.shape)  # 调试信息
        #
        # adjusted_output_weights = self.output_weights + tf.reshape(context_adjustments[:, :self.output_weights.shape[1]], shape=self.output_weights.shape)
        # adjusted_output_bias = self.output_bias + context_adjustments[:, self.output_weights.shape[1]:]
        #
        #
        batch_size = tf.shape(inputs)[0]

        # 上下文感知模块生成的效用函数参数调整量
        context_adjustments = self.context_net(inputs)
        # print("Shape of context_adjustments:", context_adjustments.shape)  # 调试信息
        # print("Shape of context_adjustments:", context_adjustments.shape)  # 调试信息   (64, 55)
        # print("Shape of output_weights:", self.output_weights.shape)  # 调试信息
        # print("Shape of inputs goal:", inputs.shape)     # (64, 8)

        # print('self.output_weights_goal', self.output_weights.shape)  # (10, 5)

        # 分离权重和偏置的调整量
        weight_adjustments = context_adjustments[:, :self.output_weights.shape[0] * self.output_weights.shape[1]]
        bias_adjustments = context_adjustments[:, self.output_weights.shape[0] * self.output_weights.shape[1]:]
        # print("Weight adjustments shape:", weight_adjustments.shape)  # 调试信息
        # print("Bias adjustments shape:", bias_adjustments.shape)  # 调试信息
        # print('bias_adjustments', bias_adjustments.shape)   #  (64, 5)
        # print('weight_adjustments11', weight_adjustments.shape, self.output_weights.shape)  #  (64, 50)  (10, 5)
        # Reshape weight adjustments to match output_weights shape
        adjusted_output_weights = self.output_weights + tf.reshape(
            weight_adjustments,
            shape=(batch_size, self.output_weights.shape[0], self.output_weights.shape[1])
        )[-1]  # 取最后一个批次的调整值
        # print('output_bias', self.output_bias.shape)   # (5,)
        # print('adjusted_output_weights', adjusted_output_weights.shape)  # (10, 5)
        adjusted_output_bias = self.output_bias + bias_adjustments   # (5,) + (64, 5)
        # print('adjusted_output_bias', adjusted_output_bias.shape)   # (64, 5)

        # 确保输入是浮点数类型
        inputs = tf.cast(inputs, dtype=tf.float32)

        # 自定义输入层计算
        hidden_input = tf.nn.relu(tf.matmul(inputs, self.input_weights) + self.input_bias)
        # 多个隐藏层的计算
        hidden_output = hidden_input
        for i in range(3):  # 假设有3个隐藏层
            hidden_output = tf.nn.relu(tf.matmul(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])

        # print('hidden_output', hidden_output.shape, adjusted_output_bias.shape)   #  (64, 10) (64, 5)

        # 自定义输出层计算
        logits = tf.matmul(hidden_output, adjusted_output_weights) + adjusted_output_bias   # (64, 10) * (10, 5) + (64, 5)

        # 应用 Softmax 函数 ≈ 应用MNL logit模型
        outputs = tf.nn.softmax(logits)

        # 保存动态效用参数到类属性
        self.latest_adjusted_weights = adjusted_output_weights
        self.latest_adjusted_bias = adjusted_output_bias

        # 保存动态效用参数
        print("Calling save_dynamic_parameters...")  # 添加调试信息
        self.save_dynamic_parameters(adjusted_output_weights, adjusted_output_bias)

        return outputs

    def save_dynamic_parameters(self, adjusted_weights, adjusted_bias):
        """
        保存动态效用参数到历史记录中。
        """
        try:
            # 确保在Eager Execution模式下运行
            if not tf.executing_eagerly():
                tf.print("Warning: Not in eager execution mode. Dynamic parameters won't be saved.")
                return

            # 将张量转换为numpy数组
            weights_np = adjusted_weights.numpy() if tf.is_tensor(adjusted_weights) else adjusted_weights
            bias_np = adjusted_bias.numpy() if tf.is_tensor(adjusted_bias) else adjusted_bias
            print('weights_np', weights_np, bias_np)

            self.dynamic_params_history.append({
                "weights": weights_np,
                "bias": bias_np
            })


        except Exception as e:
            tf.print(f"Error saving dynamic parameters: {e}")

        print('self.dynamic_params_history', self.dynamic_params_history)


    def export_dynamic_parameters(self, file_name):
        """
        将动态效用参数导出到CSV文件。
        """

        if not hasattr(self, "latest_adjusted_weights") or not hasattr(self, "latest_adjusted_bias"):
            raise AttributeError("Dynamic parameters have not been initialized. Call the 'call' method first.")

        # 保存最新的动态参数
        self.save_dynamic_parameters(self.latest_adjusted_weights, self.latest_adjusted_bias)

        print('self.dynamic_params_history11', self.dynamic_params_history)
        if not self.dynamic_params_history:
            print("Warning: No dynamic parameters to export.")
            return

        try:
            # 准备数据
            data = []
            for i, record in enumerate(self.dynamic_params_history):
                weights_flat = record["weights"].flatten()
                bias_flat = record["bias"].flatten()

                row = {
                    "record_id": i,
                    "weights": ",".join(map(str, weights_flat)),
                    "bias": ",".join(map(str, bias_flat))
                }
                data.append(row)

            # 创建DataFrame并保存
            df = pd.DataFrame(data)

            # 确保目录存在
            os.makedirs(os.path.dirname(file_name), exist_ok=True)

            df.to_csv(file_name, index=False)
            print(f"Successfully exported dynamic parameters to {file_name}")

        except Exception as e:
            print(f"Error exporting dynamic parameters: {e}")

    def compute_input_impact(self):
        """
        计算输入特征对调整量的影响。

        返回:
            input_impact_weights: 输入特征对权重调整量的影响矩阵。
            input_impact_bias: 输入特征对偏置调整量的影响矩阵。
        """
        # 获取上下文模块的权重
        context_weights = [layer.get_weights() for layer in self.context_net.layers]

        # 第一层权重 (9, 32)
        w1 = context_weights[0][0]  # 输入特征到隐藏层的权重
        b1 = context_weights[0][1]  # 输入特征到隐藏层的偏置

        # 第二层权重 (32, num_outputs * (hidden_units + 1))
        w2 = context_weights[1][0]  # 隐藏层到调整量的权重
        b2 = context_weights[1][1]  # 隐藏层到调整量的偏置

        # 权重连乘
        combined_weights = np.dot(w1, w2)  # (9, num_outputs * (hidden_units + 1))

        # 分离权重和偏置的贡献
        num_outputs = self.output_weights.shape[1]
        hidden_units = self.output_weights.shape[0]

        # 权重调整量的贡献 (9, num_outputs * hidden_units)
        input_impact_weights = combined_weights[:, :num_outputs * hidden_units]

        # 偏置调整量的贡献 (9, num_outputs)
        input_impact_bias = combined_weights[:, num_outputs * hidden_units:]

        return input_impact_weights, input_impact_bias


    # def save_dynamic_parameters(self, adjusted_weights, adjusted_bias):
    #     """
    #     保存动态效用参数到历史记录中。
    #
    #     Args:
    #         adjusted_weights (tf.Tensor): 调整后的权重。
    #         adjusted_bias (tf.Tensor): 调整后的偏置。
    #     """
    #     print("Saving dynamic parameters...")  # 添加调试信息
    #     if adjusted_weights is None or adjusted_bias is None:
    #         print("Warning: adjusted_weights or adjusted_bias is None.")
    #         return
    #
    #     weights_np = adjusted_weights.numpy()
    #     bias_np = adjusted_bias.numpy()
    #
    #     print("Weights shape:", weights_np.shape)  # 调试信息
    #     print("Bias shape:", bias_np.shape)  # 调试信息
    #
    #     self.dynamic_params_history.append({
    #         "weights": weights_np,
    #         "bias": bias_np
    #     })
    #
    #     print(f"Dynamic parameters history length: {len(self.dynamic_params_history)}")  # 调试信息
    #
    # def export_dynamic_parameters(self, file_name="dynamic_parameters.csv"):
    #     """
    #     将动态效用参数导出到 CSV 文件。
    #
    #     Args:
    #         file_name (str): 输出的 CSV 文件名。
    #     """
    #     if not self.dynamic_params_history:
    #         print("Warning: No dynamic parameters to export.")
    #         return
    #
    #     print(f"Exporting {len(self.dynamic_params_history)} dynamic parameter records...")  # 调试信息
    #
    #     # 提取所有动态参数
    #     all_weights = []
    #     all_biases = []
    #     for record in self.dynamic_params_history:
    #         all_weights.append(record["weights"].flatten())
    #         all_biases.append(record["bias"].flatten())
    #
    #     # 转换为 DataFrame 并保存到 CSV 文件
    #     df = pd.DataFrame({
    #         "weights": list(map(lambda x: ",".join(map(str, x)), all_weights)),
    #         "bias": list(map(lambda x: ",".join(map(str, x)), all_biases))
    #     })
    #     df.to_csv(file_name, index=False)
    #     print(f"Dynamic parameters saved to {file_name}")


# # 示例使用
# model = GoalNetworkMain()
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.build(input_shape=(None, 8))  # 假设输入特征数量为8
# model.summary()




# class UtilityaZoneSubNetworkDepartMainFromGoal(tf.keras.Model):
#     # GoalNetworkMain 生成目标张量，输入到 UtilityaZoneSubNetworkDepartMainFromGoal 生成 utility_a_zone，输入数据和目标张量一起输入到 ZoneNetworkDepartMain，利用 utility_a_zone 调整输出
#     def __init__(self, num_features_goal=5, hidden_units=10, num_outputs=3046):
#         # input = 5 goals_main, output = 3046 av_zones
#         super(UtilityaZoneSubNetworkDepartMainFromGoal, self).__init__()
#         # 初始化输入层的权重和偏置
#         self.input_weights = self.add_weight(shape=(num_features_goal, hidden_units), initializer='random_normal', trainable=True, name="input_weights" )
#         self.input_bias = self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name="input_bias")
#
#         # 初始化隐藏层的权重和偏置
#         self.hidden_weights = [self.add_weight(shape=(hidden_units, hidden_units), initializer='random_normal', trainable=True, name=f"hidden_weights_{i}") for i in range(3)]
#         self.hidden_biases = [self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name=f"hidden_bias_{i}") for i in range(3)]
#
#         # 初始化输出层的权重和偏置
#         self.output_weights = self.add_weight(shape=(hidden_units, num_outputs), initializer='random_normal', trainable=True, name="output_weights")
#         self.output_bias = self.add_weight(shape=(num_outputs,), initializer='zeros', trainable=True, name="output_bias")
#
#         # 效用函数参数：缩放因子 & 偏移量
#         self.utility_a = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(1.0), trainable=True, name="utility_a")
#         self.utility_b = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(2.0), trainable=True, name="utility_b")
#
#     def call(self, inputs):
#         # 自定义输入层计算
#         hidden_input = tf.nn.relu(tf.matmul(inputs, self.input_weights) + self.input_bias)
#         # 多个隐藏层的计算
#         hidden_output = hidden_input
#         for i in range(3):  # 假设有3个隐藏层
#             # 自定义隐藏层计算
#             hidden_output = tf.nn.relu(tf.matmul(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])
#         # 自定义输出层计算
#         logits = tf.matmul(hidden_output, self.output_weights) + self.output_bias
#         utility = self.utility_a * logits + self.utility_b
#         # 应用 Sigmoid 函数
#         utility_a_zone = tf.sigmoid(utility)
#         # utility_a_zone = tf.squeeze(utility_a_zone, axis=-1)  # 压缩输出维度，使其成为标量
#
#         return utility_a_zone


# class UtilityaZoneSubNetworkArriveMainFromGoal(tf.keras.Model):
#     # GoalNetworkMain 生成目标张量，输入到 UtilityaZoneSubNetworkArriveMainFromGoal 生成 utility_a_zone_depart_main，输入数据和目标张量一起输入到 ZoneNetworkArriveMain，利用 utility_a_zone_depart_main 调整输出
#     def __init__(self, num_features_goal=5, hidden_units=10, num_outputs=3046):
#         # input = 5 goals_main, output = 3046 av_zones
#         super(UtilityaZoneSubNetworkArriveMainFromGoal, self).__init__()
#         # 初始化输入层的权重和偏置
#         self.input_weights = self.add_weight(shape=(num_features_goal, hidden_units), initializer='random_normal', trainable=True, name="input_weights" )
#         self.input_bias = self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name="input_bias")
#
#         # 初始化隐藏层的权重和偏置
#         self.hidden_weights = [self.add_weight(shape=(hidden_units, hidden_units), initializer='random_normal', trainable=True, name=f"hidden_weights_{i}") for i in range(3)]
#         self.hidden_biases = [self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name=f"hidden_bias_{i}") for i in range(3)]
#
#         # 初始化输出层的权重和偏置
#         self.output_weights = self.add_weight(shape=(hidden_units, num_outputs), initializer='random_normal', trainable=True, name="output_weights")
#         self.output_bias = self.add_weight(shape=(num_outputs,), initializer='zeros', trainable=True, name="output_bias")
#
#         # 效用函数参数：缩放因子 & 偏移量
#         self.utility_a = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(1.0), trainable=True, name="utility_a")
#         self.utility_b = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(2.0), trainable=True, name="utility_b")
#
#     def call(self, inputs):
#         # 自定义输入层计算
#         hidden_input = tf.nn.relu(tf.matmul(inputs, self.input_weights) + self.input_bias)
#         # 多个隐藏层的计算
#         hidden_output = hidden_input
#         for i in range(3):  # 假设有3个隐藏层
#             # 自定义隐藏层计算
#             hidden_output = tf.nn.relu(tf.matmul(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])
#         # 自定义输出层计算
#         logits = tf.matmul(hidden_output, self.output_weights) + self.output_bias
#         utility = self.utility_a * logits + self.utility_b
#         # 应用 Sigmoid 函数
#         utility_a_zone = tf.sigmoid(utility)
#         # utility_a_zone = tf.squeeze(utility_a_zone, axis=-1)  # 压缩输出维度，使其成为标量
#
#         return utility_a_zone


# class ZoneNetworkDepartMain(tf.keras.Model):
#     def __init__(self, num_features=9, hidden_units=10, num_outputs=3046, num_features_goal=5):
#         # input = 8 x + 3046 av, output = 3046 zones
#         super(ZoneNetworkDepartMain, self).__init__()
#         self.dense_layer = tf.keras.layers.Dense(num_outputs, activation='sigmoid')
#         self.num_outputs = num_outputs  # 添加 num_outputs 属性
#         # 初始化输入层的权重和偏置
#         self.input_weights = self.add_weight(shape=(num_features, hidden_units), initializer='random_normal', trainable=True, name="input_weights" )
#         self.input_bias = self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name="input_bias")
#
#         # 初始化隐藏层的权重和偏置
#         self.hidden_weights = [self.add_weight(shape=(hidden_units, hidden_units), initializer='random_normal', trainable=True, name=f"hidden_weights_{i}") for i in range(3)]
#         self.hidden_biases = [self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name=f"hidden_bias_{i}") for i in range(3)]
#
#         # 初始化输出层的权重和偏置
#         self.output_weights = self.add_weight(shape=(hidden_units, num_outputs), initializer='random_normal', trainable=True, name="output_weights")
#         self.output_bias = self.add_weight(shape=(num_outputs,), initializer='zeros', trainable=True, name="output_bias")
#
#         # 子网络生成 utility_a
#         self.utility_a_zone_depart_main_subnetwork_from_goal = UtilityaZoneSubNetworkDepartMainFromGoal(num_features_goal=num_features_goal, hidden_units=hidden_units, num_outputs=num_outputs)
#         self.utility_b = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(2.0), trainable=True, name="utility_b")
#
#     def call(self, inputs, goals_main):
#         # 子网络生成 utility_a
#         utility_a = self.utility_a_zone_depart_main_subnetwork_from_goal(goals_main)
#
#         # 主网络计算
#         # 自定义输入层计算
#         hidden_input = tf.nn.relu(tf.matmul(inputs, self.input_weights) + self.input_bias)
#         # 多个隐藏层的计算
#         hidden_output = hidden_input
#         for i in range(3):  # 假设有3个隐藏层
#             # 自定义隐藏层计算
#             hidden_output = tf.nn.relu(tf.matmul(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])
#         # 自定义输出层计算
#         logits = tf.matmul(hidden_output, self.output_weights) + self.output_bias
#         # 应用效用函数
#         utility = utility_a * logits + self.utility_b
#         # 应用 Sigmoid 函数
#         outputs = tf.sigmoid(utility)
#
#         return outputs


#
# class ZonePredictionNetwork(tf.keras.Model):
#     def __init__(self, num_features=9, hidden_units=10, num_zones=14, embedding_dim=64, num_goals=5):
#         super(ZonePredictionNetwork, self).__init__()
#
#         # 主网络部分
#         self.input_weights = self.add_weight(shape=(num_features + num_goals, hidden_units),
#                                              initializer='random_normal', trainable=True, name="input_weights")
#         self.input_bias = self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name="input_bias")
#
#         self.hidden_weights = [
#             self.add_weight(shape=(hidden_units, hidden_units), initializer='random_normal', trainable=True,
#                             name=f"hidden_weights_{i}") for i in range(3)]
#         self.hidden_biases = [
#             self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name=f"hidden_bias_{i}") for i
#             in range(3)]
#
#         self.output_weights = self.add_weight(shape=(hidden_units + embedding_dim, num_zones),
#                                               initializer='random_normal', trainable=True, name="output_weights")
#         self.output_bias = self.add_weight(shape=(num_zones,), initializer='zeros', trainable=True, name="output_bias")
#
#         # 上下文感知模块，用于动态调整输出层权重
#         self.context_net = tf.keras.Sequential([
#             layers.Dense(32, activation='relu', input_shape=(num_goals,)),
#             layers.Dense(num_zones * 2)  # 调整num_zones个权重和偏置
#         ])
#
#         # GCN Layer: 简化的GCN实现
#         self.gcn_layer = layers.Dense(embedding_dim, activation='relu')
#
#     def call(self, inputs, goals_main, node_embeddings, adjacency_matrix):
#         """
#         :param inputs: 出行者的个体特征
#         :param goals_main: 预测的出行目的
#         :param node_embeddings: Node2Vec生成的节点嵌入向量
#         :param adjacency_matrix: 交通小区之间的邻接矩阵
#         """
#         # 将输入与目标张量拼接
#         x = tf.concat([inputs, goals_main], axis=-1)
#
#         # 自定义输入层计算
#         hidden_input = tf.nn.relu(tf.matmul(x, self.input_weights) + self.input_bias)
#         # 多个隐藏层的计算
#         hidden_output = hidden_input
#         for i in range(3):  # 假设有3个隐藏层
#             hidden_output = tf.nn.relu(tf.matmul(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])
#
#         # GCN处理
#         x_gcn = tf.matmul(adjacency_matrix, node_embeddings)
#         x_gcn = self.gcn_layer(x_gcn)
#
#         # 拼接主网络输出和GCN输出
#         combined_output = tf.concat([hidden_output, x_gcn], axis=-1)
#
#         # 上下文感知模块生成的效用函数参数调整量
#         context_adjustments = self.context_net(goals_main)
#         adjusted_output_weights = self.output_weights + tf.reshape(
#             context_adjustments[:, :self.output_weights.shape[1]], shape=self.output_weights.shape)
#         adjusted_output_bias = self.output_bias + context_adjustments[:, self.output_weights.shape[1]:]
#
#         # 自定义输出层计算
#         logits = tf.matmul(combined_output, adjusted_output_weights) + adjusted_output_bias
#
#         # 应用 Softmax 函数
#         outputs = tf.nn.softmax(logits)
#
#         return outputs


# # 示例使用
# node_embeddings_example = tf.random.normal([3046, 64])  # 假设有3046个交通小区，每个小区有64维嵌入向量
# adjacency_matrix_example = tf.random.normal([3046, 3046])  # 邻接矩阵
# inputs_example = tf.random.normal([4, 8])  # 假设有4个样本，每个样本有8个特征
# goals_main_example = tf.random.normal([4, 5])  # 假设预测了5种出行目的的概率
#
# zone_model = ZonePredictionNetwork()
# outputs = zone_model(inputs_example, goals_main_example, node_embeddings_example, adjacency_matrix_example)
# print(outputs.numpy())




def compute_weighted_adjacency_matrix(distances, sigma2=1.0):
    """
    在调用AttentionalGCNLayer之前，先转换zone_adjacency_matrix

    根据交通小区间距离计算加权邻接矩阵。
    高斯权重通常用于图神经网络中，以增强邻近节点间的关系，同时削弱远离节点的影响。通过使用高斯函数对距离进行加权，可以使得距离较近的交通小区之间具有更高的连接强度，而距离较远的交通小区之间的连接强度会迅速下降。
    如果您主要关注的是不同交通小区间的实际距离对出发时间和到达时间的影响，并且希望通过模型直接学习这些距离信息，那么使用加权前的邻接矩阵可能是更合适的选择。这种方法允许模型根据实际距离来调整其预测。
    然而，如果您的目标是更好地捕捉交通小区间的局部相互作用，特别是当您认为距离较近的交通小区之间的关系比距离远的更重要时，采用高斯加权后的邻接矩阵可能会更有帮助。这有助于模型聚焦于关键区域内的交互，从而提高预测精度。

    :param distances: 二维数组，表示每对交通小区间的距离。
    :param sigma2: 高斯核的方差参数。
    :return: 加权邻接矩阵。

    函数接受一个距离矩阵distance_matrix和一个可选的参数sigma2（高斯核的方差）。该函数首先计算距离矩阵中每个元素的平方，然后应用高斯核公式。
    其中d是两点间的距离，从而生成一个新的邻接矩阵，其中值接近1表示强连接（即短距离），值接近0表示弱连接（即长距离）。

    sigma2的选择：这个参数控制了距离对边权重的影响程度。
    较小的sigma2值会使距离的影响更加显著，即即使是很小的距离差异也会导致边权重有较大变化；较大的sigma2值则会减弱这种影响，使边权重的变化更为平缓。

    将距离矩阵转换为基于高斯核的邻接矩阵。

    参数:
    - distance_matrix: 形状为(num_zones, num_zones)的距离矩阵。
    - sigma2: 高斯核的方差参数。可以根据实际情况调整，默认为1.0。

    返回:
    - adjacency_matrix: 基于高斯核计算出的邻接矩阵，形状与distance_matrix相同。
        """

    # squared_distances = tf.square(distances)
    # adjacency_matrix = tf.exp(-squared_distances / sigma2)

    squared_distances = np.square(distances)

    return np.exp(- squared_distances / sigma2)


# class AttentionalGCNLayer(layers.Layer):
#     def __init__(self, output_dim, activation=None):
#         super(AttentionalGCNLayer, self).__init__()
#         self.output_dim = output_dim
#         self.activation = activation
#
#     def build(self, input_shape):
#         # 定义用于变换节点特征的权重矩阵
#         self.kernel = self.add_weight(name='kernel',
#                                       shape=(input_shape[-1], self.output_dim),
#                                       initializer='glorot_uniform',
#                                       trainable=True)
#         # 定义用于计算注意力得分的权重向量
#         self.attention_weights = self.add_weight(name='attention_weights',
#                                                  shape=(self.output_dim, 1),
#                                                  initializer='glorot_uniform',
#                                                  trainable=True)
#
#     def call(self, inputs, adjacency_matrix):
#         # 应用线性变换
#         support = tf.matmul(inputs, self.kernel)
#
#         # 计算注意力得分
#         attention_scores = tf.nn.softmax(tf.matmul(support, self.attention_weights), axis=1)
#
#         # 应用注意力得分
#         weighted_support = support * attention_scores
#
#         # GCN传播规则
#         output = tf.matmul(adjacency_matrix, weighted_support)
#
#         if self.activation is not None:
#             output = self.activation(output)
#
#         return output



class AttentionalGCNLayer(layers.Layer):
    def __init__(self, output_dim, activation=None):
        super(AttentionalGCNLayer, self).__init__()
        self.output_dim = output_dim
        # self.activation = activation
        # 确保激活函数是一个可调用对象
        self.activation = activations.get(activation)

    def build(self, input_shape):
        # 定义用于变换节点特征的权重矩阵
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        # 定义用于计算注意力得分的权重向量
        self.attention_weights = self.add_weight(name='attention_weights',
                                                 shape=(self.output_dim * 2, 1),  # 注意力机制需要考虑两节点间的特征差异
                                                 initializer='glorot_uniform',
                                                 trainable=True)

    def call(self, inputs, adjacency_matrix, return_attention_scores=False):
        # 应用线性变换
        support = tf.matmul(inputs, self.kernel)  # (14, 64) -> (14, output_dim)

        # 归一化邻接矩阵: 添加自环并计算D^(-1/2)AD^(-1/2)
        adjacency_matrix = adjacency_matrix + tf.eye(tf.shape(adjacency_matrix)[0])
        degree_matrix = tf.reduce_sum(adjacency_matrix, axis=-1)
        degree_matrix_inv_sqrt = tf.linalg.diag(1.0 / tf.sqrt(degree_matrix))
        normalized_adjacency_matrix = tf.matmul(degree_matrix_inv_sqrt,
                                                tf.matmul(adjacency_matrix, degree_matrix_inv_sqrt))

        # 计算注意力得分
        attention_scores = []
        for i in range(support.shape[0]):
            node_i_support = tf.tile(tf.expand_dims(support[i], axis=0), [support.shape[0], 1])  # (1, output_dim)
            concatenated_features = tf.concat([node_i_support, support], axis=-1)  # (14, 2*output_dim)
            scores = tf.nn.leaky_relu(tf.matmul(concatenated_features, self.attention_weights))  # (14, 1)
            attention_scores.append(scores)
        attention_scores = tf.squeeze(tf.stack(attention_scores, axis=1), axis=-1)  # (14, 14)
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)  # 对每一行进行softmax

        # 应用注意力得分
        weighted_support = tf.matmul(attention_scores, support)  # (14, output_dim)

        # GCN传播规则
        output = tf.matmul(normalized_adjacency_matrix, weighted_support)  # (14, output_dim)

        if self.activation is not None:
            output = self.activation(output)

        # 如果需要返回注意力权重矩阵
        if return_attention_scores:
            return output, attention_scores
        else:
            return output

        # return output



# 添加Top-K池化逻辑
def top_k_pooling(node_embeddings, k=5):
    """
    根据节点的重要性评分选择前K个节点
    :param node_embeddings: GCN层的输出，形状为 (num_nodes, embedding_dim)
    :param k: 选择的节点数
    :return: 池化后的节点嵌入和索引
    """
    # 计算重要性评分，这里简单地使用L2范数作为例子
    scores = tf.norm(node_embeddings, axis=-1)
    top_k_indices = tf.argsort(scores, direction='DESCENDING')[:k]
    pooled_embeddings = tf.gather(node_embeddings, top_k_indices)
    return pooled_embeddings, top_k_indices




# # 示例使用
# node_embeddings = tf.random.normal([14, 64])  # 假设有14个交通小区，每个小区有64维嵌入向量
# zone_adjacency_matrix = tf.random.uniform([14, 14], minval=0, maxval=2, dtype=tf.int32)
# zone_adjacency_matrix = tf.cast(zone_adjacency_matrix, tf.float32)  # 转换为浮点数
#
# gcn_layer = AttentionalGCNLayer(output_dim=64, activation='relu')
# x_gcn = gcn_layer(node_embeddings, zone_adjacency_matrix)
#
# print("Shape of x_gcn:", x_gcn.shape)  # 应该输出 (14, 64)

# class ZoneNetworkArriveMain(tf.keras.Model):
#     def __init__(self, num_features=9, hidden_units=10, num_zones=14, embedding_dim=64, num_goals=5,
#                  num_districts=3046):
#         super(ZoneNetworkArriveMain, self).__init__()
#
#         # 出发交通小区编号的嵌入层
#         self.district_embedding = layers.Embedding(input_dim=num_districts, output_dim=embedding_dim,
#                                                    name="district_embedding")
#
#         # 主网络部分
#         self.input_weights = self.add_weight(shape=(num_features + embedding_dim + num_goals, hidden_units),
#                                              initializer='random_normal', trainable=True, name="input_weights")
#         self.input_bias = self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name="input_bias")
#
#         self.hidden_weights = [
#             self.add_weight(shape=(hidden_units, hidden_units), initializer='random_normal', trainable=True,
#                             name=f"hidden_weights_{i}") for i in range(3)]
#         self.hidden_biases = [
#             self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name=f"hidden_bias_{i}") for i
#             in range(3)]
#
#         self.output_weights = self.add_weight(shape=(hidden_units + embedding_dim, num_zones),
#                                               initializer='random_normal', trainable=True, name="output_weights")
#         self.output_bias = self.add_weight(shape=(num_zones,), initializer='zeros', trainable=True, name="output_bias")
#
#         # # 上下文感知模块，用于动态调整输出层权重
#         # self.context_net = tf.keras.Sequential([
#         #     layers.Dense(32, activation='relu', input_shape=(num_goals,)),
#         #     layers.Dense(num_zones * 2)  # 调整num_zones个权重和偏置
#         # ])
#
#         # 上下文感知模块，用于生成效用函数参数的调整量
#         self.context_net = tf.keras.Sequential([
#             layers.Dense(32, activation='relu', input_shape=(num_features,)),
#             layers.Dense(num_zones * (hidden_units + 1))  # 假设需要调整num_outputs个权重和偏置
#         ])
#
#         # 改进的GCN层
#         self.gcn_layer = AttentionalGCNLayer(embedding_dim, activation='relu')
#
#     def call(self, inputs, goals_main, node_embeddings, zone_adjacency_matrix, origin_district_ids):
#         """
#         :param inputs: 出行者的个体特征
#         :param goals_main: 预测的出行目的
#         :param node_embeddings: Node2Vec生成的节点嵌入向量
#         :param zone_adjacency_matrix: 加权后的交通小区之间的邻接矩阵
#         :param origin_district_ids: 出发交通小区的ID
#         """
#
#         batch_size = tf.shape(inputs)[0]
#
#         # # 获取出发交通小区的嵌入向量
#         # origin_embeddings = self.district_embedding(origin_district_ids)
#         # # 确保 origin_embeddings 的批次大小与 inputs 和 goals_main 相同
#         # origin_embeddings = tf.tile(tf.expand_dims(origin_embeddings, 0), [batch_size, 1, 1])
#         # origin_embeddings = tf.reshape(origin_embeddings, (batch_size, -1))
#
#         # 获取出发交通小区的嵌入向量
#         origin_embeddings = self.district_embedding(origin_district_ids)
#         origin_embeddings = tf.tile(tf.expand_dims(origin_embeddings, 0), [batch_size, 1, 1])
#         origin_embeddings = tf.reshape(origin_embeddings, (batch_size, -1))
#
#
#         # 将输入与目标张量及出发交通小区的嵌入向量拼接
#         x = tf.concat([inputs, goals_main, origin_embeddings], axis=-1)
#
#         # 自定义输入层计算
#         hidden_input = tf.nn.relu(tf.matmul(x, self.input_weights) + self.input_bias)
#         # 多个隐藏层的计算
#         hidden_output = hidden_input
#         for i in range(3):  # 假设有3个隐藏层
#             hidden_output = tf.nn.relu(tf.matmul(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])
#
#         # 使用改进的GCN层处理
#         x_gcn = self.gcn_layer(node_embeddings, zone_adjacency_matrix)
#
#         # 广播 GCN 输出以匹配批次大小
#         x_gcn = tf.tile(tf.expand_dims(x_gcn, 0), [batch_size, 1, 1])
#         x_gcn = tf.reshape(x_gcn, (batch_size, -1))
#
#         # 拼接主网络输出和GCN输出
#         combined_output = tf.concat([hidden_output, x_gcn], axis=-1)
#
#         # # 上下文感知模块生成的效用函数参数调整量
#         # context_adjustments = self.context_net(goals_main)
#         # adjusted_output_weights = self.output_weights + tf.reshape(
#         #     context_adjustments[:, :self.output_weights.shape[1]], shape=self.output_weights.shape)
#         # adjusted_output_bias = self.output_bias + context_adjustments[:, self.output_weights.shape[1]:]
#
#         # 上下文感知模块生成的效用函数参数调整量
#         context_adjustments = self.context_net(goals_main)
#         weight_adjustments = context_adjustments[:, :self.output_weights.shape[1] * self.output_weights.shape[0]]
#         bias_adjustments = context_adjustments[:, self.output_weights.shape[1] * self.output_weights.shape[0]:]
#
#         adjusted_output_weights = self.output_weights + tf.reshape(
#             weight_adjustments,
#             shape=(-1, self.output_weights.shape[0], self.output_weights.shape[1])
#         )[-1]  # 取最后一个批次的调整值
#
#         # 自定义输出层计算
#         logits = tf.matmul(combined_output, adjusted_output_weights) + adjusted_output_bias
#
#         # 应用 Softmax 函数
#         outputs = tf.nn.softmax(logits)
#
#         return outputs



def create_id_to_index_mapping(zone_ids_int):
    """创建一个从实际交通小区ID到zone_adjacency_matrix索引的映射"""
    id_to_index = {id: index for index, id in enumerate(zone_ids_int)}
    return id_to_index

# 将距离信息作为额外特征加入
# 假设origin_district_ids是一个包含出发交通小区ID的张量/数组
# zone_adjacency_matrix是形状为(14, 14)的距离矩阵
# def get_distance_features(origin_district_ids, zone_adjacency_matrix):
#     # 对于每个出发交通小区ID，提取其到所有其他交通小区的距离
#     batch_size = tf.shape(origin_district_ids)[0]
#     origin_district_ids = tf.cast(origin_district_ids, dtype=tf.int32)
#     # print('546765', origin_district_ids)
#     distance_features = tf.gather(zone_adjacency_matrix, origin_district_ids)
#
#     return distance_features


def get_distance_features(origin_district_ids, zone_adjacency_matrix, id_to_index):
    """
    根据实际交通小区ID和邻接矩阵，提取距离特征。

    参数:
    - origin_district_ids: 实际交通小区ID列表或张量
    - zone_adjacency_matrix: 邻接矩阵，形状为(14, 14)
    - id_to_index: 从实际交通小区ID到邻接矩阵索引的映射字典

    返回:
    - distance_features: 提取的距离特征
    """
    # 将origin_district_ids转换为索引
    # print('origin_district_ids', origin_district_ids)   #
    # indices = [id_to_index.get(id.numpy(), -1) for id in origin_district_ids]
    # # print('indices', indices)
    #
    # # 检查是否有无效的ID（即不在映射中的ID）
    # if -1 in indices:
    #     raise ValueError("Found invalid district IDs that do not exist in the mapping.")
    #
    # # 转换为Tensor
    # indices = tf.constant(indices, dtype=tf.int32)
    #
    # # 提取距离特征
    # distance_features = tf.gather(zone_adjacency_matrix, indices)
    # ==========================================
    # 确保 origin_district_ids 是 int32 类型
    origin_district_ids = tf.cast(origin_district_ids, dtype=tf.int32)
    # 将 id_to_index 转换为 TensorFlow 可用的查找表
    keys = tf.constant(list(id_to_index.keys()), dtype=tf.int32)
    values = tf.constant(list(id_to_index.values()), dtype=tf.int32)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=-1  # 默认值表示无效的 ID
    )

    # 使用 tf.map_fn 替代 Python 原生迭代
    indices = tf.map_fn(
        lambda id: table.lookup(id),  # 查找每个 ID 对应的索引
        origin_district_ids,
        fn_output_signature=tf.int32  # 输出类型
    )

    # 检查是否有无效的 ID（即不在映射中的 ID）
    has_invalid_id = tf.reduce_any(indices == -1)
    tf.debugging.assert_equal(
        has_invalid_id, False,
        message="Found invalid district IDs that do not exist in the mapping."
    )

    # 提取距离特征
    distance_features = tf.gather(zone_adjacency_matrix, indices)

    return distance_features


def extract_origin_district_ids(inputs):
    """
    从输入特征中提取出发交通小区编号（ZoneDepartMain）。

    参数:
    - inputs: 输入特征张量，形状为(batch_size, num_features)，
              其中num_features=9，且ZoneDepartMain位于索引8。

    返回:
    - origin_district_ids: 出发交通小区编号，形状为(batch_size,)。
    """
    # 提取出ZoneDepartMain这一列，即索引为8的那一列
    origin_district_ids = inputs[:, 8]
    # print('inputs',inputs)

    return origin_district_ids


# 输入：
# features=9（包含出发交通小区）
# 上一层的输出 num_goals=5
# 真实距离信息：额外特征加入 num_zone_arrive = 14，每个出发交通小区有14个到达交通小区与其对应
# 起点/出发 交通小区的嵌入向量 64
# 另外，包含距离信息的交通小区的节点嵌入向量=64，不直接作为输入，而是采用GCN（作为GCN的输入），获得其中的邻接拓扑关系
class ZoneNetworkArriveMain(tf.keras.Model):
    def __init__(self, num_features=9, hidden_units=10, num_zone_origin=14, num_zone_arrive=14, embedding_dim=64, num_goals=5, num_districts=3046, num_origin_zone=3046, k =5):
        # num_zone_arrive 是交通小区的总数，使用所有数据进行正式建模时，num_zone_arrive= 。当前这里使用了部分数据，只有14个交通小区，故num_zone_arrive=14,  # 5是指池化层的k值
        super(ZoneNetworkArriveMain, self).__init__()

        # 出发交通小区编号的嵌入层
        self.district_embedding = layers.Embedding(input_dim=num_districts, output_dim=embedding_dim, name="district_embedding")

        # 计算输入维度,将距离信息作为额外特征加入(该向量长度=邻接矩阵的长度，即=交通小区的数量)
        input_dim = num_features + num_goals + embedding_dim + num_zone_arrive    # 9+5+64+14=92

        # # 主网络部分
        # self.input_weights = self.add_weight(shape=(input_dim, hidden_units),
        #                                      initializer='random_normal', trainable=True, name="input_weights")
        # self.input_bias = self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name="input_bias")

        # 主网络部分——
        self.input_weights = self.add_weight(shape=(input_dim, hidden_units), initializer='random_normal', trainable=True, name="input_weights")
        self.input_bias = self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name="input_bias")

        self.hidden_weights = [
            self.add_weight(shape=(hidden_units, hidden_units), initializer='random_normal', trainable=True,
                            name=f"hidden_weights_{i}") for i in range(3)]
        self.hidden_biases = [
            self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name=f"hidden_bias_{i}") for i
            in range(3)]

        self.output_weights = self.add_weight(shape=(k * embedding_dim + hidden_units, num_zone_arrive),    # 5是指池化层的k值
                                              initializer='random_normal', trainable=True, name="output_weights")
        # print(self.output_weights.shape, hidden_units, embedding_dim, '11111')   # (74, 14) 10 64
        self.output_bias = self.add_weight(shape=(num_zone_arrive,), initializer='zeros', trainable=True, name="output_bias")

        # 上下文感知模块，用于动态调整输出层权重
        # self.context_net = tf.keras.Sequential([
        #     layers.Dense(32, activation='relu', input_shape=(num_goals,)),
        #     layers.Dense(num_zone_arrive * 2)  # 调整num_zone_arrive个权重和偏置
        # ])

        # 上下文感知模块，用于生成效用函数参数的调整量
        self.context_net = tf.keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(input_dim,)),
            layers.Dense(num_zone_arrive * (k * embedding_dim + hidden_units + 1))  # 假设需要调整num_outputs个权重和偏置
        ])   # 5是指池化层的k值，14*(5*64+10 + 1) = 4634

        # 改进的GCN层
        self.gcn_layer = AttentionalGCNLayer(embedding_dim, activation='relu')

    def call(self, inputs, goals_main, node_embeddings, zone_adjacency_matrix):
        batch_size = tf.shape(inputs)[0]

        # 提取出origin_district_ids
        origin_district_ids = extract_origin_district_ids(inputs)
        # print('origin_district_ids4856', origin_district_ids.shape, origin_district_ids)

        # 创建映射字典
        id_to_index = create_id_to_index_mapping(zone_ids_int)
        # print('id_to_index0000000', id_to_index)

        # 获取距离特征
        distance_features = get_distance_features(origin_district_ids, zone_adjacency_matrix, id_to_index)
        # print('distance_features5645', distance_features.shape)
        # print('zone_adjacency_matrix5634', zone_adjacency_matrix.shape)

        # # 获取出发交通小区的嵌入向量
        # origin_embeddings = self.district_embedding(origin_district_ids)
        # origin_embeddings = tf.tile(tf.expand_dims(origin_embeddings, 0), [batch_size, 1, 1])
        # origin_embeddings = tf.reshape(origin_embeddings, (batch_size, -1))

        # print('origin_district_ids', origin_district_ids)   # tf.Tensor([  0  53 417 426 433 475 485 515 526 556 562 570 893 948], shape=(14,), dtype=int32) (14,)

        # 获取出发交通小区的嵌入向量
        origin_embeddings = self.district_embedding(origin_district_ids)
        # print("Shape of origin_district_ids:", origin_district_ids.shape)  # (14,)
        # print("Shape of origin_embeddings:", origin_embeddings.shape)  # 调试信息 (14, 64)
        # print("Shape of inputs:", inputs.shape)  # 调试信息  (64, 8)

        origin_embeddings = tf.tile(tf.expand_dims(origin_embeddings, 0), [batch_size, 1, 1])
        # print("Shape of origin_embeddings2:", origin_embeddings.shape)  #  (64, 14, 64)
        # origin_embeddings = tf.reshape(origin_embeddings, (batch_size, -1))
        # print("Shape of origin_embeddings3:", origin_embeddings.shape)  # (64, 896)

        # origin_embeddings shape: (64, 14, 64)
        origin_embeddings = tf.reduce_mean(origin_embeddings, axis=1)  # 对第二个维度求平均值
        # print("Shape of origin_embeddings after pooling:", origin_embeddings.shape)  # 应该输出 (64, 64)

        inputs = tf.cast(inputs, dtype=tf.float32)  # 确保 zones_arrive_main 是 float32 类型
        goals_main = tf.cast(goals_main, dtype=tf.float32)  # 确保 distance_features 是 float32 类型
        origin_embeddings = tf.cast(origin_embeddings, dtype=tf.float32)  # 确保 origin_embeddings 是 float32 类型
        distance_features = tf.cast(distance_features, dtype=tf.float32)  # 确保 arrive_embeddings_weighted 是 float32 类型
        # print('distance_features698', distance_features.shape)   # (64, 14)

        # 将输入与目标张量及出发交通小区的嵌入向量拼接
        # 将距离信息作为额外特征加入
        x = tf.concat([inputs, goals_main, origin_embeddings, distance_features], axis=-1)
        # print("Shape of x:", x.shape, goals_main.shape, origin_embeddings.shape, distance_features.shape)  # 只加第二个数字 (64, 92) = (64, 9) + (64, 5) + (64, 64) + (64, 14)
        # 打印 x 的形状
        # print("Shape of x:", x.shape, inputs.shape, goals_main.shape, origin_embeddings.shape, distance_features.shape)

        # 自定义输入层计算
        # hidden_input = tf.nn.relu(tf.matmul(x, self.input_weights) + self.input_bias)
        # 确保self.input_weights的形状与x的最后一维相匹配
        hidden_input = tf.nn.relu(tf.matmul(x, self.input_weights) + self.input_bias)

        # 多个隐藏层的计算
        hidden_output = hidden_input
        for i in range(3):  # 假设有3个隐藏层
            hidden_output = tf.nn.relu(tf.matmul(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])

        # 使用改进的GCN层处理
        # print("Shape of node_embeddings:", node_embeddings.shape)   # (14, 64)
        # print("Shape of zone_adjacency_matrix:", zone_adjacency_matrix.shape)  # (14, 14)
        x_gcn, attention_scores = self.gcn_layer(node_embeddings, zone_adjacency_matrix, return_attention_scores=True)
        # print('x_gcn', x_gcn.shape)   # (14, 64)

        # 将注意力权重转换为NumPy数组
        attention_weights_matrix = attention_scores.numpy()
        # 保存为 .csv 文件
        np.savetxt(fr'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\attention_weights_matrix\attention_weights_matrix.csv', attention_weights_matrix, delimiter=',')
        # 打印注意力权重矩阵
        print("Attention Weights Matrix:")
        print(attention_weights_matrix)

        # # 广播 GCN 输出以匹配批次大小
        # x_gcn = tf.tile(tf.expand_dims(x_gcn, 0), [batch_size, 1, 1])
        # # print('x_gcn2', x_gcn.shape)  # (64, 14, 64)
        # x_gcn = tf.reshape(x_gcn, (batch_size, -1))
        # # print('x_gcn3', x_gcn.shape)   # (64, 896)
        # # print('batch_size', batch_size, x_gcn.shape, node_embeddings.shape, zone_adjacency_matrix.shape)   # () (64, 896) (14, 64) (14, 14)

        # 应用Top-K池化
        k = 5  # 选择前5个最重要的节点
        x_gcn_pooled, _ = top_k_pooling(x_gcn, k=k)  # 输出形状变为 (k, embedding_dim)
        # print('x_gcn_pooled0', x_gcn_pooled.shape)  # (5, 64)
        # 广播GCN输出以匹配批次大小
        x_gcn_pooled = tf.tile(tf.expand_dims(x_gcn_pooled, 0),
                               [batch_size, 1, 1])  # 形状变为 (batch_size, k, embedding_dim)
        x_gcn_pooled = tf.reshape(x_gcn_pooled, (batch_size, -1))  # 最终形状变为 (batch_size, k * embedding_dim)
        # print('x_gcn_pooled', x_gcn_pooled.shape)  # (64, 320)

        # 拼接主网络输出和GCN输出
        combined_output = tf.concat([hidden_output, x_gcn_pooled], axis=-1)
        # print('combined_output', combined_output.shape)  # (64, 906)

        # # 上下文感知模块生成的效用函数参数调整量
        # context_adjustments = self.context_net(goals_main)
        # weight_adjustments = context_adjustments[:, :self.output_weights.shape[1] * self.output_weights.shape[0]]
        # bias_adjustments = context_adjustments[:, self.output_weights.shape[1] * self.output_weights.shape[0]:]
        #
        # print(self.output_weights.shape[0],self.output_weights.shape[1],'1111')   # 74 3046
        # adjusted_output_weights = self.output_weights + tf.reshape(
        #     weight_adjustments,
        #     shape=(-1, self.output_weights.shape[0], self.output_weights.shape[1])
        # )[-1]  # 取最后一个批次的调整值
        #
        # adjusted_output_bias = self.output_bias + bias_adjustments

        # 上下文感知模块生成的效用函数参数调整量
        context_adjustments = self.context_net(x)
        # print("Shape of context_adjustments:", context_adjustments.shape)  # 调试信息  (64, 12698)  (64, 4634)
        # print("Shape of output_weights:", self.output_weights.shape)  # 调试信息
        # print("Shape of inputs goal:", inputs.shape)     # (64, 8)

        # print('self.output_weights', self.output_weights.shape)  # (906, 14)

        # 分离权重和偏置的调整量
        weight_adjustments = context_adjustments[:, :self.output_weights.shape[0] * self.output_weights.shape[1]]
        # print('weight_adjustments', weight_adjustments.shape, self.output_weights.shape[0], self.output_weights.shape[1])  # (64, 12684) 906 14  ， (64, 4634) 4620/14=330 14
        bias_adjustments = context_adjustments[:, self.output_weights.shape[0] * self.output_weights.shape[1]:]
        # print('bias_adjustments', bias_adjustments.shape)    # (64, 14)

        # Reshape weight adjustments to match output_weights shape
        # adjusted_output_weights = self.output_weights + tf.reshape(
        #     weight_adjustments,
        #     shape=(-1, self.output_weights.shape[0], self.output_weights.shape[1])
        # )[-1]  # 取最后一个批次的调整值
        #
        # adjusted_output_bias = self.output_bias + bias_adjustments

        # print('output_weights.shape', self.output_weights.shape)   # (74, 3046)

        # Reshape weight adjustments to match output_weights shape
        # adjusted_output_weights = self.output_weights + tf.reshape(
        #     weight_adjustments,
        #     shape=(combined_output,) + self.output_weights.shape
        # )

        adjusted_output_weights = self.output_weights + tf.reshape(
            weight_adjustments,
            shape=(-1, self.output_weights.shape[0], self.output_weights.shape[1])
        )[-1]

        # print('adjusted_output_weights', adjusted_output_weights.shape)    # (906, 14)

        # adjusted_output_bias = self.output_bias + tf.reshape(
        #     bias_adjustments,
        #     shape=(-1,) + self.output_bias.shape
        # )

        adjusted_output_bias = self.output_bias + bias_adjustments  #
        # print('adjusted_output_bias', adjusted_output_bias.shape)   # (64, 14)

        # 自定义输出层计算
        logits = tf.matmul(combined_output, adjusted_output_weights) + adjusted_output_bias  #  (64, 906) * (906, 14) + (64, 14)

        # 应用 Softmax 函数，输出是概率值
        outputs = tf.nn.softmax(logits)

        return outputs

# # 示例使用
# # 假设您已经有了一个基于距离的加权邻接矩阵
# distances = ...  # 这里应该是您的数据集中的实际距离
# sigma2 = 1.0  # 可以根据需要调整
# zone_adjacency_matrix = compute_weighted_adjacency_matrix(distances, sigma2)

# node_embeddings_example = tf.random.normal([3046, 64])  # 假设有3046个交通小区，每个小区有64维嵌入向量
# adjacency_matrix_example = tf.convert_to_tensor(zone_adjacency_matrix, dtype=tf.float32)  # 转换为TensorFlow张量
# inputs_example = tf.random.normal([4, 8])  # 假设有4个样本，每个样本有8个特征
# goals_main_example = tf.random.normal([4, 5])  # 假设预测了5种出行目的的概率
# origin_district_ids_example = tf.constant([100, 200, 300, 400])  # 示例出发交通小区编号
#
# zone_model = ZoneNetworkArriveMain()
# outputs = zone_model(inputs_example, goals_main_example, node_embeddings_example, adjacency_matrix_example, origin_district_ids_example)
# print(outputs.numpy())



# class UtilityaZoneSubNetworkArriveMainFromZoneDepartMain(tf.keras.Model):
#     # GoalNetworkMain 生成目标张量，输入到 UtilityaZoneSubNetworkArriveMainFromZoneDepartMain 生成 utility_a_zone_depart_main，输入数据和目标张量一起输入到 ZoneNetworkArriveMain，利用 utility_a_zone_depart_main 调整输出
#     def __init__(self, num_features_zonedepartmain=3046, hidden_units=10, num_outputs=3046):
#         # input = 3046 zonedepartmain, output = 3046 av_zones
#         super(UtilityaZoneSubNetworkArriveMainFromZoneDepartMain, self).__init__()
#         # 初始化输入层的权重和偏置
#         self.input_weights = self.add_weight(shape=(num_features_zonedepartmain, hidden_units), initializer='random_normal', trainable=True, name="input_weights" )
#         self.input_bias = self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name="input_bias")
#
#         # 初始化隐藏层的权重和偏置
#         self.hidden_weights = [self.add_weight(shape=(hidden_units, hidden_units), initializer='random_normal', trainable=True, name=f"hidden_weights_{i}") for i in range(3)]
#         self.hidden_biases = [self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name=f"hidden_bias_{i}") for i in range(3)]
#
#         # 初始化输出层的权重和偏置
#         self.output_weights = self.add_weight(shape=(hidden_units, num_outputs), initializer='random_normal', trainable=True, name="output_weights")
#         self.output_bias = self.add_weight(shape=(num_outputs,), initializer='zeros', trainable=True, name="output_bias")
#
#         # 效用函数参数：缩放因子 & 偏移量
#         self.utility_a = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(1.0), trainable=True, name="utility_a")
#         self.utility_b = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(2.0), trainable=True, name="utility_b")
#
#     def call(self, inputs):
#         # 自定义输入层计算
#         hidden_input = tf.nn.relu(tf.matmul(inputs, self.input_weights) + self.input_bias)
#         # 多个隐藏层的计算
#         hidden_output = hidden_input
#         for i in range(3):  # 假设有3个隐藏层
#             # 自定义隐藏层计算
#             hidden_output = tf.nn.relu(tf.matmul(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])
#         # 自定义输出层计算
#         logits = tf.matmul(hidden_output, self.output_weights) + self.output_bias
#         utility = self.utility_a * logits + self.utility_b
#         # 应用 Sigmoid 函数
#         utility_a_zone = tf.sigmoid(utility)
#         # utility_a_zone = tf.squeeze(utility_a_zone, axis=-1)  # 压缩输出维度，使其成为标量
#
#         return utility_a_zone


# ZoneNetworkArriveMain中的utility_a，不仅受UtilityaZoneSubNetworkArriveMainFromGoal的影响，还受ZoneNetworkDepartMain的影响，这两者共同生成了它
# class ZoneNetworkArriveMain(tf.keras.Model):
#     def __init__(self, num_features=9, hidden_units=10, num_outputs=3046, num_features_goal=5, num_features_zonedepartmain=3046):
#         # input = 8 x + 3046 av, output = 3046 zones
#         super(ZoneNetworkArriveMain, self).__init__()
#         self.num_outputs = num_outputs  # 添加 num_outputs 属性
#         # 初始化输入层的权重和偏置
#         self.input_weights = self.add_weight(shape=(num_features, hidden_units), initializer='random_normal', trainable=True, name="input_weights" )
#         self.input_bias = self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name="input_bias")
#
#         # 初始化隐藏层的权重和偏置
#         self.hidden_weights = [self.add_weight(shape=(hidden_units, hidden_units), initializer='random_normal', trainable=True, name=f"hidden_weights_{i}") for i in range(3)]
#         self.hidden_biases = [self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name=f"hidden_bias_{i}") for i in range(3)]
#
#         # 初始化输出层的权重和偏置
#         self.output_weights = self.add_weight(shape=(hidden_units, num_outputs), initializer='random_normal', trainable=True, name="output_weights")
#         self.output_bias = self.add_weight(shape=(num_outputs,), initializer='zeros', trainable=True, name="output_bias")
#
#         # 子网络生成 utility_a
#         self.utility_a_zone_arrive_main_subnetwork_from_goal = UtilityaZoneSubNetworkArriveMainFromGoal(num_features_goal=num_features_goal, hidden_units=hidden_units, num_outputs=num_outputs)
#         self.utility_a_zone_arrive_main_subnetwork_from_zonedepartmain = UtilityaZoneSubNetworkArriveMainFromZoneDepartMain(num_features_zonedepartmain=num_features_zonedepartmain, hidden_units=hidden_units, num_outputs=num_outputs)
#         self.utility_b = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(2.0), trainable=True, name="utility_b")
#
#         # 添加权重 alpha，用于控制 utility_a 的加权平均
#         self.alpha = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(0.5), trainable=True, name="alpha")
#
#     def call(self, inputs, goals_main, zones_depart_main):
#         # 子网络生成 utility_a
#         utility_a_zone_arrive_main_from_goal = self.utility_a_zone_arrive_main_subnetwork_from_goal(goals_main)
#         utility_a_zone_arrive_main_from_zonedepartmain = self.utility_a_zone_arrive_main_subnetwork_from_zonedepartmain(zones_depart_main)
#         # 结合两个 utility_a (加权平均)
#         utility_a = self.alpha * utility_a_zone_arrive_main_from_goal + (1 - self.alpha) * utility_a_zone_arrive_main_from_zonedepartmain
#
#         # 主网络计算
#         # 自定义输入层计算
#         hidden_input = tf.nn.relu(tf.matmul(inputs, self.input_weights) + self.input_bias)
#         # 多个隐藏层的计算
#         hidden_output = hidden_input
#         for i in range(3):  # 假设有3个隐藏层
#             # 自定义隐藏层计算
#             hidden_output = tf.nn.relu(tf.matmul(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])
#         # 自定义输出层计算
#         logits = tf.matmul(hidden_output, self.output_weights) + self.output_bias
#         # 应用效用函数
#         utility = utility_a * logits + self.utility_b
#         # 应用 Sigmoid 函数
#         outputs = tf.sigmoid(utility)
#
#         return outputs


# class UtilityaTimeSubNetworkDepartMainFromZoneDepartMain(tf.keras.Model):
#     # GoalNetworkMain 生成目标张量，输入到 UtilityaZoneSubNetworkDepartMainFromGoal 生成 utility_a_zone，输入数据和目标张量一起输入到 ZoneNetworkDepartMain，利用 utility_a_zone 调整输出
#     def __init__(self, num_features_zone=3046, hidden_units=10, num_outputs=24):
#         # input = 3046 zones, output = 24 av_times
#         super(UtilityaTimeSubNetworkDepartMainFromZoneDepartMain, self).__init__()
#         # 初始化输入层的权重和偏置
#         self.input_weights = self.add_weight(shape=(num_features_zone, hidden_units), initializer='random_normal', trainable=True, name="input_weights" )
#         self.input_bias = self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name="input_bias")
#
#         # 初始化隐藏层的权重和偏置
#         self.hidden_weights = [self.add_weight(shape=(hidden_units, hidden_units), initializer='random_normal', trainable=True, name=f"hidden_weights_{i}") for i in range(3)]
#         self.hidden_biases = [self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name=f"hidden_bias_{i}") for i in range(3)]
#
#         # 初始化输出层的权重和偏置
#         self.output_weights = self.add_weight(shape=(hidden_units, num_outputs), initializer='random_normal', trainable=True, name="output_weights")
#         self.output_bias = self.add_weight(shape=(num_outputs,), initializer='zeros', trainable=True, name="output_bias")
#
#         # 效用函数参数：缩放因子 & 偏移量
#         self.utility_a = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(1.0), trainable=True, name="utility_a")
#         self.utility_b = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(2.0), trainable=True, name="utility_b")
#
#     def call(self, inputs):
#         # 自定义输入层计算
#         hidden_input = tf.nn.relu(tf.matmul(inputs, self.input_weights) + self.input_bias)
#         # 多个隐藏层的计算
#         hidden_output = hidden_input
#         for i in range(3):  # 假设有3个隐藏层
#             # 自定义隐藏层计算
#             hidden_output = tf.nn.relu(tf.matmul(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])
#         # 自定义输出层计算
#         logits = tf.matmul(hidden_output, self.output_weights) + self.output_bias
#         utility = self.utility_a * logits + self.utility_b
#         # 应用 Sigmoid 函数
#         utility_a_time = tf.sigmoid(utility)
#
#         return utility_a_time


# class UtilityaTimeSubNetworkArriveMainFromZoneArriveMain(tf.keras.Model):
#     # GoalNetworkMain 生成目标张量，输入到 UtilityaZoneSubNetworkArriveMainFromGoal 生成 utility_a_zone，输入数据和目标张量一起输入到 ZoneNetworkArriveMain，利用 utility_a_zone 调整输出
#     def __init__(self, num_features_zone=3046, hidden_units=10, num_outputs=24):
#         # input = 3046 zones, output = 24 av_times
#         super(UtilityaTimeSubNetworkArriveMainFromZoneArriveMain, self).__init__()
#         # 初始化输入层的权重和偏置
#         self.input_weights = self.add_weight(shape=(num_features_zone, hidden_units), initializer='random_normal', trainable=True, name="input_weights" )
#         self.input_bias = self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name="input_bias")
#
#         # 初始化隐藏层的权重和偏置
#         self.hidden_weights = [self.add_weight(shape=(hidden_units, hidden_units), initializer='random_normal', trainable=True, name=f"hidden_weights_{i}") for i in range(3)]
#         self.hidden_biases = [self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name=f"hidden_bias_{i}") for i in range(3)]
#
#         # 初始化输出层的权重和偏置
#         self.output_weights = self.add_weight(shape=(hidden_units, num_outputs), initializer='random_normal', trainable=True, name="output_weights")
#         self.output_bias = self.add_weight(shape=(num_outputs,), initializer='zeros', trainable=True, name="output_bias")
#
#         # 效用函数参数：缩放因子 & 偏移量
#         self.utility_a = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(1.0), trainable=True, name="utility_a")
#         self.utility_b = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(2.0), trainable=True, name="utility_b")
#
#     def call(self, inputs):
#         # 自定义输入层计算
#         hidden_input = tf.nn.relu(tf.matmul(inputs, self.input_weights) + self.input_bias)
#         # 多个隐藏层的计算
#         hidden_output = hidden_input
#         for i in range(3):  # 假设有3个隐藏层
#             # 自定义隐藏层计算
#             hidden_output = tf.nn.relu(tf.matmul(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])
#         # 自定义输出层计算
#         logits = tf.matmul(hidden_output, self.output_weights) + self.output_bias
#         utility = self.utility_a * logits + self.utility_b
#         # 应用 Sigmoid 函数
#         utility_a_time = tf.sigmoid(utility)
#
#         return utility_a_time




# class TimeNetworkDepartMain(tf.keras.Model):
#     def __init__(self, num_features=9, hidden_units=10, num_outputs=24, num_features_zone=3046):
#         # input = 8 x + 4 av, output = 24 times_depart_main
#         super(TimeNetworkDepartMain, self).__init__()
#         self.num_outputs = num_outputs  # 添加 num_outputs 属性
#         # 初始化输入层的权重和偏置
#         self.input_weights = self.add_weight(shape=(num_features, hidden_units), initializer='random_normal', trainable=True, name="input_weights" )
#         self.input_bias = self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name="input_bias")
#
#         # 初始化隐藏层的权重和偏置
#         self.hidden_weights = [self.add_weight(shape=(hidden_units, hidden_units), initializer='random_normal', trainable=True, name=f"hidden_weights_{i}") for i in range(3)]
#         self.hidden_biases = [self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name=f"hidden_bias_{i}") for i in range(3)]
#
#         # 初始化输出层的权重和偏置
#         self.output_weights = self.add_weight(shape=(hidden_units, num_outputs), initializer='random_normal', trainable=True, name="output_weights")
#         self.output_bias = self.add_weight(shape=(num_outputs,), initializer='zeros', trainable=True, name="output_bias")
#
#         # 子网络生成 utility_a
#         self.utility_a_time_depart_main_subnetwork = UtilityaTimeSubNetworkDepartMainFromZoneDepartMain(num_features_zone=num_features_zone, hidden_units=hidden_units, num_outputs=num_outputs)
#         self.utility_b = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(2.0), trainable=True, name="utility_b")
#
#     def call(self, inputs, zones_depart_main):
#         # 子网络生成 utility_a
#         utility_a = self.utility_a_time_depart_main_subnetwork(zones_depart_main)
#
#         # 主网络计算
#         # 自定义输入层计算
#         hidden_input = tf.nn.relu(tf.matmul(inputs, self.input_weights) + self.input_bias)
#         # 多个隐藏层的计算
#         hidden_output = hidden_input
#         for i in range(3):  # 假设有3个隐藏层
#             # 自定义隐藏层计算
#             hidden_output = tf.nn.relu(tf.matmul(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])
#         # 自定义输出层计算
#         logits = tf.matmul(hidden_output, self.output_weights) + self.output_bias
#         # 应用效用函数
#         utility = utility_a * logits + self.utility_b
#         # 应用 Sigmoid 函数
#         outputs = tf.sigmoid(utility)
#
#         return outputs



# class ContextAwareModule(tf.keras.Model):
#     def __init__(self, num_zones=14, context_dim=5, hidden_units=10):
#         super(ContextAwareModule, self).__init__()
#         # 简单的多层感知机用于处理上下文信息，并输出调整参数
#         self.fc1 = layers.Dense(hidden_units, activation='relu', input_shape=(context_dim,))
#         self.fc2 = layers.Dense(num_zones * 2)  # 输出num_zones个权重和偏置的调整值（假设每个区有两个参数需要调整）
#
#     def call(self, context_input):
#         x = self.fc1(context_input)
#         output = self.fc2(x)
#         return output


# 根据ZoneNetworkArriveMain输出的选择不同到达交通小区的概率值，来计算一个加权平均的到达交通小区嵌入向量
def compute_weighted_embedding(probabilities, embeddings, arrive_district_ids, table):
    """
    根据到达交通小区的选择概率和它们的嵌入向量计算加权嵌入向量。
    根据每个样本的交通小区编号找到对应的概率，并将这些概率分别乘到该样本的嵌入向量

    参数:
    - probabilities: 形状为(batch_size, num_zones)的张量，表示选择每个到达交通小区的概率。   (64,14)
    - embeddings: 形状为(batch_size, embedding_dim)的张量，表示每个到达交通小区的嵌入向量。  (64,64)
    - arrive_district_ids: 形状为(batch_size,)的张量，表示每个样本的到达交通小区编号。
    - table: 从交通小区编号到其索引的查找表。

    返回:
    - weighted_embeddings: 形状为(batch_size, embedding_dim)的张量，表示每个样本的加权嵌入向量。
    """
    # 计算加权嵌入向量
    # weighted_embeddings = tf.matmul(probabilities, embeddings)     # (64, 14) * (14, 64)

    batch_size = tf.shape(embeddings)[0]      # 64
    embedding_dim = tf.shape(embeddings)[1]   # 64
    num_zones = tf.shape(probabilities)[1]    # 14

    # 使用查找表获取每个样本对应交通小区的索引
    district_indices = table.lookup(arrive_district_ids)  # (batch_size,)
    # print('district_indices', district_indices)   # (64,)  [4 4 4 4 4 4 4 4 4 4
    # 检查是否有无效的交通小区编号
    assert_all_valid = tf.Assert(tf.reduce_all(district_indices >= 0), [district_indices])

    with tf.control_dependencies([assert_all_valid]):
        # 获取每个样本的交通小区编号的概率
        gather_indices = tf.stack([tf.range(batch_size), arrive_district_ids], axis=1)
        # print('gather_indices', gather_indices) # (64, 2)  2列分别是64个样本的索引、交通小区编号  [[  0 433] [  1 433]
        # selected_probabilities = tf.gather_nd(probabilities, gather_indices)  # (batch_size,)
        # 获取每个样本的交通小区编号的概率
        selected_probabilities = tf.gather_nd(probabilities, tf.stack([tf.range(batch_size), district_indices], axis=1))  # (batch_size,)
        # print('selected_probabilities', selected_probabilities)   # (64,)

        # 将选定的概率扩展到与嵌入向量相同的维度
        selected_probabilities = tf.expand_dims(selected_probabilities, axis=-1)  # (batch_size, 1)   (64, 1)
        # print('selected_probabilities', selected_probabilities)     #
        selected_probabilities = tf.tile(selected_probabilities, [1, embedding_dim])  # (batch_size, embedding_dim)
        # print('selected_probabilities11', selected_probabilities)   # (64, 64)

        # 计算加权嵌入向量
        weighted_embeddings = embeddings * selected_probabilities  # (batch_size, embedding_dim)

    return weighted_embeddings





# 输入：
# features=9（包含出发交通小区）
# 上一层的输出 num_zone_arrive=14
# 真实距离信息：额外特征加入 num_zone_arrive = 14，每个出发交通小区有14个到达交通小区与其对应。
#       注意，这里不用加权zonearrive的概率分布。因为这里要求时间，要根据真实距离还计算
# 出发交通小区的嵌入向量 64，同上 ZoneNetworkArriveMain
# 到达交通小区的嵌入向量 64，compute_weighted_embedding函数：根据ZoneNetworkArriveMain输出的选择不同到达交通小区的概率值，来计算一个加权平均的到达交通小区嵌入向量。
#       其中，到达交通小区的嵌入向量用节点嵌入向量表示即可，因为类ZoneNetworkArriveMain无法获得到达交通小区编号，得到的是个概率分布，而不是一个确定的交通小区编号
#       or：到达交通小区id用类ZoneNetworkArriveMain的输出概率分布中，最大的那个来表示 √，假设有一个具体的到达小区
# × 不需要这个，因为GCN已经在ZoneNetworkArriveMain中使用过了，已经据此求得了到达交通小区的概率分布，另外还有真实距离信息。这里求时间时，使用zone的结果即可，是一个数据流。（另外，包含距离信息的交通小区的节点嵌入向量=64，不直接作为输入，而是采用GCN（作为GCN的输入），获得其中的邻接拓扑关系）

class TimeNetworkDepartMain(tf.keras.Model):
    def __init__(self, zone_ids_int, num_features=9, hidden_units=10, num_outputs=24, embedding_dim=64, num_goals=5, num_districts=3046, num_zones = 14, num_zone_depart = 14, num_zone_arrive = 14):
        super(TimeNetworkDepartMain, self).__init__()
        self.num_outputs = num_outputs
        # 使用整数类型的 zone_ids 初始化 zone_id_mapping，将zone_ids转换为Tensor
        # print('zone_ids_int222', zone_ids_int)
        self.zone_id_mapping = tf.constant(zone_ids_int, dtype=tf.int32)

        # 交通小区编号的嵌入层，在call中计算出发和到达的交通小区的嵌入向量
        self.district_embedding = layers.Embedding(input_dim=num_districts, output_dim=embedding_dim, name="district_embedding")

        # 计算输入维度,将距离信息作为额外特征加入(该向量长度=邻接矩阵的长度，即=交通小区的数量)
        input_dim = num_features + num_zone_arrive * 2 + embedding_dim * 2      # 9+14*2+64*2=165

        # 输入层权重和偏置
        self.input_weights = self.add_weight(shape=(input_dim, hidden_units), initializer='random_normal', trainable=True, name="input_weights")
        # print('TimeNetworkDepartMain_input_weights', self.input_weights.shape)   # 9+14*2+64*2=165     (165, 10)
        self.input_bias = self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name="input_bias")

        # 隐藏层权重和偏置
        self.hidden_weights = [
            self.add_weight(shape=(hidden_units, hidden_units), initializer='random_normal', trainable=True,
                            name=f"hidden_weights_{i}") for i in range(3)]
        self.hidden_biases = [
            self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name=f"hidden_bias_{i}") for i
            in range(3)]

        # 输出层权重和偏置
        self.output_weights = self.add_weight(shape=(hidden_units, num_outputs), initializer='random_normal',
                                              trainable=True, name="output_weights")
        self.output_bias = self.add_weight(shape=(num_outputs,), initializer='zeros', trainable=True,
                                           name="output_bias")

        # print('output_weights', self.output_weights.shape)   # (10, 24)
        # print('output_bias', self.output_bias.shape)  # (24,)

        # 上下文感知模块，用于生成效用函数参数的调整量
        self.context_net = tf.keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(input_dim,)),
            layers.Dense(num_outputs * (hidden_units+1))
            # 假设需要调整num_outputs个权重和偏置, num_outputs * 2的设计是为了生成两组调整量：一组用于调整输出层的权重，另一组用于调整输出层的偏置
        ])


    def call(self, inputs, zones_arrive_main, distances, zone_adjacency_matrix):
        """
        :param inputs: 出行者的个体特征
        :param goals_main: 预测的出行目的
        :param origin_district_ids: 出发交通小区的ID
        :param arrive_district_ids: 到达交通小区的ID
        :param distances: 包含交通小区间距离信息的邻接矩阵
        :param context_input: 上下文信息
        """

        # （1）origin_embeddings：
        # 提取出origin_district_ids
        origin_district_ids = extract_origin_district_ids(inputs)
        # print('origin_district_ids11', origin_district_ids.shape, origin_district_ids)  # (64,)

        # 创建映射字典
        id_to_index = create_id_to_index_mapping(zone_ids_int)
        # print('id_to_index45786586', id_to_index)

        # 获取距离特征
        distance_features = get_distance_features(origin_district_ids, zone_adjacency_matrix, id_to_index)

        # 获取出发交通小区的嵌入向量
        origin_embeddings = self.district_embedding(origin_district_ids)
        # print('origin_district_ids', origin_district_ids.numpy())
        # print('origin_embeddings.shape', origin_embeddings.shape)  # (64, 64)
        # print('origin_embeddings',origin_embeddings)

        # （2）求arrive_embeddings：
        # 找出每行（每个样本）最大概率的索引
        highest_prob_indices = tf.argmax(zones_arrive_main, axis=-1)   # [4 4 4 4 4 4 4 4 4 4
        # print("highest_prob_indices:", highest_prob_indices.numpy())
        # 查找最高概率交通小区的实际ID
        arrive_district_ids = tf.gather(self.zone_id_mapping, highest_prob_indices)
        # print("arrive_district_ids11:", arrive_district_ids.numpy())   # [433 433 433 433 433
        # arrive_district_ids = tf.strings.to_number(arrive_district_ids, out_type=tf.int32)
        arrive_embeddings = self.district_embedding(arrive_district_ids)
        # print("arrive_district_ids:", arrive_district_ids.numpy())   # [0 0 0 0 …… 0 0]
        # print('origin_embeddings', origin_embeddings.shape)  # (64, 64)
        # print('arrive_embeddings.shape', arrive_embeddings.shape)  # (64, 64)
        # print('arrive_embeddings', arrive_embeddings)

        # 创建查找表：从交通小区编号到其在 zone_ids_int 中的索引
        # # keys = tf.constant(zone_ids_int, dtype=tf.int32)
        # values = tf.range(len(zone_ids_int), dtype=tf.int32)
        # table = tf.lookup.StaticHashTable(
        #     initializer=tf.lookup.KeyValueTensorInitializer(self.zone_id_mapping, values),
        #     default_value=-1  # 如果找不到对应的键，则返回默认值 -1
        # )

        # 创建查找表：从交通小区编号到其索引
        keys = tf.constant(zone_ids_int, dtype=tf.int32)
        values = tf.range(len(zone_ids_int), dtype=tf.int32)
        table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=-1  # 如果找不到对应的键，则返回默认值 -1
        )
        # 计算加权的到达交通小区的嵌入向量
        arrive_embeddings_weighted = compute_weighted_embedding(zones_arrive_main, arrive_embeddings, arrive_district_ids, table)  # (64, 64) * (64, 64)
        # print('arrive_embeddings_weighted', arrive_embeddings_weighted.shape)  # (64, 64)

        inputs = tf.cast(inputs, dtype=tf.float32)
        zones_arrive_main = tf.cast(zones_arrive_main, dtype=tf.float32)  # 确保 zones_arrive_main 是 float32 类型
        distance_features = tf.cast(distance_features, dtype=tf.float32)  # 确保 distance_features 是 float32 类型
        origin_embeddings = tf.cast(origin_embeddings, dtype=tf.float32)  # 确保 origin_embeddings 是 float32 类型
        arrive_embeddings_weighted = tf.cast(arrive_embeddings_weighted, dtype=tf.float32)  # 确保 arrive_embeddings_weighted 是 float32 类型

        # # 计算出发和到达交通小区间的距离（从邻接矩阵中获取）
        # dist = tf.gather_nd(distances, tf.stack([origin_district_ids, arrive_district_ids], axis=1))
        # print('dist', dist.shape)   # (14,)
        # print('tf.expand_dims(dist, -1)', tf.expand_dims(dist, -1).shape)  # (14, 1)

        # 将输入、目标张量、出发和到达交通小区的嵌入向量以及距离拼接
        x = tf.concat([inputs, zones_arrive_main, distance_features, origin_embeddings, arrive_embeddings_weighted], axis=-1)
        # print('x depart', x.shape)  # 9+14*2+64*2=165   (64, 9) + (64, 14) + (14, 14)

        # 处理上下文信息并获得调整参数
        # adjustment_params = self.context_module(context_input)
        context_adjustments = self.context_net(x)
        # print('context_adjustments', context_adjustments.shape)  # (64, 264)

        weight_adjustments = context_adjustments[:, :self.output_weights.shape[0] * self.output_weights.shape[1]]
        bias_adjustments = context_adjustments[:, self.output_weights.shape[0] * self.output_weights.shape[1]:]

        adjusted_output_weights = self.output_weights + tf.reshape(
            weight_adjustments,
            shape=(-1, self.output_weights.shape[0], self.output_weights.shape[1])
        )[-1]
        # print('adjusted_output_weights', adjusted_output_weights.shape)    #

        adjusted_output_bias = self.output_bias + bias_adjustments  #
        # print('adjusted_output_bias', adjusted_output_bias.shape)   #

        # 自定义输入层计算
        hidden_input = tf.nn.relu(tf.matmul(x, self.input_weights) + self.input_bias)
        # 多个隐藏层的计算
        hidden_output = hidden_input
        # print('hidden_output', hidden_output.shape)  #
        for i in range(3):  # 假设有3个隐藏层
            hidden_output = tf.nn.relu(tf.matmul(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])

        # 使用调整后的参数计算输出层
        logits = tf.matmul(hidden_output, adjusted_output_weights) + adjusted_output_bias
        # 应用 Softmax 函数来得到概率分布
        outputs = tf.nn.softmax(logits)

        return outputs



# class ContextAwareModule(tf.keras.Model):
#     def __init__(self, num_zones=14, context_dim=5, hidden_units=10):
#         super(ContextAwareModule, self).__init__()
#         # 简单的多层感知机用于处理上下文信息，并输出调整参数
#         self.fc1 = layers.Dense(hidden_units, activation='relu', input_shape=(context_dim,))
#         self.fc2 = layers.Dense(num_zones * 2)  # 输出num_zones个权重和偏置的调整值（假设每个区有两个参数需要调整）
#
#     def call(self, context_input):
#         x = self.fc1(context_input)
#         output = self.fc2(x)
#         return output



# 输入：来自前两层ZoneNetworkArriveMain（真实距离和到达交通小区嵌入向量已经表示了，但是这个概率值没有完全表达出来，还是要输入上上层的输出）+TimeNetworkDepartMain
# features=9（包含出发交通小区）
# 上一层的输出 num_time_depart = 24  【这里跟上一个类不同，其余都相同】
# 上上一层的输出 num_zone_arrive = 14
# 真实距离信息：额外特征加入 num_zone_arrive = 14，每个出发交通小区有14个到达交通小区与其对应。
#       注意，这里不用加权zonearrive的概率分布。因为这里要求时间，要根据真实距离还计算
# 出发交通小区的嵌入向量 64，同上 ZoneNetworkArriveMain
# 到达交通小区的嵌入向量 64，compute_weighted_embedding函数：根据ZoneNetworkArriveMain输出的选择不同到达交通小区的概率值，来计算一个加权平均的到达交通小区嵌入向量。
#       其中，到达交通小区的嵌入向量用节点嵌入向量表示即可，因为类ZoneNetworkArriveMain无法获得到达交通小区编号，得到的是个概率分布，而不是一个确定的交通小区编号
#       or：到达交通小区id用类ZoneNetworkArriveMain的输出概率分布中，最大的那个来表示 √，假设有一个具体的到达小区
# × 不需要这个，因为GCN已经在ZoneNetworkArriveMain中使用过了，已经据此求得了到达交通小区的概率分布，另外还有真实距离信息。这里求时间时，使用zone的结果即可，是一个数据流。（另外，包含距离信息的交通小区的节点嵌入向量=64，不直接作为输入，而是采用GCN（作为GCN的输入），获得其中的邻接拓扑关系）

class TimeNetworkArriveMain(tf.keras.Model):
    def __init__(self, zone_ids_int, num_features=9, hidden_units=10, num_outputs=24, embedding_dim=64, num_goals=5, num_zone_arrive=14, num_time_depart = 24, num_districts=3046):
        super(TimeNetworkArriveMain, self).__init__()
        self.num_outputs = num_outputs
        # self.zone_id_mapping = tf.constant(zone_ids)  # 将zone_ids转换为Tensor
        self.zone_id_mapping = tf.constant(zone_ids_int, dtype=tf.int32)

        # 交通小区编号的嵌入层，在call中计算出发和到达的交通小区的嵌入向量
        self.district_embedding = layers.Embedding(input_dim=num_districts, output_dim=embedding_dim, name="district_embedding")

        # 计算输入维度,将距离信息作为额外特征加入(该向量长度=邻接矩阵的长度，即=交通小区的数量)
        input_dim = num_features + num_time_depart + num_zone_arrive + num_zone_arrive + embedding_dim * 2  # 9+24+14+14+64*2=189

        # 输入层权重和偏置
        self.input_weights = self.add_weight(shape=(input_dim, hidden_units), initializer='random_normal', trainable=True, name="input_weights")
        self.input_bias = self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name="input_bias")

        # 隐藏层权重和偏置
        self.hidden_weights = [
            self.add_weight(shape=(hidden_units, hidden_units), initializer='random_normal', trainable=True,
                            name=f"hidden_weights_{i}") for i in range(3)]
        self.hidden_biases = [
            self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name=f"hidden_bias_{i}") for i
            in range(3)]

        # 输出层权重和偏置
        self.output_weights = self.add_weight(shape=(hidden_units, num_outputs), initializer='random_normal',
                                              trainable=True, name="output_weights")
        self.output_bias = self.add_weight(shape=(num_outputs,), initializer='zeros', trainable=True,
                                           name="output_bias")

        # 上下文感知模块
        # self.context_module = ContextAwareModule(num_zone_arrive=14)
        self.context_net = tf.keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(input_dim,)),
            layers.Dense(num_outputs * (hidden_units+1))
            # 假设需要调整num_outputs个权重和偏置, num_outputs * 2的设计是为了生成两组调整量：一组用于调整输出层的权重，另一组用于调整输出层的偏置
        ])

    def call(self, inputs, zones_arrive_main, times_depart_main, distances, zone_adjacency_matrix):
        """
        :param inputs: 出行者的个体特征
        :param goals_main: 预测的出行目的
        :param origin_district_ids: 出发交通小区的ID
        :param arrive_district_ids: 到达交通小区的ID
        :param times_depart_main: 预测的出发时间段
        :param distances: 包含交通小区间距离信息的邻接矩阵
        :param context_input: 上下文信息
        """

        # # 获取出发和到达交通小区的嵌入向量
        # origin_embeddings = self.district_embedding(origin_district_ids)
        # arrive_embeddings = self.district_embedding(arrive_district_ids)
        #
        # # 计算出发和到达交通小区间的距离（从邻接矩阵中获取）
        # dist = tf.gather_nd(distances, tf.stack([origin_district_ids, arrive_district_ids], axis=1))


#         ###############
#         # 提取出origin_district_ids
#         origin_district_ids = extract_origin_district_ids(inputs)
#         print('origin_district_ids22', origin_district_ids.shape, origin_district_ids)
#
#         # 获取距离特征
#         distance_features = get_distance_features(origin_district_ids, zone_adjacency_matrix)
#
#         # 获取出发交通小区的嵌入向量
#         origin_embeddings = self.district_embedding(origin_district_ids)
#
#         # 找出每行（每个样本）最大概率的索引
#         highest_prob_indices = tf.argmax(zones_arrive_main, axis=-1)
#         # 查找最高概率交通小区的实际ID
#         arrive_district_ids = tf.gather(self.zone_id_mapping, highest_prob_indices)
#         arrive_embeddings = self.district_embedding(arrive_district_ids)
#         print('origin_embeddings', origin_embeddings.shape)  # (14, 64)
#         print('arrive_embeddings', arrive_embeddings.shape)  # (14, 64)
#         # 计算加权的到达交通小区的嵌入向量
#         arrive_embeddings_weighted = compute_weighted_embedding(zones_arrive_main, arrive_embeddings)
#         print('arrive_embeddings_weighted', arrive_embeddings_weighted.shape)  # (14, 64)
#         ####################

        # （1）origin_embeddings：
        # 提取出origin_district_ids
        origin_district_ids = extract_origin_district_ids(inputs)
        # print('origin_district_ids11', origin_district_ids.shape, origin_district_ids)  # (64,)

        # 创建映射字典
        id_to_index = create_id_to_index_mapping(zone_ids_int)

        # 获取距离特征
        distance_features = get_distance_features(origin_district_ids, zone_adjacency_matrix, id_to_index)

        # 获取出发交通小区的嵌入向量
        origin_embeddings = self.district_embedding(origin_district_ids)
        # print('origin_district_ids', origin_district_ids.numpy())
        # print('origin_embeddings.shape', origin_embeddings.shape)  # (64, 64)
        # print('origin_embeddings',origin_embeddings)

        # （2）求arrive_embeddings：
        # 找出每行（每个样本）最大概率的索引
        highest_prob_indices = tf.argmax(zones_arrive_main, axis=-1)  # [4 4 4 4 4 4 4 4 4 4
        # print("highest_prob_indices:", highest_prob_indices.numpy())  # [2 2 2 2 2 2 2 2 2 2
        # 查找最高概率交通小区的实际ID
        arrive_district_ids = tf.gather(self.zone_id_mapping, highest_prob_indices)
        # print("arrive_district_ids11:", arrive_district_ids.numpy())   # [417 417 417 417 417
        # arrive_district_ids = tf.strings.to_number(arrive_district_ids, out_type=tf.int32)
        arrive_embeddings = self.district_embedding(arrive_district_ids)
        # print("arrive_district_ids:", arrive_district_ids.numpy())   # [0 0 0 0 …… 0 0]
        # print('origin_embeddings', origin_embeddings.shape)  # (64, 64)
        # print('arrive_embeddings.shape', arrive_embeddings.shape)  # (64, 64)
        # print('arrive_embeddings', arrive_embeddings)

        # 创建查找表：从交通小区编号到其索引
        keys = tf.constant(zone_ids_int, dtype=tf.int32)
        values = tf.range(len(zone_ids_int), dtype=tf.int32)
        table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=-1  # 如果找不到对应的键，则返回默认值 -1
        )
        # 计算加权的到达交通小区的嵌入向量
        arrive_embeddings_weighted = compute_weighted_embedding(zones_arrive_main, arrive_embeddings,
                                                                arrive_district_ids, table)  # (64, 64) * (64, 64)
        # print('arrive_embeddings_weighted', arrive_embeddings_weighted.shape)  # (64, 64)

        inputs = tf.cast(inputs, dtype=tf.float32)
        zones_arrive_main = tf.cast(zones_arrive_main, dtype=tf.float32)  # 确保 zones_arrive_main 是 float32 类型
        distance_features = tf.cast(distance_features, dtype=tf.float32)  # 确保 distance_features 是 float32 类型
        origin_embeddings = tf.cast(origin_embeddings, dtype=tf.float32)  # 确保 origin_embeddings 是 float32 类型
        arrive_embeddings_weighted = tf.cast(arrive_embeddings_weighted,
                                             dtype=tf.float32)  # 确保 arrive_embeddings_weighted 是 float32 类型

        # 处理上下文信息并获得调整参数
        # adjustment_params = self.context_module(context_input)
        # context_adjustments = self.context_net(arrive_district_ids)
        # adjusted_output_weights = self.output_weights + context_adjustments[:,
        #                                                 :self.output_weights.shape[0] * self.output_weights.shape[
        #                                                     1]].reshape(self.output_weights.shape)
        # adjusted_output_bias = self.output_bias + context_adjustments[:, -self.output_bias.shape[0]:]

        # 将输入、目标张量、出发和到达交通小区的嵌入向量、距离及出发时间拼接
        x = tf.concat([inputs, times_depart_main, zones_arrive_main, distance_features, origin_embeddings, arrive_embeddings_weighted],
                      axis=-1)
        # print('x', x.shape)  # 9+24+14+14+64*2=189

        # 处理上下文信息并获得调整参数
        # adjustment_params = self.context_module(context_input)
        context_adjustments = self.context_net(x)
        # print('context_adjustments', context_adjustments.shape)  # (64, 264)

        weight_adjustments = context_adjustments[:, :self.output_weights.shape[0] * self.output_weights.shape[1]]
        bias_adjustments = context_adjustments[:, self.output_weights.shape[0] * self.output_weights.shape[1]:]

        adjusted_output_weights = self.output_weights + tf.reshape(
            weight_adjustments,
            shape=(-1, self.output_weights.shape[0], self.output_weights.shape[1])
        )[-1]
        # print('adjusted_output_weights', adjusted_output_weights.shape)    #

        adjusted_output_bias = self.output_bias + bias_adjustments  #
        # print('adjusted_output_bias', adjusted_output_bias.shape)   #

        # 自定义输入层计算
        hidden_input = tf.nn.relu(tf.matmul(x, self.input_weights) + self.input_bias)
        # 多个隐藏层的计算
        hidden_output = hidden_input
        for i in range(3):  # 假设有3个隐藏层
            hidden_output = tf.nn.relu(tf.matmul(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])

        # 使用调整后的参数计算输出层
        logits = tf.matmul(hidden_output, adjusted_output_weights) + adjusted_output_bias
        # 应用 Softmax 函数来得到概率分布
        outputs = tf.nn.softmax(logits)

        return outputs



# class UtilityaTimeSubNetworkArriveMainFromTimeDepartMain(tf.keras.Model):
#     # GoalNetworkMain 生成目标张量，输入到 UtilityaZoneSubNetworkArriveMainFromGoal 生成 utility_a_zone，输入数据和目标张量一起输入到 ZoneNetworkArriveMain，利用 utility_a_zone 调整输出
#     def __init__(self, num_features_timedepartmain=24, hidden_units=10, num_outputs=24):
#         # input = 3046 zones, output = 24 av_times
#         super(UtilityaTimeSubNetworkArriveMainFromTimeDepartMain, self).__init__()
#         # 初始化输入层的权重和偏置
#         self.input_weights = self.add_weight(shape=(num_features_timedepartmain, hidden_units), initializer='random_normal', trainable=True, name="input_weights" )
#         self.input_bias = self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name="input_bias")
#
#         # 初始化隐藏层的权重和偏置
#         self.hidden_weights = [self.add_weight(shape=(hidden_units, hidden_units), initializer='random_normal', trainable=True, name=f"hidden_weights_{i}") for i in range(3)]
#         self.hidden_biases = [self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name=f"hidden_bias_{i}") for i in range(3)]
#
#         # 初始化输出层的权重和偏置
#         self.output_weights = self.add_weight(shape=(hidden_units, num_outputs), initializer='random_normal', trainable=True, name="output_weights")
#         self.output_bias = self.add_weight(shape=(num_outputs,), initializer='zeros', trainable=True, name="output_bias")
#
#         # 效用函数参数：缩放因子 & 偏移量
#         self.utility_a = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(1.0), trainable=True, name="utility_a")
#         self.utility_b = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(2.0), trainable=True, name="utility_b")
#
#     def call(self, inputs):
#         # 自定义输入层计算
#         hidden_input = tf.nn.relu(tf.matmul(inputs, self.input_weights) + self.input_bias)
#         # 多个隐藏层的计算
#         hidden_output = hidden_input
#         for i in range(3):  # 假设有3个隐藏层
#             # 自定义隐藏层计算
#             hidden_output = tf.nn.relu(tf.matmul(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])
#         # 自定义输出层计算
#         logits = tf.matmul(hidden_output, self.output_weights) + self.output_bias
#         utility = self.utility_a * logits + self.utility_b
#         # 应用 Sigmoid 函数
#         utility_a_time = tf.sigmoid(utility)
#
#         return utility_a_time





# class TimeNetworkArriveMain(tf.keras.Model):
#     def __init__(self, num_features=9, hidden_units=10, num_outputs=24, num_features_zone=3046, num_features_timedepartmain=24):
#         # input = 8 x + 4 av, output = 24 times_arrive_main
#         super(TimeNetworkArriveMain, self).__init__()
#         self.num_outputs = num_outputs  # 添加 num_outputs 属性
#         # 初始化输入层的权重和偏置
#         self.input_weights = self.add_weight(shape=(num_features, hidden_units), initializer='random_normal', trainable=True, name="input_weights" )
#         self.input_bias = self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name="input_bias")
#
#         # 初始化隐藏层的权重和偏置
#         self.hidden_weights = [self.add_weight(shape=(hidden_units, hidden_units), initializer='random_normal', trainable=True, name=f"hidden_weights_{i}") for i in range(3)]
#         self.hidden_biases = [self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name=f"hidden_bias_{i}") for i in range(3)]
#
#         # 初始化输出层的权重和偏置
#         self.output_weights = self.add_weight(shape=(hidden_units, num_outputs), initializer='random_normal', trainable=True, name="output_weights")
#         self.output_bias = self.add_weight(shape=(num_outputs,), initializer='zeros', trainable=True, name="output_bias")
#
#         # 子网络生成 utility_a
#         self.utility_a_time_arrive_main_subnetwork_from_zonearrivemain = UtilityaTimeSubNetworkArriveMainFromZoneArriveMain(num_features_zone=num_features_zone, hidden_units=hidden_units, num_outputs=num_outputs)
#         self.utility_a_time_arrive_main_subnetwork_from_timedepartmain = UtilityaTimeSubNetworkArriveMainFromTimeDepartMain(num_features_timedepartmain=num_features_timedepartmain, hidden_units=hidden_units, num_outputs=num_outputs)
#
#         self.utility_b = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(2.0), trainable=True, name="utility_b")
#
#         # 添加权重 alpha，用于控制 utility_a 的加权平均
#         self.alpha = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(0.5), trainable=True, name="alpha")
#
#     def call(self, inputs, zones_arrive_main, times_depart_main):
#         # 子网络生成 utility_a
#         utility_a_time_arrive_main_from_zonearrivemain = self.utility_a_time_arrive_main_subnetwork_from_zonearrivemain(zones_arrive_main)
#         utility_a_time_arrive_main_from_timedepartmain = self.utility_a_time_arrive_main_subnetwork_from_timedepartmain(times_depart_main)
#
#         # 结合两个 utility_a (加权平均)
#         utility_a = self.alpha * utility_a_time_arrive_main_from_zonearrivemain + (1 - self.alpha) * utility_a_time_arrive_main_from_timedepartmain
#
#         # 主网络计算
#         # 自定义输入层计算
#         hidden_input = tf.nn.relu(tf.matmul(inputs, self.input_weights) + self.input_bias)
#         # 多个隐藏层的计算
#         hidden_output = hidden_input
#         for i in range(3):  # 假设有3个隐藏层
#             # 自定义隐藏层计算
#             hidden_output = tf.nn.relu(tf.matmul(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])
#         # 自定义输出层计算
#         logits = tf.matmul(hidden_output, self.output_weights) + self.output_bias
#         # 应用效用函数
#         utility = utility_a * logits + self.utility_b
#         # 应用 Sigmoid 函数
#         outputs = tf.sigmoid(utility)
#
#         return outputs


# class UtilityaMode1SubNetworkMainFromTimeMain(tf.keras.Model):
#     # GoalNetworkMain 生成目标张量，输入到 UtilityaZoneSubNetworkDepartMainFromGoal 生成 utility_a_zone，输入数据和目标张量一起输入到 ZoneNetworkDepartMain，利用 utility_a_zone 调整输出
#     def __init__(self, num_features_time=48, hidden_units=10, num_outputs=6):
#         # input = 24 times_depart_main + 24 times_arrive_main, output = 6 av_modes
#         super(UtilityaMode1SubNetworkMainFromTimeMain, self).__init__()
#         # 初始化输入层的权重和偏置
#         self.input_weights = self.add_weight(shape=(num_features_time, hidden_units), initializer='random_normal', trainable=True, name="input_weights" )
#         self.input_bias = self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name="input_bias")
#
#         # 初始化隐藏层的权重和偏置
#         self.hidden_weights = [self.add_weight(shape=(hidden_units, hidden_units), initializer='random_normal', trainable=True, name=f"hidden_weights_{i}") for i in range(3)]
#         self.hidden_biases = [self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name=f"hidden_bias_{i}") for i in range(3)]
#
#         # 初始化输出层的权重和偏置
#         self.output_weights = self.add_weight(shape=(hidden_units, num_outputs), initializer='random_normal', trainable=True, name="output_weights")
#         self.output_bias = self.add_weight(shape=(num_outputs,), initializer='zeros', trainable=True, name="output_bias")
#
#         # 效用函数参数：缩放因子 & 偏移量
#         self.utility_a = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(1.0), trainable=True, name="utility_a")
#         self.utility_b = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(2.0), trainable=True, name="utility_b")
#
#     def call(self, inputs):
#         # 自定义输入层计算
#         hidden_input = tf.nn.relu(tf.matmul(inputs, self.input_weights) + self.input_bias)
#         # 多个隐藏层的计算
#         hidden_output = hidden_input
#         for i in range(3):  # 假设有3个隐藏层
#             # 自定义隐藏层计算
#             hidden_output = tf.nn.relu(tf.matmul(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])
#         # 自定义输出层计算
#         logits = tf.matmul(hidden_output, self.output_weights) + self.output_bias
#         utility = self.utility_a * logits + self.utility_b
#         # 应用 Sigmoid 函数
#         utility_a_mode = tf.sigmoid(utility)
#
#         return utility_a_mode
#
# class UtilityaMode2SubNetworkMainFromTimeMain(tf.keras.Model):
#     # GoalNetworkMain 生成目标张量，输入到 UtilityaZoneSubNetworkDepartMainFromGoal 生成 utility_a_zone，输入数据和目标张量一起输入到 ZoneNetworkDepartMain，利用 utility_a_zone 调整输出
#     def __init__(self, num_features_time=48, hidden_units=10, num_outputs=6):
#         # input = 24 times_depart_main + 24 times_arrive_main , output = 6 av_modes
#         super(UtilityaMode2SubNetworkMainFromTimeMain, self).__init__()
#         # 初始化输入层的权重和偏置
#         self.input_weights = self.add_weight(shape=(num_features_time, hidden_units), initializer='random_normal', trainable=True, name="input_weights" )
#         self.input_bias = self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name="input_bias")
#
#         # 初始化隐藏层的权重和偏置
#         self.hidden_weights = [self.add_weight(shape=(hidden_units, hidden_units), initializer='random_normal', trainable=True, name=f"hidden_weights_{i}") for i in range(3)]
#         self.hidden_biases = [self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name=f"hidden_bias_{i}") for i in range(3)]
#
#         # 初始化输出层的权重和偏置
#         self.output_weights = self.add_weight(shape=(hidden_units, num_outputs), initializer='random_normal', trainable=True, name="output_weights")
#         self.output_bias = self.add_weight(shape=(num_outputs,), initializer='zeros', trainable=True, name="output_bias")
#
#         # 效用函数参数：缩放因子 & 偏移量
#         self.utility_a = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(1.0), trainable=True, name="utility_a")
#         self.utility_b = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(2.0), trainable=True, name="utility_b")
#
#     def call(self, inputs):
#         # 自定义输入层计算
#         hidden_input = tf.nn.relu(tf.matmul(inputs, self.input_weights) + self.input_bias)
#         # 多个隐藏层的计算
#         hidden_output = hidden_input
#         for i in range(3):  # 假设有3个隐藏层
#             # 自定义隐藏层计算
#             hidden_output = tf.nn.relu(tf.matmul(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])
#         # 自定义输出层计算
#         logits = tf.matmul(hidden_output, self.output_weights) + self.output_bias
#         utility = self.utility_a * logits + self.utility_b
#         # 应用 Sigmoid 函数
#         utility_a_mode = tf.sigmoid(utility)
#
#         return utility_a_mode
#

# class Mode1NetworkMain(tf.keras.Model):
#     def __init__(self, num_features=9, hidden_units=10, num_outputs=6, num_features_time=48):
#         # input = 8 x + 4 av, output = 6 modes1_main
#         super(Mode1NetworkMain, self).__init__()
#         self.num_outputs = num_outputs  # 添加 num_outputs 属性
#         # 初始化输入层的权重和偏置
#         self.input_weights = self.add_weight(shape=(num_features, hidden_units), initializer='random_normal', trainable=True, name="input_weights" )
#         self.input_bias = self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name="input_bias")
#
#         # 初始化隐藏层的权重和偏置
#         self.hidden_weights = [self.add_weight(shape=(hidden_units, hidden_units), initializer='random_normal', trainable=True, name=f"hidden_weights_{i}") for i in range(3)]
#         self.hidden_biases = [self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name=f"hidden_bias_{i}") for i in range(3)]
#
#         # 初始化输出层的权重和偏置
#         self.output_weights = self.add_weight(shape=(hidden_units, num_outputs), initializer='random_normal', trainable=True, name="output_weights")
#         self.output_bias = self.add_weight(shape=(num_outputs,), initializer='zeros', trainable=True, name="output_bias")
#
#         # 子网络生成 utility_a
#         self.utility_a_mode1_subnetwork = UtilityaMode1SubNetworkMainFromTimeMain(num_features_time=num_features_time, hidden_units=hidden_units, num_outputs=num_outputs)
#         self.utility_b = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(2.0), trainable=True, name="utility_b")
#
#     def call(self, inputs, time_combined_main):
#         # 子网络生成 utility_a
#         utility_a = self.utility_a_mode1_subnetwork(time_combined_main)
#
#         # 主网络计算
#         # 自定义输入层计算
#         hidden_input = tf.nn.relu(tf.matmul(inputs, self.input_weights) + self.input_bias)
#         # 多个隐藏层的计算
#         hidden_output = hidden_input
#         for i in range(3):  # 假设有3个隐藏层
#             # 自定义隐藏层计算
#             hidden_output = tf.nn.relu(tf.matmul(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])
#         # 自定义输出层计算
#         logits = tf.matmul(hidden_output, self.output_weights) + self.output_bias
#         # 应用效用函数
#         utility = utility_a * logits + self.utility_b
#         # 应用 Sigmoid 函数
#         outputs = tf.sigmoid(utility)
#
#         return outputs


# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
#
#
# class ContextAwareModule(tf.keras.Model):
#     def __init__(self, num_zones=14, context_dim=5, hidden_units=10):
#         super(ContextAwareModule, self).__init__()
#         # 简单的多层感知机用于处理上下文信息，并输出调整参数
#         self.fc1 = layers.Dense(hidden_units, activation='relu', input_shape=(context_dim,))
#         self.fc2 = layers.Dense(num_zones * 2)  # 输出num_zones个权重和偏置的调整值（假设每个区有两个参数需要调整）
#
#     def call(self, context_input):
#         x = self.fc1(context_input)
#         output = self.fc2(x)
#         return output


# class Mode1NetworkMain(tf.keras.Model):
#     def __init__(self, num_features=9, hidden_units=10, num_outputs=6, embedding_dim=64, num_goals=5, num_zones=14):
#         super(Mode1NetworkMain, self).__init__()
#         self.num_outputs = num_outputs
#
#         # 初始化输入层的权重和偏置
#         self.input_weights = self.add_weight(shape=(num_features + embedding_dim * 2 + num_goals + 2 + 1),
#                                              initializer='random_normal', trainable=True, name="input_weights")
#         self.input_bias = self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name="input_bias")
#
#         # 初始化隐藏层的权重和偏置
#         self.hidden_weights = [
#             self.add_weight(shape=(hidden_units, hidden_units), initializer='random_normal', trainable=True,
#                             name=f"hidden_weights_{i}") for i in range(3)]
#         self.hidden_biases = [
#             self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name=f"hidden_bias_{i}") for i
#             in range(3)]
#
#         # 初始化输出层的权重和偏置
#         self.output_weights = self.add_weight(shape=(hidden_units, num_outputs), initializer='random_normal',
#                                               trainable=True, name="output_weights")
#         self.output_bias = self.add_weight(shape=(num_outputs,), initializer='zeros', trainable=True,
#                                            name="output_bias")
#
#         # 嵌入层
#         self.district_embedding = layers.Embedding(input_dim=num_zones, output_dim=embedding_dim,
#                                                    name="district_embedding")
#
#         # 上下文感知模块
#         self.context_module = ContextAwareModule(num_zones=14)
#
#     def call(self, inputs, goals_main, origin_district_ids, arrive_district_ids, times_depart_main, times_arrive_main, distances, context_input):
#         """
#         :param inputs: 出行者的个体特征
#         :param goals_main: 预测的出行目的
#         :param origin_district_ids: 出发交通小区的ID
#         :param arrive_district_ids: 到达交通小区的ID
#         :param times_depart_main: 预测的出发时间段
#         :param times_arrive_main: 预测的到达时间段
#         :param distances: 包含交通小区间距离信息的邻接矩阵
#         :param context_input: 上下文信息
#         """
#         # 获取出发和到达交通小区的嵌入向量
#         origin_embeddings = self.district_embedding(origin_district_ids)
#         arrive_embeddings = self.district_embedding(arrive_district_ids)
#
#         # 计算出发和到达交通小区间的距离（从邻接矩阵中获取）
#         dist = tf.gather_nd(distances, tf.stack([origin_district_ids, arrive_district_ids], axis=1))
#
#         # 处理上下文信息并获得调整参数
#         adjustment_params = self.context_module(context_input)
#         adjusted_output_weights = self.output_weights + adjustment_params[:,
#                                                         :self.output_weights.shape[0] * self.output_weights.shape[
#                                                             1]].reshape(self.output_weights.shape)
#         adjusted_output_bias = self.output_bias + adjustment_params[:, -self.output_bias.shape[0]:]
#
#         # 将输入、目标张量、出发和到达交通小区的嵌入向量、距离及出发时间拼接
#         x = tf.concat(
#             [inputs, goals_main, origin_embeddings, arrive_embeddings, tf.expand_dims(dist, -1), times_depart_main,
#              times_arrive_main], axis=-1)
#
#         # 自定义输入层计算
#         hidden_input = tf.nn.relu(tf.matmul(x, self.input_weights) + self.input_bias)
#         # 多个隐藏层的计算
#         hidden_output = hidden_input
#         for i in range(3):  # 假设有3个隐藏层
#             hidden_output = tf.nn.relu(tf.matmul(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])
#         # 使用调整后的参数计算输出层
#         logits = tf.matmul(hidden_output, adjusted_output_weights) + adjusted_output_bias
#         # 应用 Sigmoid 函数
#         outputs = tf.sigmoid(logits)
#
#         return outputs
#


# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers


# class ContextAwareModule(tf.keras.Model):
#     def __init__(self, num_zones=14, context_dim=5, hidden_units=10):
#         super(ContextAwareModule, self).__init__()
#         # 简单的多层感知机用于处理上下文信息，并输出调整参数
#         self.fc1 = layers.Dense(hidden_units, activation='relu', input_shape=(context_dim,))
#         self.fc2 = layers.Dense(num_zones * 2)  # 输出num_zones个权重和偏置的调整值（假设每个区有两个参数需要调整）
#
#     def call(self, context_input):
#         x = self.fc1(context_input)
#         output = self.fc2(x)
#         return output


# def calculate_expected_time(probabilities):
#     """
#     根据概率分布计算期望时间。
#     :param probabilities: 时间段的概率分布
#     :return: 期望时间（小时）
#     """
#     print('probabilities', probabilities)   # (64, 24)
#     expected_time = sum([i * p for i, p in enumerate(probabilities)])
#     print('expected_time', expected_time)   # (24,)
#     return expected_time


def calculate_expected_time(probabilities):
    """
    根据概率分布计算每个样本的期望时间。
    :param probabilities: 形状为 (batch_size, num_times)，即 (64, 24) 的时间段的概率分布
    :return: 形状为 (batch_size,) 的期望时间（小时）
    """
    batch_size, num_times = tf.shape(probabilities)[0], tf.shape(probabilities)[1]

    # 创建一个表示时间段索引的张量，形状为 (num_times,)
    time_indices = tf.range(num_times, dtype=probabilities.dtype)

    # 将时间段索引扩展到与 probabilities 相同的形状 (batch_size, num_times)
    time_indices = tf.tile(time_indices[tf.newaxis, :], [batch_size, 1])

    # 计算加权和以得到期望时间
    expected_time = tf.reduce_sum(time_indices * probabilities, axis=1)

    # print('probabilities', probabilities)  # (64, 24)
    # print('expected_time', expected_time)  # (64,)

    return expected_time

# 输入：来自前两层与时间有关的：TimeNetworkDepartMain+TimeNetworkArriveMain，这两者联合起作用
# features=9（包含出发交通小区）
# 上一层的输出 num_time_depart = 24
# 上上一层的输出 num_time_arrive = 24
# 时间差：到达时间-出发时间，计算【期望时间】之差  维度是 1
# 真实距离信息：额外特征加入 num_zone_arrive = 14，每个出发交通小区有14个到达交通小区与其对应。
#       注意，这里不用加权zonearrive的概率分布。因为这里要求时间，要根据真实距离还计算
# × 不需要这个，因为已知真实距离和时间差了，过多特征会导致过拟合，下同。出发交通小区的嵌入向量 64，同上 ZoneNetworkArriveMain
# × 不需要这个， 到达交通小区的嵌入向量 64，compute_weighted_embedding函数：根据ZoneNetworkArriveMain输出的选择不同到达交通小区的概率值，来计算一个加权平均的到达交通小区嵌入向量。
#       其中，到达交通小区的嵌入向量用节点嵌入向量表示即可，因为类ZoneNetworkArriveMain无法获得到达交通小区编号，得到的是个概率分布，而不是一个确定的交通小区编号
#       or：到达交通小区id用类ZoneNetworkArriveMain的输出概率分布中，最大的那个来表示 √，假设有一个具体的到达小区
# × 不需要这个，因为GCN已经在ZoneNetworkArriveMain中使用过了，已经据此求得了到达交通小区的概率分布，另外还有真实距离信息。这里求时间时，使用zone的结果即可，是一个数据流。（另外，包含距离信息的交通小区的节点嵌入向量=64，不直接作为输入，而是采用GCN（作为GCN的输入），获得其中的邻接拓扑关系）

class Mode1NetworkMain(tf.keras.Model):
    def __init__(self, zone_ids_int, num_features=9, hidden_units=10, num_outputs=6, embedding_dim=64, num_goals=5, num_zones=14, num_districts=3046, num_zone_arrive = 14, num_time_depart = 24, num_time_arrive = 24):
        super(Mode1NetworkMain, self).__init__()
        self.num_outputs = num_outputs
        # self.time_diff_dense_layer = layers.Dense(8, activation='relu')  # 处理时间差特征的层

        # self.zone_id_mapping = tf.constant(zone_ids)  # 将zone_ids转换为Tensor
        self.zone_id_mapping = tf.constant(zone_ids_int, dtype=tf.int32)

        # 交通小区编号的嵌入层，在call中计算出发和到达的交通小区的嵌入向量
        self.district_embedding = layers.Embedding(input_dim=num_districts, output_dim=embedding_dim, name="district_embedding")

        input_dim = num_features + num_time_depart + num_time_arrive + 1 + num_zone_arrive   # 9+24+24+1+14=72

        # 初始化输入层的权重和偏置
        # 注意：这里我们增加了出行时长作为一个新的输入特征
        self.input_weights = self.add_weight(shape=(input_dim, hidden_units), initializer='random_normal', trainable=True, name="input_weights")
        self.input_bias = self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name="input_bias")

        # 初始化隐藏层的权重和偏置
        self.hidden_weights = [
            self.add_weight(shape=(hidden_units, hidden_units), initializer='random_normal', trainable=True,
                            name=f"hidden_weights_{i}") for i in range(3)]
        self.hidden_biases = [
            self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name=f"hidden_bias_{i}") for i
            in range(3)]

        # 初始化输出层的权重和偏置
        self.output_weights = self.add_weight(shape=(hidden_units, num_outputs), initializer='random_normal',
                                              trainable=True, name="output_weights")
        self.output_bias = self.add_weight(shape=(num_outputs,), initializer='zeros', trainable=True,
                                           name="output_bias")

        # 嵌入层
        self.district_embedding = layers.Embedding(input_dim=num_districts, output_dim=embedding_dim,
                                                   name="district_embedding")

        # 上下文感知模块
        # self.context_module = ContextAwareModule(num_zones=14)
        self.context_net = tf.keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(input_dim,)),
            layers.Dense(num_outputs * (hidden_units+1))
            # 假设需要调整num_outputs个权重和偏置, num_outputs * 2的设计是为了生成两组调整量：一组用于调整输出层的权重，另一组用于调整输出层的偏置
        ])

    def call(self, inputs, times_depart_main, times_arrive_main, zone_adjacency_matrix):
        """
        :param inputs: 出行者的个体特征
        :param goals_main: 预测的出行目的
        :param origin_district_ids: 出发交通小区的ID
        :param arrive_district_ids: 到达交通小区的ID
        :param times_depart_main: 预测的出发时间段
        :param times_arrive_main: 预测的到达时间段
        :param distances: 包含交通小区间距离信息的邻接矩阵
        :param context_input: 上下文信息
        """

        expected_departure_time = calculate_expected_time(times_depart_main)
        expected_arrival_time = calculate_expected_time(times_arrive_main)
        # print('expected_arrival_time', expected_arrival_time)   # (64,)

        # 如果期望时间是基于0-24小时的，则需要处理跨越午夜的情况
        # if expected_arrival_time < expected_departure_time:
        #     expected_arrival_time += 24  # 跨越午夜，增加24小时

        # 对于每个样本，如果 expected_arrival_time < expected_departure_time，增加24小时
        time_difference = tf.where(
            expected_arrival_time < expected_departure_time,
            expected_arrival_time + 24 - expected_departure_time,
            expected_arrival_time - expected_departure_time
        )

        # time_difference = expected_arrival_time - expected_departure_time
        # print('time_difference',time_difference)  # (64,)

        # 将时间差转换为张量并处理
        # 将 time_difference 转换为形状为 (64, 1) 的张量
        time_difference_tensor = tf.expand_dims(time_difference, axis=-1)  # 形状变为 (64, 1)
        # time_difference_tensor = tf.convert_to_tensor([[time_difference]], dtype=tf.float32)
        # print('time_difference_tensor', time_difference_tensor)   # (64, 1)
        # processed_time_diff = self.time_diff_dense_layer(time_difference_tensor)

        # 提取出origin_district_ids
        origin_district_ids = extract_origin_district_ids(inputs)
        # print('origin_district_ids33', origin_district_ids.shape, origin_district_ids)

        # 创建映射字典
        id_to_index = create_id_to_index_mapping(zone_ids_int)

        # 获取距离特征
        distance_features = get_distance_features(origin_district_ids, zone_adjacency_matrix, id_to_index)
        # print('distance_features', distance_features)  # (64, 14)

        # # 获取出发交通小区的嵌入向量
        # origin_embeddings = self.district_embedding(origin_district_ids)
        #
        # # 找出每行（每个样本）最大概率的索引
        # highest_prob_indices = tf.argmax(zones_arrive_main, axis=-1)
        # # 查找最高概率交通小区的实际ID
        # arrive_district_ids = tf.gather(self.zone_id_mapping, highest_prob_indices)
        # arrive_embeddings = self.district_embedding(arrive_district_ids)
        # print('origin_embeddings', origin_embeddings.shape)  # (14, 64)
        # print('arrive_embeddings', arrive_embeddings.shape)  # (14, 64)
        # # 计算加权的到达交通小区的嵌入向量
        # arrive_embeddings_weighted = compute_weighted_embedding(zones_arrive_main, arrive_embeddings)
        # print('arrive_embeddings_weighted', arrive_embeddings_weighted.shape)  # (14, 64)

        # # 获取出发和到达交通小区的嵌入向量
        # origin_embeddings = self.district_embedding(origin_district_ids)
        # arrive_embeddings = self.district_embedding(arrive_district_ids)
        #
        # # 计算出发和到达交通小区间的距离（从邻接矩阵中获取）
        # dist = tf.gather_nd(distances, tf.stack([origin_district_ids, arrive_district_ids], axis=1))
        #
        # # 计算出发时间和到达时间之间的时间差（预计的出行时长）
        # travel_time = times_arrive_main - times_depart_main

        # # 处理上下文信息并获得调整参数
        # # adjustment_params = self.context_module(context_input)
        # context_adjustments = self.context_net(arrive_district_ids)
        # adjusted_output_weights = self.output_weights + context_adjustments[:,
        #                                                 :self.output_weights.shape[0] * self.output_weights.shape[
        #                                                     1]].reshape(self.output_weights.shape)
        # adjusted_output_bias = self.output_bias + context_adjustments[:, -self.output_bias.shape[0]:]

        # # 处理上下文信息并获得调整参数
        # # adjustment_params = self.context_module(context_input)
        # context_adjustments = self.context_net(x)
        # print('context_adjustments', context_adjustments.shape)  # (64, 264)
        #
        # weight_adjustments = context_adjustments[:, :self.output_weights.shape[0] * self.output_weights.shape[1]]
        # bias_adjustments = context_adjustments[:, self.output_weights.shape[0] * self.output_weights.shape[1]:]
        #
        # adjusted_output_weights = self.output_weights + tf.reshape(
        #     weight_adjustments,
        #     shape=(-1, self.output_weights.shape[0], self.output_weights.shape[1])
        # )[-1]
        # # print('adjusted_output_weights', adjusted_output_weights.shape)    #
        #
        # adjusted_output_bias = self.output_bias + bias_adjustments  #
        # # print('adjusted_output_bias', adjusted_output_bias.shape)   #

        inputs = tf.cast(inputs, dtype=tf.float32)
        times_depart_main = tf.cast(times_depart_main, dtype=tf.float32)  # 确保 zones_arrive_main 是 float32 类型
        times_arrive_main = tf.cast(times_arrive_main, dtype=tf.float32)  # 确保 distance_features 是 float32 类型
        time_difference_tensor = tf.cast(time_difference_tensor, dtype=tf.float32)  # 确保 origin_embeddings 是 float32 类型
        distance_features = tf.cast(distance_features, dtype=tf.float32)  # 确保 arrive_embeddings_weighted 是 float32 类型

        # 将输入、目标张量、出发和到达交通小区的嵌入向量、距离及出发时间拼接，并加入出行时长
        x = tf.concat(
            [inputs, times_depart_main, times_arrive_main, time_difference_tensor, distance_features], axis=-1)  # 9+24+24+1+14=72

        # 处理上下文信息并获得调整参数
        # adjustment_params = self.context_module(context_input)
        context_adjustments = self.context_net(x)
        # print('context_adjustments', context_adjustments.shape)  # (64, 264)

        weight_adjustments = context_adjustments[:, :self.output_weights.shape[0] * self.output_weights.shape[1]]
        bias_adjustments = context_adjustments[:, self.output_weights.shape[0] * self.output_weights.shape[1]:]

        adjusted_output_weights = self.output_weights + tf.reshape(
            weight_adjustments,
            shape=(-1, self.output_weights.shape[0], self.output_weights.shape[1])
        )[-1]
        # print('adjusted_output_weights', adjusted_output_weights.shape)    #

        adjusted_output_bias = self.output_bias + bias_adjustments  #
        # print('adjusted_output_bias', adjusted_output_bias.shape)   #

        # 自定义输入层计算
        hidden_input = tf.nn.relu(tf.matmul(x, self.input_weights) + self.input_bias)
        # 多个隐藏层的计算
        hidden_output = hidden_input
        for i in range(3):  # 假设有3个隐藏层
            hidden_output = tf.nn.relu(tf.matmul(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])

        # 使用调整后的参数计算输出层
        logits = tf.matmul(hidden_output, adjusted_output_weights) + adjusted_output_bias

        # 应用 Sigmoid 函数
        outputs = tf.nn.softmax(logits)

        return outputs



# class UtilityaMode2SubNetworkMainFromMode1Main(tf.keras.Model):
#     # GoalNetworkMain 生成目标张量，输入到 UtilityaZoneSubNetworkDepartMainFromGoal 生成 utility_a_zone，输入数据和目标张量一起输入到 ZoneNetworkDepartMain，利用 utility_a_zone 调整输出
#     def __init__(self, num_features_mode1=6, hidden_units=10, num_outputs=6):
#         # input = 24 times_depart_main + 24 times_arrive_main , output = 6 av_modes
#         super(UtilityaMode2SubNetworkMainFromMode1Main, self).__init__()
#         # 初始化输入层的权重和偏置
#         self.input_weights = self.add_weight(shape=(num_features_mode1, hidden_units), initializer='random_normal', trainable=True, name="input_weights" )
#         self.input_bias = self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name="input_bias")
#
#         # 初始化隐藏层的权重和偏置
#         self.hidden_weights = [self.add_weight(shape=(hidden_units, hidden_units), initializer='random_normal', trainable=True, name=f"hidden_weights_{i}") for i in range(3)]
#         self.hidden_biases = [self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name=f"hidden_bias_{i}") for i in range(3)]
#
#         # 初始化输出层的权重和偏置
#         self.output_weights = self.add_weight(shape=(hidden_units, num_outputs), initializer='random_normal', trainable=True, name="output_weights")
#         self.output_bias = self.add_weight(shape=(num_outputs,), initializer='zeros', trainable=True, name="output_bias")
#
#         # 效用函数参数：缩放因子 & 偏移量
#         self.utility_a = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(1.0), trainable=True, name="utility_a")
#         self.utility_b = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(2.0), trainable=True, name="utility_b")
#
#     def call(self, inputs):
#         # 自定义输入层计算
#         hidden_input = tf.nn.relu(tf.matmul(inputs, self.input_weights) + self.input_bias)
#         # 多个隐藏层的计算
#         hidden_output = hidden_input
#         for i in range(3):  # 假设有3个隐藏层
#             # 自定义隐藏层计算
#             hidden_output = tf.nn.relu(tf.matmul(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])
#         # 自定义输出层计算
#         logits = tf.matmul(hidden_output, self.output_weights) + self.output_bias
#         utility = self.utility_a * logits + self.utility_b
#         # 应用 Sigmoid 函数
#         utility_a_mode = tf.sigmoid(utility)
#
#         return utility_a_mode
#
#
# class Mode2NetworkMain(tf.keras.Model):
#     def __init__(self, num_features=9, hidden_units=10, num_outputs=6, num_features_time=48, num_features_mode1=6):
#         # input = 8 x + 4 av, output = 6 modes1_main
#         super(Mode2NetworkMain, self).__init__()
#         self.num_outputs = num_outputs  # 添加 num_outputs 属性
#         # 初始化输入层的权重和偏置
#         self.input_weights = self.add_weight(shape=(num_features, hidden_units), initializer='random_normal', trainable=True, name="input_weights" )
#         self.input_bias = self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name="input_bias")
#
#         # 初始化隐藏层的权重和偏置
#         self.hidden_weights = [self.add_weight(shape=(hidden_units, hidden_units), initializer='random_normal', trainable=True, name=f"hidden_weights_{i}") for i in range(3)]
#         self.hidden_biases = [self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name=f"hidden_bias_{i}") for i in range(3)]
#
#         # 初始化输出层的权重和偏置
#         self.output_weights = self.add_weight(shape=(hidden_units, num_outputs), initializer='random_normal', trainable=True, name="output_weights")
#         self.output_bias = self.add_weight(shape=(num_outputs,), initializer='zeros', trainable=True, name="output_bias")
#
#         # 子网络生成 utility_a
#         self.utility_a_mode2_subnetwork_from_timemain = UtilityaMode2SubNetworkMainFromTimeMain(num_features_time=num_features_time, hidden_units=hidden_units, num_outputs=num_outputs)
#         self.utility_a_mode2_subnetwork_from_mode1main = UtilityaMode2SubNetworkMainFromMode1Main(num_features_mode1=num_features_mode1, hidden_units=hidden_units, num_outputs=num_outputs)
#
#         self.utility_b = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(2.0), trainable=True, name="utility_b")
#
#         # 添加权重 alpha，用于控制 utility_a 的加权平均
#         self.alpha = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(0.5), trainable=True, name="alpha")
#
#     def call(self, inputs, time_combined_main, modes1_main):
#         # 子网络生成 utility_a
#         utility_a_mode2_from_timemain = self.utility_a_mode2_subnetwork_from_timemain(time_combined_main)
#         utility_a_mode2_from_mode1main = self.utility_a_mode2_subnetwork_from_mode1main(modes1_main)
#
#         # 结合两个 utility_a (加权平均)
#         utility_a = self.alpha * utility_a_mode2_from_timemain + (1 - self.alpha) * utility_a_mode2_from_mode1main
#
#         # 主网络计算
#         # 自定义输入层计算
#         hidden_input = tf.nn.relu(tf.matmul(inputs, self.input_weights) + self.input_bias)
#         # 多个隐藏层的计算
#         hidden_output = hidden_input
#         for i in range(3):  # 假设有3个隐藏层
#             # 自定义隐藏层计算
#             hidden_output = tf.nn.relu(tf.matmul(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])
#         # 自定义输出层计算
#         logits = tf.matmul(hidden_output, self.output_weights) + self.output_bias
#         # 应用效用函数
#         utility = utility_a * logits + self.utility_b
#         # 应用 Sigmoid 函数
#         outputs = tf.sigmoid(utility)
#
#         return outputs


# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
#
#
# class ContextAwareModule(tf.keras.Model):
#     def __init__(self, num_zones=14, context_dim=5, hidden_units=10):
#         super(ContextAwareModule, self).__init__()
#         # 简单的多层感知机用于处理上下文信息，并输出调整参数
#         self.fc1 = layers.Dense(hidden_units, activation='relu', input_shape=(context_dim,))
#         self.fc2 = layers.Dense(num_zones * 2)  # 输出num_zones个权重和偏置的调整值（假设每个区有两个参数需要调整）
#
#     def call(self, context_input):
#         x = self.fc1(context_input)
#         output = self.fc2(x)
#         return output


# 输入：来自前两层与时间有关的：TimeNetworkDepartMain+TimeNetworkArriveMain，这两者联合起作用
# features=9（包含出发交通小区）
# 上一层的输出 num_mode1 = 6   【这里与上一层不同，其余都相同】
# 上上一层的输出 num_time_depart = 24
# 上上上一层的输出 num_time_arrive = 24
# 时间差：到达时间-出发时间，计算【期望时间】之差  维度是 1
# 真实距离信息：额外特征加入 num_zone_arrive = 14，每个出发交通小区有14个到达交通小区与其对应。
#       注意，这里不用加权zonearrive的概率分布。因为这里要求时间，要根据真实距离还计算
# × 不需要这个，因为已知真实距离和时间差了，过多特征会导致过拟合，下同。出发交通小区的嵌入向量 64，同上 ZoneNetworkArriveMain
# × 不需要这个， 到达交通小区的嵌入向量 64，compute_weighted_embedding函数：根据ZoneNetworkArriveMain输出的选择不同到达交通小区的概率值，来计算一个加权平均的到达交通小区嵌入向量。
#       其中，到达交通小区的嵌入向量用节点嵌入向量表示即可，因为类ZoneNetworkArriveMain无法获得到达交通小区编号，得到的是个概率分布，而不是一个确定的交通小区编号
#       or：到达交通小区id用类ZoneNetworkArriveMain的输出概率分布中，最大的那个来表示 √，假设有一个具体的到达小区
# × 不需要这个，因为GCN已经在ZoneNetworkArriveMain中使用过了，已经据此求得了到达交通小区的概率分布，另外还有真实距离信息。这里求时间时，使用zone的结果即可，是一个数据流。（另外，包含距离信息的交通小区的节点嵌入向量=64，不直接作为输入，而是采用GCN（作为GCN的输入），获得其中的邻接拓扑关系）

class Mode2NetworkMain(tf.keras.Model):
    def __init__(self, zone_ids_int, num_features=9, hidden_units=10, num_outputs=6, embedding_dim=64, num_goals=5, num_zones=14, num_districts=3046, num_zone_arrive = 14, num_time_depart = 24, num_time_arrive = 24, num_mode1 = 6):
        super(Mode2NetworkMain, self).__init__()
        self.num_outputs = num_outputs
        self.zone_id_mapping = tf.constant(zone_ids_int, dtype=tf.int32)

        # 交通小区编号的嵌入层，在call中计算出发和到达的交通小区的嵌入向量
        self.district_embedding = layers.Embedding(input_dim=num_districts, output_dim=embedding_dim, name="district_embedding")

        input_dim = num_features + num_time_depart + num_time_arrive + 1 + num_zone_arrive + num_mode1  # 9+24+24+1+14+6=72=78

        # 初始化输入层的权重和偏置
        # 注意：这里我们增加了出行时长及Mode1的预测结果作为额外输入特征
        self.input_weights = self.add_weight(shape=(input_dim, hidden_units),
                                             initializer='random_normal', trainable=True, name="input_weights")
        self.input_bias = self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name="input_bias")

        # 初始化隐藏层的权重和偏置
        self.hidden_weights = [
            self.add_weight(shape=(hidden_units, hidden_units), initializer='random_normal', trainable=True,
                            name=f"hidden_weights_{i}") for i in range(3)]
        self.hidden_biases = [
            self.add_weight(shape=(hidden_units,), initializer='zeros', trainable=True, name=f"hidden_bias_{i}") for i
            in range(3)]

        # 初始化输出层的权重和偏置
        self.output_weights = self.add_weight(shape=(hidden_units, num_outputs), initializer='random_normal',
                                              trainable=True, name="output_weights")
        self.output_bias = self.add_weight(shape=(num_outputs,), initializer='zeros', trainable=True,
                                           name="output_bias")

        # 上下文感知模块
        # self.context_module = ContextAwareModule(num_zones=14)
        self.context_net = tf.keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(input_dim,)),
            layers.Dense(num_outputs * (hidden_units+1))
            # 假设需要调整num_outputs个权重和偏置, num_outputs * 2的设计是为了生成两组调整量：一组用于调整输出层的权重，另一组用于调整输出层的偏置
        ])

    def call(self, inputs, times_depart_main, times_arrive_main, modes1_main, zone_adjacency_matrix):
        """
        :param inputs: 出行者的个体特征
        :param goals_main: 预测的出行目的
        :param origin_district_ids: 出发交通小区的ID
        :param arrive_district_ids: 到达交通小区的ID
        :param times_depart_main: 预测的出发时间段
        :param times_arrive_main: 预测的到达时间段
        :param distances: 包含交通小区间距离信息的邻接矩阵
        :param mode1_predictions: Mode1的预测结果
        :param context_input: 上下文信息
        """
        # # 获取出发和到达交通小区的嵌入向量
        # origin_embeddings = self.district_embedding(origin_district_ids)
        # arrive_embeddings = self.district_embedding(arrive_district_ids)
        #
        # # 计算出发和到达交通小区间的距离（从邻接矩阵中获取）
        # dist = tf.gather_nd(distances, tf.stack([origin_district_ids, arrive_district_ids], axis=1))
        #
        # # 计算出发时间和到达时间之间的时间差（预计的出行时长）
        # travel_time = times_arrive_main - times_depart_main
        #
        # # 处理上下文信息并获得调整参数
        # # adjustment_params = self.context_module(context_input)
        # context_adjustments = self.context_net(arrive_district_ids)
        # adjusted_output_weights = self.output_weights + context_adjustments[:,
        #                                                 :self.output_weights.shape[0] * self.output_weights.shape[
        #                                                     1]].reshape(self.output_weights.shape)
        # adjusted_output_bias = self.output_bias + context_adjustments[:, -self.output_bias.shape[0]:]

        expected_departure_time = calculate_expected_time(times_depart_main)
        expected_arrival_time = calculate_expected_time(times_arrive_main)
        # print('expected_arrival_time', expected_arrival_time)   # (64,)

        # 如果期望时间是基于0-24小时的，则需要处理跨越午夜的情况
        # if expected_arrival_time < expected_departure_time:
        #     expected_arrival_time += 24  # 跨越午夜，增加24小时

        # 对于每个样本，如果 expected_arrival_time < expected_departure_time，增加24小时
        time_difference = tf.where(
            expected_arrival_time < expected_departure_time,
            expected_arrival_time + 24 - expected_departure_time,
            expected_arrival_time - expected_departure_time
        )

        # time_difference = expected_arrival_time - expected_departure_time
        # print('time_difference',time_difference)  # (64,)

        # 将时间差转换为张量并处理
        # 将 time_difference 转换为形状为 (64, 1) 的张量
        time_difference_tensor = tf.expand_dims(time_difference, axis=-1)  # 形状变为 (64, 1)
        # time_difference_tensor = tf.convert_to_tensor([[time_difference]], dtype=tf.float32)
        # print('time_difference_tensor', time_difference_tensor)   # (64, 1)
        # processed_time_diff = self.time_diff_dense_layer(time_difference_tensor)

        # 提取出origin_district_ids
        origin_district_ids = extract_origin_district_ids(inputs)
        # print('origin_district_ids33', origin_district_ids.shape, origin_district_ids)

        # 创建映射字典
        id_to_index = create_id_to_index_mapping(zone_ids_int)

        # 获取距离特征
        distance_features = get_distance_features(origin_district_ids, zone_adjacency_matrix, id_to_index)

        inputs = tf.cast(inputs, dtype=tf.float32)
        times_depart_main = tf.cast(times_depart_main, dtype=tf.float32)  # 确保 zones_arrive_main 是 float32 类型
        times_arrive_main = tf.cast(times_arrive_main, dtype=tf.float32)  # 确保 distance_features 是 float32 类型
        time_difference_tensor = tf.cast(time_difference_tensor, dtype=tf.float32)  # 确保 origin_embeddings 是 float32 类型
        distance_features = tf.cast(distance_features, dtype=tf.float32)  # 确保 arrive_embeddings_weighted 是 float32 类型

        # 将输入、目标张量、出发和到达交通小区的嵌入向量、距离及出发时间拼接，并加入出行时长
        x = tf.concat(
            [inputs, times_depart_main, times_arrive_main, time_difference_tensor, distance_features, modes1_main],
            axis=-1)  # 9+24+24+1+14+6=78

        # 处理上下文信息并获得调整参数
        # adjustment_params = self.context_module(context_input)
        context_adjustments = self.context_net(x)
        # print('context_adjustments', context_adjustments.shape)  # (64, 264)

        weight_adjustments = context_adjustments[:, :self.output_weights.shape[0] * self.output_weights.shape[1]]
        bias_adjustments = context_adjustments[:, self.output_weights.shape[0] * self.output_weights.shape[1]:]

        adjusted_output_weights = self.output_weights + tf.reshape(
            weight_adjustments,
            shape=(-1, self.output_weights.shape[0], self.output_weights.shape[1])
        )[-1]
        # print('adjusted_output_weights', adjusted_output_weights.shape)    #

        adjusted_output_bias = self.output_bias + bias_adjustments  #
        # print('adjusted_output_bias', adjusted_output_bias.shape)   #

        # 自定义输入层计算
        hidden_input = tf.nn.relu(tf.matmul(x, self.input_weights) + self.input_bias)
        # 多个隐藏层的计算
        hidden_output = hidden_input
        for i in range(3):  # 假设有3个隐藏层
            hidden_output = tf.nn.relu(tf.matmul(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])
        # 使用调整后的参数计算输出层
        logits = tf.matmul(hidden_output, adjusted_output_weights) + adjusted_output_bias

        # 应用 Sigmoid 函数
        outputs = tf.nn.softmax(logits)

        return outputs






# class CombinedNetwork(tf.keras.Model):
#     def __init__(self, **kwargs):
#         super(CombinedNetwork, self).__init__(**kwargs)
#         self.goal_network_main = GoalNetworkMain()
#         self.zone_network_depart_main = ZoneNetworkDepartMain(num_features_goal=self.goal_network_main.num_outputs)
#         self.zone_network_arrive_main = ZoneNetworkArriveMain(num_features_goal=self.goal_network_main.num_outputs)
#         self.time_network_depart_main = TimeNetworkDepartMain(num_features_zone=self.zone_network_depart_main.num_outputs)
#         self.time_network_arrive_main = TimeNetworkArriveMain(num_features_zone=self.zone_network_arrive_main.num_outputs)
#         # self.mode1_network_main = Mode1NetworkMain(num_features_time=self.time_network_arrive_main.num_outputs)
#         # self.mode2_network_main = Mode2NetworkMain(num_features_time=self.time_network_arrive_main.num_outputs)
#         # time_network_depart_main 和 time_network_arrive_main 的输出特征数相同
#         num_features_time_combined = self.time_network_depart_main.num_outputs + self.time_network_arrive_main.num_outputs
#         self.mode1_network_main = Mode1NetworkMain(num_features_time=num_features_time_combined)
#         self.mode2_network_main = Mode2NetworkMain(num_features_time=num_features_time_combined)
#
#     def call(self, goal_main_inputs, zone_depart_main_inputs, zone_arrive_main_inputs, time_depart_main_inputs, time_arrive_main_inputs, mode1_main_inputs, mode2_main_inputs):
#         # 生成目标张量
#         goals_main = self.goal_network_main(goal_main_inputs)
#
#         # 调用 ZoneNetworkDepartMain 生成 utility_a_depart_main
#         zones_depart_main = self.zone_network_depart_main(zone_depart_main_inputs, goals_main)
#         # 获取 ZoneNetworkDepartMain 的 utility_a
#         # utility_a_zone_depart_main = self.zone_network_depart_main.utility_a_zone_depart_main_subnetwork_from_goal(goals_main)
#         # 调用 ZoneNetworkArriveMain
#         zones_arrive_main = self.zone_network_arrive_main(zone_arrive_main_inputs, goals_main, zones_depart_main)
#
#         # 调用 timeNetworkDepartMain 生成 utility_a_depart_main
#         times_depart_main = self.time_network_depart_main(time_depart_main_inputs, zones_depart_main)
#         # 获取 timeNetworkDepartMain 的 utility_a
#         # utility_a_time_depart_main = self.time_network_depart_main.utility_a_time_depart_main_subnetwork(zones_depart_main)
#         # 调用 timeNetworkArriveMain
#         times_arrive_main = self.time_network_arrive_main(time_arrive_main_inputs, zones_arrive_main, times_depart_main)
#
#         # 将 times_depart_main 和 times_arrive_main 拼接在一起
#         combined_times = tf.concat([times_depart_main, times_arrive_main], axis=-1)
#         # 调用 Mode1NetworkMain 和 Mode2NetworkMain
#         modes1_main = self.mode1_network_main(mode1_main_inputs, combined_times)
#         # utility_a_modes1_main = self.mode1_network_main.utility_a_mode1_subnetwork(combined_times)
#         modes2_main = self.mode2_network_main(mode2_main_inputs, combined_times, modes1_main)
#
#         return goals_main, zones_depart_main, zones_arrive_main, times_depart_main, times_arrive_main, modes1_main, modes2_main
#
#     def reset_weights(self):
#         for layer in self.layers:
#             if hasattr(layer, 'kernel_initializer'):
#                 layer.kernel.assign(layer.kernel_initializer(shape=layer.kernel.shape))
#             if hasattr(layer, 'bias_initializer'):
#                 layer.bias.assign(layer.bias_initializer(shape=layer.bias.shape))


class CombinedNetwork(tf.keras.Model):
    def __init__(self, zone_adjacency_matrix, node_embeddings, distances, num_zone_arrive=14, embedding_dim=64, num_goals=5, **kwargs):
        super(CombinedNetwork, self).__init__(**kwargs)

        # 初始化各子网络
        self.goal_network_main = GoalNetworkMain()
        self.zone_network_arrive_main = ZoneNetworkArriveMain()
        self.time_network_depart_main = TimeNetworkDepartMain(zone_ids_int)
        self.time_network_arrive_main = TimeNetworkArriveMain(zone_ids_int)
        self.mode1_network_main = Mode1NetworkMain(zone_ids_int)
        self.mode2_network_main = Mode2NetworkMain(zone_ids_int)

        # 定义输入层并保存为类的属性
        self.input_goal = tf.keras.Input(shape=(9,), name="goal_input")
        self.input_zone_arrive = tf.keras.Input(shape=(9,), name="zone_arrive_input")
        self.input_time_depart = tf.keras.Input(shape=(9,), name="time_depart_input")
        self.input_time_arrive = tf.keras.Input(shape=(9,), name="time_arrive_input")
        self.input_mode1 = tf.keras.Input(shape=(9,), name="mode1_input")
        self.input_mode2 = tf.keras.Input(shape=(9,), name="mode2_input")

        # 保存外部变量
        self.zone_adjacency_matrix = zone_adjacency_matrix
        self.node_embeddings = node_embeddings
        self.distances = distances

        # 缓存 outputs 的结果
        self._outputs = None

    def call(self, goal_inputs, zone_arrive_inputs, time_depart_inputs, time_arrive_inputs, mode1_inputs, mode2_inputs):

        # 使用类属性访问全局变量
        node_embeddings = self.node_embeddings
        zone_adjacency_matrix = self.zone_adjacency_matrix
        distances = self.distances
        # 生成目标张量
        goals_main = self.goal_network_main(goal_inputs)

        # 调用到达区域网络
        zones_arrive_main = self.zone_network_arrive_main(zone_arrive_inputs, goals_main, node_embeddings, zone_adjacency_matrix)

        # 根据zones_arrive_main生成times_depart_main
        times_depart_main = self.time_network_depart_main(time_depart_inputs, zones_arrive_main, distances, zone_adjacency_matrix)
        times_arrive_main = self.time_network_arrive_main(time_arrive_inputs, zones_arrive_main, times_depart_main, distances, zone_adjacency_matrix)
        # 根据zones_arrive_main和times_depart_main生成times_arrive_main
        # times_arrive_main = self.time_network_arrive_main(time_arrive_inputs, zones_arrive_main, times_depart_main)
        # 根据zones_arrive_main和times_depart_main生成times_arrive_main
        # times_arrive_main = self.time_network_arrive_main(
        #     time_arrive_inputs,  # 出行者的个体特征或其他输入
        #     goals_main,  # 预测的出行目的
        #     origin_district_ids,  # 出发交通小区的ID
        #     # arrive_district_ids,  # 到达交通小区的ID
        #     times_depart_main,  # 预测的出发时间段
        #     distances  # 包含交通小区间距离信息的邻接矩阵
        # )

        # 使用times_depart_main和times_arrive_main调用mode1_network_main
        modes1_main = self.mode1_network_main(mode1_inputs, times_depart_main, times_arrive_main, zone_adjacency_matrix)

        # 结合times_depart_main, times_arrive_main和modes1_main的结果来调用mode2_network_main
        modes2_main = self.mode2_network_main(mode2_inputs, times_depart_main, times_arrive_main, modes1_main, zone_adjacency_matrix)

        return goals_main, zones_arrive_main, times_depart_main, times_arrive_main, modes1_main, modes2_main

    def export_dynamic_parameters(self, file_name):
        """
        导出所有子网络的动态效用参数。
        """
        # 确保子网络支持动态参数导出
        if hasattr(self.goal_network_main, "export_dynamic_parameters"):
            print('765467823')
            self.goal_network_main.export_dynamic_parameters(
                file_name
            )

        # 调用 GoalNetworkMain 的 compute_input_impact 方法
        input_impact_weights, input_impact_bias = self.goal_network_main.compute_input_impact()

        # 定义保存函数
        def save_to_csv(data, base_dir, file_name, header=None):
            """将数据保存为 CSV 文件"""
            # 拼接完整文件路径
            file_path = os.path.join(base_dir, file_name)

            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # 创建 DataFrame 并保存
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False, header=header)
            print(f"Saved data to {file_path}")

        # 指定目标路径
        base_directory = r"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\dynamic_parameters"

        # 生成文件名
        weights_file_name = os.path.splitext(os.path.basename(file_name))[0] + "_input_to_weights.csv"
        bias_file_name = os.path.splitext(os.path.basename(file_name))[0] + "_input_to_bias.csv"

        # 保存 input_impact_weights
        save_to_csv(
            input_impact_weights,
            base_directory,
            weights_file_name,
            header=[f"Weight_{i}" for i in range(input_impact_weights.shape[1])]
        )

        # 保存 input_impact_bias
        save_to_csv(
            input_impact_bias,
            base_directory,
            bias_file_name,
            header=[f"Bias_{i}" for i in range(input_impact_bias.shape[1])]
        )

    # def export_dynamic_parameters(self, file_name="dynamic_parameters.csv"):
    #     """
    #     导出所有子网络的动态效用参数。
    #
    #     Args:
    #         file_name (str): 输出的 CSV 文件名。
    #     """
    #     # 确保子网络支持动态参数导出
    #     if hasattr(self.goal_network_main, "export_dynamic_parameters"):
    #         self.goal_network_main.export_dynamic_parameters(file_name="goal_network_" + file_name)
    #     # 如果其他子网络也需要导出动态参数，可以类似地调用它们的方法
    #     # 示例：
    #     # if hasattr(self.zone_network_arrive_main, "export_dynamic_parameters"):
    #     #     self.zone_network_arrive_main.export_dynamic_parameters(file_name="zone_network_" + file_name)
    #     else:
    #         print("GoalNetworkMain does not support dynamic parameter export.")
    #
    #     # 导出动态参数
    #     if len(model.goal_network_main.dynamic_params_history) > 0:
    #         model.goal_network_main.export_dynamic_parameters(file_name="dynamic_parameters.csv")
    #     else:
    #         print("No dynamic parameters to export.")


    def reset_weights(self):
        """重置模型权重"""
        for layer in self.layers:
            if hasattr(layer, 'kernel_initializer'):
                initializer = tf.keras.initializers.get(layer.kernel_initializer)
                if isinstance(layer.kernel, tf.Variable):
                    layer.kernel.assign(initializer(shape=layer.kernel.shape))
            if hasattr(layer, 'bias_initializer'):
                initializer = tf.keras.initializers.get(layer.bias_initializer)
                if isinstance(layer.bias, tf.Variable):
                    layer.bias.assign(initializer(shape=layer.bias.shape))

    # # @property
    # def outputs(self, goal_inputs, zone_arrive_inputs, time_depart_inputs, time_arrive_inputs, mode1_inputs, mode2_inputs,
    #          node_embeddings, zone_adjacency_matrix, distances):
    #     """返回模型的输出张量"""
    #     # 调用 call 方法生成输出张量
    #     return self.call(goal_inputs, zone_arrive_inputs, time_depart_inputs, time_arrive_inputs, mode1_inputs, mode2_inputs,
    #          node_embeddings, zone_adjacency_matrix, distances)  # 提取所有输出

    @property
    def inputs(self):
        """返回模型的输入张量"""
        return [
            self.input_goal,
            self.input_zone_arrive,
            self.input_time_depart,
            self.input_time_arrive,
            self.input_mode1,
            self.input_mode2,
            # self.zone_adjacency_matrix,
            # self.node_embeddings,
            # self.distances
        ]

    @property
    def outputs(self):
        """返回模型的符号输出张量"""
        if self._outputs is None:
            # 调用 call 方法生成符号张量
            outputs = self.call(
                *self.inputs,  # 输入张量
                # node_embeddings=self.node_embeddings,  # 外部变量
                # zone_adjacency_matrix=self.zone_adjacency_matrix,  # 外部变量
                # distances=self.distances  # 外部变量
            )

            # 缓存符号张量
            self._outputs = outputs
        return self._outputs  # 提取第一个输出 (goals_main)



class CombinedNetwork_shap_goals_main(tf.keras.Model):
    def __init__(self, zone_adjacency_matrix, node_embeddings, distances, num_zone_arrive=14, embedding_dim=64, num_goals=5, **kwargs):

        super(CombinedNetwork_shap_goals_main, self).__init__(**kwargs)

        # 初始化各子网络
        self.goal_network_main = GoalNetworkMain()
        self.zone_network_arrive_main = ZoneNetworkArriveMain()
        self.time_network_depart_main = TimeNetworkDepartMain(zone_ids_int)
        self.time_network_arrive_main = TimeNetworkArriveMain(zone_ids_int)
        self.mode1_network_main = Mode1NetworkMain(zone_ids_int)
        self.mode2_network_main = Mode2NetworkMain(zone_ids_int)

        # 定义输入层并保存为类的属性
        self.input_goal = tf.keras.Input(shape=(9,), name="goal_input")
        self.input_zone_arrive = tf.keras.Input(shape=(9,), name="zone_arrive_input")
        self.input_time_depart = tf.keras.Input(shape=(9,), name="time_depart_input")
        self.input_time_arrive = tf.keras.Input(shape=(9,), name="time_arrive_input")
        self.input_mode1 = tf.keras.Input(shape=(9,), name="mode1_input")
        self.input_mode2 = tf.keras.Input(shape=(9,), name="mode2_input")

        # 保存外部变量
        self.zone_adjacency_matrix = zone_adjacency_matrix
        self.node_embeddings = node_embeddings
        self.distances = distances

        # 缓存 outputs 的结果
        self._outputs = None


    def call(self, inputs):
        goal_inputs, zone_arrive_inputs, time_depart_inputs, time_arrive_inputs, mode1_inputs, mode2_inputs = inputs

        # 使用类属性访问全局变量（原有逻辑）
        node_embeddings = self.node_embeddings
        zone_adjacency_matrix = self.zone_adjacency_matrix
        distances = self.distances

    # def call(self, goal_inputs, zone_arrive_inputs, time_depart_inputs, time_arrive_inputs, mode1_inputs, mode2_inputs):

        # # 使用类属性访问全局变量
        # node_embeddings = self.node_embeddings
        # zone_adjacency_matrix = self.zone_adjacency_matrix
        # distances = self.distances
        # 生成目标张量
        goals_main = self.goal_network_main(goal_inputs)

        # 调用到达区域网络
        zones_arrive_main = self.zone_network_arrive_main(zone_arrive_inputs, goals_main, node_embeddings, zone_adjacency_matrix)

        # 根据zones_arrive_main生成times_depart_main
        times_depart_main = self.time_network_depart_main(time_depart_inputs, zones_arrive_main, distances, zone_adjacency_matrix)
        times_arrive_main = self.time_network_arrive_main(time_arrive_inputs, zones_arrive_main, times_depart_main, distances, zone_adjacency_matrix)
        # 根据zones_arrive_main和times_depart_main生成times_arrive_main
        # times_arrive_main = self.time_network_arrive_main(time_arrive_inputs, zones_arrive_main, times_depart_main)
        # 根据zones_arrive_main和times_depart_main生成times_arrive_main
        # times_arrive_main = self.time_network_arrive_main(
        #     time_arrive_inputs,  # 出行者的个体特征或其他输入
        #     goals_main,  # 预测的出行目的
        #     origin_district_ids,  # 出发交通小区的ID
        #     # arrive_district_ids,  # 到达交通小区的ID
        #     times_depart_main,  # 预测的出发时间段
        #     distances  # 包含交通小区间距离信息的邻接矩阵
        # )

        # 使用times_depart_main和times_arrive_main调用mode1_network_main
        modes1_main = self.mode1_network_main(mode1_inputs, times_depart_main, times_arrive_main, zone_adjacency_matrix)

        # 结合times_depart_main, times_arrive_main和modes1_main的结果来调用mode2_network_main
        modes2_main = self.mode2_network_main(mode2_inputs, times_depart_main, times_arrive_main, modes1_main, zone_adjacency_matrix)

        return goals_main

    def reset_weights(self):
        """重置模型权重"""
        for layer in self.layers:
            if hasattr(layer, 'kernel_initializer'):
                initializer = tf.keras.initializers.get(layer.kernel_initializer)
                if isinstance(layer.kernel, tf.Variable):
                    layer.kernel.assign(initializer(shape=layer.kernel.shape))
            if hasattr(layer, 'bias_initializer'):
                initializer = tf.keras.initializers.get(layer.bias_initializer)
                if isinstance(layer.bias, tf.Variable):
                    layer.bias.assign(initializer(shape=layer.bias.shape))

class CombinedNetwork_shap_zones_arrive_main(tf.keras.Model):
    def __init__(self, zone_adjacency_matrix, node_embeddings, distances, num_zone_arrive=14, embedding_dim=64, num_goals=5, **kwargs):

        super(CombinedNetwork_shap_zones_arrive_main, self).__init__(**kwargs)

        # 初始化各子网络
        self.goal_network_main = GoalNetworkMain()
        self.zone_network_arrive_main = ZoneNetworkArriveMain()
        self.time_network_depart_main = TimeNetworkDepartMain(zone_ids_int)
        self.time_network_arrive_main = TimeNetworkArriveMain(zone_ids_int)
        self.mode1_network_main = Mode1NetworkMain(zone_ids_int)
        self.mode2_network_main = Mode2NetworkMain(zone_ids_int)

        # 定义输入层并保存为类的属性
        self.input_goal = tf.keras.Input(shape=(9,), name="goal_input")
        self.input_zone_arrive = tf.keras.Input(shape=(9,), name="zone_arrive_input")
        self.input_time_depart = tf.keras.Input(shape=(9,), name="time_depart_input")
        self.input_time_arrive = tf.keras.Input(shape=(9,), name="time_arrive_input")
        self.input_mode1 = tf.keras.Input(shape=(9,), name="mode1_input")
        self.input_mode2 = tf.keras.Input(shape=(9,), name="mode2_input")

        # 保存外部变量
        self.zone_adjacency_matrix = zone_adjacency_matrix
        self.node_embeddings = node_embeddings
        self.distances = distances

        # 缓存 outputs 的结果
        self._outputs = None


    def call(self, inputs):
        goal_inputs, zone_arrive_inputs, time_depart_inputs, time_arrive_inputs, mode1_inputs, mode2_inputs = inputs

        # 使用类属性访问全局变量（原有逻辑）
        node_embeddings = self.node_embeddings
        zone_adjacency_matrix = self.zone_adjacency_matrix
        distances = self.distances

    # def call(self, goal_inputs, zone_arrive_inputs, time_depart_inputs, time_arrive_inputs, mode1_inputs, mode2_inputs):

        # # 使用类属性访问全局变量
        # node_embeddings = self.node_embeddings
        # zone_adjacency_matrix = self.zone_adjacency_matrix
        # distances = self.distances
        # 生成目标张量
        goals_main = self.goal_network_main(goal_inputs)

        # 调用到达区域网络
        zones_arrive_main = self.zone_network_arrive_main(zone_arrive_inputs, goals_main, node_embeddings, zone_adjacency_matrix)

        # 根据zones_arrive_main生成times_depart_main
        times_depart_main = self.time_network_depart_main(time_depart_inputs, zones_arrive_main, distances, zone_adjacency_matrix)
        times_arrive_main = self.time_network_arrive_main(time_arrive_inputs, zones_arrive_main, times_depart_main, distances, zone_adjacency_matrix)
        # 根据zones_arrive_main和times_depart_main生成times_arrive_main
        # times_arrive_main = self.time_network_arrive_main(time_arrive_inputs, zones_arrive_main, times_depart_main)
        # 根据zones_arrive_main和times_depart_main生成times_arrive_main
        # times_arrive_main = self.time_network_arrive_main(
        #     time_arrive_inputs,  # 出行者的个体特征或其他输入
        #     goals_main,  # 预测的出行目的
        #     origin_district_ids,  # 出发交通小区的ID
        #     # arrive_district_ids,  # 到达交通小区的ID
        #     times_depart_main,  # 预测的出发时间段
        #     distances  # 包含交通小区间距离信息的邻接矩阵
        # )

        # 使用times_depart_main和times_arrive_main调用mode1_network_main
        modes1_main = self.mode1_network_main(mode1_inputs, times_depart_main, times_arrive_main, zone_adjacency_matrix)

        # 结合times_depart_main, times_arrive_main和modes1_main的结果来调用mode2_network_main
        modes2_main = self.mode2_network_main(mode2_inputs, times_depart_main, times_arrive_main, modes1_main, zone_adjacency_matrix)

        return goals_main

    def reset_weights(self):
        """重置模型权重"""
        for layer in self.layers:
            if hasattr(layer, 'kernel_initializer'):
                initializer = tf.keras.initializers.get(layer.kernel_initializer)
                if isinstance(layer.kernel, tf.Variable):
                    layer.kernel.assign(initializer(shape=layer.kernel.shape))
            if hasattr(layer, 'bias_initializer'):
                initializer = tf.keras.initializers.get(layer.bias_initializer)
                if isinstance(layer.bias, tf.Variable):
                    layer.bias.assign(initializer(shape=layer.bias.shape))

class CombinedNetwork_shap_times_depart_main(tf.keras.Model):
    def __init__(self, zone_adjacency_matrix, node_embeddings, distances, num_zone_arrive=14, embedding_dim=64, num_goals=5, **kwargs):

        super(CombinedNetwork_shap_times_depart_main, self).__init__(**kwargs)

        # 初始化各子网络
        self.goal_network_main = GoalNetworkMain()
        self.zone_network_arrive_main = ZoneNetworkArriveMain()
        self.time_network_depart_main = TimeNetworkDepartMain(zone_ids_int)
        self.time_network_arrive_main = TimeNetworkArriveMain(zone_ids_int)
        self.mode1_network_main = Mode1NetworkMain(zone_ids_int)
        self.mode2_network_main = Mode2NetworkMain(zone_ids_int)

        # 定义输入层并保存为类的属性
        self.input_goal = tf.keras.Input(shape=(9,), name="goal_input")
        self.input_zone_arrive = tf.keras.Input(shape=(9,), name="zone_arrive_input")
        self.input_time_depart = tf.keras.Input(shape=(9,), name="time_depart_input")
        self.input_time_arrive = tf.keras.Input(shape=(9,), name="time_arrive_input")
        self.input_mode1 = tf.keras.Input(shape=(9,), name="mode1_input")
        self.input_mode2 = tf.keras.Input(shape=(9,), name="mode2_input")

        # 保存外部变量
        self.zone_adjacency_matrix = zone_adjacency_matrix
        self.node_embeddings = node_embeddings
        self.distances = distances

        # 缓存 outputs 的结果
        self._outputs = None


    def call(self, inputs):
        goal_inputs, zone_arrive_inputs, time_depart_inputs, time_arrive_inputs, mode1_inputs, mode2_inputs = inputs

        # 使用类属性访问全局变量（原有逻辑）
        node_embeddings = self.node_embeddings
        zone_adjacency_matrix = self.zone_adjacency_matrix
        distances = self.distances

    # def call(self, goal_inputs, zone_arrive_inputs, time_depart_inputs, time_arrive_inputs, mode1_inputs, mode2_inputs):

        # # 使用类属性访问全局变量
        # node_embeddings = self.node_embeddings
        # zone_adjacency_matrix = self.zone_adjacency_matrix
        # distances = self.distances
        # 生成目标张量
        goals_main = self.goal_network_main(goal_inputs)

        # 调用到达区域网络
        zones_arrive_main = self.zone_network_arrive_main(zone_arrive_inputs, goals_main, node_embeddings, zone_adjacency_matrix)

        # 根据zones_arrive_main生成times_depart_main
        times_depart_main = self.time_network_depart_main(time_depart_inputs, zones_arrive_main, distances, zone_adjacency_matrix)
        times_arrive_main = self.time_network_arrive_main(time_arrive_inputs, zones_arrive_main, times_depart_main, distances, zone_adjacency_matrix)
        # 根据zones_arrive_main和times_depart_main生成times_arrive_main
        # times_arrive_main = self.time_network_arrive_main(time_arrive_inputs, zones_arrive_main, times_depart_main)
        # 根据zones_arrive_main和times_depart_main生成times_arrive_main
        # times_arrive_main = self.time_network_arrive_main(
        #     time_arrive_inputs,  # 出行者的个体特征或其他输入
        #     goals_main,  # 预测的出行目的
        #     origin_district_ids,  # 出发交通小区的ID
        #     # arrive_district_ids,  # 到达交通小区的ID
        #     times_depart_main,  # 预测的出发时间段
        #     distances  # 包含交通小区间距离信息的邻接矩阵
        # )

        # 使用times_depart_main和times_arrive_main调用mode1_network_main
        modes1_main = self.mode1_network_main(mode1_inputs, times_depart_main, times_arrive_main, zone_adjacency_matrix)

        # 结合times_depart_main, times_arrive_main和modes1_main的结果来调用mode2_network_main
        modes2_main = self.mode2_network_main(mode2_inputs, times_depart_main, times_arrive_main, modes1_main, zone_adjacency_matrix)

        return goals_main

    def reset_weights(self):
        """重置模型权重"""
        for layer in self.layers:
            if hasattr(layer, 'kernel_initializer'):
                initializer = tf.keras.initializers.get(layer.kernel_initializer)
                if isinstance(layer.kernel, tf.Variable):
                    layer.kernel.assign(initializer(shape=layer.kernel.shape))
            if hasattr(layer, 'bias_initializer'):
                initializer = tf.keras.initializers.get(layer.bias_initializer)
                if isinstance(layer.bias, tf.Variable):
                    layer.bias.assign(initializer(shape=layer.bias.shape))

class CombinedNetwork_shap_times_arrive_main(tf.keras.Model):
    def __init__(self, zone_adjacency_matrix, node_embeddings, distances, num_zone_arrive=14, embedding_dim=64, num_goals=5, **kwargs):

        super(CombinedNetwork_shap_times_arrive_main, self).__init__(**kwargs)

        # 初始化各子网络
        self.goal_network_main = GoalNetworkMain()
        self.zone_network_arrive_main = ZoneNetworkArriveMain()
        self.time_network_depart_main = TimeNetworkDepartMain(zone_ids_int)
        self.time_network_arrive_main = TimeNetworkArriveMain(zone_ids_int)
        self.mode1_network_main = Mode1NetworkMain(zone_ids_int)
        self.mode2_network_main = Mode2NetworkMain(zone_ids_int)

        # 定义输入层并保存为类的属性
        self.input_goal = tf.keras.Input(shape=(9,), name="goal_input")
        self.input_zone_arrive = tf.keras.Input(shape=(9,), name="zone_arrive_input")
        self.input_time_depart = tf.keras.Input(shape=(9,), name="time_depart_input")
        self.input_time_arrive = tf.keras.Input(shape=(9,), name="time_arrive_input")
        self.input_mode1 = tf.keras.Input(shape=(9,), name="mode1_input")
        self.input_mode2 = tf.keras.Input(shape=(9,), name="mode2_input")

        # 保存外部变量
        self.zone_adjacency_matrix = zone_adjacency_matrix
        self.node_embeddings = node_embeddings
        self.distances = distances

        # 缓存 outputs 的结果
        self._outputs = None


    def call(self, inputs):
        goal_inputs, zone_arrive_inputs, time_depart_inputs, time_arrive_inputs, mode1_inputs, mode2_inputs = inputs

        # 使用类属性访问全局变量（原有逻辑）
        node_embeddings = self.node_embeddings
        zone_adjacency_matrix = self.zone_adjacency_matrix
        distances = self.distances

    # def call(self, goal_inputs, zone_arrive_inputs, time_depart_inputs, time_arrive_inputs, mode1_inputs, mode2_inputs):

        # # 使用类属性访问全局变量
        # node_embeddings = self.node_embeddings
        # zone_adjacency_matrix = self.zone_adjacency_matrix
        # distances = self.distances
        # 生成目标张量
        goals_main = self.goal_network_main(goal_inputs)

        # 调用到达区域网络
        zones_arrive_main = self.zone_network_arrive_main(zone_arrive_inputs, goals_main, node_embeddings, zone_adjacency_matrix)

        # 根据zones_arrive_main生成times_depart_main
        times_depart_main = self.time_network_depart_main(time_depart_inputs, zones_arrive_main, distances, zone_adjacency_matrix)
        times_arrive_main = self.time_network_arrive_main(time_arrive_inputs, zones_arrive_main, times_depart_main, distances, zone_adjacency_matrix)
        # 根据zones_arrive_main和times_depart_main生成times_arrive_main
        # times_arrive_main = self.time_network_arrive_main(time_arrive_inputs, zones_arrive_main, times_depart_main)
        # 根据zones_arrive_main和times_depart_main生成times_arrive_main
        # times_arrive_main = self.time_network_arrive_main(
        #     time_arrive_inputs,  # 出行者的个体特征或其他输入
        #     goals_main,  # 预测的出行目的
        #     origin_district_ids,  # 出发交通小区的ID
        #     # arrive_district_ids,  # 到达交通小区的ID
        #     times_depart_main,  # 预测的出发时间段
        #     distances  # 包含交通小区间距离信息的邻接矩阵
        # )

        # 使用times_depart_main和times_arrive_main调用mode1_network_main
        modes1_main = self.mode1_network_main(mode1_inputs, times_depart_main, times_arrive_main, zone_adjacency_matrix)

        # 结合times_depart_main, times_arrive_main和modes1_main的结果来调用mode2_network_main
        modes2_main = self.mode2_network_main(mode2_inputs, times_depart_main, times_arrive_main, modes1_main, zone_adjacency_matrix)

        return goals_main

    def reset_weights(self):
        """重置模型权重"""
        for layer in self.layers:
            if hasattr(layer, 'kernel_initializer'):
                initializer = tf.keras.initializers.get(layer.kernel_initializer)
                if isinstance(layer.kernel, tf.Variable):
                    layer.kernel.assign(initializer(shape=layer.kernel.shape))
            if hasattr(layer, 'bias_initializer'):
                initializer = tf.keras.initializers.get(layer.bias_initializer)
                if isinstance(layer.bias, tf.Variable):
                    layer.bias.assign(initializer(shape=layer.bias.shape))

class CombinedNetwork_shap_modes1_main(tf.keras.Model):
    def __init__(self, zone_adjacency_matrix, node_embeddings, distances, num_zone_arrive=14, embedding_dim=64, num_goals=5, **kwargs):

        super(CombinedNetwork_shap_modes1_main, self).__init__(**kwargs)

        # 初始化各子网络
        self.goal_network_main = GoalNetworkMain()
        self.zone_network_arrive_main = ZoneNetworkArriveMain()
        self.time_network_depart_main = TimeNetworkDepartMain(zone_ids_int)
        self.time_network_arrive_main = TimeNetworkArriveMain(zone_ids_int)
        self.mode1_network_main = Mode1NetworkMain(zone_ids_int)
        self.mode2_network_main = Mode2NetworkMain(zone_ids_int)

        # 定义输入层并保存为类的属性
        self.input_goal = tf.keras.Input(shape=(9,), name="goal_input")
        self.input_zone_arrive = tf.keras.Input(shape=(9,), name="zone_arrive_input")
        self.input_time_depart = tf.keras.Input(shape=(9,), name="time_depart_input")
        self.input_time_arrive = tf.keras.Input(shape=(9,), name="time_arrive_input")
        self.input_mode1 = tf.keras.Input(shape=(9,), name="mode1_input")
        self.input_mode2 = tf.keras.Input(shape=(9,), name="mode2_input")

        # 保存外部变量
        self.zone_adjacency_matrix = zone_adjacency_matrix
        self.node_embeddings = node_embeddings
        self.distances = distances

        # 缓存 outputs 的结果
        self._outputs = None


    def call(self, inputs):
        goal_inputs, zone_arrive_inputs, time_depart_inputs, time_arrive_inputs, mode1_inputs, mode2_inputs = inputs

        # 使用类属性访问全局变量（原有逻辑）
        node_embeddings = self.node_embeddings
        zone_adjacency_matrix = self.zone_adjacency_matrix
        distances = self.distances

    # def call(self, goal_inputs, zone_arrive_inputs, time_depart_inputs, time_arrive_inputs, mode1_inputs, mode2_inputs):

        # # 使用类属性访问全局变量
        # node_embeddings = self.node_embeddings
        # zone_adjacency_matrix = self.zone_adjacency_matrix
        # distances = self.distances
        # 生成目标张量
        goals_main = self.goal_network_main(goal_inputs)

        # 调用到达区域网络
        zones_arrive_main = self.zone_network_arrive_main(zone_arrive_inputs, goals_main, node_embeddings, zone_adjacency_matrix)

        # 根据zones_arrive_main生成times_depart_main
        times_depart_main = self.time_network_depart_main(time_depart_inputs, zones_arrive_main, distances, zone_adjacency_matrix)
        times_arrive_main = self.time_network_arrive_main(time_arrive_inputs, zones_arrive_main, times_depart_main, distances, zone_adjacency_matrix)
        # 根据zones_arrive_main和times_depart_main生成times_arrive_main
        # times_arrive_main = self.time_network_arrive_main(time_arrive_inputs, zones_arrive_main, times_depart_main)
        # 根据zones_arrive_main和times_depart_main生成times_arrive_main
        # times_arrive_main = self.time_network_arrive_main(
        #     time_arrive_inputs,  # 出行者的个体特征或其他输入
        #     goals_main,  # 预测的出行目的
        #     origin_district_ids,  # 出发交通小区的ID
        #     # arrive_district_ids,  # 到达交通小区的ID
        #     times_depart_main,  # 预测的出发时间段
        #     distances  # 包含交通小区间距离信息的邻接矩阵
        # )

        # 使用times_depart_main和times_arrive_main调用mode1_network_main
        modes1_main = self.mode1_network_main(mode1_inputs, times_depart_main, times_arrive_main, zone_adjacency_matrix)

        # 结合times_depart_main, times_arrive_main和modes1_main的结果来调用mode2_network_main
        modes2_main = self.mode2_network_main(mode2_inputs, times_depart_main, times_arrive_main, modes1_main, zone_adjacency_matrix)

        return goals_main

    def reset_weights(self):
        """重置模型权重"""
        for layer in self.layers:
            if hasattr(layer, 'kernel_initializer'):
                initializer = tf.keras.initializers.get(layer.kernel_initializer)
                if isinstance(layer.kernel, tf.Variable):
                    layer.kernel.assign(initializer(shape=layer.kernel.shape))
            if hasattr(layer, 'bias_initializer'):
                initializer = tf.keras.initializers.get(layer.bias_initializer)
                if isinstance(layer.bias, tf.Variable):
                    layer.bias.assign(initializer(shape=layer.bias.shape))

class CombinedNetwork_shap_modes2_main(tf.keras.Model):
    def __init__(self, zone_adjacency_matrix, node_embeddings, distances, num_zone_arrive=14, embedding_dim=64, num_goals=5, **kwargs):

        super(CombinedNetwork_shap_modes2_main, self).__init__(**kwargs)

        # 初始化各子网络
        self.goal_network_main = GoalNetworkMain()
        self.zone_network_arrive_main = ZoneNetworkArriveMain()
        self.time_network_depart_main = TimeNetworkDepartMain(zone_ids_int)
        self.time_network_arrive_main = TimeNetworkArriveMain(zone_ids_int)
        self.mode1_network_main = Mode1NetworkMain(zone_ids_int)
        self.mode2_network_main = Mode2NetworkMain(zone_ids_int)

        # 定义输入层并保存为类的属性
        self.input_goal = tf.keras.Input(shape=(9,), name="goal_input")
        self.input_zone_arrive = tf.keras.Input(shape=(9,), name="zone_arrive_input")
        self.input_time_depart = tf.keras.Input(shape=(9,), name="time_depart_input")
        self.input_time_arrive = tf.keras.Input(shape=(9,), name="time_arrive_input")
        self.input_mode1 = tf.keras.Input(shape=(9,), name="mode1_input")
        self.input_mode2 = tf.keras.Input(shape=(9,), name="mode2_input")

        # 保存外部变量
        self.zone_adjacency_matrix = zone_adjacency_matrix
        self.node_embeddings = node_embeddings
        self.distances = distances

        # 缓存 outputs 的结果
        self._outputs = None


    def call(self, inputs):
        goal_inputs, zone_arrive_inputs, time_depart_inputs, time_arrive_inputs, mode1_inputs, mode2_inputs = inputs

        # 使用类属性访问全局变量（原有逻辑）
        node_embeddings = self.node_embeddings
        zone_adjacency_matrix = self.zone_adjacency_matrix
        distances = self.distances

    # def call(self, goal_inputs, zone_arrive_inputs, time_depart_inputs, time_arrive_inputs, mode1_inputs, mode2_inputs):

        # # 使用类属性访问全局变量
        # node_embeddings = self.node_embeddings
        # zone_adjacency_matrix = self.zone_adjacency_matrix
        # distances = self.distances
        # 生成目标张量
        goals_main = self.goal_network_main(goal_inputs)

        # 调用到达区域网络
        zones_arrive_main = self.zone_network_arrive_main(zone_arrive_inputs, goals_main, node_embeddings, zone_adjacency_matrix)

        # 根据zones_arrive_main生成times_depart_main
        times_depart_main = self.time_network_depart_main(time_depart_inputs, zones_arrive_main, distances, zone_adjacency_matrix)
        times_arrive_main = self.time_network_arrive_main(time_arrive_inputs, zones_arrive_main, times_depart_main, distances, zone_adjacency_matrix)
        # 根据zones_arrive_main和times_depart_main生成times_arrive_main
        # times_arrive_main = self.time_network_arrive_main(time_arrive_inputs, zones_arrive_main, times_depart_main)
        # 根据zones_arrive_main和times_depart_main生成times_arrive_main
        # times_arrive_main = self.time_network_arrive_main(
        #     time_arrive_inputs,  # 出行者的个体特征或其他输入
        #     goals_main,  # 预测的出行目的
        #     origin_district_ids,  # 出发交通小区的ID
        #     # arrive_district_ids,  # 到达交通小区的ID
        #     times_depart_main,  # 预测的出发时间段
        #     distances  # 包含交通小区间距离信息的邻接矩阵
        # )

        # 使用times_depart_main和times_arrive_main调用mode1_network_main
        modes1_main = self.mode1_network_main(mode1_inputs, times_depart_main, times_arrive_main, zone_adjacency_matrix)

        # 结合times_depart_main, times_arrive_main和modes1_main的结果来调用mode2_network_main
        modes2_main = self.mode2_network_main(mode2_inputs, times_depart_main, times_arrive_main, modes1_main, zone_adjacency_matrix)

        return goals_main

    def reset_weights(self):
        """重置模型权重"""
        for layer in self.layers:
            if hasattr(layer, 'kernel_initializer'):
                initializer = tf.keras.initializers.get(layer.kernel_initializer)
                if isinstance(layer.kernel, tf.Variable):
                    layer.kernel.assign(initializer(shape=layer.kernel.shape))
            if hasattr(layer, 'bias_initializer'):
                initializer = tf.keras.initializers.get(layer.bias_initializer)
                if isinstance(layer.bias, tf.Variable):
                    layer.bias.assign(initializer(shape=layer.bias.shape))




# 加载数据：读取CSV文件并提取特征和标签。
# 划分数据集：将数据集划分为训练集和测试集。
# 10次交叉验证：在训练集上进行10次交叉验证。（训练集=训练集+验证集）
# 使用all所有训练数据重新训练模型：使用最佳超参数和所有训练数据重新训练模型。
# 评估最终模型：在独立的测试集上评估最终模型的性能。

def load_data(batch_size):
    features_name = ['Gender', 'Age', 'Job', 'Homesize', 'HomeStudentJunior', 'HomeCar', 'IncomeMonth', 'District', 'ZoneDepartMain']

    goals_main = ['WorkMain', 'SchoolMain', 'VisitMain', 'ShoppingMain', 'OtherGoalMain']

    # # 定义模板列表
    # goals_templates = [
    #     'WorkSub{}', 'SchoolSub{}', 'VisitSub{}', 'ShoppingSub{}',
    #     'OutSub{}', 'WorkBackSub{}', 'HomeBackSub{}', 'OtherGoalSub{}'
    # ]
    # # 指定 i 的范围
    # i_range = range(1, 6)  # i 从 1 到 5
    # # 初始化一个空列表来存储生成的列名
    # goals_sub = []
    # # 循环生成所有可能的子目标列名
    # for template in goals_templates:
    #     for i in i_range:
    #         goals_sub.append(template.format(i))

    zones_depart_main = []
    for i in range(1, 3047):  # 从1到3046，因为range的结束索引是不包含的
        zones_depart_main.append('ZoneDepartMain' + str(i))

    zones_arrive_main = []
    for i in range(1, 3047):  # 从1到3046，因为range的结束索引是不包含的
        zones_arrive_main.append('ZoneArriveMain' + str(i))
    # print(len(zones_arrive_main),'len(zones_arrive_main)')   # 3046

    times_depart_main = []
    for i in range(1, 25):  # 从1到10，因为range的结束索引是不包含的
        times_depart_main.append('TimeDepartMain' + str(i))

    times_arrive_main = []
    for i in range(1, 25):  # 从1到10，因为range的结束索引是不包含的
        times_arrive_main.append('TimeArriveMain' + str(i))

    modes1_main = ['Car1Main', 'Bus1Main', 'Metro1Main', 'Taxi1Main', 'Bike1Main', 'Walk1Main']
    modes2_main = ['Car2Main', 'Bus2Main', 'Metro2Main', 'Taxi2Main', 'Bike2Main', 'Walk2Main']

    # 读取 CSV 文件
    file_path = 'Household survey_Person4_10_classify_prob.csv'
    df = pd.read_csv(file_path)

    # 检查数据长度是否一致
    assert len(df) == len(df[features_name]), "Features and labels length mismatch."

    # 提取特征和标签
    features_x = df[['Gender', 'Age', 'Job', 'Homesize', 'HomeStudentJunior', 'HomeCar', 'IncomeMonth', 'District', 'ZoneDepartMain']]
    
    labels_goals_main_name = [goal + '_prob' for goal in goals_main]
    labels_goals_main = df[labels_goals_main_name]

    # labels_zones_depart_main_name = [zone + '_prob' for zone in zones_depart_main]
    # if labels_zones_depart_main_name in df.columns:
    #     labels_zones_depart_main = df[labels_zones_depart_main_name]
    #
    # labels_zones_arrive_main_name = [zone + '_prob' for zone in zones_arrive_main]
    # if labels_zones_arrive_main_name in df.columns:
    #     labels_zones_arrive_main = df[labels_zones_arrive_main_name]

    # 初始化一个空列表来保存存在的列
    labels_zones_depart_main = pd.DataFrame()
    # 单独检查每一列
    labels_zones_depart_main_name = [zone + '_prob' for zone in zones_depart_main]
    # for name in labels_zones_depart_main_name:
    #     if name in df.columns:
    #         labels_zones_depart_main[name] = df[name]
    for name in labels_zones_depart_main_name:
        # if name in df.columns:
        labels_zones_depart_main[name] = df[name]
    # print(labels_zones_depart_main.shape,'labels_zones_depart_main.shape')   # (702, 82)

    # 获取 labels_zones_depart_main 的第二维大小
    # num_outputs_zonenetworkdepartmain = labels_zones_depart_main.shape[1]


    # labels_zones_depart_main_name = [zone + '_prob' for zone in zones_depart_main]
    # for name in labels_zones_depart_main_name:
    #     if name in df.columns:
    #         labels_zones_depart_main.append(df[name])
    # # 将列表中的元素转换为DataFrame
    # if labels_zones_depart_main:
    #     labels_zones_depart_main = pd.concat(labels_zones_depart_main, axis=1)

    # 对到达区域做同样的检查
    labels_zones_arrive_main = pd.DataFrame()
    labels_zones_arrive_main_name = [zone + '_prob' for zone in zones_arrive_main]
    for name in labels_zones_arrive_main_name:
        if name in df.columns:
            # labels_zones_arrive_main.append(df[name])
            labels_zones_arrive_main[name] = df[name]
    # print(labels_zones_arrive_main.shape,'labels_zones_arrive_main.shape')   # (702, 3046)
    # if labels_zones_arrive_main:
    #     labels_zones_arrive_main = pd.concat(labels_zones_arrive_main, axis=1)


    labels_times_depart_main_name = [time + '_prob' for time in times_depart_main]
    labels_times_depart_main = df[labels_times_depart_main_name]

    labels_times_arrive_main_name = [time + '_prob' for time in times_arrive_main]
    labels_times_arrive_main = df[labels_times_arrive_main_name]

    labels_mode1_main_name = [mode + '_prob' for mode in modes1_main]
    labels_mode1_main = df[labels_mode1_main_name]

    labels_mode2_main_name = [mode + '_prob' for mode in modes2_main]
    labels_mode2_main = df[labels_mode2_main_name]

    # labels_av_goal_name = ['av' + '_' + goal for goal in goals]
    # labels_av_goal = df[labels_av_goal_name]
    #
    # labels_av_zone_name = ['av' + '_' + zone for zone in zones]
    # labels_av_zone = df[labels_av_zone_name]
    #
    # labels_av_time_name = ['av' + '_' + time for time in times]
    # labels_av_time = df[labels_av_time_name]
    #
    # labels_av_mode_name = ['av' + '_' + mode for mode in modes]
    # labels_av_mode = df[labels_av_mode_name]

    # 检查特征和标签的数量是否一致
    assert len(features_x) == len(labels_goals_main), "Features and labels length mismatch."

    # 将 Pandas DataFrame 转换为 NumPy 数组
    features_x_array = features_x.to_numpy().astype(np.float32)

    labels_goals_main_array = labels_goals_main.to_numpy().astype(np.float32)  # 保持二维数组
    # print(labels_goals_main_array.shape,'labels_goals_main_array.shape')   # (702, 5)

    labels_zones_depart_main_array = labels_zones_depart_main.to_numpy().astype(np.float32)
    labels_zones_arrive_main_array = labels_zones_arrive_main.to_numpy().astype(np.float32)


    # # 检查 labels_zones_depart_main 是否为空
    # if labels_zones_depart_main.empty:
    #     labels_zones_depart_main_array = np.zeros((len(features_x), len(zones_depart_main)), dtype=np.float32)
    # else:
    #     labels_zones_depart_main_array = labels_zones_depart_main.to_numpy().astype(np.float32)
    # print(labels_zones_depart_main_array.shape,'labels_zones_depart_main_array.shape')    # (702, 82)
    #
    # if labels_zones_arrive_main.empty:
    #     labels_zones_arrive_main_array = np.zeros((len(features_x), len(zones_arrive_main)), dtype=np.float32)
    #     print(len(zones_arrive_main),'len(zones_arrive_main)')
    # else:
    #     labels_zones_arrive_main_array = labels_zones_arrive_main.to_numpy().astype(np.float32)
    # # print(labels_zones_arrive_main_array.shape, 'labels_zones_arrive_main_array.shape')   # (702, 6)

    labels_times_depart_main_array = labels_times_depart_main.to_numpy().astype(np.float32)
    # print(labels_times_depart_main_array.shape,'labels_times_depart_main_array.shape')   # (702, 24)
    labels_times_arrive_main_array = labels_times_arrive_main.to_numpy().astype(np.float32)
    labels_mode1_main_array = labels_mode1_main.to_numpy().astype(np.float32)
    labels_mode2_main_array = labels_mode2_main.to_numpy().astype(np.float32)
    # print(labels_goals_main,'labels_goals_main')
    # print(labels_goals_main_array,'labels_goals_main_array')

    # labels_av_goal_array = labels_av_goal.to_numpy().astype(float)
    # labels_av_zone_array = labels_av_zone.to_numpy().astype(float)
    # labels_av_time_array = labels_av_time.to_numpy().astype(float)
    # labels_av_mode_array = labels_av_mode.to_numpy().astype(float)

    # 创建 TensorFlow 数据集
    dataset_goal_main = tf.data.Dataset.from_tensor_slices((features_x_array, labels_goals_main_array))    # x & y
    dataset_zones_depart_main = tf.data.Dataset.from_tensor_slices((features_x_array, labels_zones_depart_main_array))
    dataset_zones_arrive_main = tf.data.Dataset.from_tensor_slices((features_x_array, labels_zones_arrive_main_array))
    dataset_times_depart_main = tf.data.Dataset.from_tensor_slices((features_x_array, labels_times_depart_main_array))
    dataset_times_arrive_main = tf.data.Dataset.from_tensor_slices((features_x_array, labels_times_arrive_main_array))
    dataset_mode1_main = tf.data.Dataset.from_tensor_slices((features_x_array, labels_mode1_main_array))
    dataset_mode2_main = tf.data.Dataset.from_tensor_slices((features_x_array, labels_mode2_main_array))

    # print(dataset_goal_main,'dataset_goal_main1')

    # 设置训练集和验证集
    # 获取数据集的大小
    num_samples = tf.data.experimental.cardinality(dataset_goal_main).numpy()
    # 计算训练集和验证集的大小
    train_size = int(0.8 * num_samples)
    validation_size = num_samples - train_size
    # print(num_samples,'num_samples')   # 702
    # print(train_size, validation_size)      # 561 141

    # 使用 `take()` 和 `skip()` 来分割数据集
    train_dataset_goal_main = dataset_goal_main.take(train_size)
    validation_dataset_goal_main = dataset_goal_main.skip(train_size).take(validation_size)
    
    train_dataset_zones_depart_main = dataset_zones_depart_main.take(train_size)
    validation_dataset_zones_depart_main = dataset_zones_depart_main.skip(train_size).take(validation_size)

    train_dataset_zones_arrive_main = dataset_zones_arrive_main.take(train_size)
    validation_dataset_zones_arrive_main = dataset_zones_arrive_main.skip(train_size).take(validation_size)

    train_dataset_times_depart_main = dataset_times_depart_main.take(train_size)
    validation_dataset_times_depart_main = dataset_times_depart_main.skip(train_size).take(validation_size)

    train_dataset_times_arrive_main = dataset_times_arrive_main.take(train_size)
    validation_dataset_times_arrive_main = dataset_times_arrive_main.skip(train_size).take(validation_size)

    train_dataset_mode1_main = dataset_mode1_main.take(train_size)
    validation_dataset_mode1_main = dataset_mode1_main.skip(train_size).take(validation_size)

    train_dataset_mode2_main = dataset_mode2_main.take(train_size)
    validation_dataset_mode2_main = dataset_mode2_main.skip(train_size).take(validation_size)


    # 设置批量大小，批次大小（batch size）是指每个批次中包含的样本数
    # batch_size = 64
    # 0   # 700+个样本数据，100个批次，每个批次包含700+/100=8个样本，每个epoch（周期）迭代8次
    # 较大的批次大小，通常能提高训练速度，但可能会牺牲一定的模型性能，准确性不好。较大的批次大小需要更多的内存和计算资源。
    # 总迭代次数 = 数据集总样本数 / 批次大小 × 周期次数epoch


    # 批量处理数据，并启用自动预加载
    train_dataset_goal_main = train_dataset_goal_main.shuffle(buffer_size=len(features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    train_dataset_zones_depart_main = train_dataset_zones_depart_main.shuffle(buffer_size=len(features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    train_dataset_zones_arrive_main = train_dataset_zones_arrive_main.shuffle(buffer_size=len(features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    train_dataset_times_depart_main = train_dataset_times_depart_main.shuffle(buffer_size=len(features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    train_dataset_times_arrive_main = train_dataset_times_arrive_main.shuffle(buffer_size=len(features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    train_dataset_mode1_main = train_dataset_mode1_main.shuffle(buffer_size=len(features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    train_dataset_mode2_main = train_dataset_mode2_main.shuffle(buffer_size=len(features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    validation_dataset_goal_main = validation_dataset_goal_main.shuffle(buffer_size=len(features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    validation_dataset_zones_depart_main = validation_dataset_zones_depart_main.shuffle(buffer_size=len(features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    validation_dataset_zones_arrive_main = validation_dataset_zones_arrive_main.shuffle(buffer_size=len(features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    validation_dataset_times_depart_main = validation_dataset_times_depart_main.shuffle(buffer_size=len(features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    validation_dataset_times_arrive_main = validation_dataset_times_arrive_main.shuffle(buffer_size=len(features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    validation_dataset_mode1_main = validation_dataset_mode1_main.shuffle(buffer_size=len(features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    validation_dataset_mode2_main = validation_dataset_mode2_main.shuffle(buffer_size=len(features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset_goal_main, train_dataset_zones_arrive_main, train_dataset_times_depart_main, train_dataset_times_arrive_main, train_dataset_mode1_main, train_dataset_mode2_main, validation_dataset_goal_main, validation_dataset_zones_arrive_main, validation_dataset_times_depart_main, validation_dataset_times_arrive_main, validation_dataset_mode1_main, validation_dataset_mode2_main


def load_data_train_trainvalidation():
    features_name = ['Gender', 'Age', 'Job', 'Homesize', 'HomeStudentJunior', 'HomeCar', 'IncomeMonth', 'District', 'ZoneDepartMain']

    # features_name_zonearrive = ['Gender', 'Age', 'Job', 'Homesize', 'HomeStudentJunior', 'HomeCar', 'IncomeMonth', 'District', 'ZoneDepartMain']


    goals_main = ['WorkMain', 'SchoolMain', 'VisitMain', 'ShoppingMain', 'OtherGoalMain']

    zones_depart_main = []
    for i in range(1, 3047):  # 从1到3046，因为range的结束索引是不包含的
        zones_depart_main.append('ZoneDepartMain' + str(i))

    zones_arrive_main = []
    for i in range(1, 3047):  # 从1到3046，因为range的结束索引是不包含的
        zones_arrive_main.append('ZoneArriveMain' + str(i))
    # print(len(zones_arrive_main),'len(zones_arrive_main)')   # 3046

    times_depart_main = []
    for i in range(1, 25):  # 从1到10，因为range的结束索引是不包含的
        times_depart_main.append('TimeDepartMain' + str(i))

    times_arrive_main = []
    for i in range(1, 25):  # 从1到10，因为range的结束索引是不包含的
        times_arrive_main.append('TimeArriveMain' + str(i))

    modes1_main = ['Car1Main', 'Bus1Main', 'Metro1Main', 'Taxi1Main', 'Bike1Main', 'Walk1Main']
    modes2_main = ['Car2Main', 'Bus2Main', 'Metro2Main', 'Taxi2Main', 'Bike2Main', 'Walk2Main']

    # 读取 CSV 文件
    file_path = 'Household survey_Person4_10_classify_prob.csv'
    df = pd.read_csv(file_path)

    # 检查数据长度是否一致
    assert len(df) == len(df[features_name]), "Features and labels length mismatch."

    # 提取特征和标签
    features_x = df[['Gender', 'Age', 'Job', 'Homesize', 'HomeStudentJunior', 'HomeCar', 'IncomeMonth', 'District', 'ZoneDepartMain']]

    # features_x_zonearrive = df[['Gender', 'Age', 'Job', 'Homesize', 'HomeStudentJunior', 'HomeCar', 'IncomeMonth', 'District', 'ZoneDepartMain']]

    labels_goals_main_name = [goal + '_prob' for goal in goals_main]
    labels_goals_main = df[labels_goals_main_name]

    # 初始化一个空列表来保存存在的列
    labels_zones_depart_main = pd.DataFrame()
    # 单独检查每一列
    labels_zones_depart_main_name = [zone + '_prob' for zone in zones_depart_main]

    for name in labels_zones_depart_main_name:
        # if name in df.columns:
        labels_zones_depart_main[name] = df[name]

    # 对到达区域做同样的检查
    labels_zones_arrive_main = pd.DataFrame()
    labels_zones_arrive_main_name = [zone + '_prob' for zone in zones_arrive_main]
    for name in labels_zones_arrive_main_name:
        if name in df.columns:
            # labels_zones_arrive_main.append(df[name])
            labels_zones_arrive_main[name] = df[name]

    labels_times_depart_main_name = [time + '_prob' for time in times_depart_main]
    labels_times_depart_main = df[labels_times_depart_main_name]

    labels_times_arrive_main_name = [time + '_prob' for time in times_arrive_main]
    labels_times_arrive_main = df[labels_times_arrive_main_name]

    labels_mode1_main_name = [mode + '_prob' for mode in modes1_main]
    labels_mode1_main = df[labels_mode1_main_name]

    labels_mode2_main_name = [mode + '_prob' for mode in modes2_main]
    labels_mode2_main = df[labels_mode2_main_name]

    # 检查特征和标签的数量是否一致
    assert len(features_x) == len(labels_goals_main), "Features and labels length mismatch."

    # 将 Pandas DataFrame 转换为 NumPy 数组
    features_x_array = features_x.to_numpy().astype(np.float32)

    labels_goals_main_array = labels_goals_main.to_numpy().astype(np.float32)  # 保持二维数组
    labels_zones_depart_main_array = labels_zones_depart_main.to_numpy().astype(np.float32)
    labels_zones_arrive_main_array = labels_zones_arrive_main.to_numpy().astype(np.float32)
    labels_times_depart_main_array = labels_times_depart_main.to_numpy().astype(np.float32)
    labels_times_arrive_main_array = labels_times_arrive_main.to_numpy().astype(np.float32)
    labels_mode1_main_array = labels_mode1_main.to_numpy().astype(np.float32)
    labels_mode2_main_array = labels_mode2_main.to_numpy().astype(np.float32)

    # 划分训练集和测试集，其中测试集testing data留着最后测试用，训练集下面会再划分为训练集和验证集用于10次交叉验证
    test_size = 0.2   # 测试集占20%
    random_state = 42

    # 以下所有与test有关的都无需在训练和验证时考虑
    X_train_features_x_array, X_test_features_x_array, y_trainvalidation_goals_main, y_test_goals_main = train_test_split(features_x_array, labels_goals_main_array, test_size=test_size, random_state=random_state)

    _, _, y_trainvalidation_zones_depart_main, y_test_zones_depart_main = train_test_split(features_x_array, labels_zones_depart_main_array, test_size=test_size, random_state=random_state)
    _, _, y_trainvalidation_zones_arrive_main, y_test_zones_arrive_main = train_test_split(features_x_array, labels_zones_arrive_main_array, test_size=test_size, random_state=random_state)

    _, _, y_trainvalidation_times_depart_main, y_test_times_depart_main = train_test_split(features_x_array, labels_times_depart_main_array, test_size=test_size, random_state=random_state)
    _, _, y_trainvalidation_times_arrive_main, y_test_times_arrive_main = train_test_split(features_x_array, labels_times_arrive_main_array, test_size=test_size, random_state=random_state)

    _, _, y_trainvalidation_mode1_main, y_test_mode1_main = train_test_split(features_x_array, labels_mode1_main_array, test_size=test_size, random_state=random_state)
    _, _, y_trainvalidation_mode2_main, y_test_mode2_main = train_test_split(features_x_array, labels_mode2_main_array, test_size=test_size, random_state=random_state)


    datasets = (X_train_features_x_array, X_test_features_x_array, y_trainvalidation_goals_main, y_test_goals_main,
                y_trainvalidation_zones_arrive_main, y_test_zones_arrive_main,
                y_trainvalidation_times_depart_main, y_test_times_depart_main,
                y_trainvalidation_times_arrive_main, y_test_times_arrive_main,
                y_trainvalidation_mode1_main, y_test_mode1_main,
                y_trainvalidation_mode2_main, y_test_mode2_main)

    # print(f"Returning {len(datasets)} datasets")  # 添加调试信息
    # assert len(datasets) == 30, "Not enough datasets returned"

    return datasets



def load_data_Kfold_cross_validation_no(batch_size, n_splits=10):
    features_name = ['Gender', 'Age', 'Job', 'Homesize', 'HomeStudentJunior', 'HomeCar', 'IncomeMonth', 'District', 'ZoneDepartMain']

    goals_main = ['WorkMain', 'SchoolMain', 'VisitMain', 'ShoppingMain', 'OtherGoalMain']

    zones_depart_main = []
    for i in range(1, 3047):  # 从1到3046，因为range的结束索引是不包含的
        zones_depart_main.append('ZoneDepartMain' + str(i))

    zones_arrive_main = []
    for i in range(1, 3047):  # 从1到3046，因为range的结束索引是不包含的
        zones_arrive_main.append('ZoneArriveMain' + str(i))
    # print(len(zones_arrive_main),'len(zones_arrive_main)')   # 3046

    times_depart_main = []
    for i in range(1, 25):  # 从1到10，因为range的结束索引是不包含的
        times_depart_main.append('TimeDepartMain' + str(i))

    times_arrive_main = []
    for i in range(1, 25):  # 从1到10，因为range的结束索引是不包含的
        times_arrive_main.append('TimeArriveMain' + str(i))

    modes1_main = ['Car1Main', 'Bus1Main', 'Metro1Main', 'Taxi1Main', 'Bike1Main', 'Walk1Main']
    modes2_main = ['Car2Main', 'Bus2Main', 'Metro2Main', 'Taxi2Main', 'Bike2Main', 'Walk2Main']

    # 读取 CSV 文件
    file_path = 'Household survey_Person4_10_classify_prob.csv'
    df = pd.read_csv(file_path)

    # 检查数据长度是否一致
    assert len(df) == len(df[features_name]), "Features and labels length mismatch."

    # 提取特征和标签
    features_x = df[['Gender', 'Age', 'Job', 'Homesize', 'HomeStudentJunior', 'HomeCar', 'IncomeMonth', 'District', 'ZoneDepartMain']]

    labels_goals_main_name = [goal + '_prob' for goal in goals_main]
    labels_goals_main = df[labels_goals_main_name]

    # 初始化一个空列表来保存存在的列
    labels_zones_depart_main = pd.DataFrame()
    # 单独检查每一列
    labels_zones_depart_main_name = [zone + '_prob' for zone in zones_depart_main]

    for name in labels_zones_depart_main_name:
        # if name in df.columns:
        labels_zones_depart_main[name] = df[name]

    # 对到达区域做同样的检查
    labels_zones_arrive_main = pd.DataFrame()
    labels_zones_arrive_main_name = [zone + '_prob' for zone in zones_arrive_main]
    for name in labels_zones_arrive_main_name:
        if name in df.columns:
            # labels_zones_arrive_main.append(df[name])
            labels_zones_arrive_main[name] = df[name]

    labels_times_depart_main_name = [time + '_prob' for time in times_depart_main]
    labels_times_depart_main = df[labels_times_depart_main_name]

    labels_times_arrive_main_name = [time + '_prob' for time in times_arrive_main]
    labels_times_arrive_main = df[labels_times_arrive_main_name]

    labels_mode1_main_name = [mode + '_prob' for mode in modes1_main]
    labels_mode1_main = df[labels_mode1_main_name]

    labels_mode2_main_name = [mode + '_prob' for mode in modes2_main]
    labels_mode2_main = df[labels_mode2_main_name]

    # 检查特征和标签的数量是否一致
    assert len(features_x) == len(labels_goals_main), "Features and labels length mismatch."

    # 将 Pandas DataFrame 转换为 NumPy 数组
    features_x_array = features_x.to_numpy().astype(np.float32)

    labels_goals_main_array = labels_goals_main.to_numpy().astype(np.float32)  # 保持二维数组
    labels_zones_depart_main_array = labels_zones_depart_main.to_numpy().astype(np.float32)
    labels_zones_arrive_main_array = labels_zones_arrive_main.to_numpy().astype(np.float32)
    labels_times_depart_main_array = labels_times_depart_main.to_numpy().astype(np.float32)
    labels_times_arrive_main_array = labels_times_arrive_main.to_numpy().astype(np.float32)
    labels_mode1_main_array = labels_mode1_main.to_numpy().astype(np.float32)
    labels_mode2_main_array = labels_mode2_main.to_numpy().astype(np.float32)

    # 划分训练集和测试集，其中测试集testing data留着最后测试用，训练集下面会再划分为训练集和验证集用于10次交叉验证
    test_size = 0.2   # 测试集占20%
    random_state = 42

    # 以下所有与test有关的都无需在训练和验证时考虑
    X_train_features_x_array, X_test_features_x_array, y_trainvalidation_goals_main, y_test_goals_main = train_test_split(features_x_array, labels_goals_main_array, test_size=test_size, random_state=random_state)

    _, _, y_trainvalidation_zones_depart_main, y_test_zones_depart_main = train_test_split(features_x_array, labels_zones_depart_main_array, test_size=test_size, random_state=random_state)
    _, _, y_trainvalidation_zones_arrive_main, y_test_zones_arrive_main = train_test_split(features_x_array, labels_zones_arrive_main_array, test_size=test_size, random_state=random_state)

    _, _, y_trainvalidation_times_depart_main, y_test_times_depart_main = train_test_split(features_x_array, labels_times_depart_main_array, test_size=test_size, random_state=random_state)
    _, _, y_trainvalidation_times_arrive_main, y_test_times_arrive_main = train_test_split(features_x_array, labels_times_arrive_main_array, test_size=test_size, random_state=random_state)

    _, _, y_trainvalidation_mode1_main, y_test_mode1_main = train_test_split(features_x_array, labels_mode1_main_array, test_size=test_size, random_state=random_state)
    _, _, y_trainvalidation_mode2_main, y_test_mode2_main = train_test_split(features_x_array, labels_mode2_main_array, test_size=test_size, random_state=random_state)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    datasets = []

    for train_index, validation_index in kf.split(X_train_features_x_array):
        X_train, X_val = X_train_features_x_array[train_index], X_train_features_x_array[validation_index]
        y_train_goal_main, y_validation_goal_main = y_trainvalidation_goals_main[train_index], y_trainvalidation_goals_main[validation_index]
        y_train_zones_depart_main, y_validation_zones_depart_main = y_trainvalidation_zones_depart_main[train_index], y_trainvalidation_zones_depart_main[validation_index]
        y_train_zones_arrive_main, y_validation_zones_arrive_main = y_trainvalidation_zones_arrive_main[train_index], y_trainvalidation_zones_arrive_main[validation_index]
        y_train_times_depart_main, y_validation_times_depart_main = y_trainvalidation_times_depart_main[train_index], y_trainvalidation_times_depart_main[validation_index]
        y_train_times_arrive_main, y_validation_times_arrive_main = y_trainvalidation_times_arrive_main[train_index], y_trainvalidation_times_arrive_main[validation_index]
        y_train_mode1_main, y_validation_mode1_main = y_trainvalidation_mode1_main[train_index], y_trainvalidation_mode1_main[validation_index]
        y_train_mode2_main, y_validation_mode2_main = y_trainvalidation_mode2_main[train_index], y_trainvalidation_mode2_main[validation_index]

        train_dataset_goal_main = tf.data.Dataset.from_tensor_slices((X_train, y_train_goal_main)).shuffle(buffer_size=len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        validation_dataset_goal_main = tf.data.Dataset.from_tensor_slices((X_val, y_validation_goal_main)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        train_dataset_zones_depart_main = tf.data.Dataset.from_tensor_slices((X_train, y_train_zones_depart_main)).shuffle(buffer_size=len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        validation_dataset_zones_depart_main = tf.data.Dataset.from_tensor_slices((X_val, y_validation_zones_depart_main)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        train_dataset_zones_arrive_main = tf.data.Dataset.from_tensor_slices((X_train, y_train_zones_arrive_main)).shuffle(buffer_size=len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        validation_dataset_zones_arrive_main = tf.data.Dataset.from_tensor_slices((X_val, y_validation_zones_arrive_main)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        train_dataset_times_depart_main = tf.data.Dataset.from_tensor_slices((X_train, y_train_times_depart_main)).shuffle(buffer_size=len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        validation_dataset_times_depart_main = tf.data.Dataset.from_tensor_slices((X_val, y_validation_times_depart_main)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        train_dataset_times_arrive_main = tf.data.Dataset.from_tensor_slices((X_train, y_train_times_arrive_main)).shuffle(buffer_size=len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        validation_dataset_times_arrive_main = tf.data.Dataset.from_tensor_slices((X_val, y_validation_times_arrive_main)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        train_dataset_mode1_main = tf.data.Dataset.from_tensor_slices((X_train, y_train_mode1_main)).shuffle(buffer_size=len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        validation_dataset_mode1_main = tf.data.Dataset.from_tensor_slices((X_val, y_validation_mode1_main)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        train_dataset_mode2_main = tf.data.Dataset.from_tensor_slices((X_train, y_train_mode2_main)).shuffle(buffer_size=len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        validation_dataset_mode2_main = tf.data.Dataset.from_tensor_slices((X_val, y_validation_mode2_main)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # 确保返回14个数据集
        datasets = (train_dataset_goal_main, train_dataset_zones_arrive_main,
                    train_dataset_times_depart_main, train_dataset_times_arrive_main, train_dataset_mode1_main,
                    train_dataset_mode2_main,
                    validation_dataset_goal_main, validation_dataset_zones_arrive_main,
                    validation_dataset_times_depart_main, validation_dataset_times_arrive_main, validation_dataset_mode1_main,
                    validation_dataset_mode2_main,
                    X_train_features_x_array, y_trainvalidation_goals_main, y_trainvalidation_zones_arrive_main, y_trainvalidation_times_depart_main, y_trainvalidation_times_arrive_main, y_trainvalidation_mode1_main, y_trainvalidation_mode2_main,
                    X_test_features_x_array, y_test_goals_main, y_test_zones_arrive_main, y_test_times_depart_main, y_test_times_arrive_main, y_test_mode1_main, y_test_mode2_main)

        # print(f"Returning {len(datasets)} datasets")  # 添加调试信息
        assert len(datasets) == 26, "Not enough datasets returned"

        yield datasets



def load_data_Kfold_cross_validation(batch_size, datasets_all_train, n_splits=10):
    (X_train_features_x_array, X_test_features_x_array, y_trainvalidation_goals_main, y_test_goals_main,
     y_trainvalidation_zones_arrive_main, y_test_zones_arrive_main,
     y_trainvalidation_times_depart_main, y_test_times_depart_main,
     y_trainvalidation_times_arrive_main, y_test_times_arrive_main,
     y_trainvalidation_mode1_main, y_test_mode1_main,
     y_trainvalidation_mode2_main, y_test_mode2_main) = datasets_all_train

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    datasets = []

    for train_index, validation_index in kf.split(X_train_features_x_array):
        X_train, X_val = X_train_features_x_array[train_index], X_train_features_x_array[validation_index]
        y_train_goal_main, y_validation_goal_main = y_trainvalidation_goals_main[train_index], y_trainvalidation_goals_main[validation_index]
        # y_train_zones_depart_main, y_validation_zones_depart_main = y_trainvalidation_zones_depart_main[train_index], y_trainvalidation_zones_depart_main[validation_index]
        y_train_zones_arrive_main, y_validation_zones_arrive_main = y_trainvalidation_zones_arrive_main[train_index], y_trainvalidation_zones_arrive_main[validation_index]
        y_train_times_depart_main, y_validation_times_depart_main = y_trainvalidation_times_depart_main[train_index], y_trainvalidation_times_depart_main[validation_index]
        y_train_times_arrive_main, y_validation_times_arrive_main = y_trainvalidation_times_arrive_main[train_index], y_trainvalidation_times_arrive_main[validation_index]
        y_train_mode1_main, y_validation_mode1_main = y_trainvalidation_mode1_main[train_index], y_trainvalidation_mode1_main[validation_index]
        y_train_mode2_main, y_validation_mode2_main = y_trainvalidation_mode2_main[train_index], y_trainvalidation_mode2_main[validation_index]

        train_dataset_goal_main = tf.data.Dataset.from_tensor_slices((X_train, y_train_goal_main)).shuffle(buffer_size=len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        validation_dataset_goal_main = tf.data.Dataset.from_tensor_slices((X_val, y_validation_goal_main)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # train_dataset_zones_depart_main = tf.data.Dataset.from_tensor_slices((X_train, y_train_zones_depart_main)).shuffle(buffer_size=len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        # validation_dataset_zones_depart_main = tf.data.Dataset.from_tensor_slices((X_val, y_validation_zones_depart_main)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        train_dataset_zones_arrive_main = tf.data.Dataset.from_tensor_slices((X_train, y_train_zones_arrive_main)).shuffle(buffer_size=len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        validation_dataset_zones_arrive_main = tf.data.Dataset.from_tensor_slices((X_val, y_validation_zones_arrive_main)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        train_dataset_times_depart_main = tf.data.Dataset.from_tensor_slices((X_train, y_train_times_depart_main)).shuffle(buffer_size=len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        validation_dataset_times_depart_main = tf.data.Dataset.from_tensor_slices((X_val, y_validation_times_depart_main)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        train_dataset_times_arrive_main = tf.data.Dataset.from_tensor_slices((X_train, y_train_times_arrive_main)).shuffle(buffer_size=len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        validation_dataset_times_arrive_main = tf.data.Dataset.from_tensor_slices((X_val, y_validation_times_arrive_main)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        train_dataset_mode1_main = tf.data.Dataset.from_tensor_slices((X_train, y_train_mode1_main)).shuffle(buffer_size=len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        validation_dataset_mode1_main = tf.data.Dataset.from_tensor_slices((X_val, y_validation_mode1_main)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        train_dataset_mode2_main = tf.data.Dataset.from_tensor_slices((X_train, y_train_mode2_main)).shuffle(buffer_size=len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        validation_dataset_mode2_main = tf.data.Dataset.from_tensor_slices((X_val, y_validation_mode2_main)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # 确保返回14个数据集
        datasets = (train_dataset_goal_main, train_dataset_zones_arrive_main,
                    train_dataset_times_depart_main, train_dataset_times_arrive_main, train_dataset_mode1_main,
                    train_dataset_mode2_main,
                    validation_dataset_goal_main, validation_dataset_zones_arrive_main,
                    validation_dataset_times_depart_main, validation_dataset_times_arrive_main, validation_dataset_mode1_main,
                    validation_dataset_mode2_main,
                    X_train_features_x_array, y_trainvalidation_goals_main, y_trainvalidation_zones_arrive_main, y_trainvalidation_times_depart_main, y_trainvalidation_times_arrive_main, y_trainvalidation_mode1_main, y_trainvalidation_mode2_main,
                    X_test_features_x_array, y_test_goals_main, y_test_zones_arrive_main, y_test_times_depart_main, y_test_times_arrive_main, y_test_mode1_main, y_test_mode2_main)

        # print(f"Returning {len(datasets)} datasets")  # 添加调试信息
        assert len(datasets) == 26, "Not enough datasets returned"

        yield datasets

# 最后用所有训练数据进行模型训练，并测试
def load_data_train_test(datasets_all_train, batch_size):
    (X_train_features_x_array, X_test_features_x_array,
     y_trainvalidation_goals_main, y_test_goals_main,
     y_trainvalidation_zones_arrive_main, y_test_zones_arrive_main,
     y_trainvalidation_times_depart_main, y_test_times_depart_main,
     y_trainvalidation_times_arrive_main, y_test_times_arrive_main,
     y_trainvalidation_mode1_main, y_test_mode1_main,
     y_trainvalidation_mode2_main, y_test_mode2_main) = datasets_all_train

    # print('datasets_all_train', type(datasets_all_train))     #
    # print('X_train_features_x_array', X_train_features_x_array.shape)     #  (145, 9)
    # print('y_trainvalidation_goals_main', y_trainvalidation_goals_main.shape)     #  (145, 5)

    # 创建 TensorFlow 数据集
    dataset_goal_main_train = tf.data.Dataset.from_tensor_slices((X_train_features_x_array, y_trainvalidation_goals_main))  # x & y
    print('dataset_goal_main_train', dataset_goal_main_train)   #
    # dataset_zones_depart_main_train = tf.data.Dataset.from_tensor_slices((X_train_features_x_array, y_trainvalidation_zones_depart_main))
    dataset_zones_arrive_main_train = tf.data.Dataset.from_tensor_slices((X_train_features_x_array, y_trainvalidation_zones_arrive_main))
    dataset_times_depart_main_train = tf.data.Dataset.from_tensor_slices((X_train_features_x_array, y_trainvalidation_times_depart_main))
    dataset_times_arrive_main_train = tf.data.Dataset.from_tensor_slices((X_train_features_x_array, y_trainvalidation_times_arrive_main))
    dataset_mode1_main_train = tf.data.Dataset.from_tensor_slices((X_train_features_x_array, y_trainvalidation_mode1_main))
    dataset_mode2_main_train = tf.data.Dataset.from_tensor_slices((X_train_features_x_array, y_trainvalidation_mode2_main))

    dataset_goal_main_test = tf.data.Dataset.from_tensor_slices((X_test_features_x_array, y_test_goals_main))  # x & y
    # dataset_zones_depart_main_test = tf.data.Dataset.from_tensor_slices((X_test_features_x_array, y_test_zones_depart_main))
    dataset_zones_arrive_main_test = tf.data.Dataset.from_tensor_slices((X_test_features_x_array, y_test_zones_arrive_main))
    dataset_times_depart_main_test = tf.data.Dataset.from_tensor_slices((X_test_features_x_array, y_test_times_depart_main))
    dataset_times_arrive_main_test = tf.data.Dataset.from_tensor_slices((X_test_features_x_array, y_test_times_arrive_main))
    dataset_mode1_main_test = tf.data.Dataset.from_tensor_slices((X_test_features_x_array, y_test_mode1_main))
    dataset_mode2_main_test = tf.data.Dataset.from_tensor_slices((X_test_features_x_array, y_test_mode2_main))

    dataset_goal_main_train = dataset_goal_main_train.shuffle(buffer_size=len(X_train_features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # dataset_zones_depart_main_train = dataset_zones_depart_main_train.shuffle(buffer_size=len(X_train_features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    dataset_zones_arrive_main_train = dataset_zones_arrive_main_train.shuffle(buffer_size=len(X_train_features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    dataset_times_depart_main_train = dataset_times_depart_main_train.shuffle(buffer_size=len(X_train_features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    dataset_times_arrive_main_train = dataset_times_arrive_main_train.shuffle(buffer_size=len(X_train_features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    dataset_mode1_main_train = dataset_mode1_main_train.shuffle(buffer_size=len(X_train_features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    dataset_mode2_main_train = dataset_mode2_main_train.shuffle(buffer_size=len(X_train_features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # dataset_goal_main_test = dataset_goal_main_test.shuffle(buffer_size=len(X_test_features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # dataset_zones_depart_main_test = dataset_zones_depart_main_test.shuffle(buffer_size=len(X_test_features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # dataset_zones_arrive_main_test = dataset_zones_arrive_main_test.shuffle(buffer_size=len(X_test_features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # dataset_times_depart_main_test = dataset_times_depart_main_test.shuffle(buffer_size=len(X_test_features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # dataset_times_arrive_main_test = dataset_times_arrive_main_test.shuffle(buffer_size=len(X_test_features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # dataset_mode1_main_test = dataset_mode1_main_test.shuffle(buffer_size=len(X_test_features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # dataset_mode2_main_test = dataset_mode2_main_test.shuffle(buffer_size=len(X_test_features_x_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # shuffle 的作用是在数据集创建时打乱数据的顺序。这个操作通常在训练阶段使用，但在测试或验证阶段一般不需要。测试集的主要目的是评估模型的泛化性能，即模型在未见过的数据上的表现。因此，测试集应该保持原始顺序，以确保评估结果的一致性和可重复性。
    dataset_goal_main_test = dataset_goal_main_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # dataset_zones_depart_main_test = dataset_zones_depart_main_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    dataset_zones_arrive_main_test = dataset_zones_arrive_main_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    dataset_times_depart_main_test = dataset_times_depart_main_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    dataset_times_arrive_main_test = dataset_times_arrive_main_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    dataset_mode1_main_test = dataset_mode1_main_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    dataset_mode2_main_test = dataset_mode2_main_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset_goal_main_train, dataset_zones_arrive_main_train, dataset_times_depart_main_train, dataset_times_arrive_main_train, dataset_mode1_main_train, dataset_mode2_main_train, dataset_goal_main_test, dataset_zones_arrive_main_test, dataset_times_depart_main_test, dataset_times_arrive_main_test, dataset_mode1_main_test, dataset_mode2_main_test


# 在绘图前，对数据应用滤波算法（例如移动平均滤波）以减少噪声和波折。
def moving_average(data, window_size=5):
    ret = np.cumsum(data, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:] / window_size


def compute_r2(true_values, pred_values):
    # 确保数据类型一致
    true_values = tf.cast(true_values, tf.float32)
    pred_values = tf.cast(pred_values, tf.float32)

    # print("true_values shape:", true_values.shape)
    # print("pred_values shape:", pred_values.shape)

    # 计算真实值的均值
    mean_true = tf.reduce_mean(true_values)

    # 计算分子（预测值与真实值的平方误差和）
    numerator = tf.reduce_sum(tf.square(true_values - pred_values))

    # 计算分母（真实值与其均值的平方误差和）
    denominator = tf.reduce_sum(tf.square(true_values - mean_true))

    # 计算 R²
    r2 = 1 - (numerator / denominator)
    return r2.numpy()


def compute_rmse(true_values, pred_values):
    mse = tf.reduce_mean(tf.square(true_values - pred_values))
    rmse = tf.sqrt(mse)
    return rmse.numpy()


def compute_mae(true_values, pred_values):
    mae = tf.reduce_mean(tf.abs(true_values - pred_values))
    return mae.numpy()



# 定义训练步骤
# @tf.function
def train_step(batch_inputs_goal_main, batch_labels_goals_main, batch_inputs_zone_arrive_main, batch_labels_zone_arrive_main,
               batch_inputs_time_depart_main, batch_labels_time_depart_main, batch_inputs_time_arrive_main, batch_labels_time_arrive_main,
               batch_inputs_mode1_main, batch_labels_mode1_main, batch_inputs_mode2_main, batch_labels_mode2_main,
               node_embeddings, zone_adjacency_matrix, distances, model, optimizer):

    with (tf.GradientTape() as tape):
        # 前向传播
        goals_main, zones_arrive_main, times_depart_main, times_arrive_main, modes1_main, modes2_main = model(
            batch_inputs_goal_main, batch_inputs_zone_arrive_main, batch_inputs_time_depart_main, batch_inputs_time_arrive_main, batch_inputs_mode1_main, batch_inputs_mode2_main)
        # print(goals_main,'=====', batch_labels_goals_main,'111')
        # print('11111',goals_main.shape, batch_labels_goals_main.shape)

        # 计算损失 MSE损失误差   7个子损失函数
        loss_goal_main = tf.reduce_mean(tf.square(goals_main - batch_labels_goals_main))
        # loss_zone_depart_main = tf.reduce_mean(tf.square(zones_depart_main - batch_labels_zone_depart_main))
        # print('zones_arrive_main', zones_arrive_main, batch_labels_zone_arrive_main.shape)    #  (32, 14) (32, 3046)
        loss_zone_arrive_main = tf.reduce_mean(tf.square(zones_arrive_main - batch_labels_zone_arrive_main))
        loss_time_depart_main = tf.reduce_mean(tf.square(times_depart_main - batch_labels_time_depart_main))
        loss_time_arrive_main = tf.reduce_mean(tf.square(times_arrive_main - batch_labels_time_arrive_main))
        loss_mode1_main = tf.reduce_mean(tf.square(modes1_main - batch_labels_mode1_main))
        loss_mode2_main = tf.reduce_mean(tf.square(modes2_main - batch_labels_mode2_main))
        # print(loss_goal_main,'loss_goal_main')

        # 组合损失（总损失）
        total_loss = 0.25 * loss_goal_main + 0.25 * loss_zone_arrive_main + 0.125 * loss_time_depart_main + 0.125 * loss_time_arrive_main + 0.125 * loss_mode1_main + 0.125 * loss_mode2_main
        # print(total_loss,'total_loss')

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # print("Trainable variables:", [v.name for v in model.trainable_variables])  # 添加调试信息

    # 计算其他指标
    r2_goals_main = compute_r2(batch_labels_goals_main, goals_main)
    r2_zones_arrive_main = compute_r2(batch_labels_zone_arrive_main, zones_arrive_main)
    r2_times_depart_main = compute_r2(batch_labels_time_depart_main, times_depart_main)
    r2_times_arrive_main = compute_r2(batch_labels_time_arrive_main, times_arrive_main)
    r2_modes1_main = compute_r2(batch_labels_mode1_main, modes1_main)
    r2_modes2_main = compute_r2(batch_labels_mode2_main, modes2_main)
    total_r2_loss = 0.25 * r2_goals_main + 0.25 * r2_zones_arrive_main + 0.125 * r2_times_depart_main + 0.125 * r2_times_arrive_main + 0.125 * r2_modes1_main + 0.125 * r2_modes2_main

    rmse_goals_main = compute_rmse(batch_labels_goals_main, goals_main)
    rmse_zones_arrive_main = compute_rmse(batch_labels_zone_arrive_main, zones_arrive_main)
    rmse_times_depart_main = compute_rmse(batch_labels_time_depart_main, times_depart_main)
    rmse_times_arrive_main = compute_rmse(batch_labels_time_arrive_main, times_arrive_main)
    rmse_modes1_main = compute_rmse(batch_labels_mode1_main, modes1_main)
    rmse_modes2_main = compute_rmse(batch_labels_mode2_main, modes2_main)
    total_rmse_loss = 0.25 * rmse_goals_main + 0.25 * rmse_zones_arrive_main + 0.125 * rmse_times_depart_main + 0.125 * rmse_times_arrive_main + 0.125 * rmse_modes1_main + 0.125 * rmse_modes2_main
    
    mae_goals_main = compute_mae(batch_labels_goals_main, goals_main)
    mae_zones_arrive_main = compute_mae(batch_labels_zone_arrive_main, zones_arrive_main)
    mae_times_depart_main = compute_mae(batch_labels_time_depart_main, times_depart_main)
    mae_times_arrive_main = compute_mae(batch_labels_time_arrive_main, times_arrive_main)
    mae_modes1_main = compute_mae(batch_labels_mode1_main, modes1_main)
    mae_modes2_main = compute_mae(batch_labels_mode2_main, modes2_main)
    total_mae_loss = 0.25 * mae_goals_main + 0.25 * mae_zones_arrive_main + 0.125 * mae_times_depart_main + 0.125 * mae_times_arrive_main + 0.125 * mae_modes1_main + 0.125 * mae_modes2_main
    
    return loss_goal_main, loss_zone_arrive_main, loss_time_depart_main, loss_time_arrive_main, loss_mode1_main, loss_mode2_main, total_loss, r2_goals_main, r2_zones_arrive_main, r2_times_depart_main, r2_times_arrive_main, r2_modes1_main, r2_modes2_main, total_r2_loss, rmse_goals_main, rmse_zones_arrive_main, rmse_times_depart_main, rmse_times_arrive_main, rmse_modes1_main, rmse_modes2_main, total_rmse_loss, mae_goals_main, mae_zones_arrive_main, mae_times_depart_main, mae_times_arrive_main, mae_modes1_main, mae_modes2_main, total_mae_loss, goals_main, zones_arrive_main, times_depart_main, times_arrive_main, modes1_main, modes2_main


# @tf.function
def validation_step(batch_inputs_goal_main, batch_labels_goals_main, batch_inputs_zone_arrive_main, batch_labels_zone_arrive_main, batch_inputs_time_depart_main, batch_labels_time_depart_main, batch_inputs_time_arrive_main, batch_labels_time_arrive_main, batch_inputs_mode1_main, batch_labels_mode1_main, batch_inputs_mode2_main, batch_labels_mode2_main, node_embeddings, zone_adjacency_matrix, distances, model, optimizer):
    with tf.GradientTape() as tape:
        # 前向传播
        goals_main, zones_arrive_main, times_depart_main, times_arrive_main, modes1_main, modes2_main = model(batch_inputs_goal_main, batch_inputs_zone_arrive_main, batch_inputs_time_depart_main, batch_inputs_time_arrive_main, batch_inputs_mode1_main, batch_inputs_mode2_main)

        # 计算损失 MSE损失误差   7个子损失函数
        loss_goal_main = tf.reduce_mean(tf.square(goals_main - batch_labels_goals_main))
        # loss_zone_depart_main = tf.reduce_mean(tf.square(zones_depart_main - batch_labels_zone_depart_main))
        loss_zone_arrive_main = tf.reduce_mean(tf.square(zones_arrive_main - batch_labels_zone_arrive_main))
        loss_time_depart_main = tf.reduce_mean(tf.square(times_depart_main - batch_labels_time_depart_main))
        loss_time_arrive_main = tf.reduce_mean(tf.square(times_arrive_main - batch_labels_time_arrive_main))
        loss_mode1_main = tf.reduce_mean(tf.square(modes1_main - batch_labels_mode1_main))
        loss_mode2_main = tf.reduce_mean(tf.square(modes2_main - batch_labels_mode2_main))
        # print(loss_goal_main,'loss_goal_main')

        # 组合损失（总损失）
        total_loss = 0.25 * loss_goal_main + 0.25 * loss_zone_arrive_main + 0.125 * loss_time_depart_main + 0.125 * loss_time_arrive_main + 0.125 * loss_mode1_main + 0.125 * loss_mode2_main
        # print(total_loss,'total_loss')
    # 在验证阶段，不需要进行梯度计算和权重更新。这是因为验证集的主要目的是评估模型在未见过的数据上的表现，而不是用来进一步训练模型。
    # 因此，在验证阶段我们只需要前向传播并计算损失，而不需要计算梯度或更新模型参数。

    return loss_goal_main, loss_zone_arrive_main, loss_time_depart_main, loss_time_arrive_main, loss_mode1_main, loss_mode2_main, total_loss, goals_main, zones_arrive_main, times_depart_main, times_arrive_main, modes1_main, modes2_main


# 在测试阶段，我们通常不需要更新模型的参数（即不进行反向传播和优化），因此 optimizer 参数在这个上下文中是不必要的。此外，测试阶段的主要目的是评估模型性能，而不是计算损失值用于训练。
# 因此，可以创建一个专门用于测试的 test_step 函数，它与 validation_step 类似，但省略了任何与训练相关的部分，如梯度计算和参数更新。
# 为了计算分类模型的准确率（accuracy）、精确度（precision）、召回率（recall）等指标，你需要将模型输出的概率值转换为具体的类别预测。对于多分类问题，通常使用 argmax 函数来选择每个样本中概率最高的类别作为预测结果
# @tf.function
def test_step(batch_inputs_goal_main, batch_labels_goals_main,
              batch_inputs_zone_arrive_main, batch_labels_zone_arrive_main,
              batch_inputs_time_depart_main, batch_labels_time_depart_main,
              batch_inputs_time_arrive_main, batch_labels_time_arrive_main,
              batch_inputs_mode1_main, batch_labels_mode1_main,
              batch_inputs_mode2_main, batch_labels_mode2_main, node_embeddings, zone_adjacency_matrix, distances, model):
    # 前向传播，不计算梯度
    goals_main, zones_arrive_main, times_depart_main, times_arrive_main, modes1_main, modes2_main = model(
        batch_inputs_goal_main, batch_inputs_zone_arrive_main,
        batch_inputs_time_depart_main, batch_inputs_time_arrive_main, batch_inputs_mode1_main, batch_inputs_mode2_main,
        training=False  # 确保在推理模式下运行
    )
    # print('goals_main:',goals_main)

    # 将概率值转换为分类预测,以求accuracy等
    predictions_goal_main = tf.argmax(goals_main, axis=1)
    # print('predictions_goal_main:', predictions_goal_main)
    # predictions_zone_depart_main = tf.argmax(zones_depart_main, axis=1)
    predictions_zone_arrive_main = tf.argmax(zones_arrive_main, axis=1)
    predictions_time_depart_main = tf.argmax(times_depart_main, axis=1)
    predictions_time_arrive_main = tf.argmax(times_arrive_main, axis=1)
    predictions_mode1_main = tf.argmax(modes1_main, axis=1)
    predictions_mode2_main = tf.argmax(modes2_main, axis=1)

    # 计算损失 MSE损失误差   7个子损失函数
    loss_goal_main = tf.reduce_mean(tf.square(goals_main - batch_labels_goals_main))
    # loss_zone_depart_main = tf.reduce_mean(tf.square(zones_depart_main - batch_labels_zone_depart_main))
    loss_zone_arrive_main = tf.reduce_mean(tf.square(zones_arrive_main - batch_labels_zone_arrive_main))
    loss_time_depart_main = tf.reduce_mean(tf.square(times_depart_main - batch_labels_time_depart_main))
    loss_time_arrive_main = tf.reduce_mean(tf.square(times_arrive_main - batch_labels_time_arrive_main))
    loss_mode1_main = tf.reduce_mean(tf.square(modes1_main - batch_labels_mode1_main))
    loss_mode2_main = tf.reduce_mean(tf.square(modes2_main - batch_labels_mode2_main))

    # 组合损失（总损失）
    total_loss = 0.25 * loss_goal_main + 0.25 * loss_zone_arrive_main + 0.125 * loss_time_depart_main + 0.125 * loss_time_arrive_main + 0.125 * loss_mode1_main + 0.125 * loss_mode2_main

    # 计算其他指标
    r2_goals_main = compute_r2(batch_labels_goals_main, goals_main)
    r2_zones_arrive_main = compute_r2(batch_labels_zone_arrive_main, zones_arrive_main)
    r2_times_depart_main = compute_r2(batch_labels_time_depart_main, times_depart_main)
    r2_times_arrive_main = compute_r2(batch_labels_time_arrive_main, times_arrive_main)
    r2_modes1_main = compute_r2(batch_labels_mode1_main, modes1_main)
    r2_modes2_main = compute_r2(batch_labels_mode2_main, modes2_main)
    total_rmse_loss = 0.25 * r2_goals_main + 0.25 * r2_zones_arrive_main + 0.125 * r2_times_depart_main + 0.125 * r2_times_arrive_main + 0.125 * r2_modes1_main + 0.125 * r2_modes2_main

    rmse_goals_main = compute_rmse(batch_labels_goals_main, goals_main)
    rmse_zones_arrive_main = compute_rmse(batch_labels_zone_arrive_main, zones_arrive_main)
    rmse_times_depart_main = compute_rmse(batch_labels_time_depart_main, times_depart_main)
    rmse_times_arrive_main = compute_rmse(batch_labels_time_arrive_main, times_arrive_main)
    rmse_modes1_main = compute_rmse(batch_labels_mode1_main, modes1_main)
    rmse_modes2_main = compute_rmse(batch_labels_mode2_main, modes2_main)
    total_r2_loss = 0.25 * rmse_goals_main + 0.25 * rmse_zones_arrive_main + 0.125 * rmse_times_depart_main + 0.125 * rmse_times_arrive_main + 0.125 * rmse_modes1_main + 0.125 * rmse_modes2_main

    mae_goals_main = compute_mae(batch_labels_goals_main, goals_main)
    mae_zones_arrive_main = compute_mae(batch_labels_zone_arrive_main, zones_arrive_main)
    mae_times_depart_main = compute_mae(batch_labels_time_depart_main, times_depart_main)
    mae_times_arrive_main = compute_mae(batch_labels_time_arrive_main, times_arrive_main)
    mae_modes1_main = compute_mae(batch_labels_mode1_main, modes1_main)
    mae_modes2_main = compute_mae(batch_labels_mode2_main, modes2_main)
    total_mae_loss = 0.25 * mae_goals_main + 0.25 * mae_zones_arrive_main + 0.125 * mae_times_depart_main + 0.125 * mae_times_arrive_main + 0.125 * mae_modes1_main + 0.125 * mae_modes2_main


    return predictions_goal_main, predictions_zone_arrive_main, predictions_time_depart_main, predictions_time_arrive_main, predictions_mode1_main, predictions_mode2_main, loss_goal_main, loss_zone_arrive_main, loss_time_depart_main, loss_time_arrive_main, loss_mode1_main, loss_mode2_main, total_loss, r2_goals_main, r2_zones_arrive_main, r2_times_depart_main, r2_times_arrive_main, r2_modes1_main, r2_modes2_main, total_r2_loss, rmse_goals_main, rmse_zones_arrive_main, rmse_times_depart_main, rmse_times_arrive_main, rmse_modes1_main, rmse_modes2_main, total_rmse_loss, mae_goals_main, mae_zones_arrive_main, mae_times_depart_main, mae_times_arrive_main, mae_modes1_main, mae_modes2_main, total_mae_loss, goals_main, zones_arrive_main, times_depart_main, times_arrive_main, modes1_main, modes2_main



def epoch_train_validation(batch_size, num_epochs, train_dataset_goal_main, train_dataset_zones_arrive_main, train_dataset_times_depart_main, train_dataset_times_arrive_main, train_dataset_mode1_main, train_dataset_mode2_main, validation_dataset_goal_main, validation_dataset_zones_arrive_main, validation_dataset_times_depart_main, validation_dataset_times_arrive_main, validation_dataset_mode1_main, validation_dataset_mode2_main, node_embeddings, zone_adjacency_matrix, distances, model, optimizer):
    # 设置早停法参数
    patience = 15
    best_validation_loss = float('inf')
    no_improvement_count = 0
    epoch_early_stopping = None   # 要先初始化，否则不可作为return的内容
    # epoch_early_stopping_df = pd.DataFrame(columns=['batch_size', 'num_epochs', 'epoch_early_stopping'])

    # 训练模型
    train_list_iter_loss_goal_main = []    # collect each iter in each epoch ' loss  (batchsize=10, 每个 epoch 会有 13 次迭代iteration)
    train_list_epoch_loss_goal_main = []
    train_list_iter_loss_zone_depart_main = []
    train_list_epoch_loss_zone_depart_main = []
    train_list_iter_loss_zone_arrive_main = []
    train_list_epoch_loss_zone_arrive_main = []
    train_list_iter_loss_time_depart_main = []
    train_list_epoch_loss_time_depart_main = []
    train_list_iter_loss_time_arrive_main = []
    train_list_epoch_loss_time_arrive_main = []
    train_list_iter_loss_mode1_main = []
    train_list_epoch_loss_mode1_main = []
    train_list_iter_loss_mode2_main = []
    train_list_epoch_loss_mode2_main = []
    train_list_iter_loss = []
    train_list_epoch_loss = []

    validation_list_iter_loss_goal_main = []    # collect each iter in each epoch ' loss  (batchsize=10, 每个 epoch 会有 13 次迭代iteration)
    validation_list_epoch_loss_goal_main = []
    validation_list_iter_loss_zone_depart_main = []
    validation_list_epoch_loss_zone_depart_main = []
    validation_list_iter_loss_zone_arrive_main = []
    validation_list_epoch_loss_zone_arrive_main = []
    validation_list_iter_loss_time_depart_main = []
    validation_list_epoch_loss_time_depart_main = []
    validation_list_iter_loss_time_arrive_main = []
    validation_list_epoch_loss_time_arrive_main = []
    validation_list_iter_loss_mode1_main = []
    validation_list_epoch_loss_mode1_main = []
    validation_list_iter_loss_mode2_main = []
    validation_list_epoch_loss_mode2_main = []
    validation_list_iter_loss = []
    validation_list_epoch_loss = []

    # 初始化时间记录变量
    # 在机器学习模型训练过程中，第一次训练的时间通常会比后续的训练时间长一些。这是由于多种因素导致的，see Notion notebook,预热
    start_time_epoch = time.time()  # 开始时间
    epoch_start_times = []  # 每个epoch的开始时间
    epoch_durations = []  # 每个epoch的持续时间
    train_start_times = []
    train_durations = []
    iter_start_times =[]

    # 重新实例化模型以确保每次训练从头开始
    # model = CombinedNetwork()
    # # 重新实例化优化器
    # optimizer = tf.keras.optimizers.Adam()

    # # 重置模型的可训练变量和优化器状态
    # model.reset_weights()
    # optimizer = tf.keras.optimizers.Adam()


    for epoch in range(num_epochs):

        epoch_start_times.append(time.time())  # 记录当前 epoch 的开始时间
        # print(epoch_start_times,'epoch_start_times')

        filename = fr'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_epochs\01epochs.csv'
        # 将当前 epoch 记录到 CSV 文件中，查看早停法在epoch=几的时候停止
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch])

        train_epoch_losses_goal_main = []  # 存储每个 train_epoch 的所有批次（13批次）损失
        train_epoch_losses_zone_depart_main = []
        train_epoch_losses_zone_arrive_main = []
        train_epoch_losses_time_depart_main = []
        train_epoch_losses_time_arrive_main = []
        train_epoch_losses_mode1_main = []
        train_epoch_losses_mode2_main = []
        train_epoch_losses = []

        validation_epoch_losses_goal_main = []  # 存储每个 validation_epoch 的所有批次（13批次）损失
        validation_epoch_losses_zone_depart_main = []
        validation_epoch_losses_zone_arrive_main = []
        validation_epoch_losses_time_depart_main = []
        validation_epoch_losses_time_arrive_main = []
        validation_epoch_losses_mode1_main = []
        validation_epoch_losses_mode2_main = []
        validation_epoch_losses = []

        # 重置迭代器状态
        train_iterator_goal_main = iter(train_dataset_goal_main)
        # train_iterator_zone_depart_main = iter(train_dataset_zones_depart_main)
        train_iterator_zone_arrive_main = iter(train_dataset_zones_arrive_main)
        train_iterator_time_depart_main = iter(train_dataset_times_depart_main)
        train_iterator_time_arrive_main = iter(train_dataset_times_arrive_main)
        train_iterator_mode1_main = iter(train_dataset_mode1_main)
        train_iterator_mode2_main = iter(train_dataset_mode2_main)

        validation_iterator_goal_main = iter(validation_dataset_goal_main)
        # validation_iterator_zone_depart_main = iter(validation_dataset_zones_depart_main)
        validation_iterator_zone_arrive_main = iter(validation_dataset_zones_arrive_main)
        validation_iterator_time_depart_main = iter(validation_dataset_times_depart_main)
        validation_iterator_time_arrive_main = iter(validation_dataset_times_arrive_main)
        validation_iterator_mode1_main = iter(validation_dataset_mode1_main)
        validation_iterator_mode2_main = iter(validation_dataset_mode2_main)

        # print(next(iter(train_dataset_goal_main)),'iter(train_dataset_goal_main)')
        # print(next(iter(train_dataset_zones_depart_main)))

        # 使用预先创建的迭代器
        train_num_batches = min(tf.data.experimental.cardinality(train_dataset_goal_main), tf.data.experimental.cardinality(train_dataset_zones_arrive_main), tf.data.experimental.cardinality(train_dataset_times_depart_main), tf.data.experimental.cardinality(train_dataset_times_arrive_main), tf.data.experimental.cardinality(train_dataset_mode1_main), tf.data.experimental.cardinality(train_dataset_mode2_main))   # 13
        # print(train_num_batches,'train_num_batches')   # tf.Tensor(6, shape=(), dtype=int64)，共有6批数据，每个周期epoch迭代6次
        validation_num_batches = min(tf.data.experimental.cardinality(validation_dataset_goal_main), tf.data.experimental.cardinality(validation_dataset_zones_arrive_main), tf.data.experimental.cardinality(validation_dataset_times_depart_main), tf.data.experimental.cardinality(validation_dataset_times_arrive_main), tf.data.experimental.cardinality(validation_dataset_mode1_main), tf.data.experimental.cardinality(validation_dataset_mode2_main))  # 13

        i = 1
        train_start_times.append(time.time())
        # print(train_start_times,'train_start_times')
        start_time_iter = time.time()

        for _ in range(train_num_batches):
            iter_start_times.append(time.time())

            batch_inputs_goal_main, batch_labels_goals_main = next(train_iterator_goal_main)
            # batch_inputs_zone_depart_main, batch_labels_zone_depart_main = next(train_iterator_zone_depart_main)
            batch_inputs_zone_arrive_main, batch_labels_zone_arrive_main = next(train_iterator_zone_arrive_main)
            batch_inputs_time_depart_main, batch_labels_time_depart_main = next(train_iterator_time_depart_main)
            batch_inputs_time_arrive_main, batch_labels_time_arrive_main = next(train_iterator_time_arrive_main)
            batch_inputs_mode1_main, batch_labels_mode1_main = next(train_iterator_mode1_main)
            batch_inputs_mode2_main, batch_labels_mode2_main = next(train_iterator_mode2_main)
            # print('555')   # 迭代次数 = num_epochs * train_num_batches

            # 从[batch_size, 3047]的batch_labels_zone_arrive_main中提取14列，以使得形状与train_step中的zones_arrive_main 匹配：
            # 创建从 zone_ids_int 到索引的映射
            id_to_index = {id: index for index, id in enumerate(zone_ids_int)}

            # 根据 zone_ids_int 提取相应的列
            def extract_relevant_zones(batch_labels_zone_arrive_main, zone_ids_int):
                indices = [id_to_index[id] for id in zone_ids_int]
                extracted_zones = tf.gather(batch_labels_zone_arrive_main, indices, axis=-1)
                return extracted_zones

            # 提取相关交通小区
            extracted_batch_labels_zone_arrive_main = extract_relevant_zones(batch_labels_zone_arrive_main,
                                                                             zone_ids_int)
            # 现在 extracted_batch_labels_zone_arrive_main 的形状应该是 [batch_size, 14]
            # print("Extracted Batch Labels Shape:", extracted_batch_labels_zone_arrive_main.shape)

            # 运行train_step
            loss_goal_main, loss_zone_arrive_main, loss_time_depart_main, loss_time_arrive_main, loss_mode1_main, loss_mode2_main, total_loss, r2_goals_main, r2_zones_arrive_main, r2_times_depart_main, r2_times_arrive_main, r2_modes1_main, r2_modes2_main, total_r2_loss, rmse_goals_main, rmse_zones_arrive_main, rmse_times_depart_main, rmse_times_arrive_main, rmse_modes1_main, rmse_modes2_main, total_rmse_loss, mae_goals_main, mae_zones_arrive_main, mae_times_depart_main, mae_times_arrive_main, mae_modes1_main, mae_modes2_main, total_mae_loss, goals_main, zones_arrive_main, times_depart_main, times_arrive_main, modes1_main, modes2_main\
                = train_step(batch_inputs_goal_main, batch_labels_goals_main, batch_inputs_zone_arrive_main, extracted_batch_labels_zone_arrive_main, batch_inputs_time_depart_main, batch_labels_time_depart_main, batch_inputs_time_arrive_main, batch_labels_time_arrive_main, batch_inputs_mode1_main, batch_labels_mode1_main, batch_inputs_mode2_main, batch_labels_mode2_main, node_embeddings, zone_adjacency_matrix, distances, model, optimizer)
            # print(loss_goal_main,'loss_goal_main1')

            # print(f"Losses: Goal={loss_goal}, Zone={loss_zone}, Time={loss_time}, Mode={loss_mode}")

            train_list_iter_loss_goal_main.append(loss_goal_main.numpy())
            train_epoch_losses_goal_main.append(loss_goal_main.numpy())  # 将每个批次的损失值添加到列表中
            # train_list_iter_loss_zone_depart_main.append(loss_zone_depart_main.numpy())
            # train_epoch_losses_zone_depart_main.append(loss_zone_depart_main.numpy())
            train_list_iter_loss_zone_arrive_main.append(loss_zone_arrive_main.numpy())
            train_epoch_losses_zone_arrive_main.append(loss_zone_arrive_main.numpy())
            train_list_iter_loss_time_depart_main.append(loss_time_depart_main.numpy())
            train_epoch_losses_time_depart_main.append(loss_time_depart_main.numpy())
            train_list_iter_loss_time_arrive_main.append(loss_time_arrive_main.numpy())
            train_epoch_losses_time_arrive_main.append(loss_time_arrive_main.numpy())
            train_list_iter_loss_mode1_main.append(loss_mode1_main.numpy())
            train_epoch_losses_mode1_main.append(loss_mode1_main.numpy())
            train_list_iter_loss_mode2_main.append(loss_mode2_main.numpy())
            train_epoch_losses_mode2_main.append(loss_mode2_main.numpy())
            train_list_iter_loss.append(total_loss.numpy())
            train_epoch_losses.append(total_loss.numpy())

            # print(time.time(),'time.time()')
            train_duration = time.time() - iter_start_times[-1]   # 每次迭代/每个批次的训练时间
            # print(train_duration,'train_duration')
            train_durations.append(train_duration)
            # print(f"Train {i} took {train_duration:.5f} seconds.")
            i += 1

        # # 计算总训练时间
        # total_iter_training_time = time.time() - start_time_iter
        # # print(time.time(),'time.time()1')
        # # print(start_time_iter,'start_time_iter')
        # # 打印或记录总训练时间
        # print(f"Total iter training took {total_iter_training_time:.5f} seconds.")

        # 保存每个 iter（每次迭代） 的持续时间到 CSV 文件
        df_iter_durations = pd.DataFrame({'iter_duration': train_durations})
        df_iter_durations.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_time\01output_iter_time_durations.csv", index=False)  # 每次迭代iter的时间

        # # 保存总训练时间到文件
        # with open("01output_total_iter_training_time.txt", "w") as f:
        #     f.write(f"Total iter training took {total_iter_training_time:.5f} seconds.")


        for _ in range(validation_num_batches):
            batch_inputs_goal_main, batch_labels_goals_main = next(validation_iterator_goal_main)
            # batch_inputs_zone_depart_main, batch_labels_zone_depart_main = next(validation_iterator_zone_depart_main)
            batch_inputs_zone_arrive_main, batch_labels_zone_arrive_main = next(validation_iterator_zone_arrive_main)
            batch_inputs_time_depart_main, batch_labels_time_depart_main = next(validation_iterator_time_depart_main)
            batch_inputs_time_arrive_main, batch_labels_time_arrive_main = next(validation_iterator_time_arrive_main)
            batch_inputs_mode1_main, batch_labels_mode1_main = next(validation_iterator_mode1_main)
            batch_inputs_mode2_main, batch_labels_mode2_main = next(validation_iterator_mode2_main)
            # print('555')   # 迭代次数 = num_epochs * validation_num_batches

            # 从[batch_size, 3047]的batch_labels_zone_arrive_main中提取14列，以使得形状与train_step中的zones_arrive_main 匹配：
            # 创建从 zone_ids_int 到索引的映射
            id_to_index = {id: index for index, id in enumerate(zone_ids_int)}

            # 根据 zone_ids_int 提取相应的列
            def extract_relevant_zones(batch_labels_zone_arrive_main, zone_ids_int):
                indices = [id_to_index[id] for id in zone_ids_int]
                extracted_zones = tf.gather(batch_labels_zone_arrive_main, indices, axis=-1)
                return extracted_zones

            # 提取相关交通小区
            extracted_batch_labels_zone_arrive_main = extract_relevant_zones(batch_labels_zone_arrive_main,
                                                                             zone_ids_int)
            # 现在 extracted_batch_labels_zone_arrive_main 的形状应该是 [batch_size, 14]
            # print("Extracted Batch Labels Shape:", extracted_batch_labels_zone_arrive_main.shape)

            # 运行validation_step
            loss_goal_main, loss_zone_arrive_main, loss_time_depart_main, loss_time_arrive_main, loss_mode1_main, loss_mode2_main, total_loss, r2_goals_main, r2_zones_arrive_main, r2_times_depart_main, r2_times_arrive_main, r2_modes1_main, r2_modes2_main, total_r2_loss, rmse_goals_main, rmse_zones_arrive_main, rmse_times_depart_main, rmse_times_arrive_main, rmse_modes1_main, rmse_modes2_main, total_r2_loss, mae_goals_main, mae_zones_arrive_main, mae_times_depart_main, mae_times_arrive_main, mae_modes1_main, mae_modes2_main, total_mae_loss, goals_main, zones_arrive_main, times_depart_main, times_arrive_main, modes1_main, modes2_main \
                = validation_step(batch_inputs_goal_main, batch_labels_goals_main, batch_inputs_zone_arrive_main, extracted_batch_labels_zone_arrive_main, batch_inputs_time_depart_main, batch_labels_time_depart_main, batch_inputs_time_arrive_main, batch_labels_time_arrive_main, batch_inputs_mode1_main, batch_labels_mode1_main, batch_inputs_mode2_main, batch_labels_mode2_main, node_embeddings, zone_adjacency_matrix, distances, model, optimizer)
            # print(loss_goal_main,'loss_goal_main1')

            # print(f"Losses: Goal={loss_goal}, Zone={loss_zone}, Time={loss_time}, Mode={loss_mode}")

            validation_list_iter_loss_goal_main.append(loss_goal_main.numpy())
            validation_epoch_losses_goal_main.append(loss_goal_main.numpy())  # 将每个批次的损失值添加到列表中
            # validation_list_iter_loss_zone_depart_main.append(loss_zone_depart_main.numpy())
            # validation_epoch_losses_zone_depart_main.append(loss_zone_depart_main.numpy())
            validation_list_iter_loss_zone_arrive_main.append(loss_zone_arrive_main.numpy())
            validation_epoch_losses_zone_arrive_main.append(loss_zone_arrive_main.numpy())
            validation_list_iter_loss_time_depart_main.append(loss_time_depart_main.numpy())
            validation_epoch_losses_time_depart_main.append(loss_time_depart_main.numpy())
            validation_list_iter_loss_time_arrive_main.append(loss_time_arrive_main.numpy())
            validation_epoch_losses_time_arrive_main.append(loss_time_arrive_main.numpy())
            validation_list_iter_loss_mode1_main.append(loss_mode1_main.numpy())
            validation_epoch_losses_mode1_main.append(loss_mode1_main.numpy())
            validation_list_iter_loss_mode2_main.append(loss_mode2_main.numpy())
            validation_epoch_losses_mode2_main.append(loss_mode2_main.numpy())
            validation_list_iter_loss.append(total_loss.numpy())
            validation_epoch_losses.append(total_loss.numpy())

            # print(f"Loss Goal: {loss_goal.numpy()}, Loss Zone: {loss_zone.numpy()}")

        validation_end_time = time.time()
        # 记录每个 epoch 的持续时间
        # print(time.time(),'time.time()')
        epoch_duration = time.time() - epoch_start_times[-1]
        # print(epoch_duration,'epoch_duration')
        epoch_durations.append(epoch_duration)
        # 打印或记录每个 epoch 的持续时间
        # print(f"Epoch {epoch + 1} took {epoch_duration:.5f} seconds.")

        # print(time.time(),'123')
        # 计算 epoch 的平均损失
        train_avg_epoch_loss_goal_main = np.mean(train_epoch_losses_goal_main)
        train_list_epoch_loss_goal_main.append(train_avg_epoch_loss_goal_main)  # 添加 epoch 的平均损失到总的列表中
        # train_avg_epoch_loss_zone_depart_main = np.mean(train_epoch_losses_zone_depart_main)
        # train_list_epoch_loss_zone_depart_main.append(train_avg_epoch_loss_zone_depart_main)
        train_avg_epoch_loss_zone_arrive_main = np.mean(train_epoch_losses_zone_arrive_main)
        train_list_epoch_loss_zone_arrive_main.append(train_avg_epoch_loss_zone_arrive_main)
        train_avg_epoch_loss_time_depart_main = np.mean(train_epoch_losses_time_depart_main)
        train_list_epoch_loss_time_depart_main.append(train_avg_epoch_loss_time_depart_main)
        train_avg_epoch_loss_time_arrive_main = np.mean(train_epoch_losses_time_arrive_main)
        train_list_epoch_loss_time_arrive_main.append(train_avg_epoch_loss_time_arrive_main)
        train_avg_epoch_loss_mode1_main = np.mean(train_epoch_losses_mode1_main)
        train_list_epoch_loss_mode1_main.append(train_avg_epoch_loss_mode1_main)
        train_avg_epoch_loss_mode2_main = np.mean(train_epoch_losses_mode2_main)
        train_list_epoch_loss_mode2_main.append(train_avg_epoch_loss_mode2_main)
        train_avg_epoch_loss = np.mean(train_epoch_losses)
        train_list_epoch_loss.append(train_avg_epoch_loss)

        # print(f"Epoch_goal {epoch + 1}, Loss_goal: {loss_goal:.4f}, avg_epoch_loss_goal: {avg_epoch_loss_goal:.4f}, list_epoch_loss_goal: {list_epoch_loss_goal}")
        # print(f"Epoch_zone {epoch + 1}, Loss_zone: {loss_zone:.4f}, avg_epoch_loss_zone: {avg_epoch_loss_zone:.4f}, list_epoch_loss_zone: {list_epoch_loss_zone}")
        validation_avg_epoch_loss_goal_main = np.mean(validation_epoch_losses_goal_main)
        validation_list_epoch_loss_goal_main.append(validation_avg_epoch_loss_goal_main)  # 添加 epoch 的平均损失到总的列表中
        # validation_avg_epoch_loss_zone_depart_main = np.mean(validation_epoch_losses_zone_depart_main)
        # validation_list_epoch_loss_zone_depart_main.append(validation_avg_epoch_loss_zone_depart_main)
        validation_avg_epoch_loss_zone_arrive_main = np.mean(validation_epoch_losses_zone_arrive_main)
        validation_list_epoch_loss_zone_arrive_main.append(validation_avg_epoch_loss_zone_arrive_main)
        validation_avg_epoch_loss_time_depart_main = np.mean(validation_epoch_losses_time_depart_main)
        validation_list_epoch_loss_time_depart_main.append(validation_avg_epoch_loss_time_depart_main)
        validation_avg_epoch_loss_time_arrive_main = np.mean(validation_epoch_losses_time_arrive_main)
        validation_list_epoch_loss_time_arrive_main.append(validation_avg_epoch_loss_time_arrive_main)
        validation_avg_epoch_loss_mode1_main = np.mean(validation_epoch_losses_mode1_main)
        validation_list_epoch_loss_mode1_main.append(validation_avg_epoch_loss_mode1_main)
        validation_avg_epoch_loss_mode2_main = np.mean(validation_epoch_losses_mode2_main)
        validation_list_epoch_loss_mode2_main.append(validation_avg_epoch_loss_mode2_main)
        validation_avg_epoch_loss = np.mean(validation_epoch_losses)
        validation_list_epoch_loss.append(validation_avg_epoch_loss)


        # 早停法检查（Early Stopping）
        if validation_avg_epoch_loss < best_validation_loss:
            best_validation_loss = validation_avg_epoch_loss
            no_improvement_count = 0
            # print(no_improvement_count,'no_improvement_count')
            # 保存当前模型状态

            # 获取当前时间的时间戳，并格式化为字符串
            # 通过添加时间戳来创建一个唯一的文件名，确保每次保存的权重文件都有不同的名称
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            # 创建带有时间戳的新文件名
            weights_file_path = fr'E:\Desktop\08[Win]CQ_GoalZoneTimeMode20250102\01output_modelweights_checkpoint\best_model_CQ_{timestamp}.weights.h5'
            group_name = f'group_{timestamp}'

            # 如果文件已经存在，则删除它（以防万一）
            if os.path.exists(weights_file_path):
                os.remove(weights_file_path)

            # 如果你希望避免 HDF5 文件格式的问题，还可以将权重保存为 TensorFlow Checkpoint 格式，用于保存和恢复训练过程中模型的状态。checkpoint.restore可以重新加载模型，eg：
            # 创建一个 Checkpoint 对象
            # checkpoint = tf.train.Checkpoint(model=model)
            # # 保存 Checkpoint
            # checkpoint.save('path_to_save_checkpoint')
            # # 加载 Checkpoint
            # checkpoint.restore(tf.train.latest_checkpoint('path_to_save_checkpoint'))
            try:
                # 使用新文件名保存模型权重
                # model.save(weights_file_path, save_format='tf')
                # print(f'Model weights saved to {weights_file_path}')

                # 创建一个 Checkpoint 对象
                checkpoint = tf.train.Checkpoint(model=model)
                # 保存 Checkpoint
                checkpoint.save(fr'E:\Desktop\08[Win]CQ_GoalZoneTimeMode20250102\01output_modelweights_checkpoint\path_to_save_checkpoint_{timestamp}')

            except Exception as e:
                print(f'Failed to save model checkpoint: {e}')

            # save model weights，保存模型权重：
            try:
                # 使用新文件名保存模型权重
                # model.save_weights(weights_file_path)
                # print(f'Model weights saved to {weights_file_path}')
                with h5py.File(weights_file_path, 'a') as f:
                    if group_name not in f:
                        grp = f.create_group(group_name)
                        for layer in model.layers:
                            if layer.weights:
                                layer_group = grp.create_group(layer.name)

                                # converted_weights = []
                                for weight in layer.weights:
                                #     if weight.dtype == 'O':  # 检查是否为 object 类型
                                #         print(f"Converting object dtype to float32 for layer {layer.name}")
                                #         converted_weights.append(np.asarray(weight, dtype=np.float32))
                                #     else:
                                #         converted_weights.append(weight)

                                    weight_value = model.get_layer(layer.name).get_weights()
                                    # 确保权重是 NumPy 数组并且具有支持的 dtype
                                    weight_array = np.asarray(weight_value[0], dtype=np.float32)
                                    layer_group.create_dataset(weight.name, data=weight_array)

                print(f'Model weights saved to {weights_file_path} under group {group_name}')

            except Exception as e:
                print(f'Failed to save model weights: {e}')



            # def check_hdf5_content(file_path):
            #     with h5py.File(file_path, 'r') as f:
            #         print("Keys in the HDF5 file:", list(f.keys()))
            # # 检查现有的 HDF5 文件
            # check_hdf5_content(weights_file_path)


        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            # print(f"Early stopping at epoch {epoch}")
            epoch_early_stopping = epoch
            break



        # # 早停法检查（Early Stopping）
        # if validation_avg_epoch_loss < best_validation_loss:
        #     best_validation_loss = validation_avg_epoch_loss
        #     no_improvement_count = 0
        #     # print(no_improvement_count,'no_improvement_count')
        #     # 保存当前模型状态
        #     model.save_weights('best_model.weights.h5')
        # else:
        #     no_improvement_count += 1
        #
        # if no_improvement_count >= patience:
        #     # print(f"Early stopping at epoch {epoch}")
        #     epoch_early_stopping = epoch
        #     break


    # 计算总训练时间
    # time1 = time.time()
    # print(time1-validation_end_time,'980')
    total_epoch_training_and_validation_time = validation_end_time - start_time_epoch
    # 打印或记录总训练时间
    # print(f"Total epoch training and validation took {total_epoch_training_and_validation_time:.7f} seconds.")

    # 保存每个 epoch 的持续时间到 CSV 文件
    df_epoch_durations = pd.DataFrame({'epoch_duration': epoch_durations})
    df_epoch_durations.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_time\01output_epoch(training_and_validation)_time_durations.csv", index=False)  # 每个周期epoch的时间

    # 保存总训练时间到文件
    with open(r"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_time\01output_total_epoch_training_and_validation_time.txt", "w") as f:
        f.write(f"Total epoch training and validation took {total_epoch_training_and_validation_time:.5f} seconds.")

    # print(validation_list_epoch_loss[-1], '897968')


    return (train_list_iter_loss_goal_main, train_list_iter_loss_zone_arrive_main, train_list_iter_loss_time_depart_main, train_list_iter_loss_time_arrive_main, train_list_iter_loss_mode1_main, train_list_iter_loss_mode2_main, train_list_iter_loss,
            train_list_epoch_loss_goal_main, train_list_epoch_loss_zone_arrive_main, train_list_epoch_loss_time_depart_main, train_list_epoch_loss_time_arrive_main, train_list_epoch_loss_mode1_main, train_list_epoch_loss_mode2_main, train_list_epoch_loss,
            validation_list_iter_loss_goal_main, validation_list_iter_loss_zone_arrive_main, validation_list_iter_loss_time_depart_main, validation_list_iter_loss_time_arrive_main, validation_list_iter_loss_mode1_main, validation_list_iter_loss_mode2_main, validation_list_iter_loss,
            validation_list_epoch_loss_goal_main, validation_list_epoch_loss_zone_arrive_main, validation_list_epoch_loss_time_depart_main, validation_list_epoch_loss_time_arrive_main, validation_list_epoch_loss_mode1_main, validation_list_epoch_loss_mode2_main, validation_list_epoch_loss, epoch_early_stopping)



# 用于epoch_train_test函数中 计算测试集中所有批次的总损失和平均损失：
def compute_sum_and_average(loss_list):
    if not loss_list:  # 检查列表是否为空
        return 0, 0

    total_loss = sum(loss_list)
    print(total_loss,'total_loss')
    average_loss = total_loss / len(loss_list)
    print('average_loss', average_loss)

    return total_loss, average_loss



# 绘制真实值与预测值之间的R2散点图。横纵坐标分别是真实值和预测值，并在图中给出拟合曲线和R2、mse、rmse、mae（已经通过train_step计算得到）
def plot_true_vs_predicted(true_values, predicted_values, metric_name, r2, mse, rmse, mae):
    """
    绘制真实值与预测值的散点图，并添加拟合曲线和评估指标。

    参数:
        true_values (array-like): 真实值
        predicted_values (array-like): 预测值
        metric_name (str): 指标名称（如 "Goal", "Zone Arrive", "Time Depart" 等）
        r2 (float): R² 值
        mse (float): 均方误差 (MSE)
        rmse (float): 均方根误差 (RMSE)
        mae (float): 平均绝对误差 (MAE)
    """
    # 设置全局字体为 Times New Roman，字号为 12
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    # 计算线性回归拟合曲线
    slope, intercept, _, _, _ = linregress(true_values, predicted_values)
    x_range = np.linspace(min(true_values), max(true_values), 100)
    y_fit = slope * x_range + intercept

    # 创建散点图
    plt.figure(figsize=(7, 6))
    plt.scatter(true_values, predicted_values, color='blue', alpha=0.6, label='Data Points')

    # 添加拟合曲线
    plt.plot(x_range, y_fit, color='red', linestyle='--', label=f'Fit: y={slope:.2f}x+{intercept:.2f}')

    # 添加对角线 (y=x)
    plt.plot(x_range, x_range, color='black', linestyle='-', linewidth=1, label='y=x')

    # 添加标题和标签
    # plt.title(f'{metric_name}: True vs Predicted Values', fontsize=14, fontfamily='Times New Roman')
    plt.xlabel('True Values', fontsize=12, fontfamily='Times New Roman')
    plt.ylabel('Predicted Values', fontsize=12, fontfamily='Times New Roman')

    # 添加评估指标到图中
    metrics_text = f'R² = {r2:.4f}\nMSE = {mse:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
    plt.text(0.05, 0.85, metrics_text, transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))

    # 调整布局
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()

    # 保存图像
    plt.savefig(fr'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_png_scatter\01true_vs_predicted_{metric_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')

    # 显示图像
    # plt.show()



def epoch_train_test(batch_size, num_epochs, train_dataset_goal_main, train_dataset_zones_arrive_main,
                     train_dataset_times_depart_main, train_dataset_times_arrive_main, train_dataset_mode1_main, train_dataset_mode2_main,
                     test_dataset_goal_main, test_dataset_zones_arrive_main, test_dataset_times_depart_main, test_dataset_times_arrive_main,
                     test_dataset_mode1_main, test_dataset_mode2_main,
                     node_embeddings, zone_adjacency_matrix, distances,
                     model, optimizer):

    print('train_dataset_goal_main', train_dataset_goal_main)      # (None, 5)
    # 设置早停法参数
    patience = 15
    best_test_loss = float('inf')
    no_improvement_count = 0
    epoch_early_stopping = None   # 要先初始化，否则不可作为return的内容
    # epoch_early_stopping_df = pd.DataFrame(columns=['batch_size', 'num_epochs', 'epoch_early_stopping'])

    # 训练模型
    train_list_iter_loss_goal_main = []    # collect each iter in each epoch ' loss  (batchsize=10, 每个 epoch 会有 13 次迭代iteration)
    train_list_epoch_loss_goal_main = []
    train_list_iter_loss_zone_depart_main = []
    train_list_epoch_loss_zone_depart_main = []
    train_list_iter_loss_zone_arrive_main = []
    train_list_epoch_loss_zone_arrive_main = []
    train_list_iter_loss_time_depart_main = []
    train_list_epoch_loss_time_depart_main = []
    train_list_iter_loss_time_arrive_main = []
    train_list_epoch_loss_time_arrive_main = []
    train_list_iter_loss_mode1_main = []
    train_list_epoch_loss_mode1_main = []
    train_list_iter_loss_mode2_main = []
    train_list_epoch_loss_mode2_main = []
    train_list_iter_loss = []
    train_list_epoch_loss = []

    train_list_iter_loss_r2_goal_main = []
    train_list_iter_loss_r2_zone_arrive_main = []
    train_list_iter_loss_r2_time_depart_main = []
    train_list_iter_loss_r2_time_arrive_main = []
    train_list_iter_loss_r2_mode1_main = []
    train_list_iter_loss_r2_mode2_main = []
    train_list_iter_loss_r2 = []

    train_list_iter_loss_rmse_goal_main = []
    train_list_iter_loss_rmse_zone_arrive_main = []
    train_list_iter_loss_rmse_time_depart_main = []
    train_list_iter_loss_rmse_time_arrive_main = []
    train_list_iter_loss_rmse_mode1_main = []
    train_list_iter_loss_rmse_mode2_main = []
    train_list_iter_loss_rmse = []

    train_list_iter_loss_mae_goal_main = []
    train_list_iter_loss_mae_zone_arrive_main = []
    train_list_iter_loss_mae_time_depart_main = []
    train_list_iter_loss_mae_time_arrive_main = []
    train_list_iter_loss_mae_mode1_main = []
    train_list_iter_loss_mae_mode2_main = []
    train_list_iter_loss_mae = []

    train_list_epoch_loss_r2_goal_main = []
    train_list_epoch_loss_r2_zone_arrive_main = []
    train_list_epoch_loss_r2_time_depart_main = []
    train_list_epoch_loss_r2_time_arrive_main = []
    train_list_epoch_loss_r2_mode1_main = []
    train_list_epoch_loss_r2_mode2_main = []
    train_list_epoch_loss_r2 = []

    train_list_epoch_loss_rmse_goal_main = []
    train_list_epoch_loss_rmse_zone_arrive_main = []
    train_list_epoch_loss_rmse_time_depart_main = []
    train_list_epoch_loss_rmse_time_arrive_main = []
    train_list_epoch_loss_rmse_mode1_main = []
    train_list_epoch_loss_rmse_mode2_main = []
    train_list_epoch_loss_rmse = []

    train_list_epoch_loss_mae_goal_main = []
    train_list_epoch_loss_mae_zone_arrive_main = []
    train_list_epoch_loss_mae_time_depart_main = []
    train_list_epoch_loss_mae_time_arrive_main = []
    train_list_epoch_loss_mae_mode1_main = []
    train_list_epoch_loss_mae_mode2_main = []
    train_list_epoch_loss_mae = []
    
    # 初始化时间记录变量
    # 在机器学习模型训练过程中，第一次训练的时间通常会比后续的训练时间长一些。这是由于多种因素导致的，see Notion notebook,预热

    epoch_start_times = []  # 每个epoch的开始时间
    epoch_durations = []  # 每个epoch的持续时间
    train_start_times = []
    train_durations = []
    iter_start_times =[]

    # 重新实例化模型以确保每次训练从头开始
    # model = CombinedNetwork()
    # # 重新实例化优化器
    # optimizer = tf.keras.optimizers.Adam()

    # # 重置模型的可训练变量和优化器状态
    # model.reset_weights()
    # optimizer = tf.keras.optimizers.Adam()

    start_time_epoch = time.time()  # 开始时间
    train_time_start = time.time()

    for epoch in range(num_epochs):
        print('epoch',epoch)
        epoch_start_times.append(time.time())  # 记录当前 epoch 的开始时间
        # print(epoch_start_times,'epoch_start_times')

        train_epoch_losses_goal_main = []  # 存储每个 train_epoch 的所有批次（13批次）损失
        train_epoch_losses_zone_depart_main = []
        train_epoch_losses_zone_arrive_main = []
        train_epoch_losses_time_depart_main = []
        train_epoch_losses_time_arrive_main = []
        train_epoch_losses_mode1_main = []
        train_epoch_losses_mode2_main = []
        train_epoch_losses = []

        train_epoch_losses_r2_goal_main = []
        train_epoch_losses_r2_zone_arrive_main = []
        train_epoch_losses_r2_time_depart_main = []
        train_epoch_losses_r2_time_arrive_main = []
        train_epoch_losses_r2_mode1_main = []
        train_epoch_losses_r2_mode2_main = []
        train_epoch_losses_r2 = []

        train_epoch_losses_rmse_goal_main = []
        train_epoch_losses_rmse_zone_arrive_main = []
        train_epoch_losses_rmse_time_depart_main = []
        train_epoch_losses_rmse_time_arrive_main = []
        train_epoch_losses_rmse_mode1_main = []
        train_epoch_losses_rmse_mode2_main = []
        train_epoch_losses_rmse = []

        train_epoch_losses_mae_goal_main = []
        train_epoch_losses_mae_zone_arrive_main = []
        train_epoch_losses_mae_time_depart_main = []
        train_epoch_losses_mae_time_arrive_main = []
        train_epoch_losses_mae_mode1_main = []
        train_epoch_losses_mae_mode2_main = []
        train_epoch_losses_mae = []
        

        # 重置迭代器状态
        train_iterator_goal_main = iter(train_dataset_goal_main)
        # print('train_iterator_goal_main11', train_iterator_goal_main)   # (1, 9)
        # train_iterator_zone_depart_main = iter(train_dataset_zones_depart_main)
        train_iterator_zone_arrive_main = iter(train_dataset_zones_arrive_main)
        train_iterator_time_depart_main = iter(train_dataset_times_depart_main)
        train_iterator_time_arrive_main = iter(train_dataset_times_arrive_main)
        train_iterator_mode1_main = iter(train_dataset_mode1_main)
        train_iterator_mode2_main = iter(train_dataset_mode2_main)

        # test_iterator_goal_main = iter(test_dataset_goal_main)
        # test_iterator_zone_depart_main = iter(test_dataset_zones_depart_main)
        # test_iterator_zone_arrive_main = iter(test_dataset_zones_arrive_main)
        # test_iterator_time_depart_main = iter(test_dataset_times_depart_main)
        # test_iterator_time_arrive_main = iter(test_dataset_times_arrive_main)
        # test_iterator_mode1_main = iter(test_dataset_mode1_main)
        # test_iterator_mode2_main = iter(test_dataset_mode2_main)

        # print(next(iter(train_dataset_goal_main)),'iter(train_dataset_goal_main)')
        # print(next(iter(train_dataset_zones_depart_main)))

        # 使用预先创建的迭代器
        train_num_batches = min(tf.data.experimental.cardinality(train_dataset_goal_main), tf.data.experimental.cardinality(train_dataset_zones_arrive_main), tf.data.experimental.cardinality(train_dataset_times_depart_main), tf.data.experimental.cardinality(train_dataset_times_arrive_main), tf.data.experimental.cardinality(train_dataset_mode1_main), tf.data.experimental.cardinality(train_dataset_mode2_main))   # 13
        # print(train_num_batches,'train_num_batches')   # tf.Tensor(6, shape=(), dtype=int64)，共有6批数据，每个周期epoch迭代6次
        # test_num_batches = min(tf.data.experimental.cardinality(test_dataset_goal_main), tf.data.experimental.cardinality(test_dataset_zones_depart_main), tf.data.experimental.cardinality(test_dataset_zones_arrive_main), tf.data.experimental.cardinality(test_dataset_times_depart_main), tf.data.experimental.cardinality(test_dataset_times_arrive_main), tf.data.experimental.cardinality(test_dataset_mode1_main), tf.data.experimental.cardinality(test_dataset_mode2_main))  # 13

        i = 1
        train_start_times.append(time.time())
        # print(train_start_times,'train_start_times')
        start_time_iter = time.time()

        # 为了绘制r2散点图，把每一次batch_labels_goals_main和loss_goal_main都保存下来，最后保存为一个一维数组
        # 初始化空列表用于保存 batch_labels_goals_main 和 loss_goal_main
        all_batch_labels_goals_main = []
        all_goal_main = []

        all_labels_goals_main = []
        all_labels_zone_arrive_main = []
        all_labels_time_depart_main = []
        all_labels_time_arrive_main = []
        all_labels_mode1_main = []
        all_labels_mode2_main = []


        for _ in range(train_num_batches):
            # print('4567468468')
            iter_start_times.append(time.time())

            batch_inputs_goal_main, batch_labels_goals_main = next(train_iterator_goal_main)
            # batch_inputs_zone_depart_main, batch_labels_zone_depart_main = next(train_iterator_zone_depart_main)
            batch_inputs_zone_arrive_main, batch_labels_zone_arrive_main = next(train_iterator_zone_arrive_main)
            batch_inputs_time_depart_main, batch_labels_time_depart_main = next(train_iterator_time_depart_main)
            batch_inputs_time_arrive_main, batch_labels_time_arrive_main = next(train_iterator_time_arrive_main)
            batch_inputs_mode1_main, batch_labels_mode1_main = next(train_iterator_mode1_main)
            batch_inputs_mode2_main, batch_labels_mode2_main = next(train_iterator_mode2_main)
            # print('555')   # 迭代次数 = num_epochs * train_num_batches

            # print('batch_inputs_goal_main11', batch_inputs_goal_main)    # (16,9)  batchsize=16

            all_labels_goals_main.append(batch_labels_goals_main)
            all_labels_zone_arrive_main.append(batch_labels_zone_arrive_main)
            all_labels_time_depart_main.append(batch_labels_time_depart_main)
            all_labels_time_arrive_main.append(batch_labels_time_arrive_main)
            all_labels_mode1_main.append(batch_labels_mode1_main)
            all_labels_mode2_main.append(batch_labels_mode2_main)


            # 从[batch_size, 3047]的batch_labels_zone_arrive_main中提取14列，以使得形状与train_step中的zones_arrive_main 匹配：
            # 创建从 zone_ids_int 到索引的映射
            id_to_index = {id: index for index, id in enumerate(zone_ids_int)}

            # 根据 zone_ids_int 提取相应的列
            def extract_relevant_zones(batch_labels_zone_arrive_main, zone_ids_int):
                indices = [id_to_index[id] for id in zone_ids_int]
                extracted_zones = tf.gather(batch_labels_zone_arrive_main, indices, axis=-1)
                return extracted_zones

            # 提取相关交通小区
            extracted_batch_labels_zone_arrive_main = extract_relevant_zones(batch_labels_zone_arrive_main, zone_ids_int)
            # 现在 extracted_batch_labels_zone_arrive_main 的形状应该是 [batch_size, 14]
            # print("Extracted Batch Labels Shape:", extracted_batch_labels_zone_arrive_main.shape)

            # 运行train_step
            loss_goal_main, loss_zone_arrive_main, loss_time_depart_main, loss_time_arrive_main, loss_mode1_main, loss_mode2_main, total_loss, r2_goals_main, r2_zones_arrive_main, r2_times_depart_main, r2_times_arrive_main, r2_modes1_main, r2_modes2_main, total_r2_loss, rmse_goals_main, rmse_zones_arrive_main, rmse_times_depart_main, rmse_times_arrive_main, rmse_modes1_main, rmse_modes2_main, total_rmse_loss, mae_goals_main, mae_zones_arrive_main, mae_times_depart_main, mae_times_arrive_main, mae_modes1_main, mae_modes2_main, total_mae_loss, goals_main, zones_arrive_main, times_depart_main, times_arrive_main, modes1_main, modes2_main \
                = train_step(batch_inputs_goal_main, batch_labels_goals_main, batch_inputs_zone_arrive_main, extracted_batch_labels_zone_arrive_main,
                             batch_inputs_time_depart_main, batch_labels_time_depart_main, batch_inputs_time_arrive_main, batch_labels_time_arrive_main,
                             batch_inputs_mode1_main, batch_labels_mode1_main, batch_inputs_mode2_main, batch_labels_mode2_main,
                             node_embeddings, zone_adjacency_matrix, distances,
                             model, optimizer)
            # print(loss_goal_main,'loss_goal_main1')

            # print('batch_inputs_goal_main33',batch_inputs_goal_main)   #(16, 9)


            train_end_time = time.time()

            # print(f"Losses: Goal={loss_goal}, Zone={loss_zone}, Time={loss_time}, Mode={loss_mode}")

            # print(time.time(),'time.time()')
            train_duration = time.time() - iter_start_times[-1]   # 每次迭代/每个批次的训练时间
            # print(train_duration,'train_duration')
            train_durations.append(train_duration)
            # print(f"Train {i} took {train_duration:.5f} seconds.")
            i += 1

            # 保存 batch_labels_goals_main 和 loss_goal_main，只保存第一次epoch的标签值即可，即为全部的label
            # if epoch == 0:
            # print('86574684', batch_labels_goals_main)   # (32, 5)
            # print('5774856785', goals_main)  # (32, 5)
            # all_batch_labels_goals_main.append(batch_labels_goals_main.numpy().ravel())  # 展平并保存
            all_goal_main.append(goals_main)


            train_list_iter_loss_goal_main.append(loss_goal_main.numpy())
            train_epoch_losses_goal_main.append(loss_goal_main.numpy())  # 将每个批次的损失值添加到列表中
            # train_list_iter_loss_zone_depart_main.append(loss_zone_depart_main.numpy())
            # train_epoch_losses_zone_depart_main.append(loss_zone_depart_main.numpy())
            train_list_iter_loss_zone_arrive_main.append(loss_zone_arrive_main.numpy())
            train_epoch_losses_zone_arrive_main.append(loss_zone_arrive_main.numpy())
            train_list_iter_loss_time_depart_main.append(loss_time_depart_main.numpy())
            train_epoch_losses_time_depart_main.append(loss_time_depart_main.numpy())
            train_list_iter_loss_time_arrive_main.append(loss_time_arrive_main.numpy())
            train_epoch_losses_time_arrive_main.append(loss_time_arrive_main.numpy())
            train_list_iter_loss_mode1_main.append(loss_mode1_main.numpy())
            train_epoch_losses_mode1_main.append(loss_mode1_main.numpy())
            train_list_iter_loss_mode2_main.append(loss_mode2_main.numpy())
            train_epoch_losses_mode2_main.append(loss_mode2_main.numpy())
            train_list_iter_loss.append(total_loss.numpy())
            train_epoch_losses.append(total_loss.numpy())
            
            # =========== cal R2, RMSE, MAE =======
            # print("r2_goals_main type:", type(r2_goals_main))   # 本身就是numpy类型
            # print("r2_goals_main value:", r2_goals_main)
            train_list_iter_loss_r2_goal_main.append(r2_goals_main)
            train_epoch_losses_r2_goal_main.append(r2_goals_main)  # 将每个批次的损失值添加到列表中
            train_list_iter_loss_r2_zone_arrive_main.append(r2_zones_arrive_main)
            train_epoch_losses_r2_zone_arrive_main.append(r2_zones_arrive_main)
            train_list_iter_loss_r2_time_depart_main.append(r2_times_depart_main)
            train_epoch_losses_r2_time_depart_main.append(r2_times_depart_main)
            train_list_iter_loss_r2_time_arrive_main.append(r2_times_arrive_main)
            train_epoch_losses_r2_time_arrive_main.append(r2_times_arrive_main)
            train_list_iter_loss_r2_mode1_main.append(r2_modes1_main)
            train_epoch_losses_r2_mode1_main.append(r2_modes1_main)
            train_list_iter_loss_r2_mode2_main.append(r2_modes2_main)
            train_epoch_losses_r2_mode2_main.append(r2_modes2_main)
            train_list_iter_loss_r2.append(total_r2_loss)
            train_epoch_losses_r2.append(total_r2_loss)

            train_list_iter_loss_rmse_goal_main.append(rmse_goals_main)
            train_epoch_losses_rmse_goal_main.append(rmse_goals_main)  # 将每个批次的损失值添加到列表中
            train_list_iter_loss_rmse_zone_arrive_main.append(rmse_zones_arrive_main)
            train_epoch_losses_rmse_zone_arrive_main.append(rmse_zones_arrive_main)
            train_list_iter_loss_rmse_time_depart_main.append(rmse_times_depart_main)
            train_epoch_losses_rmse_time_depart_main.append(rmse_times_depart_main)
            train_list_iter_loss_rmse_time_arrive_main.append(rmse_times_arrive_main)
            train_epoch_losses_rmse_time_arrive_main.append(rmse_times_arrive_main)
            train_list_iter_loss_rmse_mode1_main.append(rmse_modes1_main)
            train_epoch_losses_rmse_mode1_main.append(rmse_modes1_main)
            train_list_iter_loss_rmse_mode2_main.append(rmse_modes2_main)
            train_epoch_losses_rmse_mode2_main.append(rmse_modes2_main)
            train_list_iter_loss_rmse.append(total_rmse_loss)
            train_epoch_losses_rmse.append(total_rmse_loss)
            
            train_list_iter_loss_mae_goal_main.append(mae_goals_main)
            train_epoch_losses_mae_goal_main.append(mae_goals_main)  # 将每个批次的损失值添加到列表中
            train_list_iter_loss_mae_zone_arrive_main.append(mae_zones_arrive_main)
            train_epoch_losses_mae_zone_arrive_main.append(mae_zones_arrive_main)
            train_list_iter_loss_mae_time_depart_main.append(mae_times_depart_main)
            train_epoch_losses_mae_time_depart_main.append(mae_times_depart_main)
            train_list_iter_loss_mae_time_arrive_main.append(mae_times_arrive_main)
            train_epoch_losses_mae_time_arrive_main.append(mae_times_arrive_main)
            train_list_iter_loss_mae_mode1_main.append(mae_modes1_main)
            train_epoch_losses_mae_mode1_main.append(mae_modes1_main)
            train_list_iter_loss_mae_mode2_main.append(mae_modes2_main)
            train_epoch_losses_mae_mode2_main.append(mae_modes2_main)
            train_list_iter_loss_mae.append(total_mae_loss)
            train_epoch_losses_mae.append(total_mae_loss)

        # 将所有保存的值合并为一维数组
        # all_batch_labels_goals_main = np.concatenate(all_batch_labels_goals_main)  # 合并为一维数组
        # all_goal_main = np.array(all_goal_main)  # 转换为 NumPy 数组
        all_goal_main = np.concatenate(all_goal_main)
        all_labels_goals_main = np.concatenate(all_labels_goals_main, axis=0)
        all_labels_zone_arrive_main = np.concatenate(all_labels_zone_arrive_main, axis=0)
        all_labels_time_depart_main = np.concatenate(all_labels_time_depart_main, axis=0)
        all_labels_time_arrive_main = np.concatenate(all_labels_time_arrive_main, axis=0)
        all_labels_mode1_main = np.concatenate(all_labels_mode1_main, axis=0)
        all_labels_mode2_main = np.concatenate(all_labels_mode2_main, axis=0)
        # print("All Labels Goals Main Shape:", all_labels_goals_main.shape, all_labels_goals_main)   # (145, 5)

        # 打印结果（可选）
        # print("All Batch Labels Goals Main (Shape):", all_batch_labels_goals_main.shape)   # (725,)
        # print("All Goal Main (Shape):", all_goal_main)   # (145, 5)

        # train_end_time = time.time()
        # print('batch_inputs_goal_main4', batch_inputs_goal_main)  # (1, 9)

        # # 计算总训练时间
        # total_iter_training_time = time.time() - start_time_iter
        # # print(time.time(),'time.time()1')
        # # print(start_time_iter,'start_time_iter')
        # # 打印或记录总训练时间
        # print(f"Total iter training took {total_iter_training_time:.5f} seconds.")

        # 保存每个 iter（每次迭代） 的持续时间到 CSV 文件
        df_iter_durations = pd.DataFrame({'iter_duration': train_durations})
        df_iter_durations.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_time\01output_iter_time_durations.csv", index=False)  # 每次迭代iter的时间

        # # 保存总训练时间到文件
        # with open("01output_total_iter_training_time.txt", "w") as f:
        #     f.write(f"Total iter training took {total_iter_training_time:.5f} seconds.")

        # print(f"Loss Goal: {loss_goal.numpy()}, Loss Zone: {loss_zone.numpy()}")


        # 记录每个 epoch 的持续时间
        # print(time.time(),'time.time()')
        epoch_duration = time.time() - epoch_start_times[-1]
        # print(epoch_duration,'epoch_duration')
        epoch_durations.append(epoch_duration)
        # 打印或记录每个 epoch 的持续时间
        # print(f"Epoch {epoch + 1} took {epoch_duration:.5f} seconds.")

        # print(time.time(),'123')
        # 计算 epoch 的平均损失，用ave而不是sum，因为每个iter上计算的是MSE均方误差，是指每个样本上的平均损失。那么每个epoch上的损失也是求average
        train_avg_epoch_loss_goal_main = np.mean(train_epoch_losses_goal_main)
        train_list_epoch_loss_goal_main.append(train_avg_epoch_loss_goal_main)  # 添加 epoch 的平均损失到总的列表中
        # train_avg_epoch_loss_zone_depart_main = np.mean(train_epoch_losses_zone_depart_main)
        # train_list_epoch_loss_zone_depart_main.append(train_avg_epoch_loss_zone_depart_main)
        train_avg_epoch_loss_zone_arrive_main = np.mean(train_epoch_losses_zone_arrive_main)
        train_list_epoch_loss_zone_arrive_main.append(train_avg_epoch_loss_zone_arrive_main)
        train_avg_epoch_loss_time_depart_main = np.mean(train_epoch_losses_time_depart_main)
        train_list_epoch_loss_time_depart_main.append(train_avg_epoch_loss_time_depart_main)
        train_avg_epoch_loss_time_arrive_main = np.mean(train_epoch_losses_time_arrive_main)
        train_list_epoch_loss_time_arrive_main.append(train_avg_epoch_loss_time_arrive_main)
        train_avg_epoch_loss_mode1_main = np.mean(train_epoch_losses_mode1_main)
        train_list_epoch_loss_mode1_main.append(train_avg_epoch_loss_mode1_main)
        train_avg_epoch_loss_mode2_main = np.mean(train_epoch_losses_mode2_main)
        train_list_epoch_loss_mode2_main.append(train_avg_epoch_loss_mode2_main)
        train_avg_epoch_loss = np.mean(train_epoch_losses)
        train_list_epoch_loss.append(train_avg_epoch_loss)

        # =========== cal R2, RMSE, MAE =======   
        train_avg_epoch_loss_r2_goal_main = np.mean(train_epoch_losses_r2_goal_main)
        train_list_epoch_loss_r2_goal_main.append(train_avg_epoch_loss_r2_goal_main)  # 添加 epoch 的平均损失到总的列表中
        train_avg_epoch_loss_r2_zone_arrive_main = np.mean(train_epoch_losses_zone_arrive_main)
        train_list_epoch_loss_r2_zone_arrive_main.append(train_avg_epoch_loss_r2_zone_arrive_main)
        train_avg_epoch_loss_r2_time_depart_main = np.mean(train_epoch_losses_time_depart_main)
        train_list_epoch_loss_r2_time_depart_main.append(train_avg_epoch_loss_r2_time_depart_main)
        train_avg_epoch_loss_r2_time_arrive_main = np.mean(train_epoch_losses_time_arrive_main)
        train_list_epoch_loss_r2_time_arrive_main.append(train_avg_epoch_loss_r2_time_arrive_main)
        train_avg_epoch_loss_r2_mode1_main = np.mean(train_epoch_losses_mode1_main)
        train_list_epoch_loss_r2_mode1_main.append(train_avg_epoch_loss_r2_mode1_main)
        train_avg_epoch_loss_r2_mode2_main = np.mean(train_epoch_losses_mode2_main)
        train_list_epoch_loss_r2_mode2_main.append(train_avg_epoch_loss_r2_mode2_main)
        train_avg_epoch_loss_r2 = np.mean(train_epoch_losses)
        train_list_epoch_loss_r2.append(train_avg_epoch_loss_r2)

        train_avg_epoch_loss_rmse_goal_main = np.mean(train_epoch_losses_rmse_goal_main)
        train_list_epoch_loss_rmse_goal_main.append(train_avg_epoch_loss_rmse_goal_main)  # 添加 epoch 的平均损失到总的列表中
        train_avg_epoch_loss_rmse_zone_arrive_main = np.mean(train_epoch_losses_rmse_zone_arrive_main)
        train_list_epoch_loss_rmse_zone_arrive_main.append(train_avg_epoch_loss_rmse_zone_arrive_main)
        train_avg_epoch_loss_rmse_time_depart_main = np.mean(train_epoch_losses_rmse_time_depart_main)
        train_list_epoch_loss_rmse_time_depart_main.append(train_avg_epoch_loss_rmse_time_depart_main)
        train_avg_epoch_loss_rmse_time_arrive_main = np.mean(train_epoch_losses_rmse_time_arrive_main)
        train_list_epoch_loss_rmse_time_arrive_main.append(train_avg_epoch_loss_rmse_time_arrive_main)
        train_avg_epoch_loss_rmse_mode1_main = np.mean(train_epoch_losses_rmse_mode1_main)
        train_list_epoch_loss_rmse_mode1_main.append(train_avg_epoch_loss_rmse_mode1_main)
        train_avg_epoch_loss_rmse_mode2_main = np.mean(train_epoch_losses_rmse_mode2_main)
        train_list_epoch_loss_rmse_mode2_main.append(train_avg_epoch_loss_rmse_mode2_main)
        train_avg_epoch_loss_rmse = np.mean(train_epoch_losses_rmse)
        train_list_epoch_loss_rmse.append(train_avg_epoch_loss_rmse)

        train_avg_epoch_loss_mae_goal_main = np.mean(train_epoch_losses_mae_goal_main)
        train_list_epoch_loss_mae_goal_main.append(train_avg_epoch_loss_mae_goal_main)  # 添加 epoch 的平均损失到总的列表中
        train_avg_epoch_loss_mae_zone_arrive_main = np.mean(train_epoch_losses_mae_zone_arrive_main)
        train_list_epoch_loss_mae_zone_arrive_main.append(train_avg_epoch_loss_mae_zone_arrive_main)
        train_avg_epoch_loss_mae_time_depart_main = np.mean(train_epoch_losses_mae_time_depart_main)
        train_list_epoch_loss_mae_time_depart_main.append(train_avg_epoch_loss_mae_time_depart_main)
        train_avg_epoch_loss_mae_time_arrive_main = np.mean(train_epoch_losses_mae_time_arrive_main)
        train_list_epoch_loss_mae_time_arrive_main.append(train_avg_epoch_loss_mae_time_arrive_main)
        train_avg_epoch_loss_mae_mode1_main = np.mean(train_epoch_losses_mae_mode1_main)
        train_list_epoch_loss_mae_mode1_main.append(train_avg_epoch_loss_mae_mode1_main)
        train_avg_epoch_loss_mae_mode2_main = np.mean(train_epoch_losses_mae_mode2_main)
        train_list_epoch_loss_mae_mode2_main.append(train_avg_epoch_loss_mae_mode2_main)
        train_avg_epoch_loss_mae = np.mean(train_epoch_losses_mae)
        train_list_epoch_loss_mae.append(train_avg_epoch_loss_mae)


        # print(f"Epoch_goal {epoch + 1}, Loss_goal: {loss_goal:.4f}, avg_epoch_loss_goal: {avg_epoch_loss_goal:.4f}, list_epoch_loss_goal: {list_epoch_loss_goal}")
        # print(f"Epoch_zone {epoch + 1}, Loss_zone: {loss_zone:.4f}, avg_epoch_loss_zone: {avg_epoch_loss_zone:.4f}, list_epoch_loss_zone: {list_epoch_loss_zone}")


        # test_avg_epoch_loss_goal_main = np.mean(test_epoch_losses_goal_main)
        # test_list_epoch_loss_goal_main.append(test_avg_epoch_loss_goal_main)  # 添加 epoch 的平均损失到总的列表中
        # test_avg_epoch_loss_zone_depart_main = np.mean(test_epoch_losses_zone_depart_main)
        # test_list_epoch_loss_zone_depart_main.append(test_avg_epoch_loss_zone_depart_main)
        # test_avg_epoch_loss_zone_arrive_main = np.mean(test_epoch_losses_zone_arrive_main)
        # test_list_epoch_loss_zone_arrive_main.append(test_avg_epoch_loss_zone_arrive_main)
        # test_avg_epoch_loss_time_depart_main = np.mean(test_epoch_losses_time_depart_main)
        # test_list_epoch_loss_time_depart_main.append(test_avg_epoch_loss_time_depart_main)
        # test_avg_epoch_loss_time_arrive_main = np.mean(test_epoch_losses_time_arrive_main)
        # test_list_epoch_loss_time_arrive_main.append(test_avg_epoch_loss_time_arrive_main)
        # test_avg_epoch_loss_mode1_main = np.mean(test_epoch_losses_mode1_main)
        # test_list_epoch_loss_mode1_main.append(test_avg_epoch_loss_mode1_main)
        # test_avg_epoch_loss_mode2_main = np.mean(test_epoch_losses_mode2_main)
        # test_list_epoch_loss_mode2_main.append(test_avg_epoch_loss_mode2_main)
        # test_avg_epoch_loss = np.mean(test_epoch_losses)
        # test_list_epoch_loss.append(test_avg_epoch_loss)


        #
        #
        #
        # # 早停法检查（Early Stopping）
        # if test_avg_epoch_loss < best_test_loss:
        #     best_test_loss = test_avg_epoch_loss
        #     no_improvement_count = 0
        #     # print(no_improvement_count,'no_improvement_count')
        #     # 保存当前模型状态
        #     model.save_weights('best_model.weights.h5')
        # else:
        #     no_improvement_count += 1
        #
        # if no_improvement_count >= patience:
        #     # print(f"Early stopping at epoch {epoch}")
        #     epoch_early_stopping = epoch
        #     break

    train_time_end = time.time()
    train_time_duration_epoch = train_time_end - train_time_start

    # 计算总训练时间
    # time1 = time.time()
    # print(time1-test_end_time,'980')
    total_epoch_training_time = train_end_time - start_time_epoch
    # 打印或记录总训练时间
    # print(f"Total epoch training and test took {total_epoch_training_time:.7f} seconds.")

    # 保存每个 epoch 的持续时间到 CSV 文件
    df_epoch_durations = pd.DataFrame({'epoch_duration': epoch_durations})
    df_epoch_durations.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_time\01output_each_epoch(training)_time_durations.csv", index=False)  # 每个周期epoch的时间

    print('Finished: Training Model based on the best hyper-parameters.')


    # ============= test =============
    # test_epoch_losses_goal_main = []  # 存储每个 test_epoch 的所有批次（13批次）损失
    # test_epoch_losses_zone_depart_main = []
    # test_epoch_losses_zone_arrive_main = []
    # test_epoch_losses_time_depart_main = []
    # test_epoch_losses_time_arrive_main = []
    # test_epoch_losses_mode1_main = []
    # test_epoch_losses_mode2_main = []
    # test_epoch_losses = []

    # 初始化变量来存储累加的总损失和批次数量
    total_test_loss = 0
    num_batches = 0

    test_list_iter_loss_goal_main = []  # collect each iter in each epoch ' loss  (batchsize=10, 每个 epoch 会有 13 次迭代iteration)
    # test_list_epoch_loss_goal_main = []
    test_list_iter_loss_zone_depart_main = []
    # test_list_epoch_loss_zone_depart_main = []
    test_list_iter_loss_zone_arrive_main = []
    # test_list_epoch_loss_zone_arrive_main = []
    test_list_iter_loss_time_depart_main = []
    # test_list_epoch_loss_time_depart_main = []
    test_list_iter_loss_time_arrive_main = []
    # test_list_epoch_loss_time_arrive_main = []
    test_list_iter_loss_mode1_main = []
    # test_list_epoch_loss_mode1_main = []
    test_list_iter_loss_mode2_main = []
    # test_list_epoch_loss_mode2_main = []
    test_list_iter_loss = []
    # test_list_epoch_loss = []

    test_list_iter_loss_r2_goal_main = []
    test_list_iter_loss_r2_zone_arrive_main = []
    test_list_iter_loss_r2_time_depart_main = []
    test_list_iter_loss_r2_time_arrive_main = []
    test_list_iter_loss_r2_mode1_main = []
    test_list_iter_loss_r2_mode2_main = []
    test_list_iter_loss_r2 = []
    
    test_list_iter_loss_rmse_goal_main = []
    test_list_iter_loss_rmse_zone_arrive_main = []
    test_list_iter_loss_rmse_time_depart_main = []
    test_list_iter_loss_rmse_time_arrive_main = []
    test_list_iter_loss_rmse_mode1_main = []
    test_list_iter_loss_rmse_mode2_main = []
    test_list_iter_loss_rmse = []
    
    test_list_iter_loss_mae_goal_main = []
    test_list_iter_loss_mae_zone_arrive_main = []
    test_list_iter_loss_mae_time_depart_main = []
    test_list_iter_loss_mae_time_arrive_main = []
    test_list_iter_loss_mae_mode1_main = []
    test_list_iter_loss_mae_mode2_main = []
    test_list_iter_loss_mae = []
    
    
    # 初始化累积变量
    all_predictions = {
        'goals_main': [],
        'zones_depart_main': [],
        'zones_arrive_main': [],
        'times_depart_main': [],
        'times_arrive_main': [],
        'modes1_main': [],
        'modes2_main': []
    }

    all_labels = {
        'goals_main': [],
        'zones_depart_main': [],
        'zones_arrive_main': [],
        'times_depart_main': [],
        'times_arrive_main': [],
        'modes1_main': [],
        'modes2_main': []
    }

    test_time_start = time.time()
    # 遍历测试集的所有批次,使用 zip 同步遍历多个数据集
    for batch_data in zip(test_dataset_goal_main, test_dataset_zones_arrive_main, test_dataset_times_depart_main, test_dataset_times_arrive_main,
                          test_dataset_mode1_main, test_dataset_mode2_main):

        num_batches += 1
        # print('num_batches',num_batches)

        # 解包批次数据
        (batch_inputs_goal_main, batch_labels_goals_main), \
            (batch_inputs_zone_arrive_main, batch_labels_zone_arrive_main), \
            (batch_inputs_time_depart_main, batch_labels_time_depart_main), \
            (batch_inputs_time_arrive_main, batch_labels_time_arrive_main), \
            (batch_inputs_mode1_main, batch_labels_mode1_main), \
            (batch_inputs_mode2_main, batch_labels_mode2_main) = batch_data

        # 从[batch_size, 3047]的batch_labels_zone_arrive_main中提取14列，以使得形状与train_step中的zones_arrive_main 匹配：
        # 创建从 zone_ids_int 到索引的映射
        id_to_index = {id: index for index, id in enumerate(zone_ids_int)}

        # 根据 zone_ids_int 提取相应的列
        def extract_relevant_zones(batch_labels_zone_arrive_main, zone_ids_int):
            indices = [id_to_index[id] for id in zone_ids_int]
            extracted_zones = tf.gather(batch_labels_zone_arrive_main, indices, axis=-1)
            return extracted_zones

        # 提取相关交通小区
        extracted_batch_labels_zone_arrive_main = extract_relevant_zones(batch_labels_zone_arrive_main, zone_ids_int)
        # 现在 extracted_batch_labels_zone_arrive_main 的形状应该是 [batch_size, 14]
        # print("Extracted Batch Labels Shape_test:", extracted_batch_labels_zone_arrive_main.shape)

        # 测试时，不需要循环迭代多次，num_epoch=1，只需要前向传递一次即可
        # 前向传递 test_step
        (predictions_goal_main, predictions_zone_arrive_main,
         predictions_time_depart_main, predictions_time_arrive_main, predictions_mode1_main, predictions_mode2_main,
         loss_goal_main, loss_zone_arrive_main, loss_time_depart_main, \
            loss_time_arrive_main, loss_mode1_main, loss_mode2_main, total_loss, 
         r2_goals_main, r2_zones_arrive_main, r2_times_depart_main, r2_times_arrive_main, r2_modes1_main, r2_modes2_main, total_r2_loss, 
         rmse_goals_main, rmse_zones_arrive_main, rmse_times_depart_main, rmse_times_arrive_main, rmse_modes1_main, rmse_modes2_main, total_rmse_loss, 
         mae_goals_main, mae_zones_arrive_main, mae_times_depart_main, mae_times_arrive_main, mae_modes1_main, mae_modes2_main, total_mae_loss,
         goals_main, zones_arrive_main, times_depart_main, times_arrive_main, modes1_main, modes2_main) \
            = test_step(batch_inputs_goal_main, batch_labels_goals_main, batch_inputs_zone_arrive_main, extracted_batch_labels_zone_arrive_main,
            batch_inputs_time_depart_main, batch_labels_time_depart_main,
            batch_inputs_time_arrive_main, batch_labels_time_arrive_main,
            batch_inputs_mode1_main, batch_labels_mode1_main,
            batch_inputs_mode2_main, batch_labels_mode2_main, node_embeddings, zone_adjacency_matrix, distances, model)

        test_time_end = time.time()  # 每个for循环中，都会更新，for之外用最后一次的

        test_list_iter_loss_goal_main.append(loss_goal_main.numpy())
        # test_epoch_losses_goal_main.append(loss_goal_main.numpy())  # 将每个批次的损失值添加到列表中，测试集中只有一个epoch
        # test_list_iter_loss_zone_depart_main.append(loss_zone_depart_main.numpy())
        # test_epoch_losses_zone_depart_main.append(loss_zone_depart_main.numpy())
        test_list_iter_loss_zone_arrive_main.append(loss_zone_arrive_main.numpy())
        # test_epoch_losses_zone_arrive_main.append(loss_zone_arrive_main.numpy())
        test_list_iter_loss_time_depart_main.append(loss_time_depart_main.numpy())
        # test_epoch_losses_time_depart_main.append(loss_time_depart_main.numpy())
        test_list_iter_loss_time_arrive_main.append(loss_time_arrive_main.numpy())
        # test_epoch_losses_time_arrive_main.append(loss_time_arrive_main.numpy())
        test_list_iter_loss_mode1_main.append(loss_mode1_main.numpy())
        # test_epoch_losses_mode1_main.append(loss_mode1_main.numpy())
        test_list_iter_loss_mode2_main.append(loss_mode2_main.numpy())
        # test_epoch_losses_mode2_main.append(loss_mode2_main.numpy())
        test_list_iter_loss.append(total_loss.numpy())
        # test_epoch_losses.append(total_loss.numpy())

        test_list_iter_loss_r2_goal_main.append(r2_goals_main)
        test_list_iter_loss_r2_zone_arrive_main.append(r2_zones_arrive_main)
        test_list_iter_loss_r2_time_depart_main.append(r2_times_depart_main)
        test_list_iter_loss_r2_time_arrive_main.append(r2_times_arrive_main)
        test_list_iter_loss_r2_mode1_main.append(r2_modes1_main)
        test_list_iter_loss_r2_mode2_main.append(r2_modes2_main)
        test_list_iter_loss_r2.append(total_r2_loss)

        test_list_iter_loss_rmse_goal_main.append(rmse_goals_main)
        test_list_iter_loss_rmse_zone_arrive_main.append(rmse_zones_arrive_main)
        test_list_iter_loss_rmse_time_depart_main.append(rmse_times_depart_main)
        test_list_iter_loss_rmse_time_arrive_main.append(rmse_times_arrive_main)
        test_list_iter_loss_rmse_mode1_main.append(rmse_modes1_main)
        test_list_iter_loss_rmse_mode2_main.append(rmse_modes2_main)
        test_list_iter_loss_rmse.append(total_rmse_loss)

        test_list_iter_loss_mae_goal_main.append(mae_goals_main)
        test_list_iter_loss_mae_zone_arrive_main.append(mae_zones_arrive_main)
        test_list_iter_loss_mae_time_depart_main.append(mae_times_depart_main)
        test_list_iter_loss_mae_time_arrive_main.append(mae_times_arrive_main)
        test_list_iter_loss_mae_mode1_main.append(mae_modes1_main)
        test_list_iter_loss_mae_mode2_main.append(mae_modes2_main)
        test_list_iter_loss_mae.append(total_mae_loss)

        # 累加总损失
        total_test_loss += total_loss.numpy()

        # 收集预测和标签，用以计算accuracy等分类指标
        all_predictions['goals_main'].extend(predictions_goal_main.numpy())
        # all_predictions['zones_depart_main'].extend(predictions_zone_depart_main.numpy())
        all_predictions['zones_arrive_main'].extend(predictions_zone_arrive_main.numpy())
        all_predictions['times_depart_main'].extend(predictions_time_depart_main.numpy())
        all_predictions['times_arrive_main'].extend(predictions_time_arrive_main.numpy())
        all_predictions['modes1_main'].extend(predictions_mode1_main.numpy())
        all_predictions['modes2_main'].extend(predictions_mode2_main.numpy())

        # print('all_predictions:', all_predictions)

        all_labels['goals_main'].extend(tf.argmax(batch_labels_goals_main, axis=1).numpy())
        # all_labels['zones_depart_main'].extend(tf.argmax(batch_labels_zone_depart_main, axis=1).numpy())
        all_labels['zones_arrive_main'].extend(tf.argmax(batch_labels_zone_arrive_main, axis=1).numpy())
        all_labels['times_depart_main'].extend(tf.argmax(batch_labels_time_depart_main, axis=1).numpy())
        all_labels['times_arrive_main'].extend(tf.argmax(batch_labels_time_arrive_main, axis=1).numpy())
        all_labels['modes1_main'].extend(tf.argmax(batch_labels_mode1_main, axis=1).numpy())
        all_labels['modes2_main'].extend(tf.argmax(batch_labels_mode2_main, axis=1).numpy())
        # print('all_labels:', all_labels)

    # test_time_end = time.time()

    test_time_duration = test_time_end - test_time_start



    # 输入数据列表
    inputs_dynamic_parameters = [
        batch_inputs_goal_main,
        batch_inputs_zone_arrive_main,
        batch_inputs_time_depart_main, batch_inputs_time_arrive_main, batch_inputs_mode1_main, batch_inputs_mode2_main
    ]

    file_name_dynamic_parameters = fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\dynamic_parameters.csv"
    # 提取动态参数并保存到 CSV 文件
    # extract_and_save_dynamic_parameters(model, inputs_dynamic_parameters, node_embeddings, zone_adjacency_matrix, distances, file_name_dynamic_parameters)




    # 保存总训练时间到文件
    with open(r"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_time\01output_total_epoch_training_and_test_time.txt", "w") as f:
        f.write(f"Total epoch training took {total_epoch_training_time:.5f} seconds. \n ")
        f.write(f"Total epoch training took (including save loss value) {train_time_duration_epoch:.5f} seconds. \n ")
        f.write(f"Total iter testing took {test_time_duration:.5f} seconds. \n")
        f.write(f"Total iter training and testing took {total_epoch_training_time + test_time_duration:.5f} seconds.")

    # 计算平均损失
    average_test_loss = total_test_loss / num_batches

    # print(f"Total Test Loss: {total_test_loss:.4f}")
    # print(f"Average Test Loss: {average_test_loss:.4f}")

    # 定义文件名
    output_filename = fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss\01test_loss_results.txt"

    # 将结果保存到 txt 文件中
    with open(output_filename, 'w') as f:
        f.write(f"Total Test Loss: {total_test_loss:.4f}\n")
        f.write(f"Average Test Loss: {average_test_loss:.4f}\n")
    # print(f"Test loss results have been saved to {output_filename}")


    # 计算所有批次的总损失和平均损失，其中total_loss的总损失和平均损失已在上一行中计算出来
    # 定义所有需要处理的损失列表
    loss_lists = {
        'goal_main': test_list_iter_loss_goal_main,
        'zone_arrive_main': test_list_iter_loss_zone_arrive_main,
        'time_depart_main': test_list_iter_loss_time_depart_main,
        'time_arrive_main': test_list_iter_loss_time_arrive_main,
        'mode1_main': test_list_iter_loss_mode1_main,
        'mode2_main': test_list_iter_loss_mode2_main,
        'total_loss': test_list_iter_loss
    }

    loss_r2_lists = {
        'goal_main': test_list_iter_loss_r2_goal_main,
        'zone_arrive_main': test_list_iter_loss_r2_zone_arrive_main,
        'time_depart_main': test_list_iter_loss_r2_time_depart_main,
        'time_arrive_main': test_list_iter_loss_r2_time_arrive_main,
        'mode1_main': test_list_iter_loss_r2_mode1_main,
        'mode2_main': test_list_iter_loss_r2_mode2_main,
        'total_loss': test_list_iter_loss_r2
    }
 
    loss_rmse_lists = {
        'goal_main': test_list_iter_loss_rmse_goal_main,
        'zone_arrive_main': test_list_iter_loss_rmse_zone_arrive_main,
        'time_depart_main': test_list_iter_loss_rmse_time_depart_main,
        'time_arrive_main': test_list_iter_loss_rmse_time_arrive_main,
        'mode1_main': test_list_iter_loss_rmse_mode1_main,
        'mode2_main': test_list_iter_loss_rmse_mode2_main,
        'total_loss': test_list_iter_loss_rmse
    }

    loss_mae_lists = {
        'goal_main': test_list_iter_loss_mae_goal_main,
        'zone_arrive_main': test_list_iter_loss_mae_zone_arrive_main,
        'time_depart_main': test_list_iter_loss_mae_time_depart_main,
        'time_arrive_main': test_list_iter_loss_mae_time_arrive_main,
        'mode1_main': test_list_iter_loss_mae_mode1_main,
        'mode2_main': test_list_iter_loss_mae_mode2_main,
        'total_loss': test_list_iter_loss_mae
    }
    
    # 存储结果的字典
    results = {}
    # 对每个损失列表计算总和和平均值
    for name, loss_list in loss_lists.items():
        total, avg = compute_sum_and_average(loss_list)
        # results[name] = {'Total_Loss': total, 'Average_Loss': avg}
        print(f"{name}: Total_Loss = {total:.4f}, Average_Loss = {avg:.4f}")
        last_iter_loss = loss_list[-1] if loss_list else None  # 获取最后一次迭代的损失值
        # print('loss_list', loss_list)
        # print('last_iter_loss', last_iter_loss)

        # 检查是否有 None 值，并处理：如果是none值，也保存为“none”
        formatted_total = f"{total:.4f}" if total is not None else "None"
        formatted_avg = f"{avg:.4f}" if avg is not None else "None"
        formatted_last_iter_loss = f"{last_iter_loss:.4f}" if last_iter_loss is not None else "None"

        results[name] = {'Total_Loss': total, 'Average_Loss': avg, 'Last_Iter_Test_Loss': last_iter_loss}
        # print(f"{name}: Total_Loss = {total:.4f}, Average_Loss = {avg:.4f}, Last_Iter_Test_Loss = {formatted_last_iter_loss:.4f}")
        # print(f"{name}: Total_Loss = {formatted_total}, Average_Loss = {formatted_avg}, Last_Iter_Test_Loss = {formatted_last_iter_loss}")

    # 将结果字典转换为 DataFrame，以保存到csv中
    df_results = pd.DataFrame.from_dict(results, orient='index')
    # 重置索引以便更好地显示
    df_results = df_results.reset_index().rename(columns={'index': 'Loss Type'})
    # 打印结果（可选）
    # print(df_results)
    # 保存到 CSV 文件
    output_csv_path = r'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss\01test_losses_mse_summary_sum&average_loss.csv'
    df_results.to_csv(output_csv_path, index=False)
    # print(f"Results saved to {output_csv_path}")

    # 存储结果的字典
    results_r2 = {}
    # 对每个损失列表计算总和和平均值
    for name, loss_list in loss_r2_lists.items():
        total, avg = compute_sum_and_average(loss_list)
        # results_r2[name] = {'Total_Loss': total, 'Average_Loss': avg}
        print(f"r2 {name}: Total_Loss = {total:.4f}, Average_Loss = {avg:.4f}")
        last_iter_loss = loss_list[-1] if loss_list else None  # 获取最后一次迭代的损失值
        # print('loss_list', loss_list)
        # print('last_iter_loss', last_iter_loss)

        # 检查是否有 None 值，并处理：如果是none值，也保存为“none”
        formatted_total = f"{total:.4f}" if total is not None else "None"
        formatted_avg = f"{avg:.4f}" if avg is not None else "None"
        formatted_last_iter_loss = f"{last_iter_loss:.4f}" if last_iter_loss is not None else "None"

        results_r2[name] = {'Total_Loss': total, 'Average_Loss': avg, 'Last_Iter_Test_Loss': last_iter_loss}
        # print(f"{name}: Total_Loss = {total:.4f}, Average_Loss = {avg:.4f}, Last_Iter_Test_Loss = {formatted_last_iter_loss:.4f}")
        # print(f"{name}: Total_Loss = {formatted_total}, Average_Loss = {formatted_avg}, Last_Iter_Test_Loss = {formatted_last_iter_loss}")

    # 将结果字典转换为 DataFrame，以保存到csv中
    df_results_r2 = pd.DataFrame.from_dict(results_r2, orient='index')
    # 重置索引以便更好地显示
    df_results_r2 = df_results_r2.reset_index().rename(columns={'index': 'Loss Type'})
    # 打印结果（可选）
    # print(df_results_r2)
    # 保存到 CSV 文件
    output_csv_path_r2 = r'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_r2\01test_losses_r2_summary_sum&average_loss.csv'
    df_results_r2.to_csv(output_csv_path_r2, index=False)
    # print(f"results_r2 saved to {output_csv_path}")

    # 存储结果的字典
    results_rmse = {}
    # 对每个损失列表计算总和和平均值
    for name, loss_list in loss_rmse_lists.items():
        total, avg = compute_sum_and_average(loss_list)
        # results_rmse[name] = {'Total_Loss': total, 'Average_Loss': avg}
        print(f"r2 {name}: Total_Loss = {total:.4f}, Average_Loss = {avg:.4f}")
        last_iter_loss = loss_list[-1] if loss_list else None  # 获取最后一次迭代的损失值
        # print('loss_list', loss_list)
        # print('last_iter_loss', last_iter_loss)

        # 检查是否有 None 值，并处理：如果是none值，也保存为“none”
        formatted_total = f"{total:.4f}" if total is not None else "None"
        formatted_avg = f"{avg:.4f}" if avg is not None else "None"
        formatted_last_iter_loss = f"{last_iter_loss:.4f}" if last_iter_loss is not None else "None"

        results_rmse[name] = {'Total_Loss': total, 'Average_Loss': avg, 'Last_Iter_Test_Loss': last_iter_loss}
        # print(f"{name}: Total_Loss = {total:.4f}, Average_Loss = {avg:.4f}, Last_Iter_Test_Loss = {formatted_last_iter_loss:.4f}")
        # print(f"{name}: Total_Loss = {formatted_total}, Average_Loss = {formatted_avg}, Last_Iter_Test_Loss = {formatted_last_iter_loss}")

    # 将结果字典转换为 DataFrame，以保存到csv中
    df_results_rmse = pd.DataFrame.from_dict(results_rmse, orient='index')
    # 重置索引以便更好地显示
    df_results_rmse = df_results_rmse.reset_index().rename(columns={'index': 'Loss Type'})
    # 打印结果（可选）
    # print(df_results_rmse)
    # 保存到 CSV 文件
    output_csv_path_rmse = r'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_rmse\01test_losses_rmse_summary_sum&average_loss.csv'
    df_results_rmse.to_csv(output_csv_path_rmse, index=False)
    # print(f"results_rmse saved to {output_csv_path}")
    
    # 存储结果的字典
    results_mae = {}
    # 对每个损失列表计算总和和平均值
    for name, loss_list in loss_mae_lists.items():
        total, avg = compute_sum_and_average(loss_list)
        # results_mae[name] = {'Total_Loss': total, 'Average_Loss': avg}
        print(f"r2 {name}: Total_Loss = {total:.4f}, Average_Loss = {avg:.4f}")
        last_iter_loss = loss_list[-1] if loss_list else None  # 获取最后一次迭代的损失值
        # print('loss_list', loss_list)
        # print('last_iter_loss', last_iter_loss)

        # 检查是否有 None 值，并处理：如果是none值，也保存为“none”
        formatted_total = f"{total:.4f}" if total is not None else "None"
        formatted_avg = f"{avg:.4f}" if avg is not None else "None"
        formatted_last_iter_loss = f"{last_iter_loss:.4f}" if last_iter_loss is not None else "None"

        results_mae[name] = {'Total_Loss': total, 'Average_Loss': avg, 'Last_Iter_Test_Loss': last_iter_loss}
        # print(f"{name}: Total_Loss = {total:.4f}, Average_Loss = {avg:.4f}, Last_Iter_Test_Loss = {formatted_last_iter_loss:.4f}")
        # print(f"{name}: Total_Loss = {formatted_total}, Average_Loss = {formatted_avg}, Last_Iter_Test_Loss = {formatted_last_iter_loss}")

    # 将结果字典转换为 DataFrame，以保存到csv中
    df_results_mae = pd.DataFrame.from_dict(results_mae, orient='index')
    # 重置索引以便更好地显示
    df_results_mae = df_results_mae.reset_index().rename(columns={'index': 'Loss Type'})
    # 打印结果（可选）
    # print(df_results_mae)
    # 保存到 CSV 文件
    output_csv_path_mae = r'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_mae\01test_losses_mae_summary_sum&average_loss.csv'
    df_results_mae.to_csv(output_csv_path_mae, index=False)
    # print(f"results_mae saved to {output_csv_path}")

    # ============= 计算标签级的accuracy等评估指标（goal_main预测正确 即正确），并保存到csv中 =============
    metrics = {}
    support_metrics = {}

    for key in all_predictions.keys():
        # 计算分类报告并获取字典形式的输出
        report = classification_report(all_labels[key], all_predictions[key], output_dict=True)

        # 提取主要评估指标
        metrics[key] = {
            'accuracy': report['accuracy'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score']
        }

        # 提取 support 信息
        support_metrics[key] = {cls: info['support'] for cls, info in report.items() if
                                isinstance(info, dict) and 'support' in info}
    # print('3all_predictions:', all_predictions)
    # print('3all_labels:', all_labels)

    # 打印评估结果
    for key, value in metrics.items():
        print(f"Metrics for {key}:")
        for metric, score in value.items():
            print(f"{metric}: {score:.4f}")
        print("\n")
    print(f"Total number of batches processed: {num_batches}")

    # save respectively：
    metrics_dfs = {}
    # 遍历每个任务并创建对应的 DataFrame 和 CSV 文件
    for key, value in metrics.items():
        # 创建 DataFrame 并添加 'Label Type' 列
        df = pd.DataFrame([value])
        df.insert(0, 'Label Type', key)  # 插入 'Label Type' 作为第一列
        metrics_dfs[key] = df

        # 添加 support 信息
        support_info = support_metrics.get(key, {})
        for cls, supp in support_info.items():
            df[f'support_class_{cls}'] = supp

        # 保存到 CSV 文件
        csv_filename = fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_accuracy\01{key}_metrics_accuracy.csv"
        df.to_csv(csv_filename, index=False)
        # print(f"Saved metrics for {key} to {csv_filename}")

    # save sum:
    # 创建一个汇总 DataFrame
    summary_df = pd.DataFrame()
    # 收集所有任务的评估指标
    for key, df in metrics_dfs.items():
        # 为每个任务添加一个任务名称列
        df['task'] = key
        support_info = support_metrics.get(key, {})
        for cls, supp in support_info.items():
            df[f'support_class_{cls}'] = supp
        summary_df = pd.concat([summary_df, df], ignore_index=True)

    # 保存汇总结果到 CSV 文件
    summary_csv_filename = fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_accuracy\01all_metrics_summary_accuracy.csv"
    summary_df.to_csv(summary_csv_filename, index=False)
    print(f"Saved all accuracy metrics summary to {summary_csv_filename}")

    # # ============= （不用） 计算样本级的accuracy等评估指标（goal_main等7个标签都预测正确时 才算正确），并保存到csv中 =============
    # # why不用样本集的准确率？：划分的交通小区【很细】，只有zone预测不准确，则这一样本即看作预测不准确。这样最后的准确率特别小。所以不合适。MNL模型的样本集准确率才只有0.71%。
    # # 所以，最后还是用标签级的accuracy
    # # 将列表转换为 NumPy 数组以便于操作
    # for key in all_predictions.keys():
    #     all_predictions[key] = np.array(all_predictions[key])
    #     all_labels[key] = np.array(all_labels[key])
    # # print('2all_predictions[key]:', all_predictions[key])
    # # print('2all_labels[key]:', all_labels[key])
    #
    # # 计算每个样本是否全部预测正确
    # correct_predictions = np.ones(len(all_predictions['goals_main']), dtype=bool)
    # # print('correct_predictions:', correct_predictions)
    #
    # for key in all_predictions.keys():
    #     # print('all_predictions[key]:', all_predictions[key])
    #     # print('all_labels[key]:', all_labels[key])
    #     correct_predictions &= (all_predictions[key] == all_labels[key])  # 取并集
    #     # print('correct_predictions:', correct_predictions)
    #
    # # print('2correct_predictions:', correct_predictions)
    # # 样本级别的准确率是所有样本中预测完全正确的比例
    # sample_level_accuracy = np.mean(correct_predictions)
    # print(f"Sample-level accuracy: {sample_level_accuracy:.8f}")
    # =========================================
    # 测试阶段
    # print(f"Testing using CombinedNetwork instance ID: {id(model)}")

    # 绘制真实值与预测值之间的散点图
    plot_true_vs_predicted(
        true_values=all_labels_goals_main[0],
        predicted_values=all_goal_main[0],
        metric_name="Goal",
        r2=train_list_epoch_loss_r2[-1],
        mse=train_list_epoch_loss[-1],
        rmse=train_list_epoch_loss_rmse[-1],
        mae=train_list_epoch_loss_mae[-1]
    )

    # 导出动态效用参数
    model.call(batch_inputs_goal_main, batch_inputs_zone_arrive_main,
        batch_inputs_time_depart_main, batch_inputs_time_arrive_main, batch_inputs_mode1_main, batch_inputs_mode2_main)
    # print('====455544=')   # 只出现一次
    file_name_ynamic_parameters_goal = fr'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\dynamic_parameters\dynamic_parameters_goal.csv'
    model.export_dynamic_parameters(file_name_ynamic_parameters_goal)

    return (train_list_iter_loss_goal_main, train_list_iter_loss_zone_arrive_main, train_list_iter_loss_time_depart_main, train_list_iter_loss_time_arrive_main, train_list_iter_loss_mode1_main, train_list_iter_loss_mode2_main, train_list_iter_loss,
            train_list_epoch_loss_goal_main, train_list_epoch_loss_zone_arrive_main, train_list_epoch_loss_time_depart_main, train_list_epoch_loss_time_arrive_main, train_list_epoch_loss_mode1_main, train_list_epoch_loss_mode2_main, train_list_epoch_loss,
            test_list_iter_loss_goal_main, test_list_iter_loss_zone_arrive_main, test_list_iter_loss_time_depart_main, test_list_iter_loss_time_arrive_main, test_list_iter_loss_mode1_main, test_list_iter_loss_mode2_main, test_list_iter_loss, df_results,
            train_list_iter_loss_r2_goal_main, train_list_iter_loss_r2_zone_arrive_main, train_list_iter_loss_r2_time_depart_main, train_list_iter_loss_r2_time_arrive_main, train_list_iter_loss_r2_mode1_main, train_list_iter_loss_r2_mode2_main, train_list_iter_loss_r2,
            train_list_epoch_loss_r2_goal_main, train_list_epoch_loss_r2_zone_arrive_main, train_list_epoch_loss_r2_time_depart_main, train_list_epoch_loss_r2_time_arrive_main, train_list_epoch_loss_r2_mode1_main, train_list_epoch_loss_r2_mode2_main, train_list_epoch_loss_r2,
            test_list_iter_loss_r2_goal_main, test_list_iter_loss_r2_zone_arrive_main, test_list_iter_loss_r2_time_depart_main, test_list_iter_loss_r2_time_arrive_main, test_list_iter_loss_r2_mode1_main, test_list_iter_loss_r2_mode2_main, test_list_iter_loss_r2,
            train_list_iter_loss_rmse_goal_main, train_list_iter_loss_rmse_zone_arrive_main, train_list_iter_loss_rmse_time_depart_main, train_list_iter_loss_rmse_time_arrive_main, train_list_iter_loss_rmse_mode1_main, train_list_iter_loss_rmse_mode2_main, train_list_iter_loss_rmse,
            train_list_epoch_loss_rmse_goal_main, train_list_epoch_loss_rmse_zone_arrive_main, train_list_epoch_loss_rmse_time_depart_main, train_list_epoch_loss_rmse_time_arrive_main, train_list_epoch_loss_rmse_mode1_main, train_list_epoch_loss_rmse_mode2_main, train_list_epoch_loss_rmse,
            test_list_iter_loss_rmse_goal_main, test_list_iter_loss_rmse_zone_arrive_main, test_list_iter_loss_rmse_time_depart_main, test_list_iter_loss_rmse_time_arrive_main, test_list_iter_loss_rmse_mode1_main, test_list_iter_loss_rmse_mode2_main, test_list_iter_loss_rmse,
            train_list_iter_loss_mae_goal_main, train_list_iter_loss_mae_zone_arrive_main, train_list_iter_loss_mae_time_depart_main, train_list_iter_loss_mae_time_arrive_main, train_list_iter_loss_mae_mode1_main, train_list_iter_loss_mae_mode2_main, train_list_iter_loss_mae,
            train_list_epoch_loss_mae_goal_main, train_list_epoch_loss_mae_zone_arrive_main, train_list_epoch_loss_mae_time_depart_main, train_list_epoch_loss_mae_time_arrive_main, train_list_epoch_loss_mae_mode1_main, train_list_epoch_loss_mae_mode2_main, train_list_epoch_loss_mae,
            test_list_iter_loss_mae_goal_main, test_list_iter_loss_mae_zone_arrive_main, test_list_iter_loss_mae_time_depart_main, test_list_iter_loss_mae_time_arrive_main, test_list_iter_loss_mae_mode1_main, test_list_iter_loss_mae_mode2_main, test_list_iter_loss_mae,
            goals_main, zones_arrive_main, times_depart_main, times_arrive_main, modes1_main, modes2_main)


# ================================= 【可解释性】动态效用参数的可视化分析 ===================================
# 绘制动态效用参数参数
# 提取动态参数并保存到 CSV 文件
def extract_and_save_dynamic_parameters(model, inputs, node_embeddings, zone_adjacency_matrix, distances, file_name_dynamic_parameters):
    """
    提取模型中所有子网络的动态参数，并保存到 CSV 文件。

    Args:
        model (tf.keras.Model): CombinedNetwork 模型实例。
        inputs (list): 输入数据列表，包含每个子网络的输入张量。
        file_name (str): 输出的 CSV 文件名。
    """
    # 获取模型的所有子网络
    subnetworks = [
        model.goal_network_main,
        model.zone_network_arrive_main,
        model.time_network_depart_main,
        model.time_network_arrive_main,
        model.mode1_network_main,
        model.mode2_network_main
    ]

    # 存储动态参数的字典
    dynamic_params_dict = {}
    # 创建一个列表来存储所有子网络的输出
    all_previous_outputs = []
    # 初始化输入
    current_input = inputs

    for i, subnetwork in enumerate(subnetworks):
        # 获取子网络的输入
        # subnetwork_input = inputs[i]

        # 调用当前子网络以生成动态调整量
        if i == 0:
            # 第一个子网络直接使用初始输入
            # subnetwork_input = current_input[i]
            # 调用子网络以生成动态调整量
            context_adjustments = subnetwork.context_net(current_input[i])
        elif i == 1:
            context_adjustments = subnetwork.context_net(current_input[i], all_previous_outputs[i-1], node_embeddings, zone_adjacency_matrix)
            # subnetwork_input = tf.concat([
            #     current_input[i],  # 当前输入
            #     all_previous_outputs[i-1],  # 前一个子网络的输出
            #     node_embeddings, zone_adjacency_matrix
            # ], axis=-1)  # 在最后一个维度上拼接
        elif i == 2:
            context_adjustments = subnetwork.context_net(current_input[i], all_previous_outputs[i - 1], distances,
                                                         zone_adjacency_matrix)
            # subnetwork_input = tf.concat([
            #     current_input[i],  # 当前输入
            #     all_previous_outputs[i-1],  # 前一个子网络的输出
            #     distances, zone_adjacency_matrix
            # ], axis=-1)  # 在最后一个维度上拼接
        elif i == 3:

            subnetwork_input = tf.concat([
                current_input[i],
                all_previous_outputs[i-2],  #
                all_previous_outputs[i-1],
                distances, zone_adjacency_matrix
            ], axis=-1)  # 在最后一个维度上拼接
        elif i == 4:
            subnetwork_input = tf.concat([
                current_input[i],  # 当前输入
                all_previous_outputs[i-2],
                all_previous_outputs[i-1],
                zone_adjacency_matrix
            ], axis=-1)  # 在最后一个维度上拼接
        elif i == 5:
            # 其他子网络使用前一个子网络的输出作为输入
            subnetwork_input = tf.concat([
                current_input[i],  # 当前输入
                all_previous_outputs[i - 3],
                all_previous_outputs[i - 2],
                all_previous_outputs[i - 1],
                zone_adjacency_matrix
            ], axis=-1)  # 在最后一个维度上拼接

        # # 调用子网络以生成动态调整量
        # context_adjustments = subnetwork.context_net(subnetwork_input)

        # 提取权重和偏置的调整量
        weight_adjustments = context_adjustments[:,
                             :subnetwork.output_weights.shape[0] * subnetwork.output_weights.shape[1]]
        bias_adjustments = context_adjustments[:,
                           subnetwork.output_weights.shape[0] * subnetwork.output_weights.shape[1]:]

        # 计算调整后的权重和偏置
        adjusted_output_weights = subnetwork.output_weights + tf.reshape(
            weight_adjustments,
            shape=(-1, subnetwork.output_weights.shape[0], subnetwork.output_weights.shape[1])
        )[-1]  # 取最后一个批次的调整值
        adjusted_output_bias = subnetwork.output_bias + bias_adjustments[-1]  # 取最后一个批次的偏置调整值

        # 将结果存储到字典中
        dynamic_params_dict[f"Subnetwork_{i + 1}_Weights"] = adjusted_output_weights.numpy().flatten()
        dynamic_params_dict[f"Subnetwork_{i + 1}_Bias"] = adjusted_output_bias.numpy().flatten()

        # 更新前一个子网络的输出，并将其添加到 all_previous_outputs 列表中
        if i == 0:
            previous_output = subnetwork(current_input[i])
        elif i == 1:
            previous_output = subnetwork(current_input[i], all_previous_outputs[i-1], node_embeddings, zone_adjacency_matrix)

        all_previous_outputs.append(previous_output)

    # 转换为 DataFrame 并保存到 CSV 文件
    df = pd.DataFrame(dynamic_params_dict)
    df.to_csv(file_name_dynamic_parameters, index=False)
    print(f"Dynamic parameters saved to {file_name_dynamic_parameters}")


def epoch_train(batch_size, num_epochs, train_dataset_goal_main, train_dataset_zones_arrive_main, train_dataset_times_depart_main, train_dataset_times_arrive_main, train_dataset_mode1_main, train_dataset_mode2_main, model, optimizer):
    # 设置早停法参数
    patience = 15
    best_validation_loss = float('inf')
    no_improvement_count = 0
    epoch_early_stopping = None   # 要先初始化，否则不可作为return的内容
    # epoch_early_stopping_df = pd.DataFrame(columns=['batch_size', 'num_epochs', 'epoch_early_stopping'])

    # 训练模型
    train_list_iter_loss_goal_main = []    # collect each iter in each epoch ' loss  (batchsize=10, 每个 epoch 会有 13 次迭代iteration)
    train_list_epoch_loss_goal_main = []
    # train_list_iter_loss_zone_depart_main = []
    # train_list_epoch_loss_zone_depart_main = []
    train_list_iter_loss_zone_arrive_main = []
    train_list_epoch_loss_zone_arrive_main = []
    train_list_iter_loss_time_depart_main = []
    train_list_epoch_loss_time_depart_main = []
    train_list_iter_loss_time_arrive_main = []
    train_list_epoch_loss_time_arrive_main = []
    train_list_iter_loss_mode1_main = []
    train_list_epoch_loss_mode1_main = []
    train_list_iter_loss_mode2_main = []
    train_list_epoch_loss_mode2_main = []
    train_list_iter_loss = []
    train_list_epoch_loss = []

    # 初始化时间记录变量
    # 在机器学习模型训练过程中，第一次训练的时间通常会比后续的训练时间长一些。这是由于多种因素导致的，see Notion notebook,预热
    start_time_epoch = time.time()  # 开始时间
    epoch_start_times = []  # 每个epoch的开始时间
    epoch_durations = []  # 每个epoch的持续时间
    train_start_times = []
    train_durations = []
    iter_start_times =[]

    # 重新实例化模型以确保每次训练从头开始
    # model = CombinedNetwork()
    # # 重新实例化优化器
    # optimizer = tf.keras.optimizers.Adam()

    # # 重置模型的可训练变量和优化器状态
    # model.reset_weights()
    # optimizer = tf.keras.optimizers.Adam()


    for epoch in range(num_epochs):
        epoch_start_times.append(time.time())  # 记录当前 epoch 的开始时间
        # print(epoch_start_times,'epoch_start_times')

        train_epoch_losses_goal_main = []  # 存储每个 train_epoch 的所有批次（13批次）损失
        # train_epoch_losses_zone_depart_main = []
        train_epoch_losses_zone_arrive_main = []
        train_epoch_losses_time_depart_main = []
        train_epoch_losses_time_arrive_main = []
        train_epoch_losses_mode1_main = []
        train_epoch_losses_mode2_main = []
        train_epoch_losses = []

        # 重置迭代器状态
        train_iterator_goal_main = iter(train_dataset_goal_main)
        # train_iterator_zone_depart_main = iter(train_dataset_zones_depart_main)
        train_iterator_zone_arrive_main = iter(train_dataset_zones_arrive_main)
        train_iterator_time_depart_main = iter(train_dataset_times_depart_main)
        train_iterator_time_arrive_main = iter(train_dataset_times_arrive_main)
        train_iterator_mode1_main = iter(train_dataset_mode1_main)
        train_iterator_mode2_main = iter(train_dataset_mode2_main)

        # print(next(iter(train_dataset_goal_main)),'iter(train_dataset_goal_main)')
        # print(next(iter(train_dataset_zones_depart_main)))

        # 使用预先创建的迭代器
        train_num_batches = min(tf.data.experimental.cardinality(train_dataset_goal_main), tf.data.experimental.cardinality(train_dataset_zones_arrive_main), tf.data.experimental.cardinality(train_dataset_times_depart_main), tf.data.experimental.cardinality(train_dataset_times_arrive_main), tf.data.experimental.cardinality(train_dataset_mode1_main), tf.data.experimental.cardinality(train_dataset_mode2_main))   # 13
        # print(train_num_batches,'train_num_batches')   # tf.Tensor(6, shape=(), dtype=int64)，共有6批数据，每个周期epoch迭代6次

        i = 1
        train_start_times.append(time.time())
        # print(train_start_times,'train_start_times')
        start_time_iter = time.time()

        for _ in range(train_num_batches):
            iter_start_times.append(time.time())

            batch_inputs_goal_main, batch_labels_goals_main = next(train_iterator_goal_main)
            # batch_inputs_zone_depart_main, batch_labels_zone_depart_main = next(train_iterator_zone_depart_main)
            batch_inputs_zone_arrive_main, batch_labels_zone_arrive_main = next(train_iterator_zone_arrive_main)
            batch_inputs_time_depart_main, batch_labels_time_depart_main = next(train_iterator_time_depart_main)
            batch_inputs_time_arrive_main, batch_labels_time_arrive_main = next(train_iterator_time_arrive_main)
            batch_inputs_mode1_main, batch_labels_mode1_main = next(train_iterator_mode1_main)
            batch_inputs_mode2_main, batch_labels_mode2_main = next(train_iterator_mode2_main)
            # print('555')   # 迭代次数 = num_epochs * train_num_batches

            # 运行train_step
            loss_goal_main, loss_zone_arrive_main, loss_time_depart_main, loss_time_arrive_main, loss_mode1_main, loss_mode2_main, total_loss, r2_goals_main, r2_zones_arrive_main, r2_times_depart_main, r2_times_arrive_main, r2_modes1_main, r2_modes2_main, total_r2_loss, rmse_goals_main, rmse_zones_arrive_main, rmse_times_depart_main, rmse_times_arrive_main, rmse_modes1_main, rmse_modes2_main, total_rmse_loss, mae_goals_main, mae_zones_arrive_main, mae_times_depart_main, mae_times_arrive_main, mae_modes1_main, mae_modes2_main, total_mae_loss, goals_main, zones_arrive_main, times_depart_main, times_arrive_main, modes1_main, modes2_main\
                = train_step(batch_inputs_goal_main, batch_labels_goals_main, batch_inputs_zone_arrive_main, batch_labels_zone_arrive_main, batch_inputs_time_depart_main, batch_labels_time_depart_main, batch_inputs_time_arrive_main, batch_labels_time_arrive_main, batch_inputs_mode1_main, batch_labels_mode1_main, batch_inputs_mode2_main, batch_labels_mode2_main, node_embeddings, zone_adjacency_matrix, distances, model, optimizer)
            # print(loss_goal_main,'loss_goal_main1')

            # print(f"Losses: Goal={loss_goal}, Zone={loss_zone}, Time={loss_time}, Mode={loss_mode}")

            train_list_iter_loss_goal_main.append(loss_goal_main.numpy())
            train_epoch_losses_goal_main.append(loss_goal_main.numpy())  # 将每个批次的损失值添加到列表中
            # train_list_iter_loss_zone_depart_main.append(loss_zone_depart_main.numpy())
            # train_epoch_losses_zone_depart_main.append(loss_zone_depart_main.numpy())
            train_list_iter_loss_zone_arrive_main.append(loss_zone_arrive_main.numpy())
            train_epoch_losses_zone_arrive_main.append(loss_zone_arrive_main.numpy())
            train_list_iter_loss_time_depart_main.append(loss_time_depart_main.numpy())
            train_epoch_losses_time_depart_main.append(loss_time_depart_main.numpy())
            train_list_iter_loss_time_arrive_main.append(loss_time_arrive_main.numpy())
            train_epoch_losses_time_arrive_main.append(loss_time_arrive_main.numpy())
            train_list_iter_loss_mode1_main.append(loss_mode1_main.numpy())
            train_epoch_losses_mode1_main.append(loss_mode1_main.numpy())
            train_list_iter_loss_mode2_main.append(loss_mode2_main.numpy())
            train_epoch_losses_mode2_main.append(loss_mode2_main.numpy())
            train_list_iter_loss.append(total_loss.numpy())
            train_epoch_losses.append(total_loss.numpy())

            # print(time.time(),'time.time()')
            train_duration = time.time() - iter_start_times[-1]   # 每次迭代/每个批次的训练时间
            # print(train_duration,'train_duration')
            train_durations.append(train_duration)
            # print(f"Train {i} took {train_duration:.5f} seconds.")
            i += 1

        # # 计算总训练时间
        # total_iter_training_time = time.time() - start_time_iter
        # # print(time.time(),'time.time()1')
        # # print(start_time_iter,'start_time_iter')
        # # 打印或记录总训练时间
        # print(f"Total iter training took {total_iter_training_time:.5f} seconds.")

        # 保存每个 iter（每次迭代） 的持续时间到 CSV 文件
        df_iter_durations = pd.DataFrame({'iter_duration': train_durations})
        df_iter_durations.to_csv(fr"02output_csv_time\01output_iter_time_durations.csv", index=False)  # 每次迭代iter的时间

        # # 保存总训练时间到文件
        # with open("01output_total_iter_training_time.txt", "w") as f:
        #     f.write(f"Total iter training took {total_iter_training_time:.5f} seconds.")

        train_end_time = time.time()
        # 记录每个 epoch 的持续时间
        # print(time.time(),'time.time()')
        epoch_duration = time.time() - epoch_start_times[-1]
        # print(epoch_duration,'epoch_duration')
        epoch_durations.append(epoch_duration)
        # 打印或记录每个 epoch 的持续时间
        # print(f"Epoch {epoch + 1} took {epoch_duration:.5f} seconds.")

        # print(time.time(),'123')
        # 计算 epoch 的平均损失
        train_avg_epoch_loss_goal_main = np.mean(train_epoch_losses_goal_main)
        train_list_epoch_loss_goal_main.append(train_avg_epoch_loss_goal_main)  # 添加 epoch 的平均损失到总的列表中
        # train_avg_epoch_loss_zone_depart_main = np.mean(train_epoch_losses_zone_depart_main)
        # train_list_epoch_loss_zone_depart_main.append(train_avg_epoch_loss_zone_depart_main)
        train_avg_epoch_loss_zone_arrive_main = np.mean(train_epoch_losses_zone_arrive_main)
        train_list_epoch_loss_zone_arrive_main.append(train_avg_epoch_loss_zone_arrive_main)
        train_avg_epoch_loss_time_depart_main = np.mean(train_epoch_losses_time_depart_main)
        train_list_epoch_loss_time_depart_main.append(train_avg_epoch_loss_time_depart_main)
        train_avg_epoch_loss_time_arrive_main = np.mean(train_epoch_losses_time_arrive_main)
        train_list_epoch_loss_time_arrive_main.append(train_avg_epoch_loss_time_arrive_main)
        train_avg_epoch_loss_mode1_main = np.mean(train_epoch_losses_mode1_main)
        train_list_epoch_loss_mode1_main.append(train_avg_epoch_loss_mode1_main)
        train_avg_epoch_loss_mode2_main = np.mean(train_epoch_losses_mode2_main)
        train_list_epoch_loss_mode2_main.append(train_avg_epoch_loss_mode2_main)
        train_avg_epoch_loss = np.mean(train_epoch_losses)
        train_list_epoch_loss.append(train_avg_epoch_loss)

        # print(f"Epoch_goal {epoch + 1}, Loss_goal: {loss_goal:.4f}, avg_epoch_loss_goal: {avg_epoch_loss_goal:.4f}, list_epoch_loss_goal: {list_epoch_loss_goal}")
        # print(f"Epoch_zone {epoch + 1}, Loss_zone: {loss_zone:.4f}, avg_epoch_loss_zone: {avg_epoch_loss_zone:.4f}, list_epoch_loss_zone: {list_epoch_loss_zone}")

        # 早停法检查（Early Stopping）
        if train_avg_epoch_loss < best_train_loss:
            best_train_loss = train_avg_epoch_loss
            no_improvement_count = 0
            # print(no_improvement_count,'no_improvement_count')
            # 保存当前模型状态
            # model.save_weights('best_model.weights.h5')
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            # print(f"Early stopping at epoch {epoch}")
            epoch_early_stopping = epoch
            break



    # 计算总训练时间
    # time1 = time.time()
    # print(time1-validation_end_time,'980')
    total_epoch_training_time = train_end_time - start_time_epoch
    # 打印或记录总训练时间
    # print(f"Total epoch training took {total_epoch_training_time:.7f} seconds.")

    # 保存每个 epoch 的持续时间到 CSV 文件
    df_epoch_durations = pd.DataFrame({'epoch_duration': epoch_durations})
    df_epoch_durations.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_time\01output_epoch(all_training)_time_durations.csv", index=False)  # 每个周期epoch的时间

    # 保存总训练时间到文件
    with open(r"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_time\01output_total_epoch_all_training_time.txt", "w") as f:
        f.write(f"Total epoch all training took {total_epoch_training_time:.5f} seconds.")


    return (train_list_iter_loss_goal_main,  train_list_iter_loss_zone_arrive_main, train_list_iter_loss_time_depart_main, train_list_iter_loss_time_arrive_main, train_list_iter_loss_mode1_main, train_list_iter_loss_mode2_main, train_list_iter_loss,
            train_list_epoch_loss_goal_main, train_list_epoch_loss_zone_arrive_main, train_list_epoch_loss_time_depart_main, train_list_epoch_loss_time_arrive_main, train_list_epoch_loss_mode1_main, train_list_epoch_loss_mode2_main, train_list_epoch_loss, epoch_early_stopping)



def save_loss_csv(train_list_iter_loss_goal_main,
             train_list_iter_loss_zone_arrive_main,
             train_list_iter_loss_time_depart_main, train_list_iter_loss_time_arrive_main,
             train_list_iter_loss_mode1_main,
             train_list_iter_loss_mode2_main, train_list_iter_loss,
             train_list_epoch_loss_goal_main,
             train_list_epoch_loss_zone_arrive_main,
             train_list_epoch_loss_time_depart_main, train_list_epoch_loss_time_arrive_main,
             train_list_epoch_loss_mode1_main,
             train_list_epoch_loss_mode2_main, train_list_epoch_loss,
             validation_list_iter_loss_goal_main,
             validation_list_iter_loss_zone_arrive_main, validation_list_iter_loss_time_depart_main,
             validation_list_iter_loss_time_arrive_main, validation_list_iter_loss_mode1_main,
             validation_list_iter_loss_mode2_main, validation_list_iter_loss,
             validation_list_epoch_loss_goal_main,
             validation_list_epoch_loss_zone_arrive_main, validation_list_epoch_loss_time_depart_main,
             validation_list_epoch_loss_time_arrive_main, validation_list_epoch_loss_mode1_main,
             validation_list_epoch_loss_mode2_main, validation_list_epoch_loss, epoch_early_stopping):
    # Save loss value as csv file: each iter & each epoch
    train_df_iter_loss_goal_main = pd.DataFrame({'train_list_loss_goal_main': train_list_iter_loss_goal_main})
    train_df_iter_loss_goal_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_iter_loss_goal_main.csv", index=False)
    train_df_epoch_loss_goal_main = pd.DataFrame({'train_list_epoch_loss_goal_main': train_list_epoch_loss_goal_main})
    train_df_epoch_loss_goal_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_epoch_loss_goal_main.csv", index=False)

    # train_df_iter_loss_zone_depart_main = pd.DataFrame(
    #     {'train_list_loss_zone_depart_main': train_list_iter_loss_zone_depart_main})
    # train_df_iter_loss_zone_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_iter_loss_zone_depart_main.csv", index=False)
    # train_df_epoch_loss_zone_depart_main = pd.DataFrame(
    #     {'train_list_epoch_loss_zone_depart_main': train_list_epoch_loss_zone_depart_main})
    # train_df_epoch_loss_zone_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_epoch_loss_zone_depart_main.csv", index=False)

    train_df_iter_loss_zone_arrive_main = pd.DataFrame(
        {'train_list_loss_zone_arrive_main': train_list_iter_loss_zone_arrive_main})
    train_df_iter_loss_zone_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_iter_loss_zone_arrive_main.csv", index=False)
    train_df_epoch_loss_zone_arrive_main = pd.DataFrame(
        {'train_list_epoch_loss_zone_arrive_main': train_list_epoch_loss_zone_arrive_main})
    train_df_epoch_loss_zone_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_epoch_loss_zone_arrive_main.csv", index=False)

    train_df_iter_loss_time_depart_main = pd.DataFrame(
        {'train_list_loss_time_depart_main': train_list_iter_loss_time_depart_main})
    train_df_iter_loss_time_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_iter_loss_time_depart_main.csv", index=False)
    train_df_epoch_loss_time_depart_main = pd.DataFrame(
        {'train_list_epoch_loss_time_depart_main': train_list_epoch_loss_time_depart_main})
    train_df_epoch_loss_time_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_epoch_loss_time_depart_main.csv", index=False)

    train_df_iter_loss_time_arrive_main = pd.DataFrame(
        {'train_list_loss_time_arrive_main': train_list_iter_loss_time_arrive_main})
    train_df_iter_loss_time_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_iter_loss_time_arrive_main.csv", index=False)
    train_df_epoch_loss_time_arrive_main = pd.DataFrame(
        {'train_list_epoch_loss_time_arrive_main': train_list_epoch_loss_time_arrive_main})
    train_df_epoch_loss_time_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_epoch_loss_time_arrive_main.csv", index=False)

    train_df_iter_loss_mode1_main = pd.DataFrame({'train_list_loss_mode1_main': train_list_iter_loss_mode1_main})
    train_df_iter_loss_mode1_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_iter_loss_mode1_main.csv", index=False)
    train_df_epoch_loss_mode1_main = pd.DataFrame(
        {'train_list_epoch_loss_mode1_main': train_list_epoch_loss_mode1_main})
    train_df_epoch_loss_mode1_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_epoch_loss_mode1_main.csv", index=False)

    train_df_iter_loss_mode2_main = pd.DataFrame({'train_list_loss_mode2_main': train_list_iter_loss_mode2_main})
    train_df_iter_loss_mode2_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_iter_loss_mode2_main.csv", index=False)
    train_df_epoch_loss_mode2_main = pd.DataFrame(
        {'train_list_epoch_loss_mode2_main': train_list_epoch_loss_mode2_main})
    train_df_epoch_loss_mode2_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_epoch_loss_mode2_main.csv", index=False)

    train_df_iter_loss = pd.DataFrame({'train_list_loss': train_list_iter_loss})
    train_df_iter_loss.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_iter_loss.csv", index=False)
    train_df_epoch_loss = pd.DataFrame({'train_list_epoch_loss': train_list_epoch_loss})
    train_df_epoch_loss.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_epoch_loss.csv", index=False)
    # ======
    validation_df_iter_loss_goal_main = pd.DataFrame(
        {'validation_list_loss_goal_main': validation_list_iter_loss_goal_main})
    validation_df_iter_loss_goal_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01validation_output_iter_loss_goal_main.csv", index=False)
    validation_df_epoch_loss_goal_main = pd.DataFrame(
        {'validation_list_epoch_loss_goal_main': validation_list_epoch_loss_goal_main})
    validation_df_epoch_loss_goal_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01validation_output_epoch_loss_goal_main.csv", index=False)

    # validation_df_iter_loss_zone_depart_main = pd.DataFrame(
    #     {'validation_list_loss_zone_depart_main': validation_list_iter_loss_zone_depart_main})
    # validation_df_iter_loss_zone_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01validation_output_iter_loss_zone_depart_main.csv", index=False)
    # validation_df_epoch_loss_zone_depart_main = pd.DataFrame(
    #     {'validation_list_epoch_loss_zone_depart_main': validation_list_epoch_loss_zone_depart_main})
    # validation_df_epoch_loss_zone_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01validation_output_epoch_loss_zone_depart_main.csv", index=False)

    validation_df_iter_loss_zone_arrive_main = pd.DataFrame(
        {'validation_list_loss_zone_arrive_main': validation_list_iter_loss_zone_arrive_main})
    validation_df_iter_loss_zone_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01validation_output_iter_loss_zone_arrive_main.csv", index=False)
    validation_df_epoch_loss_zone_arrive_main = pd.DataFrame(
        {'validation_list_epoch_loss_zone_arrive_main': validation_list_epoch_loss_zone_arrive_main})
    validation_df_epoch_loss_zone_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01validation_output_epoch_loss_zone_arrive_main.csv", index=False)

    validation_df_iter_loss_time_depart_main = pd.DataFrame(
        {'validation_list_loss_time_depart_main': validation_list_iter_loss_time_depart_main})
    validation_df_iter_loss_time_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01validation_output_iter_loss_time_depart_main.csv", index=False)
    validation_df_epoch_loss_time_depart_main = pd.DataFrame(
        {'validation_list_epoch_loss_time_depart_main': validation_list_epoch_loss_time_depart_main})
    validation_df_epoch_loss_time_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01validation_output_epoch_loss_time_depart_main.csv", index=False)

    validation_df_iter_loss_time_arrive_main = pd.DataFrame(
        {'validation_list_loss_time_arrive_main': validation_list_iter_loss_time_arrive_main})
    validation_df_iter_loss_time_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01validation_output_iter_loss_time_arrive_main.csv", index=False)
    validation_df_epoch_loss_time_arrive_main = pd.DataFrame(
        {'validation_list_epoch_loss_time_arrive_main': validation_list_epoch_loss_time_arrive_main})
    validation_df_epoch_loss_time_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01validation_output_epoch_loss_time_arrive_main.csv", index=False)

    validation_df_iter_loss_mode1_main = pd.DataFrame(
        {'validation_list_loss_mode1_main': validation_list_iter_loss_mode1_main})
    validation_df_iter_loss_mode1_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01validation_output_iter_loss_mode1_main.csv", index=False)
    validation_df_epoch_loss_mode1_main = pd.DataFrame(
        {'validation_list_epoch_loss_mode1_main': validation_list_epoch_loss_mode1_main})
    validation_df_epoch_loss_mode1_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01validation_output_epoch_loss_mode1_main.csv", index=False)

    validation_df_iter_loss_mode2_main = pd.DataFrame(
        {'validation_list_loss_mode2_main': validation_list_iter_loss_mode2_main})
    validation_df_iter_loss_mode2_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01validation_output_iter_loss_mode2_main.csv", index=False)
    validation_df_epoch_loss_mode2_main = pd.DataFrame(
        {'validation_list_epoch_loss_mode2_main': validation_list_epoch_loss_mode2_main})
    validation_df_epoch_loss_mode2_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01validation_output_epoch_loss_mode2_main.csv", index=False)

    validation_df_iter_loss = pd.DataFrame({'validation_list_loss': validation_list_iter_loss})
    validation_df_iter_loss.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01validation_output_iter_loss.csv", index=False)
    validation_df_epoch_loss = pd.DataFrame({'validation_list_epoch_loss': validation_list_epoch_loss})
    validation_df_epoch_loss.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01validation_output_epoch_loss.csv", index=False)
    # =====




def save_loss_mse_csv_test(train_list_iter_loss_goal_main,
             train_list_iter_loss_zone_arrive_main,
             train_list_iter_loss_time_depart_main, train_list_iter_loss_time_arrive_main,
             train_list_iter_loss_mode1_main,
             train_list_iter_loss_mode2_main, train_list_iter_loss,
             train_list_epoch_loss_goal_main,
             train_list_epoch_loss_zone_arrive_main,
             train_list_epoch_loss_time_depart_main, train_list_epoch_loss_time_arrive_main,
             train_list_epoch_loss_mode1_main,
             train_list_epoch_loss_mode2_main, train_list_epoch_loss,
             test_list_iter_loss_goal_main,
             test_list_iter_loss_zone_arrive_main, test_list_iter_loss_time_depart_main,
             test_list_iter_loss_time_arrive_main, test_list_iter_loss_mode1_main,
             test_list_iter_loss_mode2_main, test_list_iter_loss):

    # Save loss value as csv file: each iter & each epoch
    train_df_iter_loss_goal_main = pd.DataFrame({'train_list_loss_goal_main': train_list_iter_loss_goal_main})
    train_df_iter_loss_goal_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss\01train_output_iter_loss_goal_main.csv", index=False)
    train_df_epoch_loss_goal_main = pd.DataFrame({'train_list_epoch_loss_goal_main': train_list_epoch_loss_goal_main})
    train_df_epoch_loss_goal_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss\01train_output_epoch_loss_goal_main.csv", index=False)

    # train_df_iter_loss_zone_depart_main = pd.DataFrame(
    #     {'train_list_loss_zone_depart_main': train_list_iter_loss_zone_depart_main})
    # train_df_iter_loss_zone_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss\01train_output_iter_loss_zone_depart_main.csv", index=False)
    # train_df_epoch_loss_zone_depart_main = pd.DataFrame(
    #     {'train_list_epoch_loss_zone_depart_main': train_list_epoch_loss_zone_depart_main})
    # train_df_epoch_loss_zone_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss\01train_output_epoch_loss_zone_depart_main.csv", index=False)

    train_df_iter_loss_zone_arrive_main = pd.DataFrame(
        {'train_list_loss_zone_arrive_main': train_list_iter_loss_zone_arrive_main})
    train_df_iter_loss_zone_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss\01train_output_iter_loss_zone_arrive_main.csv", index=False)
    train_df_epoch_loss_zone_arrive_main = pd.DataFrame(
        {'train_list_epoch_loss_zone_arrive_main': train_list_epoch_loss_zone_arrive_main})
    train_df_epoch_loss_zone_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss\01train_output_epoch_loss_zone_arrive_main.csv", index=False)

    train_df_iter_loss_time_depart_main = pd.DataFrame(
        {'train_list_loss_time_depart_main': train_list_iter_loss_time_depart_main})
    train_df_iter_loss_time_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss\01train_output_iter_loss_time_depart_main.csv", index=False)
    train_df_epoch_loss_time_depart_main = pd.DataFrame(
        {'train_list_epoch_loss_time_depart_main': train_list_epoch_loss_time_depart_main})
    train_df_epoch_loss_time_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss\01train_output_epoch_loss_time_depart_main.csv", index=False)

    train_df_iter_loss_time_arrive_main = pd.DataFrame(
        {'train_list_loss_time_arrive_main': train_list_iter_loss_time_arrive_main})
    train_df_iter_loss_time_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss\01train_output_iter_loss_time_arrive_main.csv", index=False)
    train_df_epoch_loss_time_arrive_main = pd.DataFrame(
        {'train_list_epoch_loss_time_arrive_main': train_list_epoch_loss_time_arrive_main})
    train_df_epoch_loss_time_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss\01train_output_epoch_loss_time_arrive_main.csv", index=False)

    train_df_iter_loss_mode1_main = pd.DataFrame({'train_list_loss_mode1_main': train_list_iter_loss_mode1_main})
    train_df_iter_loss_mode1_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss\01train_output_iter_loss_mode1_main.csv", index=False)
    train_df_epoch_loss_mode1_main = pd.DataFrame(
        {'train_list_epoch_loss_mode1_main': train_list_epoch_loss_mode1_main})
    train_df_epoch_loss_mode1_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss\01train_output_epoch_loss_mode1_main.csv", index=False)

    train_df_iter_loss_mode2_main = pd.DataFrame({'train_list_loss_mode2_main': train_list_iter_loss_mode2_main})
    train_df_iter_loss_mode2_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss\01train_output_iter_loss_mode2_main.csv", index=False)
    train_df_epoch_loss_mode2_main = pd.DataFrame(
        {'train_list_epoch_loss_mode2_main': train_list_epoch_loss_mode2_main})
    train_df_epoch_loss_mode2_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss\01train_output_epoch_loss_mode2_main.csv", index=False)

    train_df_iter_loss = pd.DataFrame({'train_list_loss': train_list_iter_loss})
    train_df_iter_loss.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss\01train_output_iter_loss.csv", index=False)
    train_df_epoch_loss = pd.DataFrame({'train_list_epoch_loss': train_list_epoch_loss})
    train_df_epoch_loss.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss\01train_output_epoch_loss.csv", index=False)

    # =============================
    test_df_iter_loss_goal_main = pd.DataFrame(
        {'test_list_loss_goal_main': test_list_iter_loss_goal_main})
    test_df_iter_loss_goal_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss\01test_output_iter_loss_goal_main.csv", index=False)
    # test_df_epoch_loss_goal_main = pd.DataFrame(
    #     {'test_list_epoch_loss_goal_main': test_list_epoch_loss_goal_main})
    # test_df_epoch_loss_goal_main.to_csv(f"01test_output_epoch_loss_goal_main.csv", index=False)

    # test_df_iter_loss_zone_depart_main = pd.DataFrame(
    #     {'test_list_loss_zone_depart_main': test_list_iter_loss_zone_depart_main})
    # test_df_iter_loss_zone_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss\01test_output_iter_loss_zone_depart_main.csv", index=False)
    # test_df_epoch_loss_zone_depart_main = pd.DataFrame(
    #     {'test_list_epoch_loss_zone_depart_main': test_list_epoch_loss_zone_depart_main})
    # test_df_epoch_loss_zone_depart_main.to_csv(f"01test_output_epoch_loss_zone_depart_main.csv", index=False)

    test_df_iter_loss_zone_arrive_main = pd.DataFrame(
        {'test_list_loss_zone_arrive_main': test_list_iter_loss_zone_arrive_main})
    test_df_iter_loss_zone_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss\01test_output_iter_loss_zone_arrive_main.csv", index=False)
    # test_df_epoch_loss_zone_arrive_main = pd.DataFrame(
    #     {'test_list_epoch_loss_zone_arrive_main': test_list_epoch_loss_zone_arrive_main})
    # test_df_epoch_loss_zone_arrive_main.to_csv(f"01test_output_epoch_loss_zone_arrive_main.csv", index=False)

    test_df_iter_loss_time_depart_main = pd.DataFrame(
        {'test_list_loss_time_depart_main': test_list_iter_loss_time_depart_main})
    test_df_iter_loss_time_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss\01test_output_iter_loss_time_depart_main.csv", index=False)
    # test_df_epoch_loss_time_depart_main = pd.DataFrame(
    #     {'test_list_epoch_loss_time_depart_main': test_list_epoch_loss_time_depart_main})
    # test_df_epoch_loss_time_depart_main.to_csv(f"01test_output_epoch_loss_time_depart_main.csv", index=False)

    test_df_iter_loss_time_arrive_main = pd.DataFrame(
        {'test_list_loss_time_arrive_main': test_list_iter_loss_time_arrive_main})
    test_df_iter_loss_time_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss\01test_output_iter_loss_time_arrive_main.csv", index=False)
    # test_df_epoch_loss_time_arrive_main = pd.DataFrame(
    #     {'test_list_epoch_loss_time_arrive_main': test_list_epoch_loss_time_arrive_main})
    # test_df_epoch_loss_time_arrive_main.to_csv(f"01test_output_epoch_loss_time_arrive_main.csv", index=False)

    test_df_iter_loss_mode1_main = pd.DataFrame(
        {'test_list_loss_mode1_main': test_list_iter_loss_mode1_main})
    test_df_iter_loss_mode1_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss\01test_output_iter_loss_mode1_main.csv", index=False)
    # test_df_epoch_loss_mode1_main = pd.DataFrame(
    #     {'test_list_epoch_loss_mode1_main': test_list_epoch_loss_mode1_main})
    # test_df_epoch_loss_mode1_main.to_csv(f"01test_output_epoch_loss_mode1_main.csv", index=False)

    test_df_iter_loss_mode2_main = pd.DataFrame(
        {'test_list_loss_mode2_main': test_list_iter_loss_mode2_main})
    test_df_iter_loss_mode2_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss\01test_output_iter_loss_mode2_main.csv", index=False)
    # test_df_epoch_loss_mode2_main = pd.DataFrame(
    #     {'test_list_epoch_loss_mode2_main': test_list_epoch_loss_mode2_main})
    # test_df_epoch_loss_mode2_main.to_csv(f"01test_output_epoch_loss_mode2_main.csv", index=False)

    test_df_iter_loss = pd.DataFrame({'test_list_loss': test_list_iter_loss})
    test_df_iter_loss.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss\01test_output_iter_loss.csv", index=False)
    # test_df_epoch_loss = pd.DataFrame({'test_list_epoch_loss': test_list_epoch_loss})
    # test_df_epoch_loss.to_csv(f"01test_output_epoch_loss.csv", index=False)
    # =====




def save_loss_r2_csv_test(train_list_iter_loss_r2_goal_main,
             train_list_iter_loss_r2_zone_arrive_main,
             train_list_iter_loss_r2_time_depart_main, train_list_iter_loss_r2_time_arrive_main,
             train_list_iter_loss_r2_mode1_main,
             train_list_iter_loss_r2_mode2_main, train_list_iter_loss_r2,
             train_list_epoch_loss_r2_goal_main,
             train_list_epoch_loss_r2_zone_arrive_main,
             train_list_epoch_loss_r2_time_depart_main, train_list_epoch_loss_r2_time_arrive_main,
             train_list_epoch_loss_r2_mode1_main,
             train_list_epoch_loss_r2_mode2_main, train_list_epoch_loss_r2,
             test_list_iter_loss_r2_goal_main,
             test_list_iter_loss_r2_zone_arrive_main, test_list_iter_loss_r2_time_depart_main,
             test_list_iter_loss_r2_time_arrive_main, test_list_iter_loss_r2_mode1_main,
             test_list_iter_loss_r2_mode2_main, test_list_iter_loss_r2):

    # Save loss_r2 value as csv file: each iter & each epoch
    train_df_iter_loss_r2_goal_main = pd.DataFrame({'train_list_loss_r2_goal_main': train_list_iter_loss_r2_goal_main})
    train_df_iter_loss_r2_goal_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_r2\01train_output_iter_loss_r2_goal_main.csv", index=False)
    train_df_epoch_loss_r2_goal_main = pd.DataFrame({'train_list_epoch_loss_r2_goal_main': train_list_epoch_loss_r2_goal_main})
    train_df_epoch_loss_r2_goal_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_r2\01train_output_epoch_loss_r2_goal_main.csv", index=False)

    # train_df_iter_loss_r2_zone_depart_main = pd.DataFrame(
    #     {'train_list_loss_r2_zone_depart_main': train_list_iter_loss_r2_zone_depart_main})
    # train_df_iter_loss_r2_zone_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_r2\01train_output_iter_loss_r2_zone_depart_main.csv", index=False)
    # train_df_epoch_loss_r2_zone_depart_main = pd.DataFrame(
    #     {'train_list_epoch_loss_r2_zone_depart_main': train_list_epoch_loss_r2_zone_depart_main})
    # train_df_epoch_loss_r2_zone_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_r2\01train_output_epoch_loss_r2_zone_depart_main.csv", index=False)

    train_df_iter_loss_r2_zone_arrive_main = pd.DataFrame(
        {'train_list_loss_r2_zone_arrive_main': train_list_iter_loss_r2_zone_arrive_main})
    train_df_iter_loss_r2_zone_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_r2\01train_output_iter_loss_r2_zone_arrive_main.csv", index=False)
    train_df_epoch_loss_r2_zone_arrive_main = pd.DataFrame(
        {'train_list_epoch_loss_r2_zone_arrive_main': train_list_epoch_loss_r2_zone_arrive_main})
    train_df_epoch_loss_r2_zone_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_r2\01train_output_epoch_loss_r2_zone_arrive_main.csv", index=False)

    train_df_iter_loss_r2_time_depart_main = pd.DataFrame(
        {'train_list_loss_r2_time_depart_main': train_list_iter_loss_r2_time_depart_main})
    train_df_iter_loss_r2_time_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_r2\01train_output_iter_loss_r2_time_depart_main.csv", index=False)
    train_df_epoch_loss_r2_time_depart_main = pd.DataFrame(
        {'train_list_epoch_loss_r2_time_depart_main': train_list_epoch_loss_r2_time_depart_main})
    train_df_epoch_loss_r2_time_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_r2\01train_output_epoch_loss_r2_time_depart_main.csv", index=False)

    train_df_iter_loss_r2_time_arrive_main = pd.DataFrame(
        {'train_list_loss_r2_time_arrive_main': train_list_iter_loss_r2_time_arrive_main})
    train_df_iter_loss_r2_time_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_r2\01train_output_iter_loss_r2_time_arrive_main.csv", index=False)
    train_df_epoch_loss_r2_time_arrive_main = pd.DataFrame(
        {'train_list_epoch_loss_r2_time_arrive_main': train_list_epoch_loss_r2_time_arrive_main})
    train_df_epoch_loss_r2_time_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_r2\01train_output_epoch_loss_r2_time_arrive_main.csv", index=False)

    train_df_iter_loss_r2_mode1_main = pd.DataFrame({'train_list_loss_r2_mode1_main': train_list_iter_loss_r2_mode1_main})
    train_df_iter_loss_r2_mode1_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_r2\01train_output_iter_loss_r2_mode1_main.csv", index=False)
    train_df_epoch_loss_r2_mode1_main = pd.DataFrame(
        {'train_list_epoch_loss_r2_mode1_main': train_list_epoch_loss_r2_mode1_main})
    train_df_epoch_loss_r2_mode1_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_r2\01train_output_epoch_loss_r2_mode1_main.csv", index=False)

    train_df_iter_loss_r2_mode2_main = pd.DataFrame({'train_list_loss_r2_mode2_main': train_list_iter_loss_r2_mode2_main})
    train_df_iter_loss_r2_mode2_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_r2\01train_output_iter_loss_r2_mode2_main.csv", index=False)
    train_df_epoch_loss_r2_mode2_main = pd.DataFrame(
        {'train_list_epoch_loss_r2_mode2_main': train_list_epoch_loss_r2_mode2_main})
    train_df_epoch_loss_r2_mode2_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_r2\01train_output_epoch_loss_r2_mode2_main.csv", index=False)

    train_df_iter_loss_r2 = pd.DataFrame({'train_list_loss_r2': train_list_iter_loss_r2})
    train_df_iter_loss_r2.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_r2\01train_output_iter_loss_r2.csv", index=False)
    train_df_epoch_loss_r2 = pd.DataFrame({'train_list_epoch_loss_r2': train_list_epoch_loss_r2})
    train_df_epoch_loss_r2.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_r2\01train_output_epoch_loss_r2.csv", index=False)

    # =============================
    test_df_iter_loss_r2_goal_main = pd.DataFrame(
        {'test_list_loss_r2_goal_main': test_list_iter_loss_r2_goal_main})
    test_df_iter_loss_r2_goal_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_r2\01test_output_iter_loss_r2_goal_main.csv", index=False)
    # test_df_epoch_loss_r2_goal_main = pd.DataFrame(
    #     {'test_list_epoch_loss_r2_goal_main': test_list_epoch_loss_r2_goal_main})
    # test_df_epoch_loss_r2_goal_main.to_csv(f"01test_output_epoch_loss_r2_goal_main.csv", index=False)

    # test_df_iter_loss_r2_zone_depart_main = pd.DataFrame(
    #     {'test_list_loss_r2_zone_depart_main': test_list_iter_loss_r2_zone_depart_main})
    # test_df_iter_loss_r2_zone_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_r2\01test_output_iter_loss_r2_zone_depart_main.csv", index=False)
    # test_df_epoch_loss_r2_zone_depart_main = pd.DataFrame(
    #     {'test_list_epoch_loss_r2_zone_depart_main': test_list_epoch_loss_r2_zone_depart_main})
    # test_df_epoch_loss_r2_zone_depart_main.to_csv(f"01test_output_epoch_loss_r2_zone_depart_main.csv", index=False)

    test_df_iter_loss_r2_zone_arrive_main = pd.DataFrame(
        {'test_list_loss_r2_zone_arrive_main': test_list_iter_loss_r2_zone_arrive_main})
    test_df_iter_loss_r2_zone_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_r2\01test_output_iter_loss_r2_zone_arrive_main.csv", index=False)
    # test_df_epoch_loss_r2_zone_arrive_main = pd.DataFrame(
    #     {'test_list_epoch_loss_r2_zone_arrive_main': test_list_epoch_loss_r2_zone_arrive_main})
    # test_df_epoch_loss_r2_zone_arrive_main.to_csv(f"01test_output_epoch_loss_r2_zone_arrive_main.csv", index=False)

    test_df_iter_loss_r2_time_depart_main = pd.DataFrame(
        {'test_list_loss_r2_time_depart_main': test_list_iter_loss_r2_time_depart_main})
    test_df_iter_loss_r2_time_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_r2\01test_output_iter_loss_r2_time_depart_main.csv", index=False)
    # test_df_epoch_loss_r2_time_depart_main = pd.DataFrame(
    #     {'test_list_epoch_loss_r2_time_depart_main': test_list_epoch_loss_r2_time_depart_main})
    # test_df_epoch_loss_r2_time_depart_main.to_csv(f"01test_output_epoch_loss_r2_time_depart_main.csv", index=False)

    test_df_iter_loss_r2_time_arrive_main = pd.DataFrame(
        {'test_list_loss_r2_time_arrive_main': test_list_iter_loss_r2_time_arrive_main})
    test_df_iter_loss_r2_time_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_r2\01test_output_iter_loss_r2_time_arrive_main.csv", index=False)
    # test_df_epoch_loss_r2_time_arrive_main = pd.DataFrame(
    #     {'test_list_epoch_loss_r2_time_arrive_main': test_list_epoch_loss_r2_time_arrive_main})
    # test_df_epoch_loss_r2_time_arrive_main.to_csv(f"01test_output_epoch_loss_r2_time_arrive_main.csv", index=False)

    test_df_iter_loss_r2_mode1_main = pd.DataFrame(
        {'test_list_loss_r2_mode1_main': test_list_iter_loss_r2_mode1_main})
    test_df_iter_loss_r2_mode1_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_r2\01test_output_iter_loss_r2_mode1_main.csv", index=False)
    # test_df_epoch_loss_r2_mode1_main = pd.DataFrame(
    #     {'test_list_epoch_loss_r2_mode1_main': test_list_epoch_loss_r2_mode1_main})
    # test_df_epoch_loss_r2_mode1_main.to_csv(f"01test_output_epoch_loss_r2_mode1_main.csv", index=False)

    test_df_iter_loss_r2_mode2_main = pd.DataFrame(
        {'test_list_loss_r2_mode2_main': test_list_iter_loss_r2_mode2_main})
    test_df_iter_loss_r2_mode2_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_r2\01test_output_iter_loss_r2_mode2_main.csv", index=False)
    # test_df_epoch_loss_r2_mode2_main = pd.DataFrame(
    #     {'test_list_epoch_loss_r2_mode2_main': test_list_epoch_loss_r2_mode2_main})
    # test_df_epoch_loss_r2_mode2_main.to_csv(f"01test_output_epoch_loss_r2_mode2_main.csv", index=False)

    test_df_iter_loss_r2 = pd.DataFrame({'test_list_loss_r2': test_list_iter_loss_r2})
    test_df_iter_loss_r2.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_r2\01test_output_iter_loss_r2.csv", index=False)
    # test_df_epoch_loss_r2 = pd.DataFrame({'test_list_epoch_loss_r2': test_list_epoch_loss_r2})
    # test_df_epoch_loss_r2.to_csv(f"01test_output_epoch_loss_r2.csv", index=False)
    # =====



def save_loss_rmse_csv_test(train_list_iter_loss_rmse_goal_main,
             train_list_iter_loss_rmse_zone_arrive_main,
             train_list_iter_loss_rmse_time_depart_main, train_list_iter_loss_rmse_time_arrive_main,
             train_list_iter_loss_rmse_mode1_main,
             train_list_iter_loss_rmse_mode2_main, train_list_iter_loss_rmse,
             train_list_epoch_loss_rmse_goal_main,
             train_list_epoch_loss_rmse_zone_arrive_main,
             train_list_epoch_loss_rmse_time_depart_main, train_list_epoch_loss_rmse_time_arrive_main,
             train_list_epoch_loss_rmse_mode1_main,
             train_list_epoch_loss_rmse_mode2_main, train_list_epoch_loss_rmse,
             test_list_iter_loss_rmse_goal_main,
             test_list_iter_loss_rmse_zone_arrive_main, test_list_iter_loss_rmse_time_depart_main,
             test_list_iter_loss_rmse_time_arrive_main, test_list_iter_loss_rmse_mode1_main,
             test_list_iter_loss_rmse_mode2_main, test_list_iter_loss_rmse):

    # Save loss_rmse value as csv file: each iter & each epoch
    train_df_iter_loss_rmse_goal_main = pd.DataFrame({'train_list_loss_rmse_goal_main': train_list_iter_loss_rmse_goal_main})
    train_df_iter_loss_rmse_goal_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_rmse\01train_output_iter_loss_rmse_goal_main.csv", index=False)
    train_df_epoch_loss_rmse_goal_main = pd.DataFrame({'train_list_epoch_loss_rmse_goal_main': train_list_epoch_loss_rmse_goal_main})
    train_df_epoch_loss_rmse_goal_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_rmse\01train_output_epoch_loss_rmse_goal_main.csv", index=False)

    # train_df_iter_loss_rmse_zone_depart_main = pd.DataFrame(
    #     {'train_list_loss_rmse_zone_depart_main': train_list_iter_loss_rmse_zone_depart_main})
    # train_df_iter_loss_rmse_zone_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_rmse\01train_output_iter_loss_rmse_zone_depart_main.csv", index=False)
    # train_df_epoch_loss_rmse_zone_depart_main = pd.DataFrame(
    #     {'train_list_epoch_loss_rmse_zone_depart_main': train_list_epoch_loss_rmse_zone_depart_main})
    # train_df_epoch_loss_rmse_zone_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_rmse\01train_output_epoch_loss_rmse_zone_depart_main.csv", index=False)

    train_df_iter_loss_rmse_zone_arrive_main = pd.DataFrame(
        {'train_list_loss_rmse_zone_arrive_main': train_list_iter_loss_rmse_zone_arrive_main})
    train_df_iter_loss_rmse_zone_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_rmse\01train_output_iter_loss_rmse_zone_arrive_main.csv", index=False)
    train_df_epoch_loss_rmse_zone_arrive_main = pd.DataFrame(
        {'train_list_epoch_loss_rmse_zone_arrive_main': train_list_epoch_loss_rmse_zone_arrive_main})
    train_df_epoch_loss_rmse_zone_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_rmse\01train_output_epoch_loss_rmse_zone_arrive_main.csv", index=False)

    train_df_iter_loss_rmse_time_depart_main = pd.DataFrame(
        {'train_list_loss_rmse_time_depart_main': train_list_iter_loss_rmse_time_depart_main})
    train_df_iter_loss_rmse_time_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_rmse\01train_output_iter_loss_rmse_time_depart_main.csv", index=False)
    train_df_epoch_loss_rmse_time_depart_main = pd.DataFrame(
        {'train_list_epoch_loss_rmse_time_depart_main': train_list_epoch_loss_rmse_time_depart_main})
    train_df_epoch_loss_rmse_time_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_rmse\01train_output_epoch_loss_rmse_time_depart_main.csv", index=False)

    train_df_iter_loss_rmse_time_arrive_main = pd.DataFrame(
        {'train_list_loss_rmse_time_arrive_main': train_list_iter_loss_rmse_time_arrive_main})
    train_df_iter_loss_rmse_time_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_rmse\01train_output_iter_loss_rmse_time_arrive_main.csv", index=False)
    train_df_epoch_loss_rmse_time_arrive_main = pd.DataFrame(
        {'train_list_epoch_loss_rmse_time_arrive_main': train_list_epoch_loss_rmse_time_arrive_main})
    train_df_epoch_loss_rmse_time_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_rmse\01train_output_epoch_loss_rmse_time_arrive_main.csv", index=False)

    train_df_iter_loss_rmse_mode1_main = pd.DataFrame({'train_list_loss_rmse_mode1_main': train_list_iter_loss_rmse_mode1_main})
    train_df_iter_loss_rmse_mode1_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_rmse\01train_output_iter_loss_rmse_mode1_main.csv", index=False)
    train_df_epoch_loss_rmse_mode1_main = pd.DataFrame(
        {'train_list_epoch_loss_rmse_mode1_main': train_list_epoch_loss_rmse_mode1_main})
    train_df_epoch_loss_rmse_mode1_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_rmse\01train_output_epoch_loss_rmse_mode1_main.csv", index=False)

    train_df_iter_loss_rmse_mode2_main = pd.DataFrame({'train_list_loss_rmse_mode2_main': train_list_iter_loss_rmse_mode2_main})
    train_df_iter_loss_rmse_mode2_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_rmse\01train_output_iter_loss_rmse_mode2_main.csv", index=False)
    train_df_epoch_loss_rmse_mode2_main = pd.DataFrame(
        {'train_list_epoch_loss_rmse_mode2_main': train_list_epoch_loss_rmse_mode2_main})
    train_df_epoch_loss_rmse_mode2_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_rmse\01train_output_epoch_loss_rmse_mode2_main.csv", index=False)

    train_df_iter_loss_rmse = pd.DataFrame({'train_list_loss_rmse': train_list_iter_loss_rmse})
    train_df_iter_loss_rmse.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_rmse\01train_output_iter_loss_rmse.csv", index=False)
    train_df_epoch_loss_rmse = pd.DataFrame({'train_list_epoch_loss_rmse': train_list_epoch_loss_rmse})
    train_df_epoch_loss_rmse.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_rmse\01train_output_epoch_loss_rmse.csv", index=False)

    # =============================
    test_df_iter_loss_rmse_goal_main = pd.DataFrame(
        {'test_list_loss_rmse_goal_main': test_list_iter_loss_rmse_goal_main})
    test_df_iter_loss_rmse_goal_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_rmse\01test_output_iter_loss_rmse_goal_main.csv", index=False)
    # test_df_epoch_loss_rmse_goal_main = pd.DataFrame(
    #     {'test_list_epoch_loss_rmse_goal_main': test_list_epoch_loss_rmse_goal_main})
    # test_df_epoch_loss_rmse_goal_main.to_csv(f"01test_output_epoch_loss_rmse_goal_main.csv", index=False)

    # test_df_iter_loss_rmse_zone_depart_main = pd.DataFrame(
    #     {'test_list_loss_rmse_zone_depart_main': test_list_iter_loss_rmse_zone_depart_main})
    # test_df_iter_loss_rmse_zone_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_rmse\01test_output_iter_loss_rmse_zone_depart_main.csv", index=False)
    # test_df_epoch_loss_rmse_zone_depart_main = pd.DataFrame(
    #     {'test_list_epoch_loss_rmse_zone_depart_main': test_list_epoch_loss_rmse_zone_depart_main})
    # test_df_epoch_loss_rmse_zone_depart_main.to_csv(f"01test_output_epoch_loss_rmse_zone_depart_main.csv", index=False)

    test_df_iter_loss_rmse_zone_arrive_main = pd.DataFrame(
        {'test_list_loss_rmse_zone_arrive_main': test_list_iter_loss_rmse_zone_arrive_main})
    test_df_iter_loss_rmse_zone_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_rmse\01test_output_iter_loss_rmse_zone_arrive_main.csv", index=False)
    # test_df_epoch_loss_rmse_zone_arrive_main = pd.DataFrame(
    #     {'test_list_epoch_loss_rmse_zone_arrive_main': test_list_epoch_loss_rmse_zone_arrive_main})
    # test_df_epoch_loss_rmse_zone_arrive_main.to_csv(f"01test_output_epoch_loss_rmse_zone_arrive_main.csv", index=False)

    test_df_iter_loss_rmse_time_depart_main = pd.DataFrame(
        {'test_list_loss_rmse_time_depart_main': test_list_iter_loss_rmse_time_depart_main})
    test_df_iter_loss_rmse_time_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_rmse\01test_output_iter_loss_rmse_time_depart_main.csv", index=False)
    # test_df_epoch_loss_rmse_time_depart_main = pd.DataFrame(
    #     {'test_list_epoch_loss_rmse_time_depart_main': test_list_epoch_loss_rmse_time_depart_main})
    # test_df_epoch_loss_rmse_time_depart_main.to_csv(f"01test_output_epoch_loss_rmse_time_depart_main.csv", index=False)

    test_df_iter_loss_rmse_time_arrive_main = pd.DataFrame(
        {'test_list_loss_rmse_time_arrive_main': test_list_iter_loss_rmse_time_arrive_main})
    test_df_iter_loss_rmse_time_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_rmse\01test_output_iter_loss_rmse_time_arrive_main.csv", index=False)
    # test_df_epoch_loss_rmse_time_arrive_main = pd.DataFrame(
    #     {'test_list_epoch_loss_rmse_time_arrive_main': test_list_epoch_loss_rmse_time_arrive_main})
    # test_df_epoch_loss_rmse_time_arrive_main.to_csv(f"01test_output_epoch_loss_rmse_time_arrive_main.csv", index=False)

    test_df_iter_loss_rmse_mode1_main = pd.DataFrame(
        {'test_list_loss_rmse_mode1_main': test_list_iter_loss_rmse_mode1_main})
    test_df_iter_loss_rmse_mode1_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_rmse\01test_output_iter_loss_rmse_mode1_main.csv", index=False)
    # test_df_epoch_loss_rmse_mode1_main = pd.DataFrame(
    #     {'test_list_epoch_loss_rmse_mode1_main': test_list_epoch_loss_rmse_mode1_main})
    # test_df_epoch_loss_rmse_mode1_main.to_csv(f"01test_output_epoch_loss_rmse_mode1_main.csv", index=False)

    test_df_iter_loss_rmse_mode2_main = pd.DataFrame(
        {'test_list_loss_rmse_mode2_main': test_list_iter_loss_rmse_mode2_main})
    test_df_iter_loss_rmse_mode2_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_rmse\01test_output_iter_loss_rmse_mode2_main.csv", index=False)
    # test_df_epoch_loss_rmse_mode2_main = pd.DataFrame(
    #     {'test_list_epoch_loss_rmse_mode2_main': test_list_epoch_loss_rmse_mode2_main})
    # test_df_epoch_loss_rmse_mode2_main.to_csv(f"01test_output_epoch_loss_rmse_mode2_main.csv", index=False)

    test_df_iter_loss_rmse = pd.DataFrame({'test_list_loss_rmse': test_list_iter_loss_rmse})
    test_df_iter_loss_rmse.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_rmse\01test_output_iter_loss_rmse.csv", index=False)
    # test_df_epoch_loss_rmse = pd.DataFrame({'test_list_epoch_loss_rmse': test_list_epoch_loss_rmse})
    # test_df_epoch_loss_rmse.to_csv(f"01test_output_epoch_loss_rmse.csv", index=False)
    # =====


def save_loss_mae_csv_test(train_list_iter_loss_mae_goal_main,
             train_list_iter_loss_mae_zone_arrive_main,
             train_list_iter_loss_mae_time_depart_main, train_list_iter_loss_mae_time_arrive_main,
             train_list_iter_loss_mae_mode1_main,
             train_list_iter_loss_mae_mode2_main, train_list_iter_loss_mae,
             train_list_epoch_loss_mae_goal_main,
             train_list_epoch_loss_mae_zone_arrive_main,
             train_list_epoch_loss_mae_time_depart_main, train_list_epoch_loss_mae_time_arrive_main,
             train_list_epoch_loss_mae_mode1_main,
             train_list_epoch_loss_mae_mode2_main, train_list_epoch_loss_mae,
             test_list_iter_loss_mae_goal_main,
             test_list_iter_loss_mae_zone_arrive_main, test_list_iter_loss_mae_time_depart_main,
             test_list_iter_loss_mae_time_arrive_main, test_list_iter_loss_mae_mode1_main,
             test_list_iter_loss_mae_mode2_main, test_list_iter_loss_mae):

    # Save loss_mae value as csv file: each iter & each epoch
    train_df_iter_loss_mae_goal_main = pd.DataFrame({'train_list_loss_mae_goal_main': train_list_iter_loss_mae_goal_main})
    train_df_iter_loss_mae_goal_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_mae\01train_output_iter_loss_mae_goal_main.csv", index=False)
    train_df_epoch_loss_mae_goal_main = pd.DataFrame({'train_list_epoch_loss_mae_goal_main': train_list_epoch_loss_mae_goal_main})
    train_df_epoch_loss_mae_goal_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_mae\01train_output_epoch_loss_mae_goal_main.csv", index=False)

    # train_df_iter_loss_mae_zone_depart_main = pd.DataFrame(
    #     {'train_list_loss_mae_zone_depart_main': train_list_iter_loss_mae_zone_depart_main})
    # train_df_iter_loss_mae_zone_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_mae\01train_output_iter_loss_mae_zone_depart_main.csv", index=False)
    # train_df_epoch_loss_mae_zone_depart_main = pd.DataFrame(
    #     {'train_list_epoch_loss_mae_zone_depart_main': train_list_epoch_loss_mae_zone_depart_main})
    # train_df_epoch_loss_mae_zone_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_mae\01train_output_epoch_loss_mae_zone_depart_main.csv", index=False)

    train_df_iter_loss_mae_zone_arrive_main = pd.DataFrame(
        {'train_list_loss_mae_zone_arrive_main': train_list_iter_loss_mae_zone_arrive_main})
    train_df_iter_loss_mae_zone_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_mae\01train_output_iter_loss_mae_zone_arrive_main.csv", index=False)
    train_df_epoch_loss_mae_zone_arrive_main = pd.DataFrame(
        {'train_list_epoch_loss_mae_zone_arrive_main': train_list_epoch_loss_mae_zone_arrive_main})
    train_df_epoch_loss_mae_zone_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_mae\01train_output_epoch_loss_mae_zone_arrive_main.csv", index=False)

    train_df_iter_loss_mae_time_depart_main = pd.DataFrame(
        {'train_list_loss_mae_time_depart_main': train_list_iter_loss_mae_time_depart_main})
    train_df_iter_loss_mae_time_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_mae\01train_output_iter_loss_mae_time_depart_main.csv", index=False)
    train_df_epoch_loss_mae_time_depart_main = pd.DataFrame(
        {'train_list_epoch_loss_mae_time_depart_main': train_list_epoch_loss_mae_time_depart_main})
    train_df_epoch_loss_mae_time_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_mae\01train_output_epoch_loss_mae_time_depart_main.csv", index=False)

    train_df_iter_loss_mae_time_arrive_main = pd.DataFrame(
        {'train_list_loss_mae_time_arrive_main': train_list_iter_loss_mae_time_arrive_main})
    train_df_iter_loss_mae_time_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_mae\01train_output_iter_loss_mae_time_arrive_main.csv", index=False)
    train_df_epoch_loss_mae_time_arrive_main = pd.DataFrame(
        {'train_list_epoch_loss_mae_time_arrive_main': train_list_epoch_loss_mae_time_arrive_main})
    train_df_epoch_loss_mae_time_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_mae\01train_output_epoch_loss_mae_time_arrive_main.csv", index=False)

    train_df_iter_loss_mae_mode1_main = pd.DataFrame({'train_list_loss_mae_mode1_main': train_list_iter_loss_mae_mode1_main})
    train_df_iter_loss_mae_mode1_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_mae\01train_output_iter_loss_mae_mode1_main.csv", index=False)
    train_df_epoch_loss_mae_mode1_main = pd.DataFrame(
        {'train_list_epoch_loss_mae_mode1_main': train_list_epoch_loss_mae_mode1_main})
    train_df_epoch_loss_mae_mode1_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_mae\01train_output_epoch_loss_mae_mode1_main.csv", index=False)

    train_df_iter_loss_mae_mode2_main = pd.DataFrame({'train_list_loss_mae_mode2_main': train_list_iter_loss_mae_mode2_main})
    train_df_iter_loss_mae_mode2_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_mae\01train_output_iter_loss_mae_mode2_main.csv", index=False)
    train_df_epoch_loss_mae_mode2_main = pd.DataFrame(
        {'train_list_epoch_loss_mae_mode2_main': train_list_epoch_loss_mae_mode2_main})
    train_df_epoch_loss_mae_mode2_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_mae\01train_output_epoch_loss_mae_mode2_main.csv", index=False)

    train_df_iter_loss_mae = pd.DataFrame({'train_list_loss_mae': train_list_iter_loss_mae})
    train_df_iter_loss_mae.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_mae\01train_output_iter_loss_mae.csv", index=False)
    train_df_epoch_loss_mae = pd.DataFrame({'train_list_epoch_loss_mae': train_list_epoch_loss_mae})
    train_df_epoch_loss_mae.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_mae\01train_output_epoch_loss_mae.csv", index=False)

    # =============================
    test_df_iter_loss_mae_goal_main = pd.DataFrame(
        {'test_list_loss_mae_goal_main': test_list_iter_loss_mae_goal_main})
    test_df_iter_loss_mae_goal_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_mae\01test_output_iter_loss_mae_goal_main.csv", index=False)
    # test_df_epoch_loss_mae_goal_main = pd.DataFrame(
    #     {'test_list_epoch_loss_mae_goal_main': test_list_epoch_loss_mae_goal_main})
    # test_df_epoch_loss_mae_goal_main.to_csv(f"01test_output_epoch_loss_mae_goal_main.csv", index=False)

    # test_df_iter_loss_mae_zone_depart_main = pd.DataFrame(
    #     {'test_list_loss_mae_zone_depart_main': test_list_iter_loss_mae_zone_depart_main})
    # test_df_iter_loss_mae_zone_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_mae\01test_output_iter_loss_mae_zone_depart_main.csv", index=False)
    # test_df_epoch_loss_mae_zone_depart_main = pd.DataFrame(
    #     {'test_list_epoch_loss_mae_zone_depart_main': test_list_epoch_loss_mae_zone_depart_main})
    # test_df_epoch_loss_mae_zone_depart_main.to_csv(f"01test_output_epoch_loss_mae_zone_depart_main.csv", index=False)

    test_df_iter_loss_mae_zone_arrive_main = pd.DataFrame(
        {'test_list_loss_mae_zone_arrive_main': test_list_iter_loss_mae_zone_arrive_main})
    test_df_iter_loss_mae_zone_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_mae\01test_output_iter_loss_mae_zone_arrive_main.csv", index=False)
    # test_df_epoch_loss_mae_zone_arrive_main = pd.DataFrame(
    #     {'test_list_epoch_loss_mae_zone_arrive_main': test_list_epoch_loss_mae_zone_arrive_main})
    # test_df_epoch_loss_mae_zone_arrive_main.to_csv(f"01test_output_epoch_loss_mae_zone_arrive_main.csv", index=False)

    test_df_iter_loss_mae_time_depart_main = pd.DataFrame(
        {'test_list_loss_mae_time_depart_main': test_list_iter_loss_mae_time_depart_main})
    test_df_iter_loss_mae_time_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_mae\01test_output_iter_loss_mae_time_depart_main.csv", index=False)
    # test_df_epoch_loss_mae_time_depart_main = pd.DataFrame(
    #     {'test_list_epoch_loss_mae_time_depart_main': test_list_epoch_loss_mae_time_depart_main})
    # test_df_epoch_loss_mae_time_depart_main.to_csv(f"01test_output_epoch_loss_mae_time_depart_main.csv", index=False)

    test_df_iter_loss_mae_time_arrive_main = pd.DataFrame(
        {'test_list_loss_mae_time_arrive_main': test_list_iter_loss_mae_time_arrive_main})
    test_df_iter_loss_mae_time_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_mae\01test_output_iter_loss_mae_time_arrive_main.csv", index=False)
    # test_df_epoch_loss_mae_time_arrive_main = pd.DataFrame(
    #     {'test_list_epoch_loss_mae_time_arrive_main': test_list_epoch_loss_mae_time_arrive_main})
    # test_df_epoch_loss_mae_time_arrive_main.to_csv(f"01test_output_epoch_loss_mae_time_arrive_main.csv", index=False)

    test_df_iter_loss_mae_mode1_main = pd.DataFrame(
        {'test_list_loss_mae_mode1_main': test_list_iter_loss_mae_mode1_main})
    test_df_iter_loss_mae_mode1_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_mae\01test_output_iter_loss_mae_mode1_main.csv", index=False)
    # test_df_epoch_loss_mae_mode1_main = pd.DataFrame(
    #     {'test_list_epoch_loss_mae_mode1_main': test_list_epoch_loss_mae_mode1_main})
    # test_df_epoch_loss_mae_mode1_main.to_csv(f"01test_output_epoch_loss_mae_mode1_main.csv", index=False)

    test_df_iter_loss_mae_mode2_main = pd.DataFrame(
        {'test_list_loss_mae_mode2_main': test_list_iter_loss_mae_mode2_main})
    test_df_iter_loss_mae_mode2_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_mae\01test_output_iter_loss_mae_mode2_main.csv", index=False)
    # test_df_epoch_loss_mae_mode2_main = pd.DataFrame(
    #     {'test_list_epoch_loss_mae_mode2_main': test_list_epoch_loss_mae_mode2_main})
    # test_df_epoch_loss_mae_mode2_main.to_csv(f"01test_output_epoch_loss_mae_mode2_main.csv", index=False)

    test_df_iter_loss_mae = pd.DataFrame({'test_list_loss_mae': test_list_iter_loss_mae})
    test_df_iter_loss_mae.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_csv_loss_mae\01test_output_iter_loss_mae.csv", index=False)
    # test_df_epoch_loss_mae = pd.DataFrame({'test_list_epoch_loss_mae': test_list_epoch_loss_mae})
    # test_df_epoch_loss_mae.to_csv(f"01test_output_epoch_loss_mae.csv", index=False)
    # =====



def save_loss_csv_all_train(train_list_iter_loss_goal_main,
             train_list_iter_loss_zone_arrive_main,
             train_list_iter_loss_time_depart_main, train_list_iter_loss_time_arrive_main,
             train_list_iter_loss_mode1_main,
             train_list_iter_loss_mode2_main, train_list_iter_loss,
             train_list_epoch_loss_goal_main,
             train_list_epoch_loss_zone_arrive_main,
             train_list_epoch_loss_time_depart_main, train_list_epoch_loss_time_arrive_main,
             train_list_epoch_loss_mode1_main,
             train_list_epoch_loss_mode2_main, train_list_epoch_loss, epoch_early_stopping):
    # Save loss value as csv file: each iter & each epoch
    train_df_iter_loss_goal_main = pd.DataFrame({'train_list_loss_goal_main': train_list_iter_loss_goal_main})
    train_df_iter_loss_goal_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_iter_loss_goal_main.csv", index=False)
    train_df_epoch_loss_goal_main = pd.DataFrame({'train_list_epoch_loss_goal_main': train_list_epoch_loss_goal_main})
    train_df_epoch_loss_goal_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_epoch_loss_goal_main.csv", index=False)

    # train_df_iter_loss_zone_depart_main = pd.DataFrame(
    #     {'train_list_loss_zone_depart_main': train_list_iter_loss_zone_depart_main})
    # train_df_iter_loss_zone_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_iter_loss_zone_depart_main.csv", index=False)
    # train_df_epoch_loss_zone_depart_main = pd.DataFrame(
    #     {'train_list_epoch_loss_zone_depart_main': train_list_epoch_loss_zone_depart_main})
    # train_df_epoch_loss_zone_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_epoch_loss_zone_depart_main.csv", index=False)

    train_df_iter_loss_zone_arrive_main = pd.DataFrame(
        {'train_list_loss_zone_arrive_main': train_list_iter_loss_zone_arrive_main})
    train_df_iter_loss_zone_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_iter_loss_zone_arrive_main.csv", index=False)
    train_df_epoch_loss_zone_arrive_main = pd.DataFrame(
        {'train_list_epoch_loss_zone_arrive_main': train_list_epoch_loss_zone_arrive_main})
    train_df_epoch_loss_zone_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_epoch_loss_zone_arrive_main.csv", index=False)

    train_df_iter_loss_time_depart_main = pd.DataFrame(
        {'train_list_loss_time_depart_main': train_list_iter_loss_time_depart_main})
    train_df_iter_loss_time_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_iter_loss_time_depart_main.csv", index=False)
    train_df_epoch_loss_time_depart_main = pd.DataFrame(
        {'train_list_epoch_loss_time_depart_main': train_list_epoch_loss_time_depart_main})
    train_df_epoch_loss_time_depart_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_epoch_loss_time_depart_main.csv", index=False)

    train_df_iter_loss_time_arrive_main = pd.DataFrame(
        {'train_list_loss_time_arrive_main': train_list_iter_loss_time_arrive_main})
    train_df_iter_loss_time_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_iter_loss_time_arrive_main.csv", index=False)
    train_df_epoch_loss_time_arrive_main = pd.DataFrame(
        {'train_list_epoch_loss_time_arrive_main': train_list_epoch_loss_time_arrive_main})
    train_df_epoch_loss_time_arrive_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_epoch_loss_time_arrive_main.csv", index=False)

    train_df_iter_loss_mode1_main = pd.DataFrame({'train_list_loss_mode1_main': train_list_iter_loss_mode1_main})
    train_df_iter_loss_mode1_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_iter_loss_mode1_main.csv", index=False)
    train_df_epoch_loss_mode1_main = pd.DataFrame(
        {'train_list_epoch_loss_mode1_main': train_list_epoch_loss_mode1_main})
    train_df_epoch_loss_mode1_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_epoch_loss_mode1_main.csv", index=False)

    train_df_iter_loss_mode2_main = pd.DataFrame({'train_list_loss_mode2_main': train_list_iter_loss_mode2_main})
    train_df_iter_loss_mode2_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_iter_loss_mode2_main.csv", index=False)
    train_df_epoch_loss_mode2_main = pd.DataFrame(
        {'train_list_epoch_loss_mode2_main': train_list_epoch_loss_mode2_main})
    train_df_epoch_loss_mode2_main.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_epoch_loss_mode2_main.csv", index=False)

    train_df_iter_loss = pd.DataFrame({'train_list_loss': train_list_iter_loss})
    train_df_iter_loss.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_iter_loss.csv", index=False)
    train_df_epoch_loss = pd.DataFrame({'train_list_epoch_loss': train_list_epoch_loss})
    train_df_epoch_loss.to_csv(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01train_output_epoch_loss.csv", index=False)


# 预热：让 TensorFlow 提前进行图构建、内存分配等初始化工作，从而缩短首次训练的时间
def warmup(model, optimizer, node_embeddings, zone_adjacency_matrix, distances):
    # 预热
    # print("Warm-up...")

    # 在train_step中执行：
    # print(batch_inputs_goal_main.shape,'len(batch_inputs_goal_main)','\n',batch_labels_goals_main.shape,'\n',  batch_inputs_zone_depart_main.shape, '\n', batch_labels_zone_depart_main.shape, '\n', batch_inputs_zone_arrive_main.shape, '\n', batch_labels_zone_arrive_main.shape, '\n', batch_inputs_time_depart_main.shape, '\n', batch_labels_time_depart_main.shape, '\n', batch_inputs_time_arrive_main.shape, '\n', batch_labels_time_arrive_main.shape, '\n', batch_inputs_mode1_main.shape, '\n', batch_labels_mode1_main.shape, '\n', batch_inputs_mode2_main.shape, '\n', batch_labels_mode2_main.shape, '\n', model, '\n', optimizer)
    # (64, 8)
    # (64, 5)
    # (64, 8)
    # (64, 3046)
    # (64, 8)
    # (64, 3046)
    # (64, 8)
    # (64, 24)
    # (64, 8)
    # (64, 24)
    # (64, 8)
    # (64, 6)
    # (64, 8)
    # (64, 6)

    # 创建模拟数据
    batch_size = 64   # 先生成64个样本
    minval = 0  # 整数的最小值
    maxval = 20  # 整数的最大值（不包括）
    
    # input_shape = (batch_size, 50)  # 假设输入张量的形状为 (batch_size, 50)
    # label_shape = (batch_size, 1)  # 假设输出张量的形状为 (batch_size, 1)

    # dummy_inputs_goal_main = tf.random.uniform(shape=(batch_size, 9))
    # dummy_inputs_goal_main = tf.cast(dummy_inputs_goal_main, dtype=tf.int32)
    # 直接生成整数类型的张量
    dummy_inputs_goal_main = tf.random.uniform(shape=(batch_size, 9), minval=minval, maxval=maxval, dtype=tf.int32)

    dummy_labels_goals_main = tf.random.uniform(shape=(batch_size, 5))
    dummy_inputs_zone_depart_main = tf.random.uniform(shape=(batch_size, 9))
    dummy_labels_zone_depart_main = tf.random.uniform(shape=(batch_size, 3046))

    # dummy_inputs_zone_arrive_main = tf.random.uniform(shape=(batch_size, 9))
    dummy_inputs_first_8_columns = tf.random.uniform(shape=(batch_size, 8), minval=minval, maxval=maxval, dtype=tf.int32)  # # 先生成前8列的数据
    # 生成最后一列的数据，范围在 zone_ids_int 内
    dummy_inputs_last_column = tf.constant([zone_ids_int[i] for i in tf.random.uniform(shape=(batch_size,), minval=0, maxval=len(zone_ids_int), dtype=tf.int32)])
    # 将前8列和最后一列合并成一个完整的张量
    dummy_inputs_zone_arrive_main = tf.concat([dummy_inputs_first_8_columns, tf.expand_dims(dummy_inputs_last_column, axis=-1)], axis=1)
    # print('dummy_inputs_zone_arrive_main', dummy_inputs_zone_arrive_main)

    dummy_labels_zone_arrive_main = tf.random.uniform(shape=(batch_size, 14))    # zone的数量

    # dummy_inputs_time_depart_main = tf.random.uniform(shape=(batch_size, 9))
    # dummy_inputs_time_depart_main = tf.random.uniform(shape=(batch_size, 9), minval=minval, maxval=maxval, dtype=tf.int32)
    dummy_inputs_time_depart_main_first_8_columns = tf.random.uniform(shape=(batch_size, 8), minval=minval, maxval=maxval, dtype=tf.int32)  # # 先生成前8列的数据
    # 生成最后一列的数据，范围在 zone_ids_int 内
    dummy_inputs_time_depart_main_last_column = tf.constant([zone_ids_int[i] for i in tf.random.uniform(shape=(batch_size,), minval=0, maxval=len(zone_ids_int), dtype=tf.int32)])
    # 将前8列和最后一列合并成一个完整的张量
    dummy_inputs_time_depart_main = tf.concat([dummy_inputs_time_depart_main_first_8_columns, tf.expand_dims(dummy_inputs_time_depart_main_last_column, axis=-1)], axis=1)
    dummy_labels_time_depart_main = tf.random.uniform(shape=(batch_size, 24))

    # dummy_inputs_time_arrive_main = tf.random.uniform(shape=(batch_size, 9))
    # dummy_inputs_time_arrive_main = tf.random.uniform(shape=(batch_size, 9), minval=minval, maxval=maxval, dtype=tf.int32)
    dummy_inputs_time_arrive_main_first_8_columns = tf.random.uniform(shape=(batch_size, 8), minval=minval,
                                                                      maxval=maxval, dtype=tf.int32)  # # 先生成前8列的数据
    # 生成最后一列的数据，范围在 zone_ids_int 内
    dummy_inputs_time_arrive_main_last_column = tf.constant([zone_ids_int[i] for i in
                                                             tf.random.uniform(shape=(batch_size,), minval=0,
                                                                               maxval=len(zone_ids_int),
                                                                               dtype=tf.int32)])
    # 将前8列和最后一列合并成一个完整的张量
    dummy_inputs_time_arrive_main = tf.concat([dummy_inputs_time_arrive_main_first_8_columns,
                                               tf.expand_dims(dummy_inputs_time_arrive_main_last_column, axis=-1)],
                                              axis=1)
    dummy_labels_time_arrive_main = tf.random.uniform(shape=(batch_size, 24))

    # dummy_inputs_mode1_main = tf.random.uniform(shape=(batch_size, 9))
    # dummy_inputs_mode1_main = tf.random.uniform(shape=(batch_size, 9), minval=minval, maxval=maxval, dtype=tf.int32)
    dummy_inputs_mode1_main_first_8_columns = tf.random.uniform(shape=(batch_size, 8), minval=minval,
                                                                      maxval=maxval, dtype=tf.int32)  # # 先生成前8列的数据
    # 生成最后一列的数据，范围在 zone_ids_int 内
    dummy_inputs_mode1_main_last_column = tf.constant([zone_ids_int[i] for i in
                                                             tf.random.uniform(shape=(batch_size,), minval=0,
                                                                               maxval=len(zone_ids_int),
                                                                               dtype=tf.int32)])
    # 将前8列和最后一列合并成一个完整的张量
    dummy_inputs_mode1_main = tf.concat([dummy_inputs_mode1_main_first_8_columns,
                                               tf.expand_dims(dummy_inputs_mode1_main_last_column, axis=-1)],
                                              axis=1)
    dummy_labels_mode1_main = tf.random.uniform(shape=(batch_size, 6))

    # dummy_inputs_mode2_main = tf.random.uniform(shape=(batch_size, 9))
    # dummy_inputs_mode2_main = tf.random.uniform(shape=(batch_size, 9), minval=minval, maxval=maxval, dtype=tf.int32)
    dummy_inputs_mode2_main_first_8_columns = tf.random.uniform(shape=(batch_size, 8), minval=minval,
                                                                maxval=maxval, dtype=tf.int32)  # # 先生成前8列的数据
    # 生成最后一列的数据，范围在 zone_ids_int 内
    dummy_inputs_mode2_main_last_column = tf.constant([zone_ids_int[i] for i in
                                                       tf.random.uniform(shape=(batch_size,), minval=0,
                                                                         maxval=len(zone_ids_int),
                                                                         dtype=tf.int32)])
    # 将前8列和最后一列合并成一个完整的张量
    dummy_inputs_mode2_main = tf.concat([dummy_inputs_mode2_main_first_8_columns,
                                         tf.expand_dims(dummy_inputs_mode2_main_last_column, axis=-1)],
                                        axis=1)
    dummy_labels_mode2_main = tf.random.uniform(shape=(batch_size, 6))

    # print('dummy_inputs_goal_main', dummy_inputs_goal_main)
    # 执行预热
    train_step(
        dummy_inputs_goal_main, dummy_labels_goals_main,
        dummy_inputs_zone_arrive_main, dummy_labels_zone_arrive_main,
        dummy_inputs_time_depart_main, dummy_labels_time_depart_main,
        dummy_inputs_time_arrive_main, dummy_labels_time_arrive_main,
        dummy_inputs_mode1_main, dummy_labels_mode1_main,
        dummy_inputs_mode2_main, dummy_labels_mode2_main,
        node_embeddings, zone_adjacency_matrix, distances,
        model, optimizer)


    return

def gridsearch(param_grid, datasets_all_train):
    # 创建一个 DataFrame 来存储结果
    df_batchsize_epoch = pd.DataFrame(columns=['batch_size', 'num_epochs', 'validation_loss'])
    df_batchsize_epoch_mean_10fold = pd.DataFrame(columns=['batch_size', 'num_epochs', 'mean_val_loss'])
    # 执行网格搜索Grid Search (属于一种超参数搜索)
    best_params = None
    best_params_mean_10fold = None
    best_validation_loss = float('inf')
    best_validation_loss_mean_10fold = float('inf')

    for params in ParameterGrid(param_grid):
        # 在进行网格搜索时，每次新的超参数组合开始训练之前，重新建立优化器（optimizer）是一个好的实践。这是因为优化器的状态（如动量、学习率等）在训练过程中会被更新，如果不在每次新的超参数组合开始时重置优化器，之前的训练状态可能会干扰新的训练过程。
        # 重新实例化模型以确保每次训练从头开始
        model = CombinedNetwork()
        # 重新实例化优化器
        optimizer = tf.keras.optimizers.Adam()
        # train_vars = model.trainable_variables
        # # 重置优化器状态
        # for var in optimizer.variables():
        #     var.assign(tf.zeros_like(var))
        # 重置优化器状态
        # reset_optimizer(optimizer, model)

        # 克隆模型并重新创建优化器
        # 克隆模型，并重新创建优化器实例是一种有效的方法，可以避免在 @tf.function 装饰器下创建新变量的问题
        # cloned_model = tf.keras.models.clone_model(model)
        # cloned_model.build((None, 1))  # 构建模型以初始化权重
        # # print("Cloned model trainable variables:", [v.name for v in cloned_model.trainable_variables])
        # optimizer = tf.keras.optimizers.Adam()

        # validation_loss = epoch_train_validation(**params)
        batch_size = params['batch_size']
        num_epochs = params['num_epochs']
        # print(batch_size, num_epochs, '7435')

        results = []
        print('Begin: load_data_Kfold_cross_validation.')


        # 开始10次交叉验证
        for fold, datasets in enumerate(load_data_Kfold_cross_validation(batch_size, datasets_all_train, n_splits=10), 1):
            if len(datasets) != 26:
                raise ValueError(f"Fold {fold} did not return 30 datasets, got {len(datasets)} instead.")

            # 重新实例化模型以确保每次训练从头开始
            model = CombinedNetwork()
            # 重新实例化优化器
            optimizer = tf.keras.optimizers.Adam()

            # enumerate 会返回一个包含索引和值的元组，因此你在解包时需要先解包索引和值，然后再解包值中的各个元素。
            (train_dataset_goal_main, train_dataset_zones_arrive_main,
                    train_dataset_times_depart_main, train_dataset_times_arrive_main, train_dataset_mode1_main,
                    train_dataset_mode2_main,
                    validation_dataset_goal_main, validation_dataset_zones_arrive_main,
                    validation_dataset_times_depart_main, validation_dataset_times_arrive_main, validation_dataset_mode1_main,
                    validation_dataset_mode2_main,
                    X_train_features_x_array, y_trainvalidation_goals_main, y_trainvalidation_zones_arrive_main, y_trainvalidation_times_depart_main, y_trainvalidation_times_arrive_main, y_trainvalidation_mode1_main, y_trainvalidation_mode2_main,
                    X_test_features_x_array, y_test_goals_main, y_test_zones_arrive_main, y_test_times_depart_main, y_test_times_arrive_main, y_test_mode1_main,
             y_test_mode2_main) = datasets

            # 获取输入和输出形状
            input_shape = train_dataset_goal_main.element_spec[0].shape[1]
            output_shape = train_dataset_goal_main.element_spec[1].shape[1]

            # # 构建模型
            # model = build_model(input_shape, output_shape)
            # optimizer = tf.keras.optimizers.Adam()

            # 执行训练和验证
            (train_list_iter_loss_goal_main,
             train_list_iter_loss_zone_arrive_main,
             train_list_iter_loss_time_depart_main, train_list_iter_loss_time_arrive_main,
             train_list_iter_loss_mode1_main,
             train_list_iter_loss_mode2_main, train_list_iter_loss,
             train_list_epoch_loss_goal_main,
             train_list_epoch_loss_zone_arrive_main,
             train_list_epoch_loss_time_depart_main, train_list_epoch_loss_time_arrive_main,
             train_list_epoch_loss_mode1_main,
             train_list_epoch_loss_mode2_main, train_list_epoch_loss,
             validation_list_iter_loss_goal_main,
             validation_list_iter_loss_zone_arrive_main, validation_list_iter_loss_time_depart_main,
             validation_list_iter_loss_time_arrive_main, validation_list_iter_loss_mode1_main,
             validation_list_iter_loss_mode2_main, validation_list_iter_loss,
             validation_list_epoch_loss_goal_main,
             validation_list_epoch_loss_zone_arrive_main, validation_list_epoch_loss_time_depart_main,
             validation_list_epoch_loss_time_arrive_main, validation_list_epoch_loss_mode1_main,
             validation_list_epoch_loss_mode2_main, validation_list_epoch_loss, epoch_early_stopping) \
                = epoch_train_validation(batch_size, num_epochs, train_dataset_goal_main,
                                         train_dataset_zones_arrive_main, train_dataset_times_depart_main,
                                         train_dataset_times_arrive_main, train_dataset_mode1_main,
                                         train_dataset_mode2_main,
                                         validation_dataset_goal_main,
                                         validation_dataset_zones_arrive_main, validation_dataset_times_depart_main,
                                         validation_dataset_times_arrive_main, validation_dataset_mode1_main,
                                         validation_dataset_mode2_main, node_embeddings, zone_adjacency_matrix, distances, model, optimizer)

            # 保存训练最佳batch_size, num_epochs和early_stopping到文件
            # with open("01the best batch_size, num_epochs and early_stopping.txt", "w") as f:
            #     f.write(f"The best batch_size, num_epochs and early_stopping are:\n batch_size: {batch_size} \n num_epochs: {num_epochs} \n epoch_early_stopping: {epoch_early_stopping}")

            # 保存损失值csv
            save_loss_csv(train_list_iter_loss_goal_main,
             train_list_iter_loss_zone_arrive_main,
             train_list_iter_loss_time_depart_main, train_list_iter_loss_time_arrive_main,
             train_list_iter_loss_mode1_main,
             train_list_iter_loss_mode2_main, train_list_iter_loss,
             train_list_epoch_loss_goal_main,
             train_list_epoch_loss_zone_arrive_main,
             train_list_epoch_loss_time_depart_main, train_list_epoch_loss_time_arrive_main,
             train_list_epoch_loss_mode1_main,
             train_list_epoch_loss_mode2_main, train_list_epoch_loss,
             validation_list_iter_loss_goal_main,
             validation_list_iter_loss_zone_arrive_main, validation_list_iter_loss_time_depart_main,
             validation_list_iter_loss_time_arrive_main, validation_list_iter_loss_mode1_main,
             validation_list_iter_loss_mode2_main, validation_list_iter_loss,
             validation_list_epoch_loss_goal_main,
             validation_list_epoch_loss_zone_arrive_main, validation_list_epoch_loss_time_depart_main,
             validation_list_epoch_loss_time_arrive_main, validation_list_epoch_loss_mode1_main,
             validation_list_epoch_loss_mode2_main, validation_list_epoch_loss, epoch_early_stopping)

            # 使用epoch而非iter的loss，是因为epoch=这一轮次所有iter的平均值（所有样本参与训练的损失值的平均数），而不是某次迭代的损失值的最小值，避免了偶然某次迭代iter的偶然性带来的数据不准确
            # 一个该轮次的总体性能指标，避免因个别 iter 的异常值而影响结果
            validation_loss = validation_list_epoch_loss[-1]   # 使用最后一次损失值，作为对比
            # print(validation_loss,'validation_loss')

            # 进一步使用所有epoch中的最小loss，避免因个别 epoch 的异常值而影响结果。
            # 最后绘制两条线，来选择最佳「batch_size和epoch」组合取值：横坐标是「batch_size和epoch」组合（如128，200），纵坐标两条线分别是validation_loss（最后一次损失值）、min_value（最低损失值）
            min_value = min(validation_list_epoch_loss)     # 找到最小值及其索引
            min_index = validation_list_epoch_loss.index(min_value)

            # 使用 pd.concat 替代 append
            df_batchsize_epoch = pd.concat([
                df_batchsize_epoch,
                pd.DataFrame([{
                    'batch_size': params['batch_size'],
                    'num_epochs': params['num_epochs'],
                    'validation_loss': validation_loss,
                    'epoch_early_stopping': epoch_early_stopping,
                    'min_value': min_value,
                    'min_index':min_index,
                    'fold': fold
                }])
            ], ignore_index=True)

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_params = params


            train_figure_iter_name = fr'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_png\01Convergence Curv of iter_loss (train) bs{batch_size}_ep{num_epochs}_fold{fold}.png'
            train_figure_epoch_name = fr'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_png\01Convergence Curv of epoch_loss (train) bs{batch_size}_ep{num_epochs}_fold{fold}.png'

            validation_figure_iter_name = fr'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_png\01Convergence Curv of iter_loss (validation) bs{batch_size}_ep{num_epochs}_fold{fold}.png'
            validation_figure_epoch_name = fr'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_png\01Convergence Curv of epoch_loss (validation) bs{batch_size}_ep{num_epochs}_fold{fold}.png'

            train_figure_iter_name_global = fr'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_png\01Convergence Curv of iter_loss (train) bs{batch_size}_ep{num_epochs}_fold{fold}_global.png'
            train_figure_epoch_name_global = fr'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_png\01Convergence Curv of epoch_loss (train) bs{batch_size}_ep{num_epochs}_fold{fold}_global.png'

            validation_figure_iter_name_global = fr'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_png\01Convergence Curv of iter_loss (validation) bs{batch_size}_ep{num_epochs}_fold{fold}_global.png'
            validation_figure_epoch_name_global = fr'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_png\01Convergence Curv of epoch_loss (validation) bs{batch_size}_ep{num_epochs}_fold{fold}_global.png'

            plot_convergence_cure(train_list_iter_loss_goal_main,
                                  train_list_iter_loss_zone_arrive_main, train_list_iter_loss_time_depart_main,
                                  train_list_iter_loss_time_arrive_main, train_list_iter_loss_mode1_main,
                                  train_list_iter_loss_mode2_main, train_list_iter_loss,
                                  train_list_epoch_loss_goal_main,
                                  train_list_epoch_loss_zone_arrive_main, train_list_epoch_loss_time_depart_main,
                                  train_list_epoch_loss_time_arrive_main, train_list_epoch_loss_mode1_main,
                                  train_list_epoch_loss_mode2_main, train_list_epoch_loss,
                                  train_figure_iter_name, train_figure_epoch_name)

            plot_convergence_cure(validation_list_iter_loss_goal_main,
                                  validation_list_iter_loss_zone_arrive_main, validation_list_iter_loss_time_depart_main,
                                  validation_list_iter_loss_time_arrive_main, validation_list_iter_loss_mode1_main,
                                  validation_list_iter_loss_mode2_main, validation_list_iter_loss,
                                  validation_list_epoch_loss_goal_main,
                                  validation_list_epoch_loss_zone_arrive_main, validation_list_epoch_loss_time_depart_main,
                                  validation_list_epoch_loss_time_arrive_main, validation_list_epoch_loss_mode1_main,
                                  validation_list_epoch_loss_mode2_main, validation_list_epoch_loss,
                                  validation_figure_iter_name, validation_figure_epoch_name)

            # 全局绘制到同一张图中
            plot_convergence_cure_global(train_list_iter_loss_goal_main,
                                  train_list_iter_loss_zone_arrive_main, train_list_iter_loss_time_depart_main,
                                  train_list_iter_loss_time_arrive_main, train_list_iter_loss_mode1_main,
                                  train_list_iter_loss_mode2_main, train_list_iter_loss,
                                  train_list_epoch_loss_goal_main,
                                  train_list_epoch_loss_zone_arrive_main, train_list_epoch_loss_time_depart_main,
                                  train_list_epoch_loss_time_arrive_main, train_list_epoch_loss_mode1_main,
                                  train_list_epoch_loss_mode2_main, train_list_epoch_loss,
                                  train_figure_iter_name_global, train_figure_epoch_name_global, is_validation=False)

            plot_convergence_cure_global(validation_list_iter_loss_goal_main,
                                  validation_list_iter_loss_zone_arrive_main,
                                  validation_list_iter_loss_time_depart_main,
                                  validation_list_iter_loss_time_arrive_main, validation_list_iter_loss_mode1_main,
                                  validation_list_iter_loss_mode2_main, validation_list_iter_loss,
                                  validation_list_epoch_loss_goal_main,
                                  validation_list_epoch_loss_zone_arrive_main,
                                  validation_list_epoch_loss_time_depart_main,
                                  validation_list_epoch_loss_time_arrive_main, validation_list_epoch_loss_mode1_main,
                                  validation_list_epoch_loss_mode2_main, validation_list_epoch_loss,
                                  validation_figure_iter_name_global, validation_figure_epoch_name_global, is_validation=False)

            # 记录验证集上的最佳性能
            best_val_loss = np.min(validation_list_epoch_loss)
            results.append(best_val_loss)

            print(f"Fold {fold} best validation loss: {best_val_loss:.4f}")

        train_figure_iter_name_global_all = fr'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_png\01Convergence Curv of iter_loss (train) bs{batch_size}_ep{num_epochs}_global_all.png'
        train_figure_epoch_name_global_all = fr'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_png\01Convergence Curv of epoch_loss (train) bs{batch_size}_ep{num_epochs}_global_all.png'
        validation_figure_iter_name_global_all = fr'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_png\01Convergence Curv of iter_loss (validation) bs{batch_size}_ep{num_epochs}_global_all.png'
        validation_figure_epoch_name_global_all = fr'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_png\01Convergence Curv of epoch_loss (validation) bs{batch_size}_ep{num_epochs}_global_all.png'

        # 在所有调用之后保存图像
        save_plots(train_figure_iter_name_global_all, validation_figure_iter_name_global_all)
        save_plots(train_figure_epoch_name_global_all, validation_figure_epoch_name_global_all)

        print('global figure finished')

        # 计算所有折叠的平均验证损失
        mean_val_loss = np.mean(results)
        print(f"Mean validation loss across all folds: {mean_val_loss:.4f}")

        df_batchsize_epoch_mean_10fold = pd.concat([
            df_batchsize_epoch_mean_10fold,
            pd.DataFrame([{
                'batch_size': params['batch_size'],
                'num_epochs': params['num_epochs'],
                'mean_val_loss': mean_val_loss,
            }])
        ], ignore_index=True)

        if mean_val_loss < best_validation_loss_mean_10fold:
            best_validation_loss_mean_10fold = mean_val_loss
            best_params_mean_10fold = params

    print(f'Best last validation loss: {best_validation_loss} with parameters {best_params}')
    print(f'Best last validation loss of mean 10 fold: {best_validation_loss_mean_10fold} with parameters {best_params_mean_10fold}')

    # 定义文件路径
    file_path = r"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_time\01output_total_epoch_training_and_validation_time.txt"

    # 打开文件以追加模式写入
    with open(file_path, "a") as f:
        # 写入总训练时间（假设这个变量已经被定义）
        # f.write(f"\nTotal epoch training and validation took {total_epoch_training_and_validation_time:.5f} seconds.\n")
        # 写入最佳验证损失及其对应的参数
        f.write(f'\n\nBest last validation loss: {best_validation_loss} with parameters {best_params}\n')
        # 写入最佳平均验证损失及其对应的参数
        f.write(f'Best last validation loss of mean 10 fold: {best_validation_loss_mean_10fold} with parameters {best_params_mean_10fold}\n')

    # 将 DataFrame 保存到 CSV 文件
    df_batchsize_epoch.to_csv(r'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01grid_search_results and epoch_early_stopping.csv', index=False)
    df_batchsize_epoch_mean_10fold.to_csv(r'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_csv_loss\01grid_search_10fold_results .csv', index=False)

    print('Grid search is complete')

    return best_validation_loss, best_params, df_batchsize_epoch, best_validation_loss_mean_10fold, best_params_mean_10fold, df_batchsize_epoch_mean_10fold


# 定义重置优化器状态的函数
def reset_optimizer(optimizer, model):
    # 重置优化器的槽变量
    for var in model.trainable_variables:
        if optimizer.get_slot(var, 'm') is not None:
            optimizer.get_slot(var, 'm').assign(tf.zeros_like(optimizer.get_slot(var, 'm')))
        if optimizer.get_slot(var, 'v') is not None:
            optimizer.get_slot(var, 'v').assign(tf.zeros_like(optimizer.get_slot(var, 'v')))

    # 重置优化器的其他状态变量
    for var in optimizer.variables():
        if isinstance(var, tf.Variable):
            var.assign(tf.zeros_like(var))



def plot_convergence_cure(list_iter_loss_goal_main, list_iter_loss_zone_arrive_main, list_iter_loss_time_depart_main, list_iter_loss_time_arrive_main, list_iter_loss_mode1_main, list_iter_loss_mode2_main, list_iter_loss,
                      list_epoch_loss_goal_main, list_epoch_loss_zone_arrive_main, list_epoch_loss_time_depart_main, list_epoch_loss_time_arrive_main, list_epoch_loss_mode1_main, list_epoch_loss_mode2_main, list_epoch_loss,
                          figure_iter_name, figure_epoch_name):

    # 设置全局字体为 Times New Roman, 字号为 12
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    # 对每个损失列表应用滤波
    # list_iter_loss_goal_main_smooth = moving_average(list_iter_loss_goal_main, window_size=5)
    # ... 对其他列表进行类似处理

    # (1) plot iter_loss
    plt.tight_layout()  # 使得子图之间的间距更加合理
    plt.figure(figsize=(7, 6))
    plt.plot(list_iter_loss_goal_main, '#9dd5ff', label="F1(goal_main)")
    # plt.plot(list_iter_loss_zone_depart_main, '#70ad47', label="F2(zone_depart_main)")
    plt.plot(list_iter_loss_zone_arrive_main, '#c0dda3', label="F2(zone_arrive_main)")
    plt.plot(list_iter_loss_time_depart_main, '#ffbd00', label="F3(time_depart_main)")
    plt.plot(list_iter_loss_time_arrive_main, '#ffe9a1', label="F4(time_arrive_main)")
    plt.plot(list_iter_loss_mode1_main, '#ce0afe', label="F5(mode1_main)")
    plt.plot(list_iter_loss_mode2_main, '#ffc2eb', label="F6(mode2_main)")
    plt.plot(list_iter_loss, '#0070c0', linestyle='--', linewidth='2.5', label="F(total_main)")

    my_font1 = {'family': 'Times New Roman', 'size': 12}
    plt.legend(loc="best", fontsize=10, prop=my_font1)
    # plt.title(figure_iter_name[2:-4], pad=20, loc='center', fontdict={'family': 'Times New Roman', 'size': 12})
    # plt.figtext(0.5, 0.03, figure_iter_name[2:-4], ha='center', va='center', fontsize=12, fontfamily='Times New Roman')
    # 调整布局，留出底部空间
    plt.subplots_adjust(bottom=0.15)  # 调整底部空间

    plt.xlabel('Iteration', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
    plt.ylabel('Loss', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变纵坐标轴标题字体
    # plt.axis([-1, 101, -1, 51])  # 改变xy坐标轴范围
    # 设置坐标轴范围
    # plt.xlim(0, 6)  # 设置 x 轴范围
    # plt.ylim(0, 40)  # 设置 y 轴范围
    x_ticks = np.arange(0, 150, 50)
    y_ticks = np.arange(0, 0.5, 0.1)  # 生成 0 到 2 之间的数，步长为 0.2
    plt.xticks(x_ticks, fontproperties='Times New Roman', size=12)
    plt.yticks(y_ticks, fontproperties='Times New Roman', size=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')  # 启用横竖网格线，并设置样式   # 启用网格线
    plt.savefig(figure_iter_name, dpi=800, format='png')
    # plt.show()
    plt.close()  # 清除当前绘图窗口

    # (2) plot epoch_loss
    plt.tight_layout()  # 使得子图之间的间距更加合理
    plt.figure(figsize=(7, 6))
    plt.plot(list_epoch_loss_goal_main, '#9dd5ff', label="F1(goal_main)")
    # plt.plot(list_epoch_loss_zone_depart_main, '#70ad47', label="F2(zone_depart_main)")
    plt.plot(list_epoch_loss_zone_arrive_main, '#c0dda3', label="F2(zone_arrive_main)")
    plt.plot(list_epoch_loss_time_depart_main, '#ffbd00', label="F3(time_depart_main)")
    plt.plot(list_epoch_loss_time_arrive_main, '#ffe9a1', label="F4(time_arrive_main)")
    plt.plot(list_epoch_loss_mode1_main, '#ce0afe', label="F5(mode1_main)")
    plt.plot(list_epoch_loss_mode2_main, '#ffc2eb', label="F6(mode2_main)")
    plt.plot(list_epoch_loss, '#0070c0', linestyle='--', linewidth='2.5', label="F(total_main)")

    my_font1 = {'family': 'Times New Roman', 'size': 12}
    plt.legend(loc="best", fontsize=10, prop=my_font1)
    # plt.title(figure_epoch_name[2:-4], pad=20, loc='center', fontdict={'family': 'Times New Roman', 'size': 12})
    # plt.figtext(0.5, 0.03, figure_epoch_name[2:-4], ha='center', va='center', fontsize=12, fontfamily='Times New Roman')
    # 调整布局，留出底部空间
    plt.subplots_adjust(bottom=0.15)  # 调整底部空间
    plt.xlabel('Epoch', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
    plt.ylabel('Loss', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变纵坐标轴标题字体
    x_ticks = np.arange(1, 10, 2)
    y_ticks = np.arange(0, 0.5, 0.1)  # 生成 0 到 2 之间的数，步长为 0.2
    plt.xticks(x_ticks, fontproperties='Times New Roman', size=12)
    plt.yticks(y_ticks, fontproperties='Times New Roman', size=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')  # 启用横竖网格线，并设置样式   # 启用网格线
    plt.savefig(figure_epoch_name, dpi=800, format='png')
    # plt.show()
    plt.close()  # 清除当前绘图窗口

    return


def plot_convergence_cure_global(list_iter_loss_goal_main, list_iter_loss_zone_arrive_main, list_iter_loss_time_depart_main, list_iter_loss_time_arrive_main, list_iter_loss_mode1_main, list_iter_loss_mode2_main, list_iter_loss,
                      list_epoch_loss_goal_main, list_epoch_loss_zone_arrive_main, list_epoch_loss_time_depart_main, list_epoch_loss_time_arrive_main, list_epoch_loss_mode1_main, list_epoch_loss_mode2_main, list_epoch_loss,
                          figure_iter_name, figure_epoch_name, is_validation=False):

    print('plot_convergence_cure_global', plot_convergence_cure_global)

    # 设置全局字体为 Times New Roman, 字号为 12
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    global fig_train, ax_train, fig_val, ax_val, train_lines_added, val_lines_added

    # 创建或获取正确的绘图对象
    if not is_validation:
        if fig_train is None:
            fig_train, ax_train = plt.subplots(figsize=(7, 6))
        ax = ax_train
        lines_added = train_lines_added
    else:
        if fig_val is None:
            fig_val, ax_val = plt.subplots(figsize=(7, 6))
        ax = ax_val
        lines_added = val_lines_added


    # 对每个损失列表应用滤波
    # list_iter_loss_goal_main_smooth = moving_average(list_iter_loss_goal_main, window_size=5)
    # ... 对其他列表进行类似处理

    # # (1) plot iter_loss
    # plt.tight_layout()  # 使得子图之间的间距更加合理
    # plt.figure(figsize=(7, 6))
    # plt.plot(list_iter_loss_goal_main, '#9dd5ff', label="F1(goal_main)")
    # plt.plot(list_iter_loss_zone_depart_main, '#70ad47', label="F2(zone_depart_main)")
    # plt.plot(list_iter_loss_zone_arrive_main, '#c0dda3', label="F3(zone_arrive_main)")
    # plt.plot(list_iter_loss_time_depart_main, '#ffbd00', label="F4(time_depart_main)")
    # plt.plot(list_iter_loss_time_arrive_main, '#ffe9a1', label="F5(time_arrive_main)")
    # plt.plot(list_iter_loss_mode1_main, '#ce0afe', label="F6(mode1_main)")
    # plt.plot(list_iter_loss_mode2_main, '#ffc2eb', label="F7(mode2_main)")
    # plt.plot(list_iter_loss, '#0070c0', linestyle='--', linewidth='2.5', label="F(total_main)")

    # (1) plot iter_loss
    labels = [
        "F1(goal_main)", "F2(zone_arrive_main)",
        "F3(time_depart_main)", "F4(time_arrive_main)", "F5(mode1_main)",
        "F6(mode2_main)", "F(total_main)"
    ]
    losses = [
        list_iter_loss_goal_main, list_iter_loss_zone_arrive_main,
        list_iter_loss_time_depart_main, list_iter_loss_time_arrive_main, list_iter_loss_mode1_main,
        list_iter_loss_mode2_main, list_iter_loss
    ]

    for loss, label in zip(losses, labels):
        ax.plot(loss, color=color_map_many[label], label=label if lines_added == 0 else "")

    # 更新已添加的线数计数器
    if not is_validation:
        train_lines_added += 8
    else:
        val_lines_added += 8

    # 设置图形属性
    my_font1 = {'family': 'Times New Roman', 'size': 12}
    ax.legend(loc="best", fontsize=10, prop=my_font1)
    ax.set_xlabel('Iteration' if not is_validation else 'Epoch', labelpad=7.5,
                  fontdict={'family': 'Times New Roman', 'size': 12})
    ax.set_ylabel('Loss', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12})
    ax.grid(True, which='both', linestyle='--', linewidth=0.001, color='gray')

    # 调整布局，留出底部空间
    plt.subplots_adjust(bottom=0.15)

    # 设置坐标轴刻度
    x_ticks = np.arange(0, 5000, 500) if not is_validation else np.arange(1, 300, 50)
    y_ticks = np.arange(0, 1.0, 0.2)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # 添加标题
    title_text = figure_iter_name[2:-4] if not is_validation else figure_epoch_name[2:-4]
    # plt.figtext(0.5, 0.03, title_text, ha='center', va='center', fontsize=12, fontfamily='Times New Roman')

    # 紧凑布局
    plt.tight_layout()



    # my_font1 = {'family': 'Times New Roman', 'size': 12}
    # plt.legend(loc="best", fontsize=10, prop=my_font1)
    # # plt.title(figure_iter_name[2:-4], pad=20, loc='center', fontdict={'family': 'Times New Roman', 'size': 12})
    # plt.figtext(0.5, 0.03, figure_iter_name[2:-4], ha='center', va='center', fontsize=12, fontfamily='Times New Roman')
    # # 调整布局，留出底部空间
    # plt.subplots_adjust(bottom=0.15)  # 调整底部空间
    #
    # plt.xlabel('Iteration', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
    # plt.ylabel('Loss', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变纵坐标轴标题字体
    # # plt.axis([-1, 101, -1, 51])  # 改变xy坐标轴范围
    # # 设置坐标轴范围
    # # plt.xlim(0, 6)  # 设置 x 轴范围
    # # plt.ylim(0, 40)  # 设置 y 轴范围
    # x_ticks = np.arange(0, 5000, 500)
    # y_ticks = np.arange(0, 1.0, 0.2)  # 生成 0 到 2 之间的数，步长为 0.2
    # plt.xticks(x_ticks, fontproperties='Times New Roman', size=12)
    # plt.yticks(y_ticks, fontproperties='Times New Roman', size=12)
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')  # 启用横竖网格线，并设置样式   # 启用网格线
    # plt.savefig(figure_iter_name, dpi=800, format='png')
    # # plt.show()
    # plt.close()  # 清除当前绘图窗口

    # (2) plot epoch_loss
    plt.tight_layout()  # 使得子图之间的间距更加合理
    plt.figure(figsize=(7, 6))
    plt.plot(list_epoch_loss_goal_main, '#9dd5ff', label="F1(goal_main)")
    # plt.plot(list_epoch_loss_zone_depart_main, '#70ad47', label="F2(zone_depart_main)")
    plt.plot(list_epoch_loss_zone_arrive_main, '#c0dda3', label="F2(zone_arrive_main)")
    plt.plot(list_epoch_loss_time_depart_main, '#ffbd00', label="F3(time_depart_main)")
    plt.plot(list_epoch_loss_time_arrive_main, '#ffe9a1', label="F4(time_arrive_main)")
    plt.plot(list_epoch_loss_mode1_main, '#ce0afe', label="F5(mode1_main)")
    plt.plot(list_epoch_loss_mode2_main, '#ffc2eb', label="F6(mode2_main)")
    plt.plot(list_epoch_loss, '#0070c0', linestyle='--', linewidth='2.5', label="F(total_main)")

    my_font1 = {'family': 'Times New Roman', 'size': 12}
    plt.legend(loc="best", fontsize=10, prop=my_font1)
    # plt.title(figure_epoch_name[2:-4], pad=20, loc='center', fontdict={'family': 'Times New Roman', 'size': 12})
    # plt.figtext(0.5, 0.03, figure_epoch_name[2:-4], ha='center', va='center', fontsize=12, fontfamily='Times New Roman')
    # 调整布局，留出底部空间
    plt.subplots_adjust(bottom=0.15)  # 调整底部空间
    plt.xlabel('Epoch', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
    plt.ylabel('Loss', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变纵坐标轴标题字体
    x_ticks = np.arange(1, 300, 50)
    y_ticks = np.arange(0, 5.0, 1.0)  # 生成 0 到 2 之间的数，步长为 0.2
    plt.xticks(x_ticks, fontproperties='Times New Roman', size=12)
    plt.yticks(y_ticks, fontproperties='Times New Roman', size=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')  # 启用横竖网格线，并设置样式   # 启用网格线
    plt.savefig(figure_epoch_name, dpi=800, format='png')
    # plt.show()
    plt.close()  # 清除当前绘图窗口

    return


def save_plots(train_figure_name, validation_figure_name):
    global fig_train, ax_train, fig_val, ax_val

    if fig_train is not None:
        fig_train.savefig(train_figure_name, dpi=800, format='png')
        # plt.show()
        plt.close(fig_train)
        fig_train = None

    if fig_val is not None:
        fig_val.savefig(validation_figure_name, dpi=800, format='png')
        # plt.show()
        plt.close(fig_val)
        fig_val = None


def plot_convergence_cure_cross_validation(list_iter_loss_goal_main, list_iter_loss_zone_arrive_main, list_iter_loss_time_depart_main, list_iter_loss_time_arrive_main, list_iter_loss_mode1_main, list_iter_loss_mode2_main, list_iter_loss,
                      list_epoch_loss_goal_main, list_epoch_loss_zone_arrive_main, list_epoch_loss_time_depart_main, list_epoch_loss_time_arrive_main, list_epoch_loss_mode1_main, list_epoch_loss_mode2_main, list_epoch_loss,
                          figure_iter_name, figure_epoch_name):

    # 设置全局字体为 Times New Roman, 字号为 12
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    # 对每个损失列表应用滤波
    # list_iter_loss_goal_main_smooth = moving_average(list_iter_loss_goal_main, window_size=5)
    # ... 对其他列表进行类似处理

    # (1) plot iter_loss
    plt.tight_layout()  # 使得子图之间的间距更加合理
    plt.figure(figsize=(7, 6))
    plt.plot(list_iter_loss_goal_main, '#9dd5ff', label="F1(goal_main)")
    # plt.plot(list_iter_loss_zone_depart_main, '#70ad47', label="F2(zone_depart_main)")
    plt.plot(list_iter_loss_zone_arrive_main, '#c0dda3', label="F2(zone_arrive_main)")
    plt.plot(list_iter_loss_time_depart_main, '#ffbd00', label="F3(time_depart_main)")
    plt.plot(list_iter_loss_time_arrive_main, '#ffe9a1', label="F4(time_arrive_main)")
    plt.plot(list_iter_loss_mode1_main, '#ce0afe', label="F5(mode1_main)")
    plt.plot(list_iter_loss_mode2_main, '#ffc2eb', label="F6(mode2_main)")
    plt.plot(list_iter_loss, '#0070c0', linestyle='--', linewidth='2.5', label="F(total_main)")

    my_font1 = {'family': 'Times New Roman', 'size': 12}
    plt.legend(loc="best", fontsize=10, prop=my_font1)
    # plt.title(figure_iter_name[2:-4], pad=20, loc='center', fontdict={'family': 'Times New Roman', 'size': 12})
    # plt.figtext(0.5, 0.03, figure_iter_name[2:-4], ha='center', va='center', fontsize=12, fontfamily='Times New Roman')
    # 调整布局，留出底部空间
    plt.subplots_adjust(bottom=0.15)  # 调整底部空间

    plt.xlabel('Iteration', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
    plt.ylabel('Loss', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变纵坐标轴标题字体
    # plt.axis([-1, 101, -1, 51])  # 改变xy坐标轴范围
    # 设置坐标轴范围
    # plt.xlim(0, 6)  # 设置 x 轴范围
    # plt.ylim(0, 40)  # 设置 y 轴范围
    x_ticks = np.arange(0, 1800, 50)
    y_ticks = np.arange(0, 0.9, 0.2)  # 生成 0 到 2 之间的数，步长为 0.2
    plt.xticks(x_ticks, fontproperties='Times New Roman', size=12)
    plt.yticks(y_ticks, fontproperties='Times New Roman', size=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')  # 启用横竖网格线，并设置样式   # 启用网格线
    plt.savefig(figure_iter_name, dpi=800, format='png')
    # plt.show()
    plt.close()  # 清除当前绘图窗口

    # (2) plot epoch_loss
    plt.tight_layout()  # 使得子图之间的间距更加合理
    plt.figure(figsize=(7, 6))
    plt.plot(list_epoch_loss_goal_main, '#9dd5ff', label="F1(goal_main)")
    # plt.plot(list_epoch_loss_zone_depart_main, '#70ad47', label="F2(zone_depart_main)")
    plt.plot(list_epoch_loss_zone_arrive_main, '#c0dda3', label="F2(zone_arrive_main)")
    plt.plot(list_epoch_loss_time_depart_main, '#ffbd00', label="F3(time_depart_main)")
    plt.plot(list_epoch_loss_time_arrive_main, '#ffe9a1', label="F4(time_arrive_main)")
    plt.plot(list_epoch_loss_mode1_main, '#ce0afe', label="F5(mode1_main)")
    plt.plot(list_epoch_loss_mode2_main, '#ffc2eb', label="F6(mode2_main)")
    plt.plot(list_epoch_loss, '#0070c0', linestyle='--', linewidth='2.5', label="F(total_main)")

    my_font1 = {'family': 'Times New Roman', 'size': 12}
    plt.legend(loc="best", fontsize=10, prop=my_font1)
    # plt.title(figure_epoch_name[2:-4], pad=20, loc='center', fontdict={'family': 'Times New Roman', 'size': 12})
    # plt.figtext(0.5, 0.03, figure_epoch_name[2:-4], ha='center', va='center', fontsize=12, fontfamily='Times New Roman')
    # 调整布局，留出底部空间
    plt.subplots_adjust(bottom=0.15)  # 调整底部空间
    plt.xlabel('Epoch', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
    plt.ylabel('Loss', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变纵坐标轴标题字体
    x_ticks = np.arange(1, 300, 20)
    y_ticks = np.arange(0, 0.9, 0.2)  # 生成 0 到 2 之间的数，步长为 0.2
    plt.xticks(x_ticks, fontproperties='Times New Roman', size=12)
    plt.yticks(y_ticks, fontproperties='Times New Roman', size=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')  # 启用横竖网格线，并设置样式   # 启用网格线
    plt.savefig(figure_epoch_name, dpi=800, format='png')
    # plt.show()
    plt.close()  # 清除当前绘图窗口

    return


def plot_gridsearch_batchsize_epoch(df_batchsize_epoch):
    # 创建横坐标的组合
    # print(df_batchsize_epoch[['batch_size', 'num_epochs']].isnull().sum())
    # 用0填充缺失值

    # 设置全局字体为 Times New Roman, 字号为 12
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    df_batchsize_epoch['batch_size'].fillna(0, inplace=True)
    df_batchsize_epoch['num_epochs'].fillna(0, inplace=True)
    df_batchsize_epoch['epoch_early_stopping'].fillna(0, inplace=True)
    df_batchsize_epoch['min_index'].fillna(0, inplace=True)

    # 将 min_index 和 epoch_early_stopping 转换为整数
    df_batchsize_epoch['min_index'] = df_batchsize_epoch['min_index'].astype(int)
    df_batchsize_epoch['epoch_early_stopping'] = df_batchsize_epoch['epoch_early_stopping'].astype(int)

    # 重新生成 batch_size_epoch 列
    df_batchsize_epoch['batch_size_epoch'] = df_batchsize_epoch['batch_size'].astype(str) + ', ' + df_batchsize_epoch['num_epochs'].astype(str)

    # 过滤掉 epoch_early_stopping 为 0 的值，不画。并在这些点的前后两个有效点之间用空心圆圈来表示缺失值
    df_filtered = df_batchsize_epoch[df_batchsize_epoch['epoch_early_stopping'] > 0]

    # 绘制曲线图 折线图
    plt.figure(figsize=(7, 6))
    # 对于validation_loss和min_value绘制柱状图，对于min_index和epoch_early_stopping绘制折线图
    # 绘制 validation_loss 曲线
    # plt.plot(df_batchsize_epoch['batch_size_epoch'], df_batchsize_epoch['validation_loss'], marker='o', label='Last Validation Loss', linewidth=2, color='#975AA4')
    # 绘制 min_value 曲线
    # plt.plot(df_batchsize_epoch['batch_size_epoch'], df_batchsize_epoch['min_value'], marker='x', label='Minimum Validation Loss', linewidth=2, color='#F7CD3B')
    # 绘制 epoch_early_stopping 曲线
    # plt.plot(df_batchsize_epoch['batch_size_epoch'], df_batchsize_epoch['epoch_early_stopping'], marker='s', label='Epoch Early Stopping', linewidth=1, color='green')
    # 绘制 min_index 曲线
    # plt.plot(df_batchsize_epoch['batch_size_epoch'], df_batchsize_epoch['min_index'], marker='^', label='Min Index', linewidth=1, color='purple')

    # 绘制柱状图
    # 定义柱子的宽度
    bar_width = 0.35
    # 定义柱子的位置
    index = range(len(df_batchsize_epoch))
    # 绘制 validation_loss 柱状图
    plt.bar([i - 0.5 * bar_width for i in index], df_batchsize_epoch['validation_loss'], bar_width, label='Last Validation Loss', color='#975AA4', alpha=0.8)
    # 绘制 min_value 柱状图
    plt.bar([i + 0.5 * bar_width for i in index], df_batchsize_epoch['min_value'], bar_width, label='Minimum Validation Loss', color='#F7CD3B', alpha=0.8)
    # plt.bar([i + 0.5 * bar_width for i in index], df_batchsize_epoch['min_index'], bar_width, label='Epoch Early Stopping', color='#1AAAC2', alpha=0.8)
    # plt.bar([i + 1.5 * bar_width for i in index], df_batchsize_epoch['epoch_early_stopping'], bar_width, label='Min Index', color='#D96161', alpha=0.8)

    # 绘制 min_index 折线图
    plt.plot(index, df_batchsize_epoch['min_index'], marker='^', label='Min Index', linewidth=1, color='#1AAAC2')
    # 绘制 epoch_early_stopping 折线图
    # plt.plot(index, df_batchsize_epoch['epoch_early_stopping'], marker='s', label='Epoch Early Stopping', linewidth=1, color='#D96161')
    # 绘制 epoch_early_stopping 折线图
    plt.plot(df_filtered.index, df_filtered['epoch_early_stopping'], marker='s', label='Epoch Early Stopping', linewidth=1, color='#D96161')

    # 找到 epoch_early_stopping 为 0 的位置，并在前后两个有效点之间用空心圆圈表示
    for i in range(1, len(df_batchsize_epoch) - 1):
        if i == 0 or i == len(df_batchsize_epoch) - 1:
            continue  # 跳过第一行和最后一行

        if df_batchsize_epoch.loc[i, 'epoch_early_stopping'] == 0:
            if df_batchsize_epoch.loc[i - 1, 'epoch_early_stopping'] > 0 and df_batchsize_epoch.loc[i + 1, 'epoch_early_stopping'] > 0:
                x = (i - 1 + i + 1) / 2
                y = (df_batchsize_epoch.loc[i - 1, 'epoch_early_stopping'] + df_batchsize_epoch.loc[i + 1, 'epoch_early_stopping']) / 2
                plt.scatter(x, y, marker='o', edgecolors='#D96161', facecolors='none', s=50, linewidth=2)
            elif df_batchsize_epoch.loc[i + 1, 'epoch_early_stopping'] == 0:
                # 处理连续两个 0 的情况
                j = i + 1
                while j < len(df_batchsize_epoch) and df_batchsize_epoch.loc[j, 'epoch_early_stopping'] == 0:
                    j += 1
                if j < len(df_batchsize_epoch) and df_batchsize_epoch.loc[j, 'epoch_early_stopping'] > 0:
                    x = (i - 1 + j) / 2
                    y = (df_batchsize_epoch.loc[i - 1, 'epoch_early_stopping'] + df_batchsize_epoch.loc[j, 'epoch_early_stopping']) / 2
                    plt.scatter(x, y, marker='o', edgecolors='#D96161', facecolors='none', s=50, linewidth=2)

    # # 找到 epoch_early_stopping 为 0 的位置，并在前后两个有效点之间用空心圆圈表示
    # for i in range(1, len(df_batchsize_epoch)):
    #     if i == 0 or i == len(df_batchsize_epoch) - 1:
    #         continue  # 跳过第一行和最后一行
    #
    #     if df_batchsize_epoch.loc[i, 'epoch_early_stopping'] == 0 and df_batchsize_epoch.loc[i - 1, 'epoch_early_stopping'] > 0 and df_batchsize_epoch.loc[i + 1, 'epoch_early_stopping'] > 0:
    #         x = (i - 1 + i + 1) / 2
    #         y = (df_batchsize_epoch.loc[i - 1, 'epoch_early_stopping'] + df_batchsize_epoch.loc[i + 1, 'epoch_early_stopping']) / 2
    #         plt.scatter(x, y, marker='o', edgecolors='#D96161', facecolors='none', s=50, linewidth=2)

    # 找到 validation_loss 的最小值及其对应的索引
    min_val_loss = df_batchsize_epoch['validation_loss'].min()
    min_val_loss_idx = df_batchsize_epoch['validation_loss'].idxmin()
    min_val_loss_label = df_batchsize_epoch.loc[min_val_loss_idx, 'batch_size_epoch']
    # 找到 min_value 的最小值及其对应的索引
    min_min_value = df_batchsize_epoch['min_value'].min()
    min_min_value_idx = df_batchsize_epoch['min_value'].idxmin()
    min_min_value_label = df_batchsize_epoch.loc[min_min_value_idx, 'batch_size_epoch']
    # 在图中标记出最小值
    # plt.scatter(min_val_loss_label, min_val_loss, color='#975AA4', s=100, marker='*')
    # plt.scatter(min_min_value_label, min_min_value, color='#F7CD3B', s=100, marker='*')
    # 在图中标记出最小值
    plt.scatter(min_val_loss_idx - 0.5 * bar_width, min_val_loss, color='#990033', s=100, marker='*')
    plt.scatter(min_min_value_idx + 0.5 * bar_width, min_min_value, color='#FF6633', s=100, marker='*')

    # 添加每个柱状图的数值，并使用与星号相同的颜色标记最小值
    for i in index:
        # 动态调整偏移量，防止重叠
        val_loss_offset = 0.03 * (1 + df_batchsize_epoch.loc[i, 'validation_loss'] / df_batchsize_epoch['validation_loss'].max())
        min_value_offset = 0.008 * (1 + df_batchsize_epoch.loc[i, 'min_value'] / df_batchsize_epoch['min_value'].max())

        # 如果 validation_loss 和 min_value 相同，增加额外的偏移量
        if df_batchsize_epoch.loc[i, 'validation_loss'] == df_batchsize_epoch.loc[i, 'min_value']:
            val_loss_offset += 0.2  # 增加额外的固定偏移量

        # 对于最小值，使用与星号相同的颜色
        if i == min_val_loss_idx:
            plt.text(i - 0.6 * bar_width, df_batchsize_epoch.loc[i, 'validation_loss'] + val_loss_offset, f"{df_batchsize_epoch.loc[i, 'validation_loss']:.4f}", ha='center', va='bottom', fontsize=10, fontfamily='Times New Roman', color='#990033', fontweight='bold')
        else:
            plt.text(i - 0.6 * bar_width, df_batchsize_epoch.loc[i, 'validation_loss'] + val_loss_offset, f"{df_batchsize_epoch.loc[i, 'validation_loss']:.4f}", ha='center', va='bottom', fontsize=10, fontfamily='Times New Roman')

        if i == min_min_value_idx:
            plt.text(i + 0.5 * bar_width, df_batchsize_epoch.loc[i, 'min_value'] + min_value_offset, f"{df_batchsize_epoch.loc[i, 'min_value']:.4f}", ha='center', va='bottom', fontsize=10, fontfamily='Times New Roman', color='#FF6633', fontweight='bold')
        else:
            plt.text(i + 0.5 * bar_width, df_batchsize_epoch.loc[i, 'min_value'] + min_value_offset, f"{df_batchsize_epoch.loc[i, 'min_value']:.4f}", ha='center', va='bottom', fontsize=10, fontfamily='Times New Roman')

    # # 添加每个柱状图的数值
    # for i in index:
    #     plt.text(i - 0.5 * bar_width, df_batchsize_epoch.loc[i, 'validation_loss'] + 0.05, f"{df_batchsize_epoch.loc[i, 'validation_loss']:.5f}", ha='center', va='bottom', fontsize=10, fontfamily='Times New Roman')
    #     plt.text(i + 0.5 * bar_width, df_batchsize_epoch.loc[i, 'min_value'] + 0.01, f"{df_batchsize_epoch.loc[i, 'min_value']:.5f}", ha='center', va='bottom', fontsize=10, fontfamily='Times New Roman')
        # plt.text(i + 0.5 * bar_width, df_batchsize_epoch.loc[i, 'min_index'] + 0.01, f"{df_batchsize_epoch.loc[i, 'min_index']:.5f}", ha='center', va='bottom', fontsize=10, fontfamily='Times New Roman')
        # plt.text(i + 1.5 * bar_width, df_batchsize_epoch.loc[i, 'epoch_early_stopping'] + 0.01, f"{df_batchsize_epoch.loc[i, 'epoch_early_stopping']:.5f}", ha='center', va='bottom', fontsize=10, fontfamily='Times New Roman')
    # 添加每个折线图的数值
    # for i in index:
    #     plt.text(i, df_batchsize_epoch.loc[i, 'min_index'] + 0.5, f"{df_batchsize_epoch.loc[i, 'min_index']:.0f}", ha='center', va='bottom', fontsize=10, fontfamily='Times New Roman')
    #     plt.text(i, df_batchsize_epoch.loc[i, 'epoch_early_stopping'] + 0.5, f"{df_batchsize_epoch.loc[i, 'epoch_early_stopping']:.0f}", ha='center', va='bottom', fontsize=10, fontfamily='Times New Roman')
    # 添加每个折线图的数值
    # for i in index:
    #     if df_batchsize_epoch.loc[i, 'min_index'] > 0:
    #         plt.text(i, df_batchsize_epoch.loc[i, 'min_index'] - 9.0, f"{df_batchsize_epoch.loc[i, 'min_index']:.0f}", ha='center', va='bottom', fontsize=10, fontfamily='Times New Roman')
    #     if df_batchsize_epoch.loc[i, 'epoch_early_stopping'] > 0:
    #         plt.text(i, df_batchsize_epoch.loc[i, 'epoch_early_stopping'] + 4.5, f"{df_batchsize_epoch.loc[i, 'epoch_early_stopping']:.0f}", ha='center', va='bottom', fontsize=10, fontfamily='Times New Roman')

    # 动态计算偏移量，同时设置最大偏移量
    def dynamic_offset_min_index(value, max_offset=7, multiplier=2.0):
        return min(max_offset, abs(value) * multiplier)

    def dynamic_offset(value, max_offset=10, multiplier=0.1):
        return min(max_offset, abs(value) * multiplier)

    for i in index:
        if df_batchsize_epoch.loc[i, 'min_index'] > 0:
            offset = -dynamic_offset_min_index(df_batchsize_epoch.loc[i, 'min_index'])
            plt.text(i, df_batchsize_epoch.loc[i, 'min_index'] + offset,
                     f"{df_batchsize_epoch.loc[i, 'min_index']:.0f}",
                     ha='center', va='top' if offset > 0 else 'bottom',
                     fontsize=10, fontfamily='Times New Roman')

        if df_batchsize_epoch.loc[i, 'epoch_early_stopping'] > 0:
            offset = dynamic_offset(df_batchsize_epoch.loc[i, 'epoch_early_stopping'])
            plt.text(i, df_batchsize_epoch.loc[i, 'epoch_early_stopping'] + offset,
                     f"{df_batchsize_epoch.loc[i, 'epoch_early_stopping']:.0f}",
                     ha='center', va='top' if offset < 0 else 'bottom',
                     fontsize=10, fontfamily='Times New Roman')

    # 使用对数尺度（logarithmic scale）可以更好地展示不同量级的数据
    plt.yscale('log')   # 设置对数尺度

    # 添加标题和标签
    # plt.title('Comparison of different batch_size and epochs', pad=20, loc='center')
    # plt.figtext(0.52, 0.02, 'Comparison of different batch_size and epochs', ha='center', va='center', fontsize=12, fontfamily='Times New Roman')
    # 调整布局，留出底部空间
    # plt.subplots_adjust(bottom=0.8)  # 调整底部空间
    plt.subplots_adjust(bottom=0.2, top=0.95, left=0.1, right=0.95, wspace=0.4, hspace=0.4)

    plt.xlabel('Batchsize and Epochs', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
    plt.ylabel('Value(Loss/Epoch)', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变纵坐标轴标题字体
    plt.ylabel('Loss', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变纵坐标轴标题字体
    my_font1 = {'family': 'Times New Roman', 'size': 12}
    # plt.legend(loc="best", fontsize=10, prop=my_font1)
    # 调整图例位置
    plt.legend(loc="upper center", bbox_to_anchor=(0.8, 0.65), fontsize=12, prop=my_font1)
    # x_ticks = np.arange(0, 350, 50)
    # y_ticks = np.arange(0, 1.1, 0.2)  # 生成 0 到 2 之间的数，步长为 0.2
    # plt.xticks(rotation=45, fontproperties='Times New Roman', size=2)
    plt.xticks(index, df_batchsize_epoch['batch_size_epoch'], rotation=45, fontproperties='Times New Roman', size=12)
    plt.yticks(rotation=0, fontproperties='Times New Roman', size=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.tight_layout()
    # 保存和显示图形
    plt.savefig(r'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_png\01Comparison of different batch_size and epochs.png', dpi=800, format='png')
    # plt.show()

    return



def plot_gridsearch_batchsize_epoch_10fold(df_batchsize_epoch_mean_10fold):
    # 设置全局字体为 Times New Roman, 字号为 12
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    # 创建横坐标的组合
    # print(df_batchsize_epoch_mean_10fold[['batch_size', 'num_epochs']].isnull().sum())
    # 用0填充缺失值
    df_batchsize_epoch_mean_10fold['batch_size'].fillna(0, inplace=True)
    df_batchsize_epoch_mean_10fold['num_epochs'].fillna(0, inplace=True)

    # 重新生成 batch_size_epoch 列
    df_batchsize_epoch_mean_10fold['batch_size_epoch'] = df_batchsize_epoch_mean_10fold['batch_size'].astype(str) + ', ' + df_batchsize_epoch_mean_10fold['num_epochs'].astype(str)

    # (1) 绘制曲线图 折线图
    plt.figure(figsize=(7, 6))

    # 绘制柱状图
    # 定义柱子的宽度
    bar_width = 0.5
    # 定义柱子的位置
    index = range(len(df_batchsize_epoch_mean_10fold))
    # 绘制 mean_val_loss 柱状图
    plt.bar([i - 0.5 * bar_width for i in index], df_batchsize_epoch_mean_10fold['mean_val_loss'], bar_width, label='mean validation loss', color='#975AA4', alpha=0.8)

    # 找到 mean_val_loss 的最小值及其对应的索引
    min_val_loss = df_batchsize_epoch_mean_10fold['mean_val_loss'].min()
    min_val_loss_idx = df_batchsize_epoch_mean_10fold['mean_val_loss'].idxmin()

    # 在图中标记出最小值
    plt.scatter(min_val_loss_idx - 0.5 * bar_width, min_val_loss, color='#990033', s=100, marker='*')

    # 添加每个柱状图的数值，并使用与星号相同的颜色标记最小值
    for i, val in enumerate(df_batchsize_epoch_mean_10fold['mean_val_loss']):
        color = '#990033' if i == min_val_loss_idx else 'black'
        plt.text(i - 0.5 * bar_width, val + 0.01, f'{val:.4f}', ha='center', va='bottom', color=color, fontsize=12, fontfamily='Times New Roman')


    # for i in index:
    #     # 动态调整偏移量，防止重叠
    #     val_loss_offset = 0.03 * (1 + df_batchsize_epoch_mean_10fold.loc[i, 'mean_val_loss'] / df_batchsize_epoch_mean_10fold['mean_val_loss'].max())
    #
    #     # 对于最小值，使用与星号相同的颜色
    #     if i == min_val_loss_idx:
    #         plt.text(i - 0.6 * bar_width, df_batchsize_epoch_mean_10fold.loc[i, 'mean_val_loss'] + val_loss_offset, f"{df_batchsize_epoch_mean_10fold.loc[i, 'mean_val_loss']:.4f}", ha='center', va='bottom', fontsize=10, fontfamily='Times New Roman', color='#990033', fontweight='bold')
    #     else:
    #         plt.text(i - 0.6 * bar_width, df_batchsize_epoch_mean_10fold.loc[i, 'mean_val_loss'] + val_loss_offset, f"{df_batchsize_epoch_mean_10fold.loc[i, 'mean_val_loss']:.4f}", ha='center', va='bottom', fontsize=10, fontfamily='Times New Roman')

    # 使用对数尺度（logarithmic scale）可以更好地展示不同量级的数据
    # plt.yscale('log')   # 设置对数尺度

    # 添加标题和标签
    # plt.title('Comparison of different batch_size and epochs', pad=20, loc='center')
    # plt.figtext(0.52, 0.02, 'Comparison of different batch_size and epochs of mean 10fold', ha='center', va='center', fontsize=12, fontfamily='Times New Roman')
    # 调整布局，留出底部空间
    # plt.subplots_adjust(bottom=0.8)  # 调整底部空间
    plt.subplots_adjust(bottom=0.2, top=0.95, left=0.1, right=0.95, wspace=0.4, hspace=0.4)

    plt.xlabel('Batchsize and Epochs', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
    plt.ylabel('Value(Loss/Epoch)', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变纵坐标轴标题字体
    plt.ylabel('Loss', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变纵坐标轴标题字体
    my_font1 = {'family': 'Times New Roman', 'size': 12}
    plt.legend(loc="best", fontsize=10, prop=my_font1)
    # 调整图例位置
    # plt.legend(loc="upper center", bbox_to_anchor=(0.8, 0.65), fontsize=12, prop=my_font1)
    # x_ticks = np.arange(0, 350, 50)
    # y_ticks = np.arange(0, 1.1, 0.2)  # 生成 0 到 2 之间的数，步长为 0.2
    # plt.xticks(rotation=45, fontproperties='Times New Roman', size=2)
    plt.xticks(index, df_batchsize_epoch_mean_10fold['batch_size_epoch'], rotation=45, fontproperties='Times New Roman', size=12)
    plt.yticks(rotation=0, fontproperties='Times New Roman', size=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.tight_layout()
    # 保存和显示图形
    plt.savefig(r'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_png\01Comparison of different batch_size and epochs of mean 10fold.png', dpi=800, format='png')
    # plt.show()


    # (2) draw 3D figure 散点图：x坐标为batch_size，y坐标为num_epochs，z坐标为mean_val_loss
    # 创建3D图表
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 设置全局字体
    plt.rcParams['font.family'] = 'Times New Roman'

    # 获取数据
    x = df_batchsize_epoch_mean_10fold['batch_size']
    y = df_batchsize_epoch_mean_10fold['num_epochs']
    z = df_batchsize_epoch_mean_10fold['mean_val_loss']

    # 绘制3D散点图
    sc = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o', s=50)  # 增大s散点大小

    # 找到 mean_val_loss 的最小值及其对应的索引
    min_val_loss = df_batchsize_epoch_mean_10fold['mean_val_loss'].min()
    min_val_loss_idx = df_batchsize_epoch_mean_10fold['mean_val_loss'].idxmin()

    # 添加每个点的数值标签
    for i in range(len(df_batchsize_epoch_mean_10fold)):
        color = '#990033' if i == min_val_loss_idx else 'black'
        ax.text(x[i], y[i], z[i], f'{z[i]:.4f}', color=color, fontsize=12, fontfamily='Times New Roman')

    # 增加一个偏移量，让文字往上挪动
    z_offset = 0.05 * (z.max() - z.min())  # 计算一个基于 z 轴范围的合理偏移量

    # 在图中标记出最小值
    ax.scatter(x[min_val_loss_idx], y[min_val_loss_idx], min_val_loss, color='#990033', s=150, marker='*', alpha=1.0)

    # 标记最小值点的 batch_size 和 num_epochs
    best_batch_size = x[min_val_loss_idx]
    best_num_epochs = y[min_val_loss_idx]
    ax.text(best_batch_size, best_num_epochs, min_val_loss + z_offset,  # 这里增加了 z_offset
            f'Best: Batch Size={best_batch_size}, Epochs={best_num_epochs}',
            color='#990033', fontsize=12, fontfamily='Times New Roman',
            verticalalignment='bottom', horizontalalignment='left')

    # 添加标题和标签
    # plt.title('3D Comparison of Different Batch Size and Epochs on Mean Validation Loss', fontsize=14, fontfamily='Times New Roman')
    ax.set_xlabel('Batch Size', labelpad=10, fontsize=12, fontfamily='Times New Roman')
    ax.set_ylabel('Number of Epochs', labelpad=10, fontsize=12, fontfamily='Times New Roman')
    ax.set_zlabel('Mean Validation Loss', labelpad=10, fontsize=12, fontfamily='Times New Roman')

    # 设置刻度标签字体
    ax.tick_params(axis='x', which='both', labelsize=12)
    ax.tick_params(axis='y', which='both', labelsize=12)
    ax.tick_params(axis='z', which='both', labelsize=12)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.zaxis.set_tick_params(labelsize=12)

    # 设置刻度标签字体为 Times New Roman
    for t in ax.xaxis.get_major_ticks():
        t.label.set_fontname('Times New Roman')
    for t in ax.yaxis.get_major_ticks():
        t.label.set_fontname('Times New Roman')
    for t in ax.zaxis.get_major_ticks():
        t.label.set_fontname('Times New Roman')

    # 添加颜色条
    cbar = fig.colorbar(sc, shrink=0.3, aspect=10)  # 减小 shrink 值使颜色条更窄
    cbar.ax.set_ylabel('Mean Validation Loss', fontsize=12, fontfamily='Times New Roman')

    # 调整视角
    ax.view_init(elev=30, azim=-45)  # 调整视角角度

    # 增加网格线
    ax.grid(True, linestyle='--', alpha=0.5)

    # 增加背景颜色
    ax.xaxis.pane.fill = False  # 不填充x轴平面
    ax.yaxis.pane.fill = False  # 不填充y轴平面
    ax.zaxis.pane.fill = False  # 不填充z轴平面

    # 设置背景颜色
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')

    # 调整轴的范围
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    ax.set_zlim([z.min(), z.max()])

    # 调整布局
    plt.tight_layout()
    # 保存和显示图形
    plt.savefig(r'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_png\01Comparison of different batch_size and epochs of mean 10fold_3D_scatter.png', dpi=800, format='png')
    # plt.show()


    # (3) draw 3D figure 柱状图：x坐标为batch_size，y坐标为num_epochs，z坐标为mean_val_loss
    # 设置全局字体
    plt.rcParams['font.family'] = 'Times New Roman'

    # 创建3D图表
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 获取数据
    df = df_batchsize_epoch_mean_10fold  # 假设你有一个包含数据的DataFrame
    x = df['batch_size'].values
    y = df['num_epochs'].values
    z = df['mean_val_loss'].values

    # 定义柱子的宽度和深度（假设所有柱子的大小相同）
    dx = 0.3 # 柱子在 x 和 y 方向上的宽度
    dy = 0.1
    dz = z  # 柱子的高度

    # 绘制3D柱形图
    for i in range(len(x)):
        color = plt.cm.viridis((z[i] - min(z)) / (max(z) - min(z))) if max(z) != min(z) else 'blue'  # 根据 z 值设定颜色
        ax.bar3d(x[i] - dx / 2, y[i] - dy / 2, 0, dx, dy, dz[i], color=color, alpha=0.6)

    # 找到 mean_val_loss 的最小值及其对应的索引
    min_val_loss = df['mean_val_loss'].min()
    min_val_loss_idx = df['mean_val_loss'].idxmin()

    # 添加每个点的数值标签
    for i in range(len(df)):
        color = '#990033' if i == min_val_loss_idx else 'black'
        ax.text(x[i], y[i], z[i], f'{z[i]:.4f}', color=color, fontsize=12, fontfamily='Times New Roman', zdir='z')

    # 在图中标记出最小值
    ax.scatter(x[min_val_loss_idx], y[min_val_loss_idx], min_val_loss, color='#990033', s=150, marker='*', alpha=1.0)

    # 标记最小值点的 batch_size 和 num_epochs
    best_batch_size = x[min_val_loss_idx]
    best_num_epochs = y[min_val_loss_idx]
    z_offset = 0.05 * (z.max() - z.min())  # 计算一个基于 z 轴范围的合理偏移量
    ax.text(best_batch_size, best_num_epochs, min_val_loss + z_offset,
            f'Best: Batch Size={best_batch_size}, Epochs={best_num_epochs}',
            color='#990033', fontsize=12, fontfamily='Times New Roman',
            verticalalignment='bottom', horizontalalignment='left')

    # 添加标题和标签
    ax.set_xlabel('Batch Size', labelpad=10, fontsize=12, fontfamily='Times New Roman')
    ax.set_ylabel('Number of Epochs', labelpad=10, fontsize=12, fontfamily='Times New Roman')
    ax.set_zlabel('Mean Validation Loss', labelpad=10, fontsize=12, fontfamily='Times New Roman')

    # 设置刻度标签字体
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.zaxis.set_tick_params(labelsize=12)

    # 添加颜色条
    mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    mappable.set_array(z)
    cbar = fig.colorbar(mappable, shrink=0.3, aspect=10)
    cbar.ax.set_ylabel('Mean Validation Loss', fontsize=12, fontfamily='Times New Roman')

    # 调整视角
    ax.view_init(elev=30, azim=-45)

    # 增加网格线
    ax.grid(True, linestyle='--', alpha=0.5)

    # 不填充轴平面
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # 设置背景颜色
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')

    # 调整轴的范围
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    ax.set_zlim([z.min(), z.max()])

    # 调整布局
    plt.tight_layout()

    # 保存和显示图形
    plt.savefig(fr'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_png\01Comparison of different batch_size and epochs of mean 10fold_3D.png', dpi=800, format='png')
    # plt.show()



    # (4) draw 3D figure 柱状图，紧挨着的柱子：x坐标为batch_size，y坐标为num_epochs，z坐标为mean_val_loss
    # 设置全局字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    # 创建3D图表
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 获取数据
    df = df_batchsize_epoch_mean_10fold  # 假设你有一个包含数据的DataFrame
    x = df['batch_size'].values
    y = df['num_epochs'].values
    z = df['mean_val_loss'].values

    # 定义柱子的宽度和深度（根据实际数值设置）
    unique_x = np.unique(x)
    unique_y = np.unique(y)

    dx = np.diff(unique_x)[0] / 2 if len(unique_x) > 1 else 1  # 柱子在 x 方向上的宽度
    dy = np.diff(unique_y)[0] / 2 if len(unique_y) > 1 else 1  # 柱子在 y 方向上的宽度
    dz = z  # 柱子的高度

    # 确保柱子之间没有间隙
    x_positions = unique_x - dx / 2
    y_positions = unique_y - dy / 2

    # 绘制3D柱形图
    for i in range(len(df)):
        xi = x[i]
        yi = y[i]
        zi = z[i]

        # 找到对应的 x 和 y 的位置索引
        x_idx = np.where(unique_x == xi)[0][0]
        y_idx = np.where(unique_y == yi)[0][0]

        color = plt.cm.viridis((zi - min(z)) / (max(z) - min(z))) if max(z) != min(z) else 'blue'  # 根据 z 值设定颜色
        ax.bar3d(x_positions[x_idx], y_positions[y_idx], 0, dx, dy, zi, color=color, alpha=0.6)

    # 找到 mean_val_loss 的最小值及其对应的索引
    min_val_loss = df['mean_val_loss'].min()
    min_val_loss_idx = df['mean_val_loss'].idxmin()

    # 添加每个点的数值标签
    for i in range(len(df)):
        color = '#990033' if i == min_val_loss_idx else 'black'
        ax.text(x[i], y[i], z[i], f'{z[i]:.4f}', color=color, fontsize=12, fontfamily='Times New Roman', zdir='z')

    # 在图中标记出最小值
    ax.scatter(x[min_val_loss_idx], y[min_val_loss_idx], min_val_loss, color='#990033', s=150, marker='*', alpha=1.0)

    # 标记最小值点的 batch_size 和 num_epochs
    best_batch_size = x[min_val_loss_idx]
    best_num_epochs = y[min_val_loss_idx]
    z_offset = 0.05 * (z.max() - z.min())  # 计算一个基于 z 轴范围的合理偏移量
    ax.text(best_batch_size, best_num_epochs, min_val_loss + z_offset,
            f'Best: Batch Size={best_batch_size}, Epochs={best_num_epochs}',
            color='#990033', fontsize=12, fontfamily='Times New Roman',
            verticalalignment='bottom', horizontalalignment='left')

    # 添加标题和标签
    ax.set_xlabel('Batch Size', labelpad=10, fontsize=12, fontfamily='Times New Roman')
    ax.set_ylabel('Number of Epochs', labelpad=10, fontsize=12, fontfamily='Times New Roman')
    ax.set_zlabel('Mean Validation Loss', labelpad=10, fontsize=12, fontfamily='Times New Roman')

    # 设置刻度标签为实际数值
    ax.set_xticks(unique_x)
    ax.set_yticks(unique_y)

    # 设置刻度标签字体
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.zaxis.set_tick_params(labelsize=12)

    # 添加颜色条
    mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    mappable.set_array(z)
    cbar = fig.colorbar(mappable, shrink=0.3, aspect=10)
    cbar.ax.set_ylabel('Mean Validation Loss', fontsize=12, fontfamily='Times New Roman')

    # 调整视角
    ax.view_init(elev=30, azim=-45)

    # 增加网格线
    ax.grid(True, linestyle='--', alpha=0.5)

    # 不填充轴平面
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # 设置背景颜色
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')

    # 调整轴的范围
    ax.set_xlim([unique_x.min(), unique_x.max()])
    ax.set_ylim([unique_y.min(), unique_y.max()])
    ax.set_zlim([z.min(), z.max()])

    # 调整布局
    plt.tight_layout()

    # 保存和显示图形
    plt.savefig(
        fr'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_png\01Comparison of different batch_size and epochs of mean 10fold_3D_close.png',
        dpi=800, format='png')
    # plt.show()


    # (5) draw 3D figure 拟合的曲面图，xyz为散点，并将其拟合为一个连续的曲面：x坐标为batch_size，y坐标为num_epochs，z坐标为mean_val_loss
    # 为了将3D柱状图转换为一个拟合的平滑曲面图，可以使用 scipy.interpolate.griddata 函数来进行插值，并用 matplotlib 的 plot_surface 方法来绘制平滑的曲面
    # 设置全局字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    # 获取数据
    x = df_batchsize_epoch_mean_10fold['batch_size'].values
    y = df_batchsize_epoch_mean_10fold['num_epochs'].values
    z = df_batchsize_epoch_mean_10fold['mean_val_loss'].values

    # 创建网格数据用于插值
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    xi, yi = np.meshgrid(xi, yi)

    # 使用插值方法创建 z 值矩阵
    zi = griddata((x, y), z, (xi, yi), method='cubic')

    # 创建3D图表
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制3D曲面图
    surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none', alpha=0.6)

    # 找到 mean_val_loss 的最小值及其对应的索引
    min_val_loss = df_batchsize_epoch_mean_10fold['mean_val_loss'].min()
    min_val_loss_idx = df_batchsize_epoch_mean_10fold['mean_val_loss'].idxmin()

    # 标记最小值点的 batch_size 和 num_epochs
    best_batch_size = x[min_val_loss_idx]
    best_num_epochs = y[min_val_loss_idx]

    # 在图中标记出最小值
    ax.scatter(best_batch_size, best_num_epochs, min_val_loss, color='#990033', s=150, marker='*', alpha=1.0)

    # 添加每个点的数值标签（可选）
    for i in range(len(df_batchsize_epoch_mean_10fold)):
        color = '#990033' if i == min_val_loss_idx else 'black'
        ax.text(x[i], y[i], z[i], f'{z[i]:.4f}', color=color, fontsize=12, fontfamily='Times New Roman', zdir='z')

    # 标记最小值点的 batch_size 和 num_epochs
    z_offset = 0.05 * (z.max() - z.min())  # 计算一个基于 z 轴范围的合理偏移量
    ax.text(best_batch_size, best_num_epochs, min_val_loss + z_offset,
            f'Best: Batch Size={best_batch_size}, Epochs={best_num_epochs}',
            color='#990033', fontsize=12, fontfamily='Times New Roman',
            verticalalignment='bottom', horizontalalignment='left')

    # 添加标题和标签
    ax.set_xlabel('Batch Size', labelpad=10, fontsize=12, fontfamily='Times New Roman')
    ax.set_ylabel('Number of Epochs', labelpad=10, fontsize=12, fontfamily='Times New Roman')
    ax.set_zlabel('Mean Validation Loss', labelpad=10, fontsize=12, fontfamily='Times New Roman')

    # 设置刻度标签为实际数值
    ax.set_xticks(np.unique(x))
    ax.set_yticks(np.unique(y))

    # 设置刻度标签字体
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.zaxis.set_tick_params(labelsize=12)

    # 添加颜色条
    mappable = plt.cm.ScalarMappable(cmap='viridis')
    mappable.set_array(z)
    cbar = fig.colorbar(surf, shrink=0.3, aspect=10)
    cbar.ax.set_ylabel('Mean Validation Loss', fontsize=12, fontfamily='Times New Roman')

    # 调整视角
    ax.view_init(elev=30, azim=-45)

    # 增加网格线
    ax.grid(True, linestyle='--', alpha=0.5)

    # 不填充轴平面
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # 设置背景颜色
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')

    # 调整轴的范围
    ax.set_xlim([min(x), max(x)])
    ax.set_ylim([min(y), max(y)])
    ax.set_zlim([min(z), max(z)])

    # 调整布局
    plt.tight_layout()

    # 保存和显示图形
    plt.savefig(
        fr'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_png\01Comparison of different batch_size and epochs of mean 10fold_3D_surface.png',
        dpi=800, format='png')
    # plt.show()

    return


# 用于绘制plot_with_labels函数中的柱形图
# def plot_with_labels(data, palette, title, output_path):
#     fig, ax = plt.subplots(figsize=(7, 6))
#
#     # 绘制柱状图并应用自定义调色板
#     sns.barplot(x='Loss Type', y='Value', data=data, palette=[palette.get(x, 'gray') for x in data['Loss Type']], ax=ax)
#
#     # 添加数值标签到柱状图
#     for p in ax.patches:
#         ax.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
#                     ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
#                     textcoords='offset points')
#
#     # 设置图表属性
#     plt.xlabel('Loss Type')
#     plt.ylabel(title.split(' ')[0])  # 使用标题的第一个词作为Y轴标签
#     plt.tight_layout()
#
#     # 保存图表
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.show()
#     plt.close()


def plot_test_cure(list_iter_loss_goal_main, list_iter_loss_zone_arrive_main, list_iter_loss_time_depart_main, list_iter_loss_time_arrive_main, list_iter_loss_mode1_main, list_iter_loss_mode2_main, list_iter_loss, figure_iter_name, df_results):
    # 设置全局字体为 Times New Roman, 字号为 12
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    # print('list_iter_loss_goal_main',list_iter_loss_goal_main)
    # 对每个损失列表应用滤波
    # list_iter_loss_goal_main_smooth = moving_average(list_iter_loss_goal_main, window_size=5)
    # ... 对其他列表进行类似处理

    # (1) plot iter_loss
    plt.tight_layout()  # 使得子图之间的间距更加合理
    plt.figure(figsize=(7, 6))
    plt.plot(list_iter_loss_goal_main, '#9dd5ff', label="F1(goal_main)")
    # plt.plot(list_iter_loss_zone_depart_main, '#70ad47', label="F2(zone_depart_main)")
    plt.plot(list_iter_loss_zone_arrive_main, '#c0dda3', label="F2(zone_arrive_main)")
    plt.plot(list_iter_loss_time_depart_main, '#ffbd00', label="F3(time_depart_main)")
    plt.plot(list_iter_loss_time_arrive_main, '#ffe9a1', label="F4(time_arrive_main)")
    plt.plot(list_iter_loss_mode1_main, '#ce0afe', label="F5(mode1_main)")
    plt.plot(list_iter_loss_mode2_main, '#ffc2eb', label="F6(mode2_main)")
    plt.plot(list_iter_loss, '#0070c0', linestyle='--', linewidth='2.5', label="F(total_main)")

    my_font1 = {'family': 'Times New Roman', 'size': 12}
    plt.legend(loc="best", fontsize=10, prop=my_font1)
    # plt.title(figure_iter_name[2:-4], pad=20, loc='center', fontdict={'family': 'Times New Roman', 'size': 12})
    # plt.figtext(0.5, 0.03, figure_iter_name[2:-4], ha='center', va='center', fontsize=12, fontfamily='Times New Roman')
    # 调整布局，留出底部空间
    plt.subplots_adjust(bottom=0.15)  # 调整底部空间

    plt.xlabel('Iteration', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
    plt.ylabel('Loss', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变纵坐标轴标题字体
    # plt.axis([-1, 101, -1, 51])  # 改变xy坐标轴范围
    # 设置坐标轴范围
    # plt.xlim(0, 6)  # 设置 x 轴范围
    # plt.ylim(0, 40)  # 设置 y 轴范围
    x_max = math.ceil((data_sample * 0.2) / batch_size)   # 横坐标轴最大值是多少：样本量*testing data的比例0.2，再/每个批次的样本量
    # x_interval = x_max / 5
    x_interval = math.ceil(x_max / 5)  # 向上取整
    # x_ticks = np.arange(0, x_max, x_interval)
    x_ticks = np.arange(0, 5, 1)
    y_ticks = np.arange(0, 0.5, 0.1)  # 生成 0 到 2 之间的数，步长为 0.2
    plt.xticks(x_ticks, fontproperties='Times New Roman', size=12)
    plt.yticks(y_ticks, fontproperties='Times New Roman', size=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')  # 启用横竖网格线，并设置样式   # 启用网格线
    plt.savefig(figure_iter_name, dpi=800, format='png')
    # plt.show()
    plt.close()  # 清除当前绘图窗口

    # (2) plot sum loss in one epoch 柱状图
    # 设置全局字体为 Times New Roman, 字号为 12
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    # 定义颜色序列
    colors = ['#9dd5ff', '#c0dda3', '#ffbd00', '#ffe9a1', '#ce0afe', '#ffc2eb', '#0070c0']
    labels = ['goal_main', 'zone_arrive_main', 'time_depart_main',
              'time_arrive_main', 'mode1_main', 'mode2_main', 'total_loss']

    # # 自定义颜色映射
    # custom_palette = {
    #     'goal_main': '#9dd5ff',
    #     'zone_depart_main': '#70ad47',
    #     'zone_arrive_main': '#c0dda3',
    #     'time_depart_main': '#ffbd00',
    #     'time_arrive_main': '#ffe9a1',
    #     'mode1_main': '#ce0afe',
    #     'mode2_main': '#ffc2eb',
    #     'total_loss': '#0070c0'
    # }

    # 创建两个DataFrame来分别存储df_results中的总损失和平均损失的数据
    df_total_loss = df_results['Total_Loss'].reset_index().rename(columns={'index': 'Loss Type', 'Total_Loss': 'Value'})
    df_average_loss = df_results['Average_Loss'].reset_index().rename(columns={'index': 'Loss Type', 'Average_Loss': 'Value'})

    # 确保 Loss Type 列的值是字符串并且去除首尾空格
    df_total_loss['Loss Type'] = df_total_loss['Loss Type'].astype(str).str.strip()
    df_average_loss['Loss Type'] = df_average_loss['Loss Type'].astype(str).str.strip()

    print('df_total_loss',df_total_loss)
    print('df_average_loss',df_average_loss)


    # =============== 绘制总损失的柱状图 ===============
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.barplot(x='Loss Type', y='Value', data=df_total_loss, palette=colors[:len(df_total_loss)], ax=ax)

    # 添加数值标签到柱状图
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                    textcoords='offset points')

    # 添加图例
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors[:len(df_total_loss)]]
    # ax.legend(handles, labels[:len(df_total_loss)], title='Loss Types')

    # 设置横坐标标签为具体的文字，并使其斜向上倾斜45°
    ax.set_xticklabels(labels[:len(df_total_loss)], rotation=45, ha='right')

    # plt.title('Total Loss for Each Loss Type')
    plt.xlabel('Loss Type')
    plt.ylabel('Total Loss')
    plt.tight_layout()

    # 保存总损失图表
    plt.savefig(r'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_png\01Total_Loss.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

    # =============== 绘制平均损失的柱状图 ===============
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.barplot(x='Loss Type', y='Value', data=df_average_loss, palette=colors[:len(df_average_loss)], ax=ax)

    # 添加数值标签到柱状图
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                    textcoords='offset points')

    # 添加图例
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors[:len(df_average_loss)]]
    # ax.legend(handles, labels[:len(df_average_loss)], title='Loss Types')

    # 设置横坐标标签为具体的文字，并使其斜向上倾斜45°
    ax.set_xticklabels(labels[:len(df_average_loss)], rotation=45, ha='right')

    # plt.title('Average Loss for Each Loss Type')
    plt.xlabel('Loss Type')
    plt.ylabel('Average Loss')
    plt.tight_layout()

    # 保存平均损失图表
    plt.savefig(r'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_png\01Average_Loss.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

    # # 绘制总损失的柱状图
    # plt.figure(figsize=(7, 6))
    # sns.barplot(x='Loss Type', y='Value', data=df_total_loss, palette='Blues_d')
    # # plt.title('Total Loss for Each Loss Type')  # 在图中不显示title，在论文中图下方标记
    # plt.xlabel('Loss Type')
    # plt.ylabel('Total Loss')
    # # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig(r'E:\Desktop\07[Win]CQ_part_GoalZoneTimeMode20250101\04output_train_test_on_bestHyperPara\04Total_loss.png')  # 保存总损失图表
    # # plt.show()
    #
    # # 绘制平均损失的柱状图
    # plt.figure(figsize=(7, 6))
    # sns.barplot(x='Loss Type', y='Value', data=df_average_loss, palette='Reds_d')
    # # plt.title('Average Loss for Each Loss Type')
    # plt.xlabel('Loss Type')
    # plt.ylabel('Average Loss')
    # # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig(r'E:\Desktop\07[Win]CQ_part_GoalZoneTimeMode20250101\04output_train_test_on_bestHyperPara\04Average_Loss.png')  # 保存平均损失图表
    # # plt.show()

    return


def plot_train_test_cure_two(train_list_iter_loss_goal_main, train_list_iter_loss_zone_arrive_main, train_list_iter_loss_time_depart_main, train_list_iter_loss_time_arrive_main, train_list_iter_loss_mode1_main, train_list_iter_loss_mode2_main, train_list_iter_loss,
                         test_list_iter_loss_goal_main, test_list_iter_loss_zone_arrive_main, test_list_iter_loss_time_depart_main, test_list_iter_loss_time_arrive_main, test_list_iter_loss_mode1_main, test_list_iter_loss_mode2_main, test_list_iter_loss, figure_iter_name_two):
    # 设置全局字体为 Times New Roman, 字号为 12
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    plt.tight_layout()  # 使得子图之间的间距更加合理
    plt.figure(figsize=(8, 6))
    plt.plot(train_list_iter_loss_goal_main, '#9dd5ff', label="F1_train(goal_main)")
    plt.plot(train_list_iter_loss_zone_arrive_main, '#c0dda3', label="F2_train(zone_arrive_main)")
    plt.plot(train_list_iter_loss_time_depart_main, '#ffbd00', label="F3_train(time_depart_main)")
    plt.plot(train_list_iter_loss_time_arrive_main, '#ffe9a1', label="F4_train(time_arrive_main)")
    plt.plot(train_list_iter_loss_mode1_main, '#ce0afe', label="F5_train(mode1_main)")
    plt.plot(train_list_iter_loss_mode2_main, '#ffc2eb', label="F6_train(mode2_main)")
    plt.plot(train_list_iter_loss, '#0070c0', linewidth='2.5', label="F_train(total_main)")

    plt.plot(test_list_iter_loss_goal_main, '#9dd5ff', linestyle='--', label="F1_test(goal_main)")
    plt.plot(test_list_iter_loss_zone_arrive_main, '#c0dda3', linestyle='--', label="F2_test(zone_arrive_main)")
    plt.plot(test_list_iter_loss_time_depart_main, '#ffbd00', linestyle='--', label="F3_test(time_depart_main)")
    plt.plot(test_list_iter_loss_time_arrive_main, '#ffe9a1', linestyle='--', label="F4_test(time_arrive_main)")
    plt.plot(test_list_iter_loss_mode1_main, '#ce0afe', linestyle='--', label="F5_test(mode1_main)")
    plt.plot(test_list_iter_loss_mode2_main, '#ffc2eb', linestyle='--', label="F6_test(mode2_main)")
    plt.plot(test_list_iter_loss, '#0070c0', linestyle='--', linewidth='2.5', label="F_test(total_main)")

    my_font1 = {'family': 'Times New Roman', 'size': 12}
    plt.legend(loc="best", bbox_to_anchor=(1, 1), fontsize=10, prop=my_font1)
    # plt.title(figure_iter_name[2:-4], pad=20, loc='center', fontdict={'family': 'Times New Roman', 'size': 12})
    # plt.figtext(0.5, 0.03, figure_iter_name[2:-4], ha='center', va='center', fontsize=12, fontfamily='Times New Roman')
    # 调整布局，留出底部空间
    plt.subplots_adjust(bottom=0.15, right=0.65)  # 调整底部空间

    plt.xlabel('Iteration', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
    plt.ylabel('Loss', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变纵坐标轴标题字体

    # 设置横坐标为对数尺度
    # plt.xscale('log')

    # plt.axis([-1, 101, -1, 51])  # 改变xy坐标轴范围
    # 设置坐标轴范围
    # plt.xlim(0, 6)  # 设置 x 轴范围
    # plt.ylim(0, 40)  # 设置 y 轴范围
    x_max = math.ceil((data_sample * 0.2) / batch_size)  # 横坐标轴最大值是多少：样本量*testing data的比例0.2，再/每个批次的样本量
    # x_interval = x_max / 5
    x_interval = math.ceil(x_max / 5)  # 向上取整
    # x_ticks = np.arange(0, x_max, x_interval)
    x_ticks = np.arange(0, 200, 50)
    # x_ticks = np.logspace(0, np.log10(200), num=6)  # 生成对数刻度的ticks
    y_ticks = np.arange(0, 0.2, 0.05)  # 生成 0 到 2 之间的数，步长为 0.2
    plt.xticks(x_ticks, fontproperties='Times New Roman', size=12)
    plt.yticks(y_ticks, fontproperties='Times New Roman', size=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')  # 启用横竖网格线，并设置样式   # 启用网格线
    plt.savefig(figure_iter_name_two, dpi=800, format='png')
    # plt.show()

    return



# def plot_train_test_cure_one(train_list_iter_loss_goal_main, train_list_iter_loss_zone_arrive_main, train_list_iter_loss_time_depart_main, train_list_iter_loss_time_arrive_main, train_list_iter_loss_mode1_main, train_list_iter_loss_mode2_main, train_list_iter_loss,
#                          test_list_iter_loss_goal_main, test_list_iter_loss_zone_arrive_main, test_list_iter_loss_time_depart_main, test_list_iter_loss_time_arrive_main, test_list_iter_loss_mode1_main, test_list_iter_loss_mode2_main, test_list_iter_loss, figure_iter_name_one):
#     # 设置全局字体为 Times New Roman, 字号为 12
#     plt.rcParams['font.family'] = 'Times New Roman'
#     plt.rcParams['font.size'] = 12
#
#     plt.tight_layout()  # 使得子图之间的间距更加合理
#     plt.figure(figsize=(8, 6))
#     plt.plot(train_list_iter_loss_goal_main, '#9dd5ff', label="F1_train(goal_main)")
#     plt.plot(train_list_iter_loss_zone_arrive_main, '#c0dda3', label="F2_train(zone_arrive_main)")
#     plt.plot(train_list_iter_loss_time_depart_main, '#ffbd00', label="F3_train(time_depart_main)")
#     plt.plot(train_list_iter_loss_time_arrive_main, '#ffe9a1', label="F4_train(time_arrive_main)")
#     plt.plot(train_list_iter_loss_mode1_main, '#ce0afe', label="F5_train(mode1_main)")
#     plt.plot(train_list_iter_loss_mode2_main, '#ffc2eb', label="F6_train(mode2_main)")
#     plt.plot(train_list_iter_loss, '#0070c0', linewidth='2.5', label="F_train(total_main)")
#
#     plt.plot(test_list_iter_loss_goal_main, '#9dd5ff', linestyle='--', label="F1_test(goal_main)")
#     plt.plot(test_list_iter_loss_zone_arrive_main, '#c0dda3', linestyle='--', label="F2_test(zone_arrive_main)")
#     plt.plot(test_list_iter_loss_time_depart_main, '#ffbd00', linestyle='--', label="F3_test(time_depart_main)")
#     plt.plot(test_list_iter_loss_time_arrive_main, '#ffe9a1', linestyle='--', label="F4_test(time_arrive_main)")
#     plt.plot(test_list_iter_loss_mode1_main, '#ce0afe', linestyle='--', label="F5_test(mode1_main)")
#     plt.plot(test_list_iter_loss_mode2_main, '#ffc2eb', linestyle='--', label="F6_test(mode2_main)")
#     plt.plot(test_list_iter_loss, '#0070c0', linestyle='--', linewidth='2.5', label="F_test(total_main)")
#
#     my_font1 = {'family': 'Times New Roman', 'size': 12}
#     plt.legend(loc="best", bbox_to_anchor=(1, 1), fontsize=10, prop=my_font1)
#     # plt.title(figure_iter_name[2:-4], pad=20, loc='center', fontdict={'family': 'Times New Roman', 'size': 12})
#     # plt.figtext(0.5, 0.03, figure_iter_name[2:-4], ha='center', va='center', fontsize=12, fontfamily='Times New Roman')
#     # 调整布局，留出底部空间
#     plt.subplots_adjust(bottom=0.15, right=0.65)  # 调整底部空间
#
#     plt.xlabel('Iteration', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变横坐标轴标题字体  labelpad=10表示坐标标题距离坐标轴的垂直距离
#     plt.ylabel('Loss', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})  # 改变纵坐标轴标题字体
#
#     # 设置横坐标为对数尺度
#     # plt.xscale('log')
#
#     # plt.axis([-1, 101, -1, 51])  # 改变xy坐标轴范围
#     # 设置坐标轴范围
#     # plt.xlim(0, 6)  # 设置 x 轴范围
#     # plt.ylim(0, 40)  # 设置 y 轴范围
#     x_max = math.ceil((data_sample * 0.2) / batch_size)  # 横坐标轴最大值是多少：样本量*testing data的比例0.2，再/每个批次的样本量
#     # x_interval = x_max / 5
#     x_interval = math.ceil(x_max / 5)  # 向上取整
#     # x_ticks = np.arange(0, x_max, x_interval)
#     x_ticks = np.arange(0, 200, 50)
#     # x_ticks = np.logspace(0, np.log10(200), num=6)  # 生成对数刻度的ticks
#     y_ticks = np.arange(0, 0.2, 0.05)  # 生成 0 到 2 之间的数，步长为 0.2
#     plt.xticks(x_ticks, fontproperties='Times New Roman', size=12)
#     plt.yticks(y_ticks, fontproperties='Times New Roman', size=12)
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')  # 启用横竖网格线，并设置样式   # 启用网格线
#     plt.savefig(figure_iter_name_one, dpi=800, format='png')
#     # plt.show()
#
#     return


def plot_train_test_cure_one(train_list_iter_loss_goal_main, train_list_iter_loss_zone_arrive_main,
                            train_list_iter_loss_time_depart_main, train_list_iter_loss_time_arrive_main,
                            train_list_iter_loss_mode1_main, train_list_iter_loss_mode2_main, train_list_iter_loss,
                            test_list_iter_loss_goal_main, test_list_iter_loss_zone_arrive_main,
                            test_list_iter_loss_time_depart_main, test_list_iter_loss_time_arrive_main,
                            test_list_iter_loss_mode1_main, test_list_iter_loss_mode2_main, test_list_iter_loss,
                            figure_iter_name_one):
    plt.figure()
    # 设置全局字体为 Times New Roman, 字号为 12
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    # 创建组图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1行2列的子图
    train_ax, test_ax = axes  # 分别对应训练集和测试集的子图

    # 绘制训练集的曲线 (左侧子图)
    train_ax.plot(train_list_iter_loss_goal_main, '#9dd5ff', label="F1(goal_main)")
    train_ax.plot(train_list_iter_loss_zone_arrive_main, '#c0dda3', label="F2(zone_arrive_main)")
    train_ax.plot(train_list_iter_loss_time_depart_main, '#ffbd00', label="F3(time_depart_main)")
    train_ax.plot(train_list_iter_loss_time_arrive_main, '#ffe9a1', label="F4(time_arrive_main)")
    train_ax.plot(train_list_iter_loss_mode1_main, '#ce0afe', label="F5(mode1_main)")
    train_ax.plot(train_list_iter_loss_mode2_main, '#ffc2eb', label="F6(mode2_main)")
    train_ax.plot(train_list_iter_loss, '#0070c0', linewidth='2.5', label="F(total_main)")

    # 绘制测试集的曲线 (右侧子图)
    test_ax.plot(test_list_iter_loss_goal_main, '#9dd5ff', label="F1_test(goal_main)")
    test_ax.plot(test_list_iter_loss_zone_arrive_main, '#c0dda3', label="F2_test(zone_arrive_main)")
    test_ax.plot(test_list_iter_loss_time_depart_main, '#ffbd00', label="F3_test(time_depart_main)")
    test_ax.plot(test_list_iter_loss_time_arrive_main, '#ffe9a1', label="F4_test(time_arrive_main)")
    test_ax.plot(test_list_iter_loss_mode1_main, '#ce0afe', label="F5_test(mode1_main)")
    test_ax.plot(test_list_iter_loss_mode2_main, '#ffc2eb', label="F6_test(mode2_main)")
    test_ax.plot(test_list_iter_loss, '#0070c0',  linewidth='2.5', label="F_test(total_main)")

    # 设置子图标题
    # train_ax.set_title("Training Loss", pad=10, fontdict={'family': 'Times New Roman', 'size': 14})
    # test_ax.set_title("Testing Loss", pad=10, fontdict={'family': 'Times New Roman', 'size': 14})

    # 设置纵坐标一致
    y_min = 0
    y_max = 0.2
    y_ticks = np.arange(y_min, y_max, 0.05)

    train_ax.set_ylim(y_min, y_max)
    test_ax.set_ylim(y_min, y_max)
    train_ax.set_yticks(y_ticks)
    test_ax.set_yticks(y_ticks)

    # 设置横坐标范围
    # x_max_train = math.ceil((data_sample * 0.8) / batch_size)  # 训练集的横坐标最大值
    # x_max_test = math.ceil((data_sample * 0.2) / batch_size)   # 测试集的横坐标最大值
    #
    # x_ticks_train = np.arange(0, x_max_train, math.ceil(x_max_train / 5))
    # x_ticks_test = np.arange(0, x_max_test, math.ceil(x_max_test / 5))

    # 直接设置固定的刻度范围
    x_ticks_train = np.arange(0, 150 + 1, 10)  # 刻度为 [0, 10, 20, 30, 40, 50]
    x_ticks_test = np.arange(0, 5 + 1, 10)  # 刻度为 [0, 10, 20, 30, 40, 50]

    # train_ax.set_xlim(0, x_max_train)
    # test_ax.set_xlim(0, x_max_test)
    train_ax.set_xticks(x_ticks_train)
    test_ax.set_xticks(x_ticks_test)

    # 设置坐标轴标签
    train_ax.set_xlabel('Iteration', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})
    train_ax.set_ylabel('Training Loss', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})

    test_ax.set_xlabel('Iteration', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})
    test_ax.set_ylabel('Test Loss', labelpad=7.5, fontdict={'family': 'Times New Roman', 'size': 12, 'color': 'k'})

    # 添加网格线
    train_ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    test_ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    # # 添加图例
    # my_font1 = {'family': 'Times New Roman', 'size': 10}
    # train_ax.legend(loc="best", bbox_to_anchor=(1, 1), fontsize=10, prop=my_font1)
    # test_ax.legend(loc="best", bbox_to_anchor=(1, 1), fontsize=10, prop=my_font1)

    # 添加统一的图例
    my_font1 = {'family': 'Times New Roman', 'size': 10}
    handles, labels = train_ax.get_legend_handles_labels()  # 获取训练集子图的图例句柄和标签
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1, 0.5), fontsize=10, prop=my_font1)  # 图例放在整体的右边

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, right=0.85)  # 留出底部和右侧空间

    # 保存图片
    plt.savefig(figure_iter_name_one, dpi=800, format='png')
    # plt.show()
    plt.close()

    return



def cal_shap(X_train_features_x_array, label_name):
    # label_name = ["goals_main", "zones_arrive_main", "times_depart_main", "times_arrive_main", "modes1_main",
    #               "modes2_main"]
    # # 5+3047+24+24+6+6

    # 实例化模型
    # model_goals_main = CombinedNetwork_shap_goals_main(zone_adjacency_matrix=zone_adjacency_matrix,
    #                         node_embeddings=node_embeddings,
    #                         distances=distances)
    number_label_name = 1  # 表示取label_name的第几个  'goals', 'zones_arrive', 'times_depart', 'times_arrive', 'modes1', 'modes2'
    # 定义一个字典存储所有模型实例
    models = {}

    # 遍历 label_name，动态创建模型实例
    for label in label_name:
        # 动态获取类名（假设类名与 label 对应）
        model_class_name = f"CombinedNetwork_shap_{label}"

        # 假设这些类已经定义在当前模块中，通过 globals() 获取类
        model_class = globals().get(model_class_name)

        if model_class is None:
            raise ValueError(f"Model class '{model_class_name}' not found!")

        # 实例化模型
        models[label] = model_class(
            zone_adjacency_matrix=zone_adjacency_matrix,
            node_embeddings=node_embeddings,
            distances=distances
        )

    # print('model_goals_main', model_goals_main)
    # print(models['zones_arrive_main'])  # 这两行输出相同
    # print(models[label_name[number_label_name]])


    # 背景数据
    num_sample = 3
    background_data_main = X_train_features_x_array[:num_sample]  # 形状 (10, 31)
    # print('background_data_main', background_data_main)    # (10, 9)

    # 合并背景数据
    background_data = np.hstack(
        [background_data_main, background_data_main, background_data_main, background_data_main,
         background_data_main, background_data_main])  # 形状 (10, 45)

    def predict_function(x):
        # print('6455864867564')
        goal_inputs = x[:, :9]
        zone_arrive_inputs = x[:, 9:18]
        time_depart_inputs = x[:, 18:27]
        time_arrive_inputs = x[:, 27:36]
        mode1_inputs = x[:, 36:45]
        mode2_inputs = x[:, 45:54]

        # 将输入按顺序放入列表
        input_list = [
            goal_inputs,
            zone_arrive_inputs,
            time_depart_inputs,
            time_arrive_inputs,
            mode1_inputs,
            mode2_inputs
        ]

        # 调用模型预测
        # # predictions = model_goals_main.predict(input_list)
        # predictions_goals_main = models['goals_main'].predict(input_list)
        # predictions_zones_arrive_main = models['zones_arrive_main'].predict(input_list)
        # predictions_times_depart_main = models['times_depart_main'].predict(input_list)
        # predictions_times_arrive_main = models['times_arrive_main'].predict(input_list)
        # predictions_modes1_main = models['modes1_main'].predict(input_list)
        # predictions_modes2_main = models['modes2_main'].predict(input_list)

        # 初始化一个字典存储每个模型的预测结果
        predictions = {}


        # 遍历 label_name，动态调用模型预测
        for label in label_name:
            if label in models:  # 确保模型已实例化
                predictions[label] = models[label].predict(input_list)
            else:
                print(f"Model for '{label}' not found!")

        # 打印调试信息（可选）
        # print("Predictions:", predictions)
        # print('predictions', predictions['zones_arrive_main'], predictions[label_name[number_label_name]])      # 二者相同
        # print('predictions54', predictions[label_name[number_label_name]])    # 60个

        # 如果 predictions_goals_main 是列表，提取第一个输出
        if isinstance(predictions[label_name[number_label_name]], list):
            predictions_goals_main = predictions[label_name[number_label_name]][0]  # 假设我们只关心第一个输出

        # 如果 predictions_goals_main 是 TensorFlow 张量，转换为 NumPy 数组
        if isinstance(predictions[label_name[number_label_name]], tf.Tensor):
            predictions[label_name[number_label_name]] = predictions[label_name[number_label_name]].numpy()

        return predictions[label_name[number_label_name]]

    # 初始化 KernelExplainer
    explainer = shap.KernelExplainer(predict_function, background_data)

    # 测试数据
    test_data_goal_inputs = X_train_features_x_array[:num_sample]
    test_data_zone_arrive_inputs = X_train_features_x_array[:num_sample]
    test_data_time_depart_inputs = X_train_features_x_array[:num_sample]
    test_data_time_arrive_inputs = X_train_features_x_array[:num_sample]
    test_data_mode1_inputs = X_train_features_x_array[:num_sample]
    test_data_mode2_inputs = X_train_features_x_array[:num_sample]

    # 合并测试数据
    test_data = np.hstack([test_data_goal_inputs, test_data_zone_arrive_inputs, test_data_time_depart_inputs,
                           test_data_time_arrive_inputs, test_data_mode1_inputs, test_data_mode2_inputs])  # 形状 (10, 54)
    # print('test_data', test_data.shape)   # (10, 54)

    # 计算 SHAP 值
    shap_values = explainer.shap_values(test_data)

    # 确保SHAP值正确
    # print(f"SHAP值形状: {np.array(shap_values).shape}")  # 应为 (num_classes, num_samples, num_features)   (10, 54, 5)

    # 如果 shap_values 是一个三维数组，将其转换为列表
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
        # print(f"转换后的 shap_values 类型: {type(shap_values)}")  # 应为 <class 'list'>
        # print(f"每个类别的 SHAP 值形状: {[sv.shape for sv in shap_values]}")  # 应为 [(10, 54), (10, 54), ...] 共5个

    # print(f"Length of shap_values: {len(shap_values)}")  # 10
    # print(f"Type of shap_values: {type(shap_values)}")   # <class 'list'>

    features_name = ['Gender', 'Age', 'Job', 'Homesize', 'HomeStudentJunior', 'HomeCar', 'IncomeMonth', 'District', 'ZoneDepartMain',
                     'Gender', 'Age', 'Job', 'Homesize', 'HomeStudentJunior', 'HomeCar', 'IncomeMonth', 'District', 'ZoneDepartMain',
                     'Gender', 'Age', 'Job', 'Homesize', 'HomeStudentJunior', 'HomeCar', 'IncomeMonth', 'District', 'ZoneDepartMain',
                     'Gender', 'Age', 'Job', 'Homesize', 'HomeStudentJunior', 'HomeCar', 'IncomeMonth', 'District', 'ZoneDepartMain',
                     'Gender', 'Age', 'Job', 'Homesize', 'HomeStudentJunior', 'HomeCar', 'IncomeMonth', 'District', 'ZoneDepartMain',
                     'Gender', 'Age', 'Job', 'Homesize', 'HomeStudentJunior', 'HomeCar', 'IncomeMonth', 'District', 'ZoneDepartMain']

    # 定义特征名称（根据实际情况调整）
    if not features_name:
        features_name = [f'Feature {i}' for i in range(test_data.shape[1])]
    # print("Length of features_name:", len(features_name))    # 54


    # 只保留前 8 个特征    # [:8] [9:17] [18:26] [27:35] [36:44] [45:53]
    # num_features_to_show_goals_main = 9
    # num_features_to_show_zones_arrive_main = 18
    # num_features_to_show_times_depart_main = 27
    # num_features_to_show_times_arrive_main = 36
    # num_features_to_show_modes1_main = 45
    # num_features_to_show_modes2_main = 54
    #
    # selected_features_goals_main = features_name[:num_features_to_show_goals_main-1]
    # selected_shap_values_goals_main = [sv[:, :num_features_to_show_goals_main-1] for sv in shap_values]  # 切片 SHAP 值
    # selected_test_data_goals_main = test_data[:, :num_features_to_show_goals_main-1]  # 切片测试数据
    #
    # selected_features_zones_arrive_main = features_name[num_features_to_show_goals_main:num_features_to_show_zones_arrive_main - 1]
    # selected_shap_values_zones_arrive_main = [sv[:, num_features_to_show_goals_main:num_features_to_show_zones_arrive_main - 1] for sv in shap_values]  # 切片 SHAP 值
    # selected_test_data_zones_arrive_main = test_data[:, num_features_to_show_goals_main:num_features_to_show_zones_arrive_main - 1]  # 切片测试数据
    #
    # selected_features_times_depart_main = features_name[num_features_to_show_zones_arrive_main:num_features_to_show_times_depart_main - 1]
    # selected_shap_values_times_depart_main = [sv[:, num_features_to_show_zones_arrive_main:num_features_to_show_times_depart_main - 1] for sv in shap_values]  # 切片 SHAP 值
    # selected_test_data_times_depart_main = test_data[:,num_features_to_show_zones_arrive_main:num_features_to_show_times_depart_main - 1]  # 切片测试数据
    #
    # selected_features_times_arrive_main = features_name[num_features_to_show_times_depart_main:num_features_to_show_times_arrive_main - 1]
    # selected_shap_values_times_arrive_main = [sv[:, num_features_to_show_times_depart_main:num_features_to_show_times_arrive_main - 1] for sv in shap_values]  # 切片 SHAP 值
    # selected_test_data_times_arrive_main = test_data[:,num_features_to_show_times_depart_main:num_features_to_show_times_arrive_main - 1]  # 切片测试数据
    #
    # selected_features_modes1_main = features_name[num_features_to_show_times_arrive_main:num_features_to_show_modes1_main - 1]
    # selected_shap_values_modes1_main = [sv[:, num_features_to_show_times_arrive_main:num_features_to_show_modes1_main - 1] for sv in shap_values]  # 切片 SHAP 值
    # selected_test_data_modes1_main = test_data[:,num_features_to_show_times_arrive_main:num_features_to_show_modes1_main - 1]  # 切片测试数据
    #
    # selected_features_modes2_main = features_name[num_features_to_show_modes1_main:num_features_to_show_modes2_main - 1]
    # selected_shap_values_modes2_main = [sv[:, num_features_to_show_modes1_main:num_features_to_show_modes2_main - 1] for sv in shap_values]  # 切片 SHAP 值
    # selected_test_data_modes2_main = test_data[:,num_features_to_show_modes1_main:num_features_to_show_modes2_main - 1]  # 切片测试数据

    # # 定义特征分组的范围
    # feature_ranges = {
    #     "goals_main": (0, 8),  # 第 1-8 列
    #     "zones_arrive_main": (9, 17),  # 第 10-17 列
    #     "times_depart_main": (18, 26),  # 第 19-26 列
    #     "times_arrive_main": (27, 35),  # 第 28-35 列
    #     "modes1_main": (36, 44),  # 第 37-44 列
    #     "modes2_main": (45, 53),  # 第 46-53 列
    # }

    # 定义特征分组的范围
    feature_ranges = {
        0: (0, 8),  # 第 1-8 列
        1: (9, 17),  # 第 10-17 列
        2: (18, 26),  # 第 19-26 列
        3: (27, 35),  # 第 28-35 列
        4: (36, 44),  # 第 37-44 列
        5: (45, 53),  # 第 46-53 列
    }

    # 初始化存储结果的字典
    selected_features = {}
    selected_shap_values = {}
    selected_test_data = {}

    # 遍历每个分组，批量生成数据
    for key, (start, end) in feature_ranges.items():
        # 切片特征名称
        selected_features[key] = features_name[start:end]
        # 切片 SHAP 值
        selected_shap_values[key] = [sv[:, start:end] for sv in shap_values]
        # 切片测试数据
        selected_test_data[key] = test_data[:, start:end]

        # 保存到 CSV 文件
        # 1. 保存特征名称
        features_df = pd.DataFrame(selected_features[key])
        features_df.to_csv(
            fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01shap_data\selected_features_{key}.csv",
            index=False)

        # 2. 保存 SHAP 值
        for i, shap_group in enumerate(selected_shap_values[key]):
            shap_df = pd.DataFrame(shap_group, columns=selected_features[key])
            shap_df.to_csv(
                fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01shap_data\selected_shap_values_{key}_group{i}.csv",
                index=False)

        # 3. 保存测试数据
        test_data_df = pd.DataFrame(selected_test_data[key], columns=selected_features[key])
        test_data_df.to_csv(
            fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01shap_data\selected_test_data_{key}.csv",
            index=False)

    # 示例：访问某个分组的数据
    print("Selected Features for goals_main:", selected_features[number_label_name], selected_features[number_label_name])
    print("Selected SHAP Values for zones_arrive_main:", selected_shap_values[number_label_name])
    print("Selected Test Data for modes2_main:", selected_test_data[number_label_name])


    # # 格式化特征名称和 SHAP 值，保留三位小数
    # selected_features_goals_main = [
    #     f"{name} = {value:.3f}"  # 保留三位小数
    #     for name, value in zip(selected_features_goals_main, selected_test_data_goals_main[0])  # 第一个样本的特征值
    # ]
    # print(f"Length of feature_names: {len(features_name)}")  # 54
    # print(f"Number of features in shap_values: {shap_values[0].shape[1]}") #  54
    assert len(features_name) == shap_values[0].shape[1], "Feature names length does not match the number of features!"
    # print('selected_shap_values_goals_main', selected_shap_values_goals_main[0].shape)  # (9, 8)
    # print('selected_test_data_goals_main', selected_test_data_goals_main.shape)   # (10, 8)
    # print('selected_shap_values[number_label_name]', selected_shap_values[number_label_name])  # (3, 9, 5)  是一个列表，包含【5 goal的输出个数】个二维数组，每个二维数组是（3，8）=（样本数，特征数）
    # print(selected_test_data[number_label_name])   # （5, 8）
    # print(selected_features[number_label_name])    # ['Gender', 'Age', 'Job', 'Homesize', 'HomeStudentJunior', 'HomeCar', 'IncomeMonth', 'District']

    # ================================ 绘图 ============================
    # 1. 条形图（平均|SHAP值|）  全局解释方法，平均绝对SHAP值, goal的5个输出一次性绘制出来，Goal(work、school ……)
    class_names_goals_main = ["work", "school", "visit", "shopping", "other"]   # from 01CQdata_input20241016.py  这里换成首字母小写了
    plt.figure(figsize=(7, 6))
    shap.summary_plot(selected_shap_values[number_label_name], selected_test_data[number_label_name], plot_type="bar", feature_names=selected_features[number_label_name], class_names=class_names_goals_main, show=False)
    # 移除默认标题
    plt.title("")  # 清空默认标题
    # 提取最后一个下划线之前的部分
    label_prefix = label_name[number_label_name].rsplit("_", 1)[0]    # 'goals', 'zones_arrive', 'times_depart', 'times_arrive', 'modes1', 'modes2'
    # 打印结果
    print("Extracted label prefix:", label_prefix)
    # 使用提取的名称设置 x 轴标签
    plt.xlabel(f'Mean(|SHAP Value|) for "{label_prefix}" (Red = Positive impact)', fontsize=12)   # 添加自定义横轴标题（放在图片下方）
    # label_name = ["goals_main", "zones_arrive_main", "times_depart_main", "times_arrive_main", "modes1_main", "modes2_main"]
    # plt.xlabel(f"Mean(|SHAP Value|) of {label_name[number_label_name]} ", fontsize=12, fontweight="bold")
    plt.savefig(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01shap\shap_bar_plot for {label_prefix} .png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. 蜂群图（所有样本的SHAP值分布）  全局解释方法，目前只绘制Goal【number_label_name】中的第一个work【number_in_label_name】。若想绘制其他的，则修改索引shap_values[0], 共画5个
    # eg. gender和age的SHAP值表现出相似的模式，这可能暗示这两个特征之间存在交互作用，可以通过进一步的【散点图】的“交互分析”来确认这一点
    # ----------------------------------------
    number_in_label_name = 4    # Goal中的第几个输出（'WorkMain', 'SchoolMain', 'VisitMain', 'ShoppingMain', 'OtherGoalMain'）
    # goals_main = ['WorkMain', 'SchoolMain', 'VisitMain', 'ShoppingMain', 'OtherGoalMain']
    plt.figure(figsize=(7, 6))
    shap.summary_plot(selected_shap_values[number_label_name][number_in_label_name], selected_test_data[number_label_name], feature_names=selected_features[number_label_name], show=False)   # 自动选择 plot_type="dot"
    # 移除默认标题
    plt.title("")  # 清空默认标题
    # label_prefix = label_name[number_label_name].rsplit("_", 1)[0]
    # label_in_prefix = class_names_goals_main[number_in_label_name].split("Main")[0]   # [0]指取分割后的第一部分，这里不动， 'Work', 'School', 'Visit', 'Shopping', 'OtherGoal'
    label_in_prefix = class_names_goals_main[number_in_label_name]
    plt.xlabel(f'SHAP Value for "{label_in_prefix} in {label_prefix}"', fontsize=12)
    plt.savefig(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01shap\shap_beeswarm_plot for {label_in_prefix} in {label_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. 散点图（依赖图，分析单个特征）   局部解释方法，特征值（x轴）与SHAP值（y轴）之间的关系，每个点都代表一个样本的特征值和SHAP值，交互第一个[0]和第二个特征[1]
    # selected_features_plot = 0
    # selected_features_interaction = 1
    # -----------------------------------------
    # number_label_name_interaction = 2
    num_feature = 3
    num_feature_interaction = 4
    plt.figure(figsize=(7, 6))
    shap.dependence_plot(
        num_feature,  # 选择第一个特征 Gender
        selected_shap_values[number_label_name][number_in_label_name],  # 指定类别（如goal有5个输出）work
        selected_test_data[number_label_name],
        feature_names=selected_features[number_label_name],
        interaction_index=num_feature_interaction,  # 可指定交互特征  Age
        show=False
    )
    plt.savefig(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01shap\shap_dependence_plot for {label_in_prefix} in {label_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. 瀑布图（单个样本的解释） √   局部解释方法，展示单个样本的SHAP值，goal的第一个输出work的第一个样本
    # 红色箭头表示预测值增加，蓝色箭头表示预测值减少。可以看到，【哪个特征】是影响出行者的“出行目的”的最重要特征。
    # ------------------------------------------------------
    # 设置 matplotlib 的全局格式
    plt.rcParams['axes.formatter.use_locale'] = True  # 启用本地化格式
    plt.rcParams['axes.formatter.limits'] = (-5, 5)  # 控制科学计数法的范围
    plt.rcParams['axes.formatter.useoffset'] = False  # 禁用偏移量

    number_sample = 0  # 第几个样本
    plt.figure(figsize=(7, 6))
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value[number_in_label_name],
        selected_shap_values[number_label_name][number_in_label_name][number_sample],  # 第一个样本，第一个类别
        feature_names=selected_features[number_label_name],
        max_display=10,    # 用于控制在图表中显示的最大特征数量
        show=False
    )

    # 使用最新的 waterfall 方法
    shap.plots.waterfall(
        shap.Explanation(
            values=selected_shap_values[number_label_name][number_in_label_name][number_sample],
            base_values=explainer.expected_value[number_in_label_name],
            data=selected_features[number_label_name],
            feature_names=selected_features[number_label_name]
        ),
        max_display=10,  # 控制显示的最大特征数量
        show=False
    )

    plt.savefig(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01shap\shap_waterfall_plot for {label_in_prefix} in {label_prefix}.png",
                dpi=30, bbox_inches="tight")
    plt.close()

    # 5. 力图 Force Plot
    # 绘制 Force Plot  局部解释方法，绘制单个样本的特征重要性（Force Plot）  假设关注第一个样本的 SHAP 值   goal的第一个输出work的第一个样本
    # 根据excel看第一个样本值 eg. 0.4，那么出行者以工作/上班为目的的概率有0.4，观察哪些特征对work的影响是正负，哪个特征影响最大等
    # 中间加粗的黑色数字代表的是模型对于特定预测的基础值（base value），是模型对所有样本的预期平均预测值。加上由各个特征值带来的影响后得到的最终预测值
    # (5-1)  html格式
    plt.figure(figsize=(7, 6))
    # sample_index = 0
    # output_index = 0
    # 绘制 Force Plot
    force_plot = shap.force_plot(
        explainer.expected_value[number_in_label_name],  # 基线值（对应第一个输出）
        selected_shap_values[number_label_name][number_in_label_name][number_sample],  # 单个样本的 SHAP 值
        selected_test_data[number_label_name][number_sample],   # 单个样本的输入数据
        feature_names=selected_features[number_label_name],
        # matplotlib=True, # 启用 matplotlib 模式, 若启用，则为静态图，savefig修改为png
        show=False
    )
    # 保存为 HTML 文件到当前工作目录
    shap.save_html(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01shap\force_plot for {label_in_prefix} in {label_prefix}.html", force_plot)

    # (5-2) 绘制 Force Plot      png格式
    plt.figure(figsize=(7, 6))
    force_plot = shap.force_plot(
        explainer.expected_value[number_in_label_name],  # 基线值（对应第一个输出）
        selected_shap_values[number_label_name][number_in_label_name][number_sample],  # 单个样本的 SHAP 值
        selected_test_data[number_label_name][number_sample],  # 单个样本的输入数据
        feature_names=selected_features[number_label_name],
        matplotlib=True,  # 启用 matplotlib 模式, 若启用，则为静态图，savefig修改为png
        show=False
    )
    plt.savefig(fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01shap\force_plot for {label_in_prefix} in {label_prefix}.png", dpi=300, bbox_inches="tight")  # 保存图像
    plt.close()  # 关闭当前画布

    # # (6) 绘制 相互作用图 Interaction Values
    # # shap_interaction_values 方法仅支持 TreeExplainer，NN不可绘制
    # plt.figure(figsize=(7, 6))
    # shap_interaction_values = explainer.shap_interaction_values(selected_shap_values[number_label_name][number_in_label_name])
    # shap.summary_plot(shap_interaction_values, selected_shap_values[number_label_name][number_in_label_name])
    # plt.savefig(
    #     fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01shap\interaction_plot for {label_in_prefix} in {label_prefix}.png",
    #     dpi=300, bbox_inches="tight")  # 保存图像
    # plt.close()  # 关闭当前画布

    # # (7) 绘制 热力图 Heatmap plot
    # plt.figure(figsize=(7, 6))
    # # shap_values_2 = explainer(selected_test_data[number_label_name])
    # shap_values_2 = explainer(test_data)
    # # selected_features_2 = {}
    # # selected_shap_values_2 = {}
    # # selected_test_data_2 = {}
    # # # 遍历每个分组，批量生成数据
    # # for key, (start, end) in feature_ranges.items():
    # #     # 切片特征名称
    # #     selected_features_2[key] = features_name[start:end]
    # #     # 切片 SHAP 值
    # #     selected_shap_values_2[key] = [sv[:, start:end] for sv in shap_values_2]
    # #     # 切片测试数据
    # #     selected_test_data_2[key] = test_data[:, start:end]
    # #
    # # shap.plots.heatmap(selected_test_data_2[number_label_name])
    # shap.plots.heatmap(shap_values_2[number_label_name])
    # plt.savefig(
    #     fr"E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01shap\heatmap_plot for {label_prefix}.png",
    #     dpi=300, bbox_inches="tight")  # 保存图像
    # plt.close()  # 关闭当前画布

    return



def plot_attention_heatmap_logscale(file_path):
    """
    从 CSV 文件中读取注意力权重矩阵并使用对数刻度绘制热图。
    :param file_path: 注意力权重矩阵的 CSV 文件路径
    """
    # 读取 CSV 文件
    attention_weights_matrix = np.loadtxt(file_path, delimiter=',')

    # 确保数据中没有零或负值
    # 将零或负值替换为 NaN，避免影响对数刻度
    attention_weights_matrix = np.where(attention_weights_matrix > 0, attention_weights_matrix, np.nan)

    # 绘制热图
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        attention_weights_matrix,
        annot=True,  # 显示注释
        fmt=".1e",  # 使用科学计数法显示注释
        cmap="viridis",  # 颜色映射
        norm=LogNorm(vmin=np.nanmin(attention_weights_matrix), vmax=np.nanmax(attention_weights_matrix)),  # 使用对数刻度
        cbar_kws={"label": "Attention Weights (Log Scale)"}  # 颜色条标签
    )
    # plt.title("Attention Weights Heatmap (Log Scale)")
    plt.xlabel("Zones")
    plt.ylabel("Zones")
    # plt.show()


def find_top_5x5_submatrix(matrix):
    """
    在给定矩阵中找到包含前25个最大值的5x5子矩阵。
    :param matrix: 输入的二维 NumPy 矩阵
    :return: 包含前25个最大值的5x5子矩阵及其左上角索引
    """
    rows, cols = matrix.shape
    max_sum = -np.inf
    top_submatrix = None
    top_left_index = (0, 0)

    # 遍历所有可能的5x5子矩阵
    for i in range(rows - 4):  # 行索引范围
        for j in range(cols - 4):  # 列索引范围
            submatrix = matrix[i:i + 5, j:j + 5]  # 提取5x5子矩阵
            current_sum = np.sum(submatrix)  # 计算子矩阵的总和

            # 更新最大值及对应的子矩阵
            if current_sum > max_sum:
                max_sum = current_sum
                top_submatrix = submatrix
                top_left_index = (i, j)

    return top_submatrix, top_left_index


def plot_heatmap(submatrix):
    """
    绘制5x5子矩阵的热图。
    :param submatrix: 5x5的子矩阵
    """

    # 定义格式化函数，用于显示注释
    def format_scientific(value):
        if value == 0:
            return "0"
        return f"{value:.1e}"

    formatted_matrix = np.vectorize(format_scientific)(submatrix)

    # 绘制热图
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        submatrix,
        annot=formatted_matrix,
        fmt="",
        cmap="viridis",
        cbar_kws={"label": "Attention Weights"}
    )
    # plt.title("Top 5x5 Submatrix Heatmap")
    plt.xlabel("Zones")
    plt.ylabel("Zones")
    plt.xticks(ticks=np.arange(5) + 0.5, labels=np.arange(1, 6))
    plt.yticks(ticks=np.arange(5) + 0.5, labels=np.arange(1, 6))

    # plt.show()

# ============================== Definition End =============================

# distances = ...  # 这里应该是您的数据集中的实际距离
# 替换为您的CSV文件路径
csv_file_path = "suevey_zone_adjacency_matrix_distances_try.csv"  # 交通小区的邻接矩阵，是一个对称矩阵
# 使用pandas读取CSV文件
distances_df = pd.read_csv(csv_file_path, index_col=0)
# 将DataFrame转换为NumPy数组
distances = distances_df.values

# 获取交通小区ID
zone_ids = distances_df.columns.tolist()  # 列名就是交通小区ID, DataFrame格式
# print('zone_ids', zone_ids)

# 将 zone_ids 转换为整数列表
zone_ids_int = [int(id) for id in zone_ids]
print('zone_ids_int', zone_ids_int)

# 获取行数，这将等于 num_zones
num_zones = distances_df.shape[0]
# print('num_zones', num_zones)   # 14


# print('CombinedNetwork')
# print("Model inputs:", model.input)

# 指定CSV文件路径
csv_file_path_qianru = "Node2Vec_GraphEmbedding_characteristic_matrix_try.csv"
# 使用pandas读取CSV文件
# embeddings_df = pd.read_csv(csv_file_path_qianru, header=None)
# 使用pandas读取CSV文件，跳过第一行（header=None）和第一列（usecols从第二列开始）
embeddings_df = pd.read_csv(csv_file_path_qianru, header=None, skiprows=1, usecols=lambda x: x != 0)
# 确认数据形状
# print("Shape of embeddings_df:", embeddings_df.shape)  # 应该输出 (14, 64)

# 将DataFrame转换为TensorFlow张量
node_embeddings = tf.convert_to_tensor(embeddings_df.values, dtype=tf.float32)
# 打印形状以确认
# print("Shape of node_embeddings:", node_embeddings.shape)  # 应该输出 (14, 64)

# # 如果你的模型只需要部分节点的嵌入（例如前14个）
# num_nodes_needed = 14
# selected_node_embeddings = node_embeddings[:num_nodes_needed]
# print("Shape of selected_node_embeddings:", selected_node_embeddings.shape)  # 应该输出 (14, 64)

# node_embeddings = tf.random.normal([14, 64])  # 假设有3046个交通小区，每个小区有64维嵌入向量
# # 实际上，这个[嵌入向量]，应该是一个csv，3046*64的，待计算

sigma2 = 1.0  # 可以根据需要调整
zone_adjacency_matrix = compute_weighted_adjacency_matrix(distances, sigma2)
# print(zone_adjacency_matrix.shape)  # (14, 14)

# 实例化模型
model = CombinedNetwork(zone_adjacency_matrix=zone_adjacency_matrix,
    node_embeddings=node_embeddings,
    distances=distances)
# print(f"CombinedNetwork instance ID: {id(model)}")
# 导出动态效用参数
# model.export_dynamic_parameters("dynamic_parameters.csv")

# optimizer = tf.keras.optimizers.Adam(learning_rate=0.5)
optimizer = tf.keras.optimizers.Adam()
train_vars = model.trainable_variables
# print("train_vars:")
# for var in train_vars:
#     print(var.name, var.shape, var)

# origin_district_ids = tf.constant([100, 200, 300, 400])  # 示例出发交通小区编号
# arrive_district_ids = tf.constant([100, 200, 300, 400])  # 示例出发交通小区编号

# 提取第一列（假设第一列为交通小区编号）
# 如果你的CSV文件使用的是0索引，则可能需要调整这里的逻辑来适应实际情况
origin_district_ids_series = distances_df.index
# 将所有索引值转换为列表
origin_district_ids_list = origin_district_ids_series.tolist()
# 将Python列表转换为TensorFlow张量
origin_district_ids = tf.constant(origin_district_ids_list)
arrive_district_ids = tf.constant(origin_district_ids_list)

# print('origin_district_ids', origin_district_ids, arrive_district_ids.shape)   #  (14,)  (14,)

# 读取更新后的CSV文件
# df_ZoneDepartMain = pd.read_csv('Household survey_Person4_10_classify_prob.csv')
# # 提取“ZoneDepartMain”列
# zone_depart_main_series = df_ZoneDepartMain['ZoneDepartMain']
# # 将该列转换为列表
# zone_depart_main_list = zone_depart_main_series.tolist()
# # 将Python列表转换为TensorFlow张量
# ZoneDepartMain_int = tf.constant(zone_depart_main_list)
# ZoneDepartMain = tf.cast(ZoneDepartMain_int, dtype=tf.float32)

# 预热
warmup(model, optimizer, node_embeddings, zone_adjacency_matrix, distances)
print('Finished: warmup')
# 预热阶段
# print(f"Warm-up using CombinedNetwork instance ID: {id(model)}")



datasets_all_train = load_data_train_trainvalidation()
print('Finished: load_data_train_trainvalidation')


#  用于绘图：
# 定义颜色映射
# color_map_many = {
#     'F1(goal_main)': '#c9e4ff',  # 比 #9dd5ff 浅
#     'F2(zone_depart_main)': '#b8d7a3',  # 比 #70ad47 浅
#     'F3(zone_arrive_main)': '#d8e6c1',  # 比 #c0dda3 浅
#     'F4(time_depart_main)': '#ffe58f',  # 比 #ffbd00 浅
#     'F5(time_arrive_main)': '#fff2cc',  # 比 #ffe9a1 浅
#     'F6(mode1_main)': '#e5caff',  # 比 #ce0afe 浅
#     'F7(mode2_main)': '#ffd9f0',  # 比 #ffc2eb 浅
#     'F(total_main)': '#9ec5f2'     # 比 #0070c0 浅
# }

color_map_many = {
    'F1(goal_main)': '#e0f0ff',  # 比 #c9e4ff 更浅
    'F2(zone_arrive_main)': '#e8f1d8',  # 比 #d8e6c1 更浅
    'F3(time_depart_main)': '#fff5cc',  # 比 #ffe58f 更浅
    'F4(time_arrive_main)': '#fff9dd',  # 比 #fff2cc 更浅
    'F5(mode1_main)': '#f3daff',  # 比 #e5caff 更浅
    'F6(mode2_main)': '#fff0fa',  # 比 #ffd9f0 更浅
    'F(total_main)': '#cbe1fd'     # 比 #9ec5f2 更浅
}

color_map_one = {
    'F1(goal_main)': '#9dd5ff',
    'F2(zone_arrive_main)': '#c0dda3',
    'F3(time_depart_main)': '#ffbd00',
    'F4(time_arrive_main)': '#ffe9a1',
    'F5(mode1_main)': '#ce0afe',
    'F6(mode2_main)': '#ffc2eb',
    'F(total_main)': '#0070c0'
}
# 初始化全局绘图对象
fig_train, ax_train = None, None
fig_val, ax_val = None, None
train_lines_added = 0
val_lines_added = 0


# 网格搜索Grid Search
# 设定超参数搜索空间
# param_grid = {
#     'batch_size': [16, 32, 64, 128, 256, 320, 512, 640, 1024],
#     'num_epochs': [10, 20, 50, 100, 150, 200, 250, 300, 400, 500, 600]
# }   # 7*12=84

# param_grid = {
#     'batch_size': [16, 32, 64, 128, 256],
#     'num_epochs': [20, 50, 100, 300]
# }   # 5*4=20
#
# param_grid = {
#     'batch_size': [32, 128, 256],
#     'num_epochs': [20, 100, 300]
# }   # 5*4=20   (1)

param_grid = {
    'batch_size': [32, 256],
    'num_epochs': [20, 30]
}   # 5*4=20


# param_grid = {
#     'batch_size': [32, 128, 256],
#     'num_epochs': [50, 200, 400, 500, 600]
# }   # 5*4=20  (2)



# ============================= gridsearch 寻找最优超参数 ==============================================
# # 执行网格搜索Grid Search，调试时，删除@function，都则会报错，查看最后生成的验证集上的收敛图，选择合适的batch_size和num_epochs，
# # 注释掉以下几行；再取消注释再下面的行。不用打开@function，否则会报错
# best_validation_loss, best_params, df_batchsize_epoch, best_validation_loss_mean_10fold, best_params_mean_10fold, df_batchsize_epoch_mean_10fold \
#     = gridsearch(param_grid, datasets_all_train)
# print('Finished: gridsearch.')
#
# # save data, in case of #
# df_batchsize_epoch = df_batchsize_epoch
# df_batchsize_epoch_mean_10fold = df_batchsize_epoch_mean_10fold
#
# # 使用最佳的batch_size和num_epochs训练模型：
# batch_size = best_params['batch_size']
# num_epochs = best_params['num_epochs']
#
# plot_gridsearch_batchsize_epoch(df_batchsize_epoch)   # 绘制batch_size & epoch的对比曲线图
# plot_gridsearch_batchsize_epoch_10fold(df_batchsize_epoch_mean_10fold)




# ======================== train and test after find the best hyper-parameters ===========================
# 通过网格搜索和10从交叉验证等方法寻找到了最佳超参数之后，重新训练所有数据，建立模型。在进行10次交叉验证之后，最常用的策略是使用所有数据重新训练模型。这种策略结合了交叉验证的优势和模型的泛化能力
batch_size = 32
num_epochs = 2
data_sample = 62000  # How many rows do input data have ?

# 正式训练，all_training data
(dataset_goal_main_train, dataset_zones_arrive_main_train, dataset_times_depart_main_train, dataset_times_arrive_main_train, dataset_mode1_main_train, dataset_mode2_main_train, dataset_goal_main_test, dataset_zones_arrive_main_test, dataset_times_depart_main_test, dataset_times_arrive_main_test, dataset_mode1_main_test,
 dataset_mode2_main_test) = load_data_train_test(datasets_all_train, batch_size)
print('Finished: load_data_train_test.')

# 执行训练train和测试test，不再需要验证，因为已经寻找到最佳超参数（epoch、batchsize等）。训练时，迭代多次，根据最佳超参数，建立一个新的模型。而在测试时，不需要迭代多次，只进行一次forwared passing即可
(train_list_iter_loss_goal_main, train_list_iter_loss_zone_arrive_main, train_list_iter_loss_time_depart_main, train_list_iter_loss_time_arrive_main, train_list_iter_loss_mode1_main, train_list_iter_loss_mode2_main, train_list_iter_loss,
 train_list_epoch_loss_goal_main, train_list_epoch_loss_zone_arrive_main, train_list_epoch_loss_time_depart_main, train_list_epoch_loss_time_arrive_main, train_list_epoch_loss_mode1_main, train_list_epoch_loss_mode2_main, train_list_epoch_loss,
 test_list_iter_loss_goal_main, test_list_iter_loss_zone_arrive_main, test_list_iter_loss_time_depart_main, test_list_iter_loss_time_arrive_main, test_list_iter_loss_mode1_main, test_list_iter_loss_mode2_main, test_list_iter_loss,
 df_results,
 train_list_iter_loss_r2_goal_main, train_list_iter_loss_r2_zone_arrive_main, train_list_iter_loss_r2_time_depart_main,
 train_list_iter_loss_r2_time_arrive_main, train_list_iter_loss_r2_mode1_main, train_list_iter_loss_r2_mode2_main,
 train_list_iter_loss_r2,
 train_list_epoch_loss_r2_goal_main, train_list_epoch_loss_r2_zone_arrive_main,
 train_list_epoch_loss_r2_time_depart_main, train_list_epoch_loss_r2_time_arrive_main,
 train_list_epoch_loss_r2_mode1_main, train_list_epoch_loss_r2_mode2_main, train_list_epoch_loss_r2,
 test_list_iter_loss_r2_goal_main, test_list_iter_loss_r2_zone_arrive_main, test_list_iter_loss_r2_time_depart_main,
 test_list_iter_loss_r2_time_arrive_main, test_list_iter_loss_r2_mode1_main, test_list_iter_loss_r2_mode2_main,
 test_list_iter_loss_r2,
 train_list_iter_loss_rmse_goal_main, train_list_iter_loss_rmse_zone_arrive_main,
 train_list_iter_loss_rmse_time_depart_main, train_list_iter_loss_rmse_time_arrive_main,
 train_list_iter_loss_rmse_mode1_main, train_list_iter_loss_rmse_mode2_main, train_list_iter_loss_rmse,
 train_list_epoch_loss_rmse_goal_main, train_list_epoch_loss_rmse_zone_arrive_main,
 train_list_epoch_loss_rmse_time_depart_main, train_list_epoch_loss_rmse_time_arrive_main,
 train_list_epoch_loss_rmse_mode1_main, train_list_epoch_loss_rmse_mode2_main, train_list_epoch_loss_rmse,
 test_list_iter_loss_rmse_goal_main, test_list_iter_loss_rmse_zone_arrive_main,
 test_list_iter_loss_rmse_time_depart_main, test_list_iter_loss_rmse_time_arrive_main,
 test_list_iter_loss_rmse_mode1_main, test_list_iter_loss_rmse_mode2_main, test_list_iter_loss_rmse,
 train_list_iter_loss_mae_goal_main, train_list_iter_loss_mae_zone_arrive_main,
 train_list_iter_loss_mae_time_depart_main, train_list_iter_loss_mae_time_arrive_main,
 train_list_iter_loss_mae_mode1_main, train_list_iter_loss_mae_mode2_main, train_list_iter_loss_mae,
 train_list_epoch_loss_mae_goal_main, train_list_epoch_loss_mae_zone_arrive_main,
 train_list_epoch_loss_mae_time_depart_main, train_list_epoch_loss_mae_time_arrive_main,
 train_list_epoch_loss_mae_mode1_main, train_list_epoch_loss_mae_mode2_main, train_list_epoch_loss_mae,
 test_list_iter_loss_mae_goal_main, test_list_iter_loss_mae_zone_arrive_main, test_list_iter_loss_mae_time_depart_main,
 test_list_iter_loss_mae_time_arrive_main, test_list_iter_loss_mae_mode1_main, test_list_iter_loss_mae_mode2_main,
 test_list_iter_loss_mae, goals_main, zones_arrive_main, times_depart_main, times_arrive_main, modes1_main, modes2_main)\
    = epoch_train_test(batch_size, num_epochs, dataset_goal_main_train, dataset_zones_arrive_main_train, dataset_times_depart_main_train, dataset_times_arrive_main_train, dataset_mode1_main_train, dataset_mode2_main_train,
                       dataset_goal_main_test, dataset_zones_arrive_main_test, dataset_times_depart_main_test, dataset_times_arrive_main_test, dataset_mode1_main_test, dataset_mode2_main_test,
                       node_embeddings, zone_adjacency_matrix, distances, model, optimizer)
# print('Finished: epoch_train_test.')
# 定义 ANSI 转义序列
RED = "\033[31m"  # 普通红色
RESET = "\033[0m"  # 重置颜色
# 使用 ANSI 转义序列打印红色文字
print(f"{RED}Finished: epoch_train_test.{RESET}")

# 保存损失值csv
save_loss_mse_csv_test(train_list_iter_loss_goal_main,
              train_list_iter_loss_zone_arrive_main,
              train_list_iter_loss_time_depart_main, train_list_iter_loss_time_arrive_main,
              train_list_iter_loss_mode1_main,
              train_list_iter_loss_mode2_main, train_list_iter_loss,
              train_list_epoch_loss_goal_main,
              train_list_epoch_loss_zone_arrive_main,
              train_list_epoch_loss_time_depart_main, train_list_epoch_loss_time_arrive_main,
              train_list_epoch_loss_mode1_main,
              train_list_epoch_loss_mode2_main, train_list_epoch_loss,
              test_list_iter_loss_goal_main,
              test_list_iter_loss_zone_arrive_main, test_list_iter_loss_time_depart_main,
              test_list_iter_loss_time_arrive_main, test_list_iter_loss_mode1_main,
              test_list_iter_loss_mode2_main, test_list_iter_loss)

save_loss_r2_csv_test(train_list_iter_loss_r2_goal_main, train_list_iter_loss_r2_zone_arrive_main, train_list_iter_loss_r2_time_depart_main,
 train_list_iter_loss_r2_time_arrive_main, train_list_iter_loss_r2_mode1_main, train_list_iter_loss_r2_mode2_main,
 train_list_iter_loss_r2,
 train_list_epoch_loss_r2_goal_main, train_list_epoch_loss_r2_zone_arrive_main,
 train_list_epoch_loss_r2_time_depart_main, train_list_epoch_loss_r2_time_arrive_main,
 train_list_epoch_loss_r2_mode1_main, train_list_epoch_loss_r2_mode2_main, train_list_epoch_loss_r2,
 test_list_iter_loss_r2_goal_main, test_list_iter_loss_r2_zone_arrive_main, test_list_iter_loss_r2_time_depart_main,
 test_list_iter_loss_r2_time_arrive_main, test_list_iter_loss_r2_mode1_main, test_list_iter_loss_r2_mode2_main,
 test_list_iter_loss_r2)

save_loss_rmse_csv_test(train_list_iter_loss_rmse_goal_main, train_list_iter_loss_rmse_zone_arrive_main,
 train_list_iter_loss_rmse_time_depart_main, train_list_iter_loss_rmse_time_arrive_main,
 train_list_iter_loss_rmse_mode1_main, train_list_iter_loss_rmse_mode2_main, train_list_iter_loss_rmse,
 train_list_epoch_loss_rmse_goal_main, train_list_epoch_loss_rmse_zone_arrive_main,
 train_list_epoch_loss_rmse_time_depart_main, train_list_epoch_loss_rmse_time_arrive_main,
 train_list_epoch_loss_rmse_mode1_main, train_list_epoch_loss_rmse_mode2_main, train_list_epoch_loss_rmse,
 test_list_iter_loss_rmse_goal_main, test_list_iter_loss_rmse_zone_arrive_main,
 test_list_iter_loss_rmse_time_depart_main, test_list_iter_loss_rmse_time_arrive_main,
 test_list_iter_loss_rmse_mode1_main, test_list_iter_loss_rmse_mode2_main, test_list_iter_loss_rmse)

save_loss_mae_csv_test(train_list_iter_loss_mae_goal_main, train_list_iter_loss_mae_zone_arrive_main,
 train_list_iter_loss_mae_time_depart_main, train_list_iter_loss_mae_time_arrive_main,
 train_list_iter_loss_mae_mode1_main, train_list_iter_loss_mae_mode2_main, train_list_iter_loss_mae,
 train_list_epoch_loss_mae_goal_main, train_list_epoch_loss_mae_zone_arrive_main,
 train_list_epoch_loss_mae_time_depart_main, train_list_epoch_loss_mae_time_arrive_main,
 train_list_epoch_loss_mae_mode1_main, train_list_epoch_loss_mae_mode2_main, train_list_epoch_loss_mae,
 test_list_iter_loss_mae_goal_main, test_list_iter_loss_mae_zone_arrive_main, test_list_iter_loss_mae_time_depart_main,
 test_list_iter_loss_mae_time_arrive_main, test_list_iter_loss_mae_mode1_main, test_list_iter_loss_mae_mode2_main,
 test_list_iter_loss_mae)

print('Finished: save_loss_csv_test.')



train_figure_iter_name = fr'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_png\01Convergence Curv of iter_loss (train).png'
train_figure_epoch_name = fr'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_png\01Convergence Curv of epoch_loss (train).png'
# 以下绘图时，需要修改函数中横纵坐标的范围值
plot_convergence_cure(train_list_iter_loss_goal_main,
                      train_list_iter_loss_zone_arrive_main, train_list_iter_loss_time_depart_main,
                      train_list_iter_loss_time_arrive_main, train_list_iter_loss_mode1_main,
                      train_list_iter_loss_mode2_main, train_list_iter_loss,
                      train_list_epoch_loss_goal_main,
                      train_list_epoch_loss_zone_arrive_main, train_list_epoch_loss_time_depart_main,
                      train_list_epoch_loss_time_arrive_main, train_list_epoch_loss_mode1_main,
                      train_list_epoch_loss_mode2_main, train_list_epoch_loss,
                      train_figure_iter_name, train_figure_epoch_name)

print('Finished: plot_convergence_cure_train.')

test_figure_iter_name = fr'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_png\01Curv of iter_loss (test).png'

plot_test_cure(test_list_iter_loss_goal_main,
              test_list_iter_loss_zone_arrive_main, test_list_iter_loss_time_depart_main,
              test_list_iter_loss_time_arrive_main, test_list_iter_loss_mode1_main,
              test_list_iter_loss_mode2_main, test_list_iter_loss, test_figure_iter_name, df_results)

# print('test_list_iter_loss_goal_main', test_list_iter_loss_goal_main)


# print('Finished: plot_test_cure.')
# 定义 ANSI 转义序列
BLUE = "\033[94m"  # 蓝色
RESET = "\033[0m"  # 重置颜色
# 使用 ANSI 转义序列打印蓝色文字
print(f"{BLUE}Finished: plot_test_cure.{RESET}")



# ================== 将把Test loss和Training loss画在一起，做对比，看是否有过拟合等现象 ========================
figure_iter_name_two = fr'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_png\01Convergence Curv of iter_loss (train and test).png'
plot_train_test_cure_two(train_list_iter_loss_goal_main, train_list_iter_loss_zone_arrive_main, train_list_iter_loss_time_depart_main, train_list_iter_loss_time_arrive_main, train_list_iter_loss_mode1_main, train_list_iter_loss_mode2_main, train_list_iter_loss, test_list_iter_loss_goal_main, test_list_iter_loss_zone_arrive_main, test_list_iter_loss_time_depart_main, test_list_iter_loss_time_arrive_main, test_list_iter_loss_mode1_main, test_list_iter_loss_mode2_main, test_list_iter_loss, figure_iter_name_two)

# 绘制【组图】同一个图，左边一个图 是training，右边一个图 是test。纵坐标相同，横坐标不同
figure_iter_name_one = fr'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\01output_traintest_png\01Convergence Curv of iter_loss (train and test)_one.png'
plot_train_test_cure_one(train_list_iter_loss_goal_main, train_list_iter_loss_zone_arrive_main, train_list_iter_loss_time_depart_main, train_list_iter_loss_time_arrive_main, train_list_iter_loss_mode1_main, train_list_iter_loss_mode2_main, train_list_iter_loss, test_list_iter_loss_goal_main, test_list_iter_loss_zone_arrive_main, test_list_iter_loss_time_depart_main, test_list_iter_loss_time_arrive_main, test_list_iter_loss_mode1_main, test_list_iter_loss_mode2_main, test_list_iter_loss, figure_iter_name_one)


# ================================= SHAP ===================================
label_name = ["goals_main", "zones_arrive_main", "times_depart_main", "times_arrive_main", "modes1_main", "modes2_main"]
# 5+3047+24+24+6+6

(X_train_features_x_array, X_test_features_x_array,
     y_trainvalidation_goals_main, y_test_goals_main,
     y_trainvalidation_zones_arrive_main, y_test_zones_arrive_main,
     y_trainvalidation_times_depart_main, y_test_times_depart_main,
     y_trainvalidation_times_arrive_main, y_test_times_arrive_main,
     y_trainvalidation_mode1_main, y_test_mode1_main,
     y_trainvalidation_mode2_main, y_test_mode2_main) = datasets_all_train


# cal_shap(X_train_features_x_array, label_name)



# ================================= 【可解释性】注意力权重的可视化分析 ===================================
# 绘制注意力得分的热图
file_path = r'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\attention_weights_matrix\attention_weights_matrix.csv'
plot_attention_heatmap_logscale(file_path)
plt.savefig(fr'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\attention_weights_matrix\attention_heatmap.png', dpi=800, format='png')

# 绘制注意力得分的热图 前5个最大数值的小区的热图
file_path = r'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\attention_weights_matrix\attention_weights_matrix.csv'
# plot_top_attention_zones(file_path)
attention_weights_matrix = np.loadtxt(file_path, delimiter=',')
# 找到包含前25个最大值的5x5子矩阵
top_submatrix, top_left_index = find_top_5x5_submatrix(attention_weights_matrix)
print(f"Top-left corner of the 5x5 submatrix: {top_left_index}")
# 绘制热图
plot_heatmap(top_submatrix)
plt.savefig(fr'E:\Desktop\09[Win]CQ_part_GoalZoneTimeMode_zoneDistance20250221\01output_test_accuracy\attention_weights_matrix\attention_heatmap_max.png', dpi=800, format='png')


