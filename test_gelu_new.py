import torch
import math
import matplotlib.pyplot as plt

# 定义gelu_new函数
def gelu_new(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

# 定义gelu函数
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

# 生成x值
x_values = torch.linspace(-5, 5, 1000)

# 计算两个函数的y值
y_gelu_new = gelu_new(x_values)
y_gelu = gelu(x_values)

# 绘制图像
plt.figure(figsize=(10, 6))
plt.plot(x_values.numpy(), y_gelu_new.numpy(), label='GELU-New', linewidth=2)
plt.plot(x_values.numpy(), y_gelu.numpy(), label='GELU', linestyle='--', linewidth=2)
plt.title('Comparison of GELU-New and GELU Functions')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.savefig('comparison_between_gelu_new_and_gelu.png')