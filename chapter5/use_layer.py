# 使用层
from layer import *
import numpy as np


# 以购买两个苹果三个橘子为例
# 设置结点
apple_mul_layer = MulLayer()
origin_mul_layer = MulLayer()
total_add_layer = AddLayer()
tax_mul_layer = MulLayer()
# 设置初值
apple_num = 2
apple_price = 100
orange_num = 3
orange_price = 150
tax = 1.1
# 正向传播
apple_sum = apple_mul_layer.forward(apple_num, apple_price)
orange_sum = origin_mul_layer.forward(orange_num, orange_price)
total_sum = total_add_layer.forward(apple_sum, orange_sum)
taxed_total_sum = tax_mul_layer.forward(total_sum, tax)
# 反向传播
dtaxed_total_sum = 1
dtotal_sum, dtax = tax_mul_layer.backward(dtaxed_total_sum)
dapple_sum, dorange_sum = total_add_layer.backward(dtotal_sum)
dapple_num, dapple_price = apple_mul_layer.backward(dapple_sum)
dorange_num, dorange_price = origin_mul_layer.backward(dorange_sum)

print(apple_num)
print(apple_price)
print(orange_num)
print(orange_price)
print(apple_sum)
print(orange_sum)
print(total_sum)
print(tax)
print(dtaxed_total_sum)
print(dtotal_sum)
print(dtax)
print(dapple_sum)
print(dorange_sum)
print(dapple_num)
print(dapple_price)
print(dorange_num)
print(dorange_price)

# 使用RuLE层
relu_layer = Relu()
x = np.array([[1.0, -0.5], [-2.0, 3.0]])
x_rule = relu_layer.forward(x)
dx = relu_layer.backward(x_rule)
print(x)
print(x_rule)
print(dx)

# 使用Sigmoid层
sigmoid_layer = Sigmoid()
x = 0
out = sigmoid_layer.forward(x)
dout = 1
dx = sigmoid_layer.backward(dout)
print(out)
print(dx)