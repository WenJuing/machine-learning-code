import torchvision.models as models
from PIL import Image
from commom import preprocess_image_to_input
import matplotlib.pyplot as plt


features = []	# 存放中间层特征
def hook(model, input, output):
  features.append(output.detach())


if __name__ == '__main__':
    vgg16 = models.vgg16(pretrained=True)
    # 打印每个卷积层的索引和卷积核形状
    for key, val in vgg16.state_dict().items():
        print(key, val.shape)
    # 可视化卷积核
    for i in range(64):
        plt.subplot(8, 8, i+1)
        # plt.imshow(feature.numpy()[0, i, :, :])
        plt.imshow(vgg16.state_dict()['features.2.weight'].numpy()[10, i, :, :])    # [num, channel, h, w]
        plt.axis("off")
    plt.show()
    # 可视化特征图
    img = Image.open("./flower.jpg")
    img = preprocess_image_to_input(img)	# 处理成输入数据的形式
    vgg16.eval()
	# 获取vgg16的features中第四层的特征图
    vgg16.features[4].register_forward_hook(hook)	
    _ = vgg16(img)
    feature = features[0]	# feature.shape = [1, 64, 112, 112]
    for i in range(64):
        plt.subplot(8, 8, i+1)
        plt.imshow(feature.numpy()[0, i, :, :])
        plt.axis("off")
    plt.show()