import torch
import torch.nn as nn
import timm

class SwinTransformerFeatureExtractor_hsi(nn.Module):
    def __init__(self):
        super(SwinTransformerFeatureExtractor_hsi, self).__init__()
        # 初始化 Swin Transformer，这里使用预定义的 swin_base_patch4_window7_224 模型
        # 注意：通常 Swin Transformer 需要 3 通道输入，你需要调整网络以接受 31 通道
        self.swin_transformer = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        self.swin_transformer.head = nn.Identity()  # 移除原来的分类头

        # 替换第一层以接受 31 通道输入
        self.swin_transformer.patch_embed.proj = nn.Conv2d(31, 128, kernel_size=(4, 4), stride=(4, 4))
        # 添加自适应平均池化层和一个卷积层来调整输出维度

        self.conv1 = nn.Conv2d(1024, 512, kernel_size=1)  # 假设中间特征维度是1024，需要确认

    def forward(self, x):
        x = self.swin_transformer(x)  # 通过 Swin Transformer 提取特征

        x = x.permute(0, 3, 1, 2)

        x = self.conv1(x)  # 减少通道数到512
        return x

class SwinTransformerFeatureExtractor_rgb(nn.Module):
    def __init__(self):
        super(SwinTransformerFeatureExtractor_rgb, self).__init__()
        # 初始化 Swin Transformer，这里使用预定义的 swin_base_patch4_window7_224 模型
        # 注意：通常 Swin Transformer 需要 3 通道输入，你需要调整网络以接受 31 通道
        self.swin_transformer = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        self.swin_transformer.head = nn.Identity()  # 移除原来的分类头

        # 替换第一层以接受 31 通道输入
        #self.swin_transformer.patch_embed.proj = nn.Conv2d(31, 128, kernel_size=(4, 4), stride=(4, 4))
        # 添加自适应平均池化层和一个卷积层来调整输出维度

        self.conv1 = nn.Conv2d(1024, 512, kernel_size=1)  # 假设中间特征维度是1024，需要确认

    def forward(self, x):
        x = self.swin_transformer(x)  # 通过 Swin Transformer 提取特征
        print(x.shape)
        x = x.permute(0, 3, 1, 2)
        print(x.shape)
        x = self.conv1(x)  # 减少通道数到512
        print(x.shape)
        return x


class SwinTransformerFeatureExtractor_joint(nn.Module):
    def __init__(self):
        super(SwinTransformerFeatureExtractor_joint, self).__init__()
        # 使用预训练的 Swin Transformer 模型
        self.swin_transformer = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        self.swin_transformer.head = nn.Identity()  # 移除分类头

        # 替换第一层以接受 31 通道输入
        first_conv_layer = self.swin_transformer.patch_embed.proj
        self.swin_transformer.patch_embed.proj = nn.Conv2d(34, first_conv_layer.out_channels,
                                                           kernel_size=first_conv_layer.kernel_size,
                                                           stride=first_conv_layer.stride,
                                                           padding=first_conv_layer.padding)

        # 确认 Swin Transformer 的输出通道数，并相应地调整卷积层
        num_features = self.swin_transformer.num_features  # 通常这是最后一层的通道数
        self.conv1 = nn.Conv2d(num_features, 512, kernel_size=1)

    def forward(self, rgb, hsi):
        x = torch.cat((rgb,hsi), dim=1)
        x = self.swin_transformer(x)  # 提取特征
        x = x.permute(0, 3, 1, 2)
        # 假设输出已经是 [batch, channels, height, width]，则不需要permute
        x = self.conv1(x)  # 调整通道数
        return x

class Joint_Swintransformer(nn.Module):
    def __init__(self):
        super(Joint_Swintransformer, self).__init__()
        self.joint_swintransformer = timm.create_model('swin_base_patch4_window7_224', pretrained=True, in_chans= 34)
        self.joint_swintransformer.head = nn.Identity()
        self.conv1 = nn.Conv2d(1024, 512, kernel_size=1, )

    def forward(self, rgb, hsi):
        print(rgb.shape)
        print(hsi.shape)
        x = torch.cat((rgb, hsi), dim=1)
        print(x.shape)
        x = self.joint_swintransformer(x)
        x = x.permute(0,3,1,2)
        x = self.conv1(x)
        return x

if __name__ =='__main__':
    # 实例化模型并创建输入张量
    model = SwinTransformerFeatureExtractor_joint()
    #print(model)
    input_tensor = torch.randn(1, 34, 224, 224)  # 假设 batch size 是 1

    # 前向传播
    output = model(input_tensor)
    print(f'Output shape: {output.shape}')