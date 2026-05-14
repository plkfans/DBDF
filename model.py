import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    vgg11_bn, VGG11_BN_Weights,
    efficientnet_b3, EfficientNet_B3_Weights
)
from utils import generate_local_images
from args import args_parser

args = args_parser()
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    def __init__(self, arch, n_classes, mode):
        super(Model, self).__init__()
        self.mode = mode

        if arch == 'resnet18':
            self.feature_dim = 512  
            self.global_model = nn.Sequential(*list(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[:-2])
            self.local_model = nn.Sequential(*list(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[:-2])

        elif arch == 'vgg11':
            self.feature_dim = 512  
            self.global_model = vgg11_bn(weights=VGG11_BN_Weights.IMAGENET1K_V1).features
            self.local_model = vgg11_bn(weights=VGG11_BN_Weights.IMAGENET1K_V1).features

        elif arch == 'efficientnet_b3':
            self.feature_dim = 1536 
            self.global_model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1).features
            self.local_model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1).features

        elif arch == 'resnet50':
            self.feature_dim = 2048
            self.global_model = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).children())[:-2])
            self.local_model = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).children())[:-2])


        # 全局分类器
        self.fc_g = nn.Linear(self.feature_dim, n_classes)
        # 本地分类器
        self.fc_l = nn.Linear(self.feature_dim, n_classes)
        # ensemble 分类器
        self.fc_en = nn.Linear(self.feature_dim*2, n_classes)

        # 池化层
        self.pooling = nn.AdaptiveAvgPool2d(1)
        

    def get_parameters(self):
        if self.mode == 'va':
            # 仅训练 global_model 和 fc_g
            for param in self.local_model.parameters():
                param.requires_grad = False
            for param in self.fc_l.parameters():
                param.requires_grad = False
            for param in self.fc_en.parameters():
                param.requires_grad = False
            params = list(self.global_model.parameters()) + list(self.fc_g.parameters())

        elif self.mode == 'kd':
            # 全部训练
            params = list(self.global_model.parameters()) + \
                     list(self.local_model.parameters()) + \
                     list(self.fc_g.parameters()) + \
                     list(self.fc_l.parameters()) + \
                     list(self.fc_en.parameters())
        return params

    def forward(self, img_s1, visualization=False):
        img_b = img_s1.size(0)

        if self.mode == 'va':
            feat_g = self.global_model(img_s1)
            vector_g = self.pooling(feat_g).view(feat_g.size(0), -1)
            logits = self.fc_g(vector_g)
            return logits

        elif self.mode == 'kd':
            # Global branch
            feat_g = self.global_model(img_s1)
            vector_g = self.pooling(feat_g).view(feat_g.size(0), -1)
            logits_g = self.fc_g(vector_g)

            # Generate local images
            with torch.no_grad():
                local_imgs, masks, masks_new, bboxes = generate_local_images(
                    img_s1, feat_g, out_size=args.img_size, return_bbox=True
                )

            # Local branch
            feat_l = self.local_model(local_imgs) 
            vector_l = self.pooling(feat_l).view(feat_l.size(0), -1)
            logits_l = self.fc_l(vector_l)

            # Ensemble branch
            feat_en = torch.cat([vector_g, vector_l], dim=1)
            logits_en = self.fc_en(feat_en)

            logits = [logits_g, logits_l, logits_en]

            return logits, [local_imgs, masks, masks_new, bboxes]