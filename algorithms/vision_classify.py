# coding: utf-8
import os
import cv2
import numpy as np
import torch
import random
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from MobileNetV3_Lite import MobileNetV3Lite

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from thop import profile
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import shutil
import json
import pickle

BINARY = True
BINARY_TH = 60
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================== 数据增强函数 ==============================
def apply_data_augmentation(img, is_training=True):
    """
    对单张图像应用数据增强：随机缩放、平移、裁剪
    参数:
        img: 输入图像 (RGB格式)
        is_training: 是否为训练模式 (训练时随机增强，测试时固定增强)
    返回:
        增强后的图像 (RGB格式)
    """
    # 确保输入是numpy数组
    if isinstance(img, str):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB
    
    # 确保图像是RGB格式
    if len(img.shape) == 2:  # 如果是灰度图，转换为RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    h, w = img.shape[:2]
    
    if not is_training:
        return img
    
    # 1. 随机缩放: 缩放因子 ∈ [0.9, 1.1]
    scale_factor = random.uniform(0.9, 1.1)
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    img_scaled = cv2.resize(img, (new_w, new_h))
    
    # 2. 随机平移: 最多±10%的图像尺寸
    max_translate_h = int(0.1 * h)
    max_translate_w = int(0.1 * w)
    translate_h = random.randint(-max_translate_h, max_translate_h)
    translate_w = random.randint(-max_translate_w, max_translate_w)
    
    # 创建平移矩阵
    M = np.float32([[1, 0, translate_w], [0, 1, translate_h]])
    borderValue = (255, 255, 255)  # 白色填充
    # img_translated = cv2.warpAffine(img_scaled, M, (new_w, new_h), borderValue=borderValue)
    img_translated = cv2.warpAffine(img_scaled, M, (new_w, new_h), borderMode=cv2.BORDER_REFLECT_101)
    
    # 3. 随机裁剪: 输出尺寸 = 原图的90%
    crop_h = int(0.9 * h)
    crop_w = int(0.9 * w)
    
    # 确保裁剪区域不超出图像边界
    if crop_h <= new_h and crop_w <= new_w:
        start_h = random.randint(0, new_h - crop_h)
        start_w = random.randint(0, new_w - crop_w)
        img_cropped = img_translated[start_h:start_h + crop_h, start_w:start_w + crop_w]
    else:
        # 如果缩放后的图像小于裁剪尺寸，则调整裁剪尺寸
        crop_h = min(crop_h, new_h)
        crop_w = min(crop_w, new_w)
        start_h = random.randint(0, new_h - crop_h)
        start_w = random.randint(0, new_w - crop_w)
        img_cropped = img_translated[start_h:start_h + crop_h, start_w:start_w + crop_w]
    
    # 将裁剪后的图像调整回原始尺寸
    img_resized = cv2.resize(img_cropped, (w, h))
    
    return img_resized

def generate_augmented_dataset(input_dir, output_dir, num_augmentations=5, skip_existing=False):
    """
    生成增强数据集
    参数:
        input_dir: 原始数据集目录
        output_dir: 增强数据集保存目录
        num_augmentations: 每张原始图像生成的增强图像数量
        skip_existing: 如果输出目录已存在，是否跳过生成
    """
    # 检查输出目录是否已存在
    if skip_existing and os.path.exists(output_dir):
        print(f"增强数据集已存在: {output_dir}，跳过生成...")
        return output_dir
    
    # 获取所有子目录（类别）
    categories = ['negative', 'positive']
    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)
        print(f"创建输入目录: {input_dir}")
        return output_dir
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    for category in categories:
        category_input_dir = os.path.join(input_dir, category)
        category_output_dir = os.path.join(output_dir, category)
        
        # 跳过不存在的类别目录
        if not os.path.exists(category_input_dir):
            print(f"跳过不存在的目录: {category_input_dir}")
            continue
        
        # 创建输出目录
        os.makedirs(category_output_dir, exist_ok=True)
        
        # 获取类别下的所有图像文件
        image_files = [f for f in os.listdir(category_input_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        print(f"处理类别 '{category}': 找到 {len(image_files)} 张图像")
        
        for img_file in tqdm(image_files, desc=f"处理 {category}"):
            img_path = os.path.join(category_input_dir, img_file)
            
            # 读取原始图像
            try:
                original_img = cv2.imread(img_path)
                if original_img is None:
                    print(f"无法读取图像: {img_path}")
                    continue
                
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # 转换为RGB
                
                # 保存原始图像
                original_save_path = os.path.join(category_output_dir, f"orig_{img_file}")
                # 保存为RGB格式
                cv2.imwrite(original_save_path, cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
                
                # 生成增强图像
                for i in range(num_augmentations):
                    # 应用增强
                    augmented_img = apply_data_augmentation(original_img, is_training=True)
                    
                    # 生成增强图像的文件名
                    name_without_ext = os.path.splitext(img_file)[0]
                    ext = os.path.splitext(img_file)[1]
                    aug_filename = f"{name_without_ext}_aug{i:03d}{ext}"
                    aug_save_path = os.path.join(category_output_dir, aug_filename)
                    
                    # 保存增强图像（转换为BGR保存）
                    cv2.imwrite(aug_save_path, cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))
                    
            except Exception as e:
                print(f"处理图像 {img_file} 时出错: {str(e)}")
        
        print(f"类别 '{category}' 处理完成，图像已保存到: {category_output_dir}")
    
    return output_dir

# ============================== 数据集相关函数 ==============================
def normalize_param(root_path = './data/vision/search'):
    '''
    计算normalize的均值和方差
    '''
    # 文件夹列表
    folders = ['negative', 'positive']

    img_list = []   # 初始化图像列表

    # 遍历文件夹
    for folder in folders:
        folder_path = os.path.join(root_path, folder)
        img_path_list = os.listdir(folder_path)

        img_len = len(img_path_list)
        print(f'{folder} img length = {img_len}')

        for item in img_path_list:
            img = cv2.imread(os.path.join(folder_path, item))
            if BINARY:
                # 转为二值化图像
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, binary_img = cv2.threshold(gray_img, BINARY_TH, 255, cv2.THRESH_BINARY)
                img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB
            img = cv2.resize(img, (224, 224))
            # 给三维图片(高度, 宽度, 通道数)后再加一维，变为四维(高度, 宽度, 通道数, 1)
            img = img[:, :, :, np.newaxis]
            img_list.append(img)
        # 沿照片最后一个维度将所有图片拼接为一个大的四维数组(高度, 宽度, 通道数, len)
        imgs = np.concatenate(img_list, axis=-1)
        imgs = imgs.astype(np.float32)/255

    # 初始化均值和方差
    mean_list = []
    std_list = []

    for i in range(3):
        # 对每个通道，将所有像素拉平成一维数组，并计算均值和方差
        pixels = imgs[:, :, i, :].ravel()
        mean_list.append(np.mean(pixels))
        std_list.append(np.std(pixels))
    
    print(f'mean = {mean_list}, std = {std_list}')

    return mean_list, std_list

class CustomDataset(Dataset):
    '''
    自定义数据集
    '''
    def __init__(self, root_path, transform=None, augment=False, binary_threshold=60):
        self.root_path = root_path
        self.transform = transform
        self.augment = augment
        self.binary_threshold = binary_threshold
        self.img_paths = []  # 初始化空列表存储图像路径
        self.labels = []     # 初始化空列表存储标签
        self.class_idx = {'negative': 0, 'positive': 1}    # 定义分类到索引的映射

        # 文件夹列表
        folders = ['negative', 'positive']
        for class_name in folders:
            folder_path = os.path.join(root_path, class_name)
            
            # 跳过不存在的目录
            if not os.path.exists(folder_path):
                print(f"警告: 目录不存在 {folder_path}")
                continue
                
            # 遍历每个分类的文件，收集所有图像路径，并根据分类为每张图像分配标签
            for root, _, fnames in sorted(os.walk(folder_path, followlinks=True)):
                for fname in sorted(fnames):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        path = os.path.join(root, fname)
                        self.img_paths.append(path)
                        self.labels.append(self.class_idx[class_name])
                        
        print(f"数据集加载完成: {len(self.img_paths)} 张图像")

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path)  # cv2读取的图像默认是BGR格式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB

        if img is None:
            raise ValueError(f"Failed to load image at path: {img_path}")
        
        # 应用数据增强（如果启用）
        if self.augment:
            img = apply_data_augmentation(img, is_training=True)
        
        # 应用二值化（如果启用）
        if BINARY:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, binary_img = cv2.threshold(gray_img, self.binary_threshold, 255, cv2.THRESH_BINARY)
            img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2RGB)
        
        if self.transform:
            img = self.transform(img)
        else:
            # 将numpy的(高度, 宽度, 通道)转换为PyTorch的(通道, 高度, 宽度)
            img = img.transpose((2, 0, 1))
            img = torch.from_numpy(img).float()     # 将图像转为张量
        
        return img, label, img_path

# 自定义 collate_fn 来处理额外的文件名数据
def collate_fn(batch):
    images, labels, img_paths = zip(*batch)
    images = torch.stack(images, 0)
    labels = torch.tensor(labels)
    return images, labels, list(img_paths)

# ============================== 主训练函数 ==============================
def main(isSaveModel=False, isSaveData=True, use_augmented_data=False, dataset_path='./data/vision/search',seed=42):
    # 设置随机种子
    set_seed(seed)
    
    # 设置路径
    path = dataset_path
    
    # 如果需要使用增强数据且不跳过增强步骤
    if use_augmented_data:
        print("=" * 50)
        print("生成增强数据集")
        print("=" * 50)
        
        # 生成增强数据集
        augmented_path = generate_augmented_dataset(
            input_dir=path,
            output_dir=path + '_augmented',
            num_augmentations=5,
            skip_existing=True
        )
        
        # 使用增强数据集
        train_val_path = augmented_path
        print(f"使用增强数据集: {train_val_path}")
    else:
        train_val_path = path
        print(f"使用原始数据集: {train_val_path}")
    
    # 获取当前时间戳用于日志
    now = datetime.now()
    timenow = now.strftime("%Y-%m-%d_%H-%M")
    
    if isSaveData:
        log_path = './logs/' + 'MobileNetV3Lite' + timenow
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        writer = SummaryWriter(log_dir=log_path)
        print(f"日志保存到: {log_path}")
    
    # 计算均值和标准差
    dataset_mean, dataset_std = normalize_param(root_path=train_val_path)
    
    # 定义预处理变换
    transform = transforms.Compose([
        transforms.ToPILImage(),  # 将numpy数组或张量转换为PIL图像
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # 将图像转换为tensor，并缩放到[0,1]范围
        transforms.Normalize(mean=dataset_mean, std=dataset_std)  # 标准化
    ])
    
    # 创建数据集
    total_dataset = CustomDataset(
        root_path=train_val_path, 
        transform=transform,
        augment=not use_augmented_data,
        binary_threshold=BINARY_TH
    )
    
    # 划分训练集、测试集、验证集
    val_scale = 0.1
    test_scale = 0.1

    test_size = int(test_scale * len(total_dataset))
    train_val_size = len(total_dataset) - test_size
    test_dataset, train_val_dataset = random_split(total_dataset, [test_size, train_val_size])

    val_size = int(val_scale * len(train_val_dataset))
    train_size = len(train_val_dataset) - val_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
    
    print(f"数据集划分: 训练集 {len(train_dataset)}, 验证集 {len(val_dataset)}, 测试集 {len(test_dataset)}")
    
    # 设置数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=0, drop_last=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=0, drop_last=True, collate_fn=collate_fn)
    
    # 模型初始化
    model = MobileNetV3Lite(num_classes=2).to(device)
    
    # 计算模型参数量和FLOPs
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(model, inputs=(input_tensor,))
    print(f"模型参数: {params/1e6:.2f}M, FLOPs: {flops/1e6:.2f}M")
    
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    # 模型训练
    train_epochs = 30
    t0 = time.time()
    best_val_acc = 0.0
    
    for epoch in range(train_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels, img_paths in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            pred = outputs.argmax(1)
            batch_loss = loss.item()
            batch_correct = (pred == labels).sum().item()
            batch_num = labels.size(0)
            
            running_loss += batch_loss
            total_train += batch_num
            correct_train += batch_correct
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        
        # 验证阶段
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels, img_paths in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
                pred = outputs.argmax(1)
                total_val += labels.size(0)
                correct_val += (pred == labels).sum().item()
        
        val_loss = running_val_loss / len(val_loader)
        val_acc = correct_val / total_val
        
        # 保存最佳模型
        if val_acc > best_val_acc and isSaveModel:
            best_val_acc = val_acc
            best_model_path = f'./model/VisionGuide search/model_search_lite_aug_best.pth'
            # 清理状态字典
            state_dict = model.state_dict()
            # 删除所有包含"total_ops"或"total_params"的键
            keys_to_remove = [key for key in state_dict.keys() if 'total_ops' in key or 'total_params' in key]
            for key in keys_to_remove:
                del state_dict[key]

            # 保存清理后的状态字典
            torch.save(state_dict, best_model_path)    
            print(f"保存最佳模型到: {best_model_path}")
        
        # 记录日志
        if isSaveData:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # 打印进度
        print(f"Epoch [{epoch+1}/{train_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    print(f'总训练时间: {time.time() - t0:.2f}秒')
    print('训练完成!\n')
    
    # 模型测试
    model.eval()
    correct_test = 0
    total_test = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels, img_paths in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            pred = outputs.argmax(1)
            total_test += labels.size(0)
            correct_test += (pred == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
    
    test_acc = correct_test / total_test
    print(f"测试集准确率: {test_acc:.4f}")
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("混淆矩阵:")
    print(cm)
    
    # class_names = ['negative', 'positive']
    
    # # 绘制混淆矩阵
    # fig, ax = plt.subplots(figsize=(12, 8))
    # # im = ax.matshow(cm, cmap=plt.cm.Blues)  # 使用matshow方法显示矩阵
    # im = ax.matshow(cm, cmap=plt.cm.YlGn)  # 使用matshow方法显示矩阵

    # font_size = 30
    # # font_name='Times New Roman'
    # font_name='Arial'


    # # 在每个单元格上添加文本注释
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         value = float(format('%.2f' % cm[i, j]))
    #         if i == j:
    #             color = 'white'  # 对角线上的文字颜色设为白色
    #         else:
    #             color = 'black'  # 非对角线上的文字颜色设为黑色
    #         ax.text(x=j, y=i, s=value, va='center', ha='center',
    #                 fontsize=font_size, fontname=font_name, color=color)

    # # 设置刻度位置和标签
    # tick_marks = np.arange(len(class_names))
    # ax.set_xticks(tick_marks)
    # ax.set_yticks(tick_marks)
    # ax.set_xticklabels(class_names, fontname=font_name, fontsize=font_size)
    # ax.set_yticklabels(class_names, fontname=font_name, fontsize=font_size, rotation=90, ha='right', va='center')

    # plt.xlabel('Predicted Label', fontname=font_name, fontsize=font_size)
    # plt.ylabel('True Label', fontname=font_name, fontsize=font_size)
    # plt.title('MobileNetV3-Lite', fontname=font_name, fontsize=font_size)
    # # plt.title('MobileNetV3-Small', fontname=font_name, fontsize=font_size)


    # cb = plt.colorbar(im)
    # cb.ax.tick_params(labelsize=font_size)  # 设置颜色条上的字体大小
    # for l in cb.ax.yaxis.get_ticklabels():
    #     l.set_family(font_name)  # 设置颜色条上的字体类型

    # # 显示图形
    # plt.savefig("test.png", dpi=500)
    # plt.show()
    
    # plt.show()
    
    # 保存最终模型
    if isSaveModel:
        final_model_path = f'./model/VisionGuide search/model_search_lite_aug.pth'
        # 清理状态字典
        state_dict = model.state_dict()
        # 删除所有包含"total_ops"或"total_params"的键
        keys_to_remove = [key for key in state_dict.keys() if 'total_ops' in key or 'total_params' in key]
        for key in keys_to_remove:
            del state_dict[key]
        
        # 保存清理后的状态字典
        torch.save(state_dict, final_model_path)
        print(f"最终模型保存到: {final_model_path}")
    
    # 关闭TensorBoard写入器
    if isSaveData:
        writer.close()
    
    # return test_acc, best_val_acc, final_model_path if isSaveModel else None
    return test_acc, best_val_acc, best_model_path if isSaveModel else None

# ============================== 鲁棒性测试函数 ==============================
def robustness_test(model_path, dataset_path, test_ratio=0.1, seed=28, isSaveData=True):
    """
    鲁棒性测试：在原始测试集和扰动测试集上评估模型性能
    参数:
        model_path: 模型路径
        dataset_path: 数据集路径
        test_ratio: 测试集比例
        seed: 随机种子
        isSaveData: 是否保存结果
    """
    print("=" * 50)
    print("鲁棒性测试")
    print("=" * 50)
    
    # 设置随机种子
    set_seed(seed)
    
    # 加载模型
    model = MobileNetV3Lite(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    print(f"加载模型 {model_path}")
    # 获取当前时间戳
    now = datetime.now()
    timenow = now.strftime("%Y-%m-%d_%H-%M")
    
    if isSaveData:
        log_path = './logs/robustness_' + timenow
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    
    # 定义评估函数
    def evaluate_model(data_loader, dataset_name=""):
        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for images, labels, _ in data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                pred = outputs.argmax(1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(pred.cpu().numpy())
        
        accuracy = correct / total
        cm = confusion_matrix(all_labels, all_preds)
        
        print(f"{dataset_name}准确率: {accuracy:.4f}")
        print(f"{dataset_name}混淆矩阵:")
        print(cm)
        
        return accuracy, all_labels, all_preds, cm
    
    # 1. 评估原始测试集
    print("\n1. 评估原始测试集...")
    
    # 计算原始数据集的均值和标准差
    orig_dataset_mean, orig_dataset_std = normalize_param(root_path=dataset_path)
    
    # 定义原始数据集的预处理变换
    orig_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=orig_dataset_mean, std=orig_dataset_std)
    ])
    
    # 创建原始数据集
    orig_dataset = CustomDataset(
        root_path=dataset_path,
        transform=orig_transform,
        augment=False,
        binary_threshold=BINARY_TH
    )
    
    # 划分原始训练集和测试集
    test_size = int(test_ratio * len(orig_dataset))
    train_size = len(orig_dataset) - test_size
    _, test_dataset = random_split(orig_dataset, [train_size, test_size])
    
    print(f"原始数据集划分: 测试集 {len(test_dataset)}")
    
    # 创建原始测试集数据加载器
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, 
                            num_workers=0, collate_fn=collate_fn)
    
    # 评估原始测试集
    orig_acc, orig_labels, orig_preds, orig_cm = evaluate_model(test_loader, "原始测试集")
    
    # 2. 评估扰动测试集
    print("\n2. 评估扰动测试集...")
    
    # 生成扰动数据集路径
    perturbed_dataset_path = dataset_path + '_augmented'
    
    # 如果扰动数据集不存在，则生成
    if not os.path.exists(perturbed_dataset_path):
        print(f"生成扰动数据集: {perturbed_dataset_path}")
        generate_augmented_dataset(
            input_dir=dataset_path,
            output_dir=perturbed_dataset_path,
            num_augmentations=1,  # 每个原始图像生成1个扰动版本
            skip_existing=False
        )
    else:
        print(f"使用已存在的扰动数据集: {perturbed_dataset_path}")
    
    # 计算扰动数据集的均值和标准差
    pert_dataset_mean, pert_dataset_std = normalize_param(root_path=perturbed_dataset_path)
    
    # 定义扰动数据集的预处理变换
    pert_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=pert_dataset_mean, std=pert_dataset_std)
    ])
    
    # 创建扰动数据集
    pert_dataset = CustomDataset(
        root_path=perturbed_dataset_path,
        transform=pert_transform,
        augment=False,  # 扰动数据集已经增强过了，不再进行在线增强
        binary_threshold=BINARY_TH
    )
    
    # 划分扰动训练集和测试集
    set_seed(seed)
    pert_test_size = int(test_ratio * len(pert_dataset))
    pert_train_size = len(pert_dataset) - pert_test_size
    _, pert_test_dataset = random_split(pert_dataset, [pert_train_size, pert_test_size])
    
    print(f"扰动数据集划分: 测试集 {len(pert_test_dataset)}")
    
    # 创建扰动测试集数据加载器
    pert_test_loader = DataLoader(pert_test_dataset, batch_size=32, shuffle=False, 
                                 num_workers=0, collate_fn=collate_fn)
    
    # 评估扰动测试集
    pert_acc, pert_labels, pert_preds, pert_cm = evaluate_model(pert_test_loader, "扰动测试集")
    
    # 计算性能下降
    performance_drop = (orig_acc - pert_acc) * 100
    print(f"\n性能下降: {performance_drop:.2f}%")
    
    # 保存结果
    if isSaveData:
        results = {
            'original_accuracy': float(orig_acc),
            'perturbed_accuracy': float(pert_acc),
            'performance_drop': float(performance_drop),
            'original_test_size': len(test_dataset),
            'perturbed_test_size': len(pert_test_dataset),
            'random_seed': seed,
            'binary_threshold': BINARY_TH,
            'binary_enabled': BINARY,
            'data_augmentation_params': {
                'scaling_range': [0.9, 1.1],
                'translation_range': 0.1,
                'crop_ratio': 0.9
            }
        }
        
        # 保存为JSON
        results_path = os.path.join(log_path, 'robustness_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"结果保存到: {results_path}")
    
    return orig_acc, pert_acc, performance_drop

# ============================== 主程序入口 ==============================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='视觉特征提取实验')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'robustness'],
                       help='运行模式: train(训练) 或 robustness(鲁棒性测试)')
    parser.add_argument('--use_augmented_data', action='store_true', default=True,
                       help='使用增强数据集')
    parser.add_argument('--save_model', action='store_true', default=True,
                       help='保存模型')
    parser.add_argument('--save_data', action='store_true', default=False,
                       help='保存训练日志和数据')
    parser.add_argument('--model_path', type=str, default='',
                       help='用于鲁棒性测试的模型路径')
    parser.add_argument('--dataset_path', type=str, default='./data/vision/search',
                       help='数据集路径')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='测试集比例')
    parser.add_argument('--seed', type=int, default=28,
                       help='随机种子')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # 训练模式
        test_acc, best_val_acc, model_path = main(
            isSaveModel=args.save_model,
            isSaveData=args.save_data,
            use_augmented_data=args.use_augmented_data,
            dataset_path=args.dataset_path,
            seed=args.seed
        )
        
        print(f"\n训练结果:")
        print(f"最佳验证集准确率: {best_val_acc:.4f}")
        print(f"测试集准确率: {test_acc:.4f}")
        
        if args.save_model and model_path:
            # 自动进行鲁棒性测试
            print("\n自动进行鲁棒性测试...")
            orig_acc, pert_acc, performance_drop = robustness_test(
                model_path=model_path,
                dataset_path=args.dataset_path,
                test_ratio=args.test_ratio,
                seed=args.seed,
                isSaveData=args.save_data
            )
            
    elif args.mode == 'robustness':
        # 鲁棒性测试模式
        if not args.model_path:
            print("错误: 鲁棒性测试需要指定模型路径 (--model_path)")
        else:
            orig_acc, pert_acc, performance_drop = robustness_test(
                model_path=args.model_path,
                dataset_path=args.dataset_path,
                test_ratio=args.test_ratio,
                seed=args.seed,
                isSaveData=args.save_data
            )
            
            print(f"\n鲁棒性测试结果:")
            print(f"原始测试集准确率: {orig_acc:.4f}")
            print(f"扰动测试集准确率: {pert_acc:.4f}")
            print(f"性能下降: {performance_drop:.2f}%")