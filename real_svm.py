"""
real_svm.py
SVM识别器 - 同时使用合成和真实字符数据训练
使用HOG特征 + SVM分类器
作者：车牌识别系统开发团队
"""

import cv2  # 导入OpenCV库，用于图像处理和HOG特征提取
import numpy as np  # 导入NumPy库，用于数值计算和数组操作
import joblib  # 导入joblib库，用于保存和加载机器学习模型
from pathlib import Path  # 导入Path类，用于处理文件路径
from sklearn.svm import SVC  # 导入SVM分类器
from sklearn.preprocessing import LabelEncoder  # 导入标签编码器，将字符标签转换为数字
from sklearn.model_selection import train_test_split  # 导入数据划分函数
from sklearn.metrics import accuracy_score  # 导入准确率计算函数
import json  # 导入json库，用于保存配置文件
import os  # 导入os库，用于操作系统相关功能


class RealSVMRecognizer:
    """
    基于SVM的字符识别器
    同时使用合成和真实字符数据进行训练
    策略：优先使用真实数据，合成数据作为补充
    """

    def __init__(self, config=None):
        """
        初始化SVM识别器

        参数：
            config: 配置对象，包含路径和参数设置
        返回：
            无
        """
        # 判断是否传入配置对象
        if config:  # 如果传入了配置对象
            self.config = config  # 使用传入的配置对象
        else:  # 如果没有传入配置对象
            # 从同级目录的all3.py文件导入Config类
            from all3 import Config  # 导入Config类
            self.config = Config() if callable(Config) else Config  # 创建配置实例或直接使用

        # 初始化HOG描述符参数
        # HOG参数说明：
        # _winSize: 窗口大小（字符图像尺寸）
        # _blockSize: 块大小（16x16像素）
        # _blockStride: 块步长（8x8像素，块重叠一半）
        # _cellSize: 细胞大小（8x8像素）
        # _nbins: 方向bin数量（9个方向）
        self.hog = cv2.HOGDescriptor(
            _winSize=(self.config.CHAR_WIDTH, self.config.CHAR_HEIGHT),  # 字符标准尺寸
            _blockSize=(16, 16),  # 块大小
            _blockStride=(8, 8),  # 块移动步长
            _cellSize=(8, 8),  # 细胞大小
            _nbins=9  # 方向直方图bin数量
        )

        # 初始化SVM模型
        self.model = None  # SVM模型，初始为空
        self.label_encoder = LabelEncoder()  # 创建标签编码器对象
        self.char_list = []  # 存储所有字符类别列表

        # 模型文件路径
        self.model_dir = Path("models")  # 模型保存目录（相对路径）
        self.model_path = self.model_dir / "svm_model.pkl"  # 模型文件路径
        self.label_path = self.model_dir / "svm_labels.json"  # 标签文件路径

        # 确保模型目录存在
        self.model_dir.mkdir(exist_ok=True)  # 创建模型目录（如果不存在）

        # 打印初始化信息
        print("✓ SVM识别器初始化完成")
        print(f"  HOG特征维度: {self.get_hog_dimension()}")  # 显示HOG特征维度
        print(f"  模型保存路径: {self.model_path}")  # 显示模型保存路径

    def get_hog_dimension(self):
        """
        计算HOG特征维度

        返回：
            int: HOG特征向量的长度
        """
        # 创建一个测试图像来计算HOG维度
        test_img = np.zeros((self.config.CHAR_HEIGHT, self.config.CHAR_WIDTH), dtype=np.uint8)  # 创建全黑测试图像
        features = self.hog.compute(test_img)  # 计算HOG特征
        return features.shape[0] if features is not None else 0  # 返回特征维度

    def preprocess_image(self, img):
        """
        预处理字符图像

        参数：
            img: 输入图像（可以是彩色或灰度）
        返回：
            np.ndarray: 预处理后的灰度图像
        """
        # 检查图像维度
        if len(img.shape) == 3:  # 如果是彩色图像（3通道）
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
        else:  # 如果是灰度图像
            gray = img.copy()  # 直接复制

        # 二值化处理
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 使用Otsu自动阈值

        # 确保黑底白字（与模板一致）
        if np.mean(binary) > 127:  # 如果图像大部分是白色（白底黑字）
            binary = 255 - binary  # 颜色反转（变为黑底白字）

        # 调整到标准尺寸
        resized = cv2.resize(binary, (self.config.CHAR_WIDTH, self.config.CHAR_HEIGHT))  # 调整到标准大小

        # 归一化到0-1范围
        normalized = resized.astype(np.float32) / 255.0  # 归一化

        # 转换为uint8（OpenCV需要）
        final_img = (normalized * 255).astype(np.uint8)  # 转换回uint8

        return final_img  # 返回预处理后的图像

    def extract_hog_features(self, img):
        """
        提取HOG特征

        参数：
            img: 输入图像
        返回：
            np.ndarray: HOG特征向量
        """
        # 预处理图像
        processed_img = self.preprocess_image(img)  # 调用预处理函数

        # 计算HOG特征
        features = self.hog.compute(processed_img)  # 计算HOG特征

        # 展平特征向量
        if features is not None:  # 如果成功提取特征
            features = features.flatten()  # 展平为1维向量
        else:  # 如果提取失败
            features = np.zeros(self.get_hog_dimension())  # 创建零向量

        return features  # 返回特征向量

    def load_synthetic_data(self):
        """
        加载合成字符数据

        返回：
            tuple: (特征矩阵, 标签列表)
        """
        print("正在加载合成字符数据...")  # 打印进度信息

        X_synth = []  # 合成数据特征列表
        y_synth = []  # 合成数据标签列表
        char_counts = {}  # 字符计数字典

        # 合成数据目录（相对路径）
        synth_dir = Path("synthetic_chars")  # 合成数据目录路径

        # 定义字符类型和对应的目录
        char_categories = [
            ("provinces", "provinces"),  # 省份汉字
            ("letters", "letters"),  # 字母
            ("digits", "digits")  # 数字
        ]

        # 遍历所有字符类别
        for char_type, dir_name in char_categories:  # 遍历字符类别
            type_dir = synth_dir / dir_name  # 构建类型目录路径

            if not type_dir.exists():  # 如果目录不存在
                print(f"  警告：合成数据目录不存在 {type_dir}")  # 打印警告
                continue  # 跳过当前类型

            # 遍历字符目录
            for char_folder in type_dir.iterdir():  # 遍历字符文件夹
                if char_folder.is_dir():  # 如果是目录
                    # 获取字符标签
                    if char_type == "provinces":  # 如果是省份汉字
                        # 从拼音找对应的汉字
                        char_label = self.get_chinese_from_pinyin(char_folder.name)  # 获取汉字
                    else:  # 如果是字母或数字
                        char_label = char_folder.name  # 直接使用文件夹名作为标签

                    if not char_label:  # 如果找不到对应汉字
                        continue  # 跳过

                    # 加载字符图像
                    image_files = list(char_folder.glob("*.png"))  # 查找所有png文件

                    # 限制每个字符的最大样本数（防止数据不平衡）
                    max_samples = 100  # 每个字符最多使用100个样本
                    sample_count = 0  # 样本计数器

                    for img_file in image_files[:max_samples]:  # 遍历图像文件
                        # 读取图像
                        img = cv2.imread(str(img_file))  # 读取图像文件
                        if img is None:  # 如果读取失败
                            continue  # 跳过

                        # 提取HOG特征
                        features = self.extract_hog_features(img)  # 提取特征

                        # 添加到数据集
                        X_synth.append(features)  # 添加特征
                        y_synth.append(char_label)  # 添加标签

                        # 更新计数
                        sample_count += 1  # 样本计数加1
                        char_counts[char_label] = char_counts.get(char_label, 0) + 1  # 更新字符计数

                    if sample_count > 0:  # 如果有加载到样本
                        print(f"   加载 {char_label}: {sample_count} 个样本")  # 打印加载信息

        # 转换数据格式
        X_synth_array = np.array(X_synth) if X_synth else np.array([])  # 转换为NumPy数组
        y_synth_array = np.array(y_synth) if y_synth else np.array([])  # 转换为NumPy数组

        print(f"  ✓ 合成数据加载完成: {len(y_synth_array)} 个样本")  # 打印完成信息
        print(f"    字符类别数: {len(set(y_synth))}")  # 打印字符类别数

        return X_synth_array, y_synth_array  # 返回合成数据

    def load_real_data(self):
        """
        加载真实字符数据 - 支持多层目录结构

        返回：
            tuple: (特征矩阵, 标签列表)
        """
        print("正在加载真实字符数据...")  # 打印进度信息

        X_real = []  # 真实数据特征列表
        y_real = []  # 真实数据标签列表
        char_counts = {}  # 字符计数字典
        total_images = 0  # 总图像计数

        # 真实数据目录（相对路径）
        real_dir = Path("real_train_chars")  # 真实数据目录路径

        if not real_dir.exists():  # 如果目录不存在
            print(f"  警告：真实数据目录不存在 {real_dir}")  # 打印警告
            return np.array([]), np.array([])  # 返回空数据

        print(f"  真实数据目录: {real_dir.absolute()}")  # 显示绝对路径

        # 定义支持的图像扩展名
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']

        # 多层目录结构：real_train_chars/{letters,digits,provinces}/{字符}/{图片}
        for category_dir in real_dir.iterdir():  # 遍历分类目录（letters, digits, provinces）
            if not category_dir.is_dir():  # 如果不是目录
                continue  # 跳过

            category_name = category_dir.name  # 分类名称
            print(f"  处理分类: {category_name}")  # 显示分类名称

            # 遍历字符目录
            for char_dir in category_dir.iterdir():  # 遍历字符文件夹
                if not char_dir.is_dir():  # 如果不是目录
                    continue  # 跳过

                # 获取字符标签（需要根据目录名转换）
                char_label = self.get_char_label_from_dirname(char_dir.name, category_name)

                if not char_label:  # 如果无法获取有效标签
                    continue  # 跳过

                # 查找所有支持的图像文件
                image_files = []
                for ext in image_extensions:
                    image_files.extend(char_dir.glob(f"*{ext}"))
                    image_files.extend(char_dir.glob(f"*{ext.upper()}"))

                if not image_files:  # 如果没有找到图像文件
                    continue  # 跳过

                print(f"    字符 {char_label}: 找到 {len(image_files)} 个图像")

                # 限制每个字符的最大样本数
                max_samples = 50  # 每个真实字符最多使用50个样本
                loaded_count = 0  # 已加载计数

                for img_file in image_files[:max_samples]:  # 遍历图像文件
                    # 读取图像
                    img = cv2.imread(str(img_file))  # 读取图像文件
                    if img is None:  # 如果读取失败
                        print(f"      警告：无法读取 {img_file.name}")  # 打印警告
                        continue  # 跳过

                    # 提取HOG特征
                    features = self.extract_hog_features(img)  # 提取特征

                    # 添加到数据集
                    X_real.append(features)  # 添加特征
                    y_real.append(char_label)  # 添加标签

                    # 更新计数
                    loaded_count += 1  # 加载计数加1
                    total_images += 1  # 总图像计数加1
                    char_counts[char_label] = char_counts.get(char_label, 0) + 1  # 更新字符计数

                print(f"      已加载: {loaded_count} 个样本")  # 显示加载数量

        # 转换数据格式
        X_real_array = np.array(X_real) if X_real else np.array([])  # 转换为NumPy数组
        y_real_array = np.array(y_real) if y_real else np.array([])  # 转换为NumPy数组

        print(f"  ✓ 真实数据加载完成: {len(y_real_array)} 个样本")  # 打印完成信息
        print(f"    总图像数: {total_images}")  # 打印总图像数
        print(f"    字符类别数: {len(set(y_real)) if y_real else 0}")  # 打印字符类别数
        print(f"    字符分布: {char_counts}")  # 打印字符分布

        return X_real_array, y_real_array  # 返回真实数据

    def get_char_label_from_dirname(self, dir_name, category_name):
        """
        根据目录名和分类名获取字符标签

        参数：
            dir_name: 目录名称
            category_name: 分类名称（letters, digits, provinces）
        返回：
            str: 字符标签
        """
        if category_name == "provinces":  # 如果是省份分类
            # 处理拼音目录名（如FUJIAN）转换为汉字
            # 这里需要根据实际的拼音-汉字映射来处理
            pinyin_map = {
                'FUJIAN': '闽', 'ZHEJIANG': '浙', 'JIANGSU': '苏',
                'BEIJING': '京', 'SHANGHAI': '沪', 'TIANJIN': '津',
                'CHONGQING': '渝', 'HEBEI': '冀', 'SHANXI': '晋',
                'LIAONING': '辽', 'JILIN': '吉', 'HEILONGJIANG': '黑',
                'ANHUI': '皖', 'FUJIAN': '闽', 'JIANGXI': '赣',
                'SHANDONG': '鲁', 'HENAN': '豫', 'HUBEI': '鄂',
                'HUNAN': '湘', 'GUANGDONG': '粤', 'GUANGXI': '桂',
                'HAINAN': '琼', 'SICHUAN': '川', 'GUIZHOU': '贵',
                'YUNNAN': '云', 'SHAANXI': '陕', 'GANSU': '甘',
                'QINGHAI': '青', 'NINGXIA': '宁', 'XINJIANG': '新'
            }
            # 尝试多种可能的格式
            if dir_name in pinyin_map:  # 如果是全大写拼音
                return pinyin_map[dir_name]
            elif dir_name.upper() in pinyin_map:  # 尝试大写转换
                return pinyin_map[dir_name.upper()]
            else:  # 如果找不到映射，尝试使用拼音到汉字的通用转换
                chinese = self.get_chinese_from_pinyin(dir_name.lower())
                return chinese if chinese else dir_name  # 返回汉字或原目录名
        elif category_name == "letters":  # 如果是字母分类
            # 字母目录可能是单个字母，也可能是字母组合
            if len(dir_name) == 1 and dir_name.isalpha():  # 如果是单个字母
                return dir_name.upper()  # 返回大写字母
            else:  # 如果是其他格式
                # 尝试从文件名中提取字母
                import re
                match = re.search(r'[A-Za-z]', dir_name)  # 查找字母
                if match:  # 如果找到字母
                    return match.group().upper()  # 返回大写字母
                else:  # 如果没有找到字母
                    return dir_name  # 返回原目录名
        elif category_name == "digits":  # 如果是数字分类
            # 数字目录可能是单个数字，也可能是数字组合
            if len(dir_name) == 1 and dir_name.isdigit():  # 如果是单个数字
                return dir_name  # 返回数字
            else:  # 如果是其他格式
                # 尝试从文件名中提取数字
                import re
                match = re.search(r'\d', dir_name)  # 查找数字
                if match:  # 如果找到数字
                    return match.group()  # 返回数字
                else:  # 如果没有找到数字
                    return dir_name  # 返回原目录名
        else:  # 其他分类
            return dir_name  # 直接返回目录名

    def get_chinese_from_pinyin(self, pinyin):
        """
        根据拼音找到对应的汉字

        参数：
            pinyin: 拼音字符串
        返回：
            str: 对应的汉字，如果找不到返回原拼音
        """
        # 拼音到汉字的映射（常用省份简称）
        pinyin_to_chinese = {
            'bei': '京', 'hu': '沪', 'jin': '津', 'yu': '渝',
            'ji': '冀', 'jin': '晋', 'liao': '辽', 'ji': '吉',
            'hei': '黑', 'su': '苏', 'zhe': '浙', 'wan': '皖',
            'min': '闽', 'gan': '赣', 'lu': '鲁', 'yu': '豫',
            'e': '鄂', 'xiang': '湘', 'yue': '粤', 'gui': '桂',
            'qiong': '琼', 'chuan': '川', 'gui': '贵', 'yun': '云',
            'zang': '藏', 'shan': '陕', 'gan': '甘', 'qing': '青',
            'ning': '宁', 'xin': '新',
            # 全拼映射
            'fujian': '闽', 'zhejiang': '浙', 'jiangsu': '苏',
            'beijing': '京', 'shanghai': '沪', 'tianjin': '津',
            'chongqing': '渝', 'hebei': '冀', 'shanxi': '晋',
            'liaoning': '辽', 'jilin': '吉', 'heilongjiang': '黑',
            'anhui': '皖', 'jiangxi': '赣', 'shandong': '鲁',
            'henan': '豫', 'hubei': '鄂', 'hunan': '湘',
            'guangdong': '粤', 'guangxi': '桂', 'hainan': '琼',
            'sichuan': '川', 'guizhou': '贵', 'yunnan': '云',
            'shaanxi': '陕', 'gansu': '甘', 'qinghai': '青',
            'ningxia': '宁', 'xinjiang': '新'
        }

        # 查找对应的汉字
        pinyin_lower = pinyin.lower()  # 转换为小写
        if pinyin_lower in pinyin_to_chinese:  # 如果找到映射
            return pinyin_to_chinese[pinyin_lower]  # 返回汉字
        else:  # 如果找不到
            # 尝试部分匹配
            for key, value in pinyin_to_chinese.items():
                if pinyin_lower.startswith(key):  # 如果拼音以某个键开头
                    return value  # 返回对应的汉字

            return None  # 返回None

    def combine_datasets(self, X1, y1, X2, y2):
        """
        合并两个数据集

        参数：
            X1, y1: 第一个数据集（真实数据）
            X2, y2: 第二个数据集（合成数据）
        返回：
            tuple: (合并后的特征矩阵, 合并后的标签数组)
        """
        # 检查数据集是否为空
        if X1.size == 0 and X2.size == 0:  # 如果两个都为空
            return np.array([]), np.array([])  # 返回空数组

        if X1.size == 0:  # 如果第一个为空
            return X2, y2  # 返回第二个

        if X2.size == 0:  # 如果第二个为空
            return X1, y1  # 返回第一个

        # 合并数据集
        X_combined = np.vstack((X1, X2))  # 垂直堆叠特征矩阵
        y_combined = np.concatenate((y1, y2))  # 连接标签数组

        return X_combined, y_combined  # 返回合并后的数据

    def prepare_training_data(self):
        """
        准备训练数据（合并合成和真实数据）

        返回：
            tuple: (训练特征, 测试特征, 训练标签, 测试标签)
        """
        # 加载数据
        X_real, y_real = self.load_real_data()  # 加载真实数据
        X_synth, y_synth = self.load_synthetic_data()  # 加载合成数据

        # 合并数据集（真实数据在前，合成数据在后）
        X_all, y_all = self.combine_datasets(X_real, y_real, X_synth, y_synth)  # 合并数据

        if X_all.size == 0:  # 如果没有数据
            raise ValueError("错误：没有找到任何训练数据！")  # 抛出异常

        # 打印数据统计
        print("=" * 50)  # 分隔线
        print("数据统计：")  # 标题
        print(f"  总样本数: {len(y_all)}")  # 总样本数
        print(f"  真实数据: {len(y_real)} 个样本")  # 真实数据数量
        print(f"  合成数据: {len(y_synth)} 个样本")  # 合成数据数量
        print(f"  字符类别: {len(set(y_all))} 类")  # 字符类别数

        # 对标签进行编码（转换为数字）
        self.label_encoder.fit(y_all)  # 拟合标签编码器
        y_encoded = self.label_encoder.transform(y_all)  # 转换标签

        # 保存字符列表
        self.char_list = self.label_encoder.classes_.tolist()  # 保存字符类别列表

        # 划分训练集和测试集
        # test_size=0.2 表示20%的数据作为测试集
        # random_state=42 确保每次划分结果一致
        # stratify=y_encoded 按类别分层抽样，保持类别比例
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        print(f"  训练集: {len(X_train)} 个样本")  # 打印训练集大小
        print(f"  测试集: {len(X_test)} 个样本")  # 打印测试集大小

        return X_train, X_test, y_train, y_test  # 返回划分好的数据

    def train(self, save_model=True):
        """
        训练SVM模型

        参数：
            save_model: 是否保存训练好的模型
        返回：
            float: 模型在测试集上的准确率
        """
        print("=" * 50)  # 分隔线
        print("开始训练SVM模型...")  # 开始训练

        # 准备数据
        try:  # 尝试准备数据
            X_train, X_test, y_train, y_test = self.prepare_training_data()  # 准备训练数据
        except ValueError as e:  # 如果发生错误
            print(f"✗ {e}")  # 打印错误信息
            print("请确保：")  # 提示信息
            print("  1. synthetic_chars/ 目录包含合成字符数据")  # 提示1
            print("  2. real_train_chars/ 目录包含真实字符数据")  # 提示2
            print("  真实数据目录结构应为：")  # 提示目录结构
            print("    real_train_chars/")  # 根目录
            print("    ├── letters/        # 字母分类")  # 字母分类
            print("    │   ├── A/          # 字母A")  # 字母A
            print("    │   │   ├── *.png   # 图像文件")  # 图像文件
            print("    │   │   └── *.jpg")  # 图像文件
            print("    │   ├── B/")  # 字母B
            print("    │   └── ...")  # 其他
            print("    ├── digits/         # 数字分类")  # 数字分类
            print("    │   ├── 0/")  # 数字0
            print("    │   ├── 1/")  # 数字1
            print("    │   └── ...")  # 其他
            print("    └── provinces/      # 省份分类")  # 省份分类
            print("        ├── FUJIAN/     # 福建（拼音）")  # 省份
            print("        ├── ZHEJIANG/   # 浙江（拼音）")  # 省份
            print("        └── ...")  # 其他
            return 0.0  # 返回0准确率

        # 创建SVM模型
        # 参数说明：
        # C=1.0: 正则化参数
        # kernel='linear': 使用线性核函数（适合高维数据）
        # probability=True: 启用概率估计
        # random_state=42: 随机种子
        self.model = SVC(C=1.0, kernel='linear', probability=True, random_state=42)  # 创建SVM模型

        # 训练模型
        print("正在训练模型...")  # 打印训练信息
        self.model.fit(X_train, y_train)  # 训练模型
        print("✓ 模型训练完成")  # 打印完成信息

        # 评估模型
        print("正在评估模型...")  # 打印评估信息
        y_pred = self.model.predict(X_test)  # 在测试集上预测
        accuracy = accuracy_score(y_test, y_pred)  # 计算准确率
        print(f"✓ 测试集准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")  # 打印准确率

        # 保存模型
        if save_model:  # 如果需要保存模型
            self.save_model()  # 调用保存模型方法

        return accuracy  # 返回准确率

    def save_model(self):
        """
        保存训练好的模型和标签编码器
        """
        # 检查模型是否已训练
        if self.model is None:  # 如果模型不存在
            print("✗ 错误：没有训练好的模型可以保存")  # 打印错误信息
            return  # 直接返回

        print("正在保存模型...")  # 打印保存信息

        # 保存SVM模型
        joblib.dump(self.model, self.model_path)  # 保存模型到文件

        # 保存标签信息
        label_info = {  # 创建标签信息字典
            'char_list': self.char_list,  # 字符列表
            'classes': self.label_encoder.classes_.tolist()  # 编码器类别
        }

        with open(self.label_path, 'w', encoding='utf-8') as f:  # 打开文件
            json.dump(label_info, f, ensure_ascii=False, indent=2)  # 保存JSON

        print(f"✓ 模型已保存到 {self.model_path}")  # 打印保存成功信息
        print(f"✓ 标签信息已保存到 {self.label_path}")  # 打印标签保存信息

    def load_model(self):
        """
        加载已训练的模型
        """
        print("正在加载模型...")  # 打印加载信息

        # 检查模型文件是否存在
        if not self.model_path.exists():  # 如果模型文件不存在
            print(f"✗ 错误：模型文件不存在 {self.model_path}")  # 打印错误信息
            print("请先运行 train() 方法训练模型")  # 提示训练
            return False  # 返回失败

        # 加载SVM模型
        self.model = joblib.load(self.model_path)  # 加载模型

        # 加载标签信息
        if self.label_path.exists():  # 如果标签文件存在
            with open(self.label_path, 'r', encoding='utf-8') as f:  # 打开文件
                label_info = json.load(f)  # 加载JSON
            self.char_list = label_info.get('char_list', [])  # 获取字符列表
            classes = label_info.get('classes', [])  # 获取类别列表

            # 重新创建标签编码器
            self.label_encoder = LabelEncoder()  # 创建新编码器
            self.label_encoder.classes_ = np.array(classes)  # 设置类别

        print("✓ 模型加载完成")  # 打印完成信息
        print(f"  支持的字符: {len(self.char_list)} 个")  # 打印字符数量
        return True  # 返回成功

    def predict_char(self, char_img, position=None):
        """
        识别单个字符

        参数：
            char_img: 字符图像
            position: 字符位置（1-7），用于约束字符类型
        返回：
            dict: 识别结果，包含字符、置信度等信息
        """
        # 检查模型是否已加载
        if self.model is None:  # 如果模型不存在
            if not self.load_model():  # 尝试加载模型
                return {'char': '?', 'confidence': 0.0, 'method': 'svm'}  # 返回未知字符

        # 提取特征
        features = self.extract_hog_features(char_img)  # 提取HOG特征
        features = features.reshape(1, -1)  # 调整形状为(1, n_features)

        # 预测
        proba = self.model.predict_proba(features)[0]  # 获取概率预测
        pred_idx = np.argmax(proba)  # 获取最大概率索引
        confidence = proba[pred_idx]  # 获取置信度

        # 解码预测结果
        char = self.label_encoder.inverse_transform([pred_idx])[0]  # 转换为字符

        # 根据位置约束结果
        if position is not None:  # 如果有位置信息
            char = self.constrain_by_position(char, position, confidence)  # 约束字符类型

        # 获取top5结果
        top5_indices = np.argsort(proba)[-5:][::-1]  # 获取top5索引
        top5_chars = self.label_encoder.inverse_transform(top5_indices)  # 转换为字符
        top5_scores = proba[top5_indices]  # 获取对应的概率

        top5 = [  # 创建top5列表
            {'char': char, 'score': float(score)}  # 每个结果包含字符和分数
            for char, score in zip(top5_chars, top5_scores)  # 遍历字符和分数
        ]

        return {  # 返回结果字典
            'char': char,  # 识别出的字符
            'confidence': float(confidence),  # 置信度
            'method': 'svm',  # 识别方法
            'top5': top5  # top5结果
        }

    def constrain_by_position(self, char, position, confidence):
        """
        根据字符位置约束字符类型

        参数：
            char: 当前识别的字符
            position: 字符位置（1-7）
            confidence: 当前置信度
        返回：
            str: 约束后的字符
        """
        # 位置规则：
        # 位置1：只能是省份汉字
        # 位置2：只能是字母
        # 位置3-7：可以是字母或数字

        # 定义字符集（简化版，可根据实际配置扩展）
        provinces = ['京', '沪', '津', '渝', '冀', '晋', '辽', '吉', '黑', '苏', '浙',
                     '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤', '桂', '琼', '川',
                     '贵', '云', '藏', '陕', '甘', '青', '宁', '新']

        letters = [chr(i) for i in range(65, 91)]  # A-Z

        digits = [str(i) for i in range(10)]  # 0-9

        # 检查字符是否在允许的集合中
        if position == 1:  # 位置1：省份汉字
            if char not in provinces:  # 如果不是省份汉字
                # 在字母和数字中寻找最相似的
                if char in letters or char in digits:  # 如果是字母或数字
                    return '?'  # 返回未知
        elif position == 2:  # 位置2：字母
            if char not in letters:  # 如果不是字母
                if char in provinces:  # 如果是省份汉字
                    return '?'  # 返回未知
        else:  # 位置3-7：字母或数字
            if char not in letters and char not in digits:  # 如果不是字母也不是数字
                if char in provinces:  # 如果是省份汉字
                    return '?'  # 返回未知

        return char  # 返回原始字符

    def recognize_plate(self, char_images):
        """
        识别整个车牌

        参数：
            char_images: 字符图像列表（7个字符）
        返回：
            tuple: (车牌字符串, 平均置信度)
        """
        if len(char_images) != 7:  # 如果不是7个字符
            print(f"警告：期望7个字符，得到{len(char_images)}个")  # 打印警告

        plate_number = ""  # 车牌号码字符串
        confidences = []  # 置信度列表

        for i, char_img in enumerate(char_images):  # 遍历字符图像
            position = i + 1  # 字符位置（1-7）
            result = self.predict_char(char_img, position)  # 识别单个字符

            plate_number += result['char']  # 添加到车牌字符串
            confidences.append(result['confidence'])  # 添加置信度

        avg_confidence = np.mean(confidences) if confidences else 0.0  # 计算平均置信度

        return plate_number, avg_confidence  # 返回车牌和平均置信度

    def print_info(self):
        """
        打印模型信息
        """
        print("=" * 50)  # 分隔线
        print("SVM识别器信息：")  # 标题
        print(f"  模型状态: {'已训练' if self.model is not None else '未训练'}")  # 模型状态
        print(f"  支持的字符数: {len(self.char_list)}")  # 字符数量
        print(f"  HOG特征维度: {self.get_hog_dimension()}")  # HOG维度
        print(f"  模型文件: {self.model_path}")  # 模型文件路径


# 测试代码
if __name__ == "__main__":
    """
    主函数：测试SVM识别器
    """
    print("=" * 50)  # 分隔线
    print("测试SVM识别器...")  # 标题

    try:  # 尝试导入配置
        # 从同级目录导入Config
        from all3 import Config  # 导入配置类

        print("✓ 配置导入成功")  # 打印成功信息
    except ImportError as e:  # 如果导入失败
        print(f"✗ 配置导入失败: {e}")  # 打印错误
        print("请确保 all3.py 文件存在且包含 Config 类")  # 提示
        exit(1)  # 退出程序

    # 初始化识别器
    try:  # 尝试初始化
        recognizer = RealSVMRecognizer(Config)  # 创建识别器实例
        print("✓ SVM识别器初始化成功")  # 打印成功信息
    except Exception as e:  # 如果初始化失败
        print(f"✗ 初始化失败: {e}")  # 打印错误
        exit(1)  # 退出程序

    # 训练模型
    try:  # 尝试训练
        accuracy = recognizer.train()  # 训练模型
        print(f"✓ 模型训练完成，准确率: {accuracy * 100:.2f}%")  # 打印准确率
    except Exception as e:  # 如果训练失败
        print(f"✗ 训练失败: {e}")  # 打印错误

    # 打印信息
    recognizer.print_info()  # 打印模型信息

    print("=" * 50)  # 分隔线
    print("SVM识别器测试完成！")  # 完成信息
    print("使用方法：")  # 使用说明
    print("  1. 创建对象: recognizer = RealSVMRecognizer(Config)")  # 步骤1
    print("  2. 训练模型: recognizer.train()")  # 步骤2
    print("  3. 识别字符: result = recognizer.predict_char(char_image)")  # 步骤3
    print("  4. 识别车牌: plate, conf = recognizer.recognize_plate(char_images)")  # 步骤4