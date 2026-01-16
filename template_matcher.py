"""
real_svm.py
真正的SVM+HOG字符识别器
使用CNN的合成数据和真实数据训练
作者：车牌识别系统开发团队
"""

import cv2
import numpy as np
from pathlib import Path
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')
import os


class RealSVMRecognizer:
    """
    真实的SVM+HOG识别器
    使用你已有的字符数据进行训练
    """

    def __init__(self, config=None):
        if config:
            self.config = config
        else:
            from all3 import Config
            self.config = Config

        # HOG参数（与你的字符尺寸匹配）
        self.winSize = (self.config.CHAR_WIDTH, self.config.CHAR_HEIGHT)  # 40x80
        self.blockSize = (20, 20)
        self.blockStride = (10, 10)
        self.cellSize = (10, 10)
        self.nbins = 9

        # 创建HOG描述符
        self.hog = cv2.HOGDescriptor(
            self.winSize,
            self.blockSize,
            self.blockStride,
            self.cellSize,
            self.nbins
        )

        # SVM模型
        self.svm_model = None
        self.label_encoder = LabelEncoder()

        # 模型文件路径
        self.model_path = self.config.MODELS_DIR / "svm_model.pkl"
        self.encoder_path = self.config.MODELS_DIR / "label_encoder.pkl"

        print(f"✓ SVM识别器初始化完成")
        print(f"  HOG特征维度: {self.get_hog_feature_dimension()}")
        print(f"  字符总数: {len(self.config.CHAR_TO_LABEL)}")

    def get_hog_feature_dimension(self):
        """计算HOG特征维度"""
        # 计算方式: (窗口大小/块步长)^2 * 每个块的细胞数 * 方向数
        cells_per_block = (self.blockSize[0] // self.cellSize[0]) * (self.blockSize[1] // self.cellSize[1])
        n_cells = ((self.winSize[0] - self.blockSize[0]) // self.blockStride[0] + 1) * \
                  ((self.winSize[1] - self.blockSize[1]) // self.blockStride[1] + 1)
        return n_cells * cells_per_block * self.nbins

    def extract_hog_features(self, image):
        """
        提取图像的HOG特征

        参数：
            image: 灰度图像，尺寸为CHAR_WIDTH x CHAR_HEIGHT

        返回：
            numpy array: HOG特征向量
        """
        # 确保是正确尺寸
        if image.shape[:2] != (self.config.CHAR_HEIGHT, self.config.CHAR_WIDTH):
            image = cv2.resize(image, (self.config.CHAR_WIDTH, self.config.CHAR_HEIGHT))

        # 确保是灰度图
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 确保黑底白字（与CNN训练一致）
        if np.mean(image) > 127:
            image = 255 - image

        # 归一化到0-1范围
        image = image.astype(np.float32) / 255.0

        # 计算HOG特征
        hog_features = self.hog.compute(image)

        # 展平并返回
        return hog_features.flatten()

    def load_training_data(self, max_samples_per_char=50, progress_callback=None):
        """
        从合成数据和真实数据加载训练数据

        参数：
            max_samples_per_char: 每个字符的最大样本数（避免内存过大）
        """
        print("开始加载SVM训练数据...")

        features_list = []
        labels_list = []

        # 获取所有字符
        all_chars = list(self.config.CHAR_TO_LABEL.keys())
        char_count = len(all_chars)

        for char_idx, char in enumerate(all_chars):
            if progress_callback:
                progress_callback(f"加载字符 {char_idx + 1}/{char_count}: {char}")

            char_samples = []

            # 1. 从合成数据加载
            synth_dirs = self.get_char_dirs(char, self.config.SYNTH_CHARS_DIR)
            for char_dir in synth_dirs:
                if char_dir.exists():
                    img_files = list(char_dir.glob("*.png"))[:max_samples_per_char]
                    for img_file in img_files:
                        try:
                            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                char_samples.append(img)
                        except:
                            pass

            # 2. 从真实数据加载
            real_dirs = self.get_char_dirs(char, self.config.REAL_TRAIN_DIR)
            for char_dir in real_dirs:
                if char_dir.exists():
                    img_files = list(char_dir.glob("*.png"))
                    for img_file in img_files:
                        try:
                            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                char_samples.append(img)
                        except:
                            pass

            # 提取特征
            for img in char_samples[:max_samples_per_char]:  # 限制数量
                try:
                    hog_features = self.extract_hog_features(img)
                    features_list.append(hog_features)
                    labels_list.append(char)
                except Exception as e:
                    print(f"  特征提取失败 {char}: {e}")
                    continue

        print(f"✓ 数据加载完成: {len(features_list)} 个样本")
        return np.array(features_list), np.array(labels_list)

    def get_char_dirs(self, char, base_dir):
        """根据字符类型获取对应的目录（复用CNN的代码）"""
        dirs = []

        if char in self.config.PROVINCES:
            pinyin = self.config.PROVINCES[char]
            dirs.append(base_dir / "provinces" / pinyin)
        if char in self.config.LETTERS:
            dirs.append(base_dir / "letters" / char)
        if char in self.config.DIGITS:
            dirs.append(base_dir / "digits" / char)

        return dirs

    def train(self, progress_callback=None, use_cached=True):
        """
        训练SVM模型

        参数：
            use_cached: 是否使用缓存的模型
        """
        # 检查是否有缓存模型
        if use_cached and self.model_path.exists() and self.encoder_path.exists():
            try:
                self.load_model()
                if progress_callback:
                    progress_callback("✓ 加载缓存的SVM模型")
                return True
            except:
                if progress_callback:
                    progress_callback("⚠ 缓存模型加载失败，重新训练")

        if progress_callback:
            progress_callback("开始训练SVM模型...")

        # 1. 加载训练数据
        X, y = self.load_training_data(max_samples_per_char=50, progress_callback=progress_callback)

        if len(X) == 0:
            if progress_callback:
                progress_callback("✗ 错误：没有训练数据")
            return False

        # 2. 编码标签（字符 -> 数字）
        y_encoded = self.label_encoder.fit_transform(y)

        # 3. 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        if progress_callback:
            progress_callback(f"训练集: {len(X_train)} 样本")
            progress_callback(f"测试集: {len(X_test)} 样本")

        # 4. 训练SVM
        if progress_callback:
            progress_callback("训练SVM分类器...")

        self.svm_model = SVC(
            C=1.0,  # 正则化参数
            kernel='rbf',  # 径向基函数核
            gamma='scale',  # 核系数
            probability=True,  # 启用概率预测
            random_state=42,
            verbose=False
        )

        self.svm_model.fit(X_train, y_train)

        # 5. 评估模型
        y_pred = self.svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        if progress_callback:
            progress_callback(f"✓ SVM训练完成")
            progress_callback(f"  测试准确率: {accuracy:.2%}")
            progress_callback(f"  类别数: {len(self.label_encoder.classes_)}")

        # 6. 保存模型
        self.save_model()

        return True

    def save_model(self):
        """保存SVM模型和标签编码器"""
        try:
            self.config.MODELS_DIR.mkdir(exist_ok=True)

            # 保存SVM模型
            joblib.dump(self.svm_model, self.model_path)

            # 保存标签编码器
            joblib.dump({
                'classes': self.label_encoder.classes_,
                'fitted': True
            }, self.encoder_path)

            print(f"✓ SVM模型保存到 {self.model_path}")
            return True
        except Exception as e:
            print(f"✗ 保存模型失败: {e}")
            return False

    def load_model(self):
        """加载SVM模型和标签编码器"""
        try:
            # 加载SVM模型
            self.svm_model = joblib.load(self.model_path)

            # 加载标签编码器
            encoder_data = joblib.load(self.encoder_path)
            self.label_encoder.classes_ = encoder_data['classes']

            print(f"✓ SVM模型加载成功")
            print(f"  类别数: {len(self.label_encoder.classes_)}")
            return True
        except Exception as e:
            print(f"✗ 加载模型失败: {e}")
            return False

    def predict_char(self, char_img, position=None):
        """
        识别单个字符

        参数：
            char_img: 字符图像
            position: 字符位置（用于分层识别）

        返回：
            dict: 识别结果
        """
        if self.svm_model is None:
            # 如果模型未加载，尝试加载
            if not self.load_model():
                return {'char': '?', 'confidence': 0.0, 'method': 'svm'}

        # 1. 提取HOG特征
        try:
            features = self.extract_hog_features(char_img)
        except Exception as e:
            print(f"SVM特征提取失败: {e}")
            return {'char': '?', 'confidence': 0.0, 'method': 'svm'}

        # 2. 根据位置过滤（分层识别）
        if position is not None:
            if position == 1:  # 汉字
                allowed_indices = [
                    idx for idx, char in enumerate(self.label_encoder.classes_)
                    if char in self.config.PROVINCES
                ]
            elif position == 2:  # 字母
                allowed_indices = [
                    idx for idx, char in enumerate(self.label_encoder.classes_)
                    if char in self.config.LETTERS
                ]
            else:  # 字母或数字
                allowed_indices = [
                    idx for idx, char in enumerate(self.label_encoder.classes_)
                    if char in (self.config.LETTERS + self.config.DIGITS)
                ]
        else:
            allowed_indices = list(range(len(self.label_encoder.classes_)))

        # 3. 预测概率
        try:
            probabilities = self.svm_model.predict_proba([features])[0]
        except:
            # 如果概率预测失败，使用普通预测
            pred_label = self.svm_model.predict([features])[0]
            pred_char = self.label_encoder.inverse_transform([pred_label])[0]
            return {
                'char': pred_char,
                'confidence': 0.8,  # 默认置信度
                'method': 'svm'
            }

        # 4. 只考虑允许的类别
        filtered_probs = []
        for idx in allowed_indices:
            if idx < len(probabilities):
                filtered_probs.append((idx, probabilities[idx]))

        if not filtered_probs:
            return {'char': '?', 'confidence': 0.0, 'method': 'svm'}

        # 5. 找到最佳匹配
        filtered_probs.sort(key=lambda x: x[1], reverse=True)
        best_idx, best_prob = filtered_probs[0]

        # 6. 解码字符
        try:
            best_char = self.label_encoder.inverse_transform([best_idx])[0]
        except:
            best_char = '?'

        # 7. 获取Top5备选
        top5 = []
        for idx, prob in filtered_probs[:5]:
            try:
                char = self.label_encoder.inverse_transform([idx])[0]
                top5.append({'char': char, 'score': float(prob)})
            except:
                pass

        return {
            'char': best_char,
            'confidence': float(best_prob),
            'method': 'svm',
            'top5': top5
        }

    def recognize_plate(self, char_images):
        """
        识别整个车牌

        参数：
            char_images: 7个字符图像列表

        返回：
            tuple: (车牌字符串, 平均置信度)
        """
        if len(char_images) != 7:
            print(f"警告：期望7个字符，得到{len(char_images)}个")

        plate_number = ""
        confidences = []

        for i, char_img in enumerate(char_images):
            position = i + 1
            result = self.predict_char(char_img, position)

            plate_number += result['char']
            confidences.append(result['confidence'])

        avg_confidence = np.mean(confidences) if confidences else 0.0

        return plate_number, avg_confidence


# 测试代码
if __name__ == "__main__":
    print("测试SVM识别器...")

    try:
        # 导入Config
        import sys

        sys.path.append('.')
        from all3 import Config

        # 初始化SVM识别器
        svm_recognizer = RealSVMRecognizer(Config)

        # 训练模型（第一次运行需要）
        svm_recognizer.train(progress_callback=print)

        # 测试预测
        print("\n✓ SVM识别器初始化成功！")
        print("使用方法：")
        print("1. 创建对象: recognizer = RealSVMRecognizer(Config)")
        print("2. 训练: recognizer.train()")
        print("3. 识别: result = recognizer.predict_char(char_image)")

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback

        traceback.print_exc()