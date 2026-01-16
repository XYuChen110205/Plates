"""
车牌识别系统 v8.1 - 回归v7.0处理流程版
专为0基础用户设计，包含详细的中文注释
版本: 8.1 回归v7.0处理流程版

修改要点：
1. 字符分割回归v7.0经典算法（更可靠）
2. 保留v8.1的评估功能和三种方法对比
3. 保留完整可视化处理流程

使用步骤：
1. 运行程序
2. 点击"创建目录"按钮
3. 点击"生成合成数据"按钮
4. 点击"选择车牌图像"选择真实车牌图片
5. 点击"处理车牌图像"查看完整处理流程
6. 点击"保存处理截图"保存所有步骤图片
7. 点击"收集训练数据"手动标注字符
8. 点击"训练CNN模型"和"训练SVM模型"
9. 点击"三种方法对比"查看识别效果
10. 点击"性能评估"生成准确率报告
"""

# ============ 第一部分：导入必要的库 ============
import os
import sys
import random
import numpy as np

from pathlib import Path
from datetime import datetime

# 图形界面相关
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText

# 图像处理相关
from PIL import Image, ImageDraw, ImageFont, ImageTk
from PIL import ImageFilter, ImageEnhance
import cv2

# 深度学习相关
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# 机器学习相关
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# 其他工具
import threading
import time
import shutil
import platform
import json
import pandas as pd

# 尝试导入EasyOCR（可选功能）
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    print("✓ EasyOCR已安装")
except ImportError:
    EASYOCR_AVAILABLE = False
    print("⚠ EasyOCR未安装，OCR对比功能将不可用")


# ============ 第二部分：配置类（系统设置） ============
class Config:
    """系统配置类 - 存储所有系统设置"""

    # 使用当前工作目录（程序运行的文件夹）
    BASE_DIR = Path.cwd()

    # 定义各个子目录的路径（全部使用英文名称，避免中文路径问题）
    SYNTH_CHARS_DIR = BASE_DIR / "synthetic_chars"
    MODELS_DIR = BASE_DIR / "models"
    TEST_IMAGES_DIR = BASE_DIR / "test_images"
    RESULTS_DIR = BASE_DIR / "results"
    REAL_TRAIN_DIR = BASE_DIR / "real_train_chars"

    # 车牌省份简称（汉字）和对应的拼音（用拼音做目录名，避免中文路径）
    PROVINCES = {
        '京': 'JING', '沪': 'HU', '津': 'TIANJIN', '渝': 'CHONGQING',
        '冀': 'HEBEI', '晋': 'SHANXI', '蒙': 'NEIMENG', '辽': 'LIAONING',
        '吉': 'JILIN', '黑': 'HEILONGJIANG', '苏': 'JIANGSU', '浙': 'ZHEJIANG',
        '皖': 'ANHUI', '闽': 'FUJIAN', '赣': 'JIANGXI', '鲁': 'SHANDONG',
        '豫': 'HENAN', '鄂': 'HUBEI', '湘': 'HUNAN', '粤': 'GUANGDONG',
        '桂': 'GUANGXI', '琼': 'HAINAN', '川': 'SICHUAN', '贵': 'GUIZHOU',
        '云': 'YUNNAN', '藏': 'XIZANG', '陕': 'SHANXI2', '甘': 'GANSU',
        '青': 'QINGHAI', '宁': 'NINGXIA', '新': 'XINJIANG'
    }

    # 车牌字母（去掉I和O，因为容易混淆）
    LETTERS = list('ABCDEFGHJKLMNPQRSTUVWXYZ')

    # 数字
    DIGITS = list('0123456789')

    # 字符映射表（空字典，稍后初始化）
    CHAR_TO_LABEL = {}
    LABEL_TO_CHAR = {}

    @staticmethod
    def init_char_maps():
        """初始化字符映射表 - 为每个字符分配一个数字ID"""
        Config.CHAR_TO_LABEL.clear()
        Config.LABEL_TO_CHAR.clear()

        all_chars = []

        # 1. 添加省份汉字
        for char in Config.PROVINCES:
            all_chars.append(char)

        # 2. 添加字母
        all_chars.extend(Config.LETTERS)

        # 3. 添加数字
        all_chars.extend(Config.DIGITS)

        # 4. 创建映射关系
        for idx, char in enumerate(all_chars):
            Config.CHAR_TO_LABEL[char] = idx
            Config.LABEL_TO_CHAR[idx] = char

        print(f"字符映射初始化: {len(Config.CHAR_TO_LABEL)} 个字符")

    # 字符图像尺寸设置
    CHAR_HEIGHT = 80
    CHAR_WIDTH = 40
    PADDING = 5

    # 训练参数设置
    BATCH_SIZE = 32
    CNN_EPOCHS = 25
    LEARNING_RATE = 0.001
    PATIENCE = 5
    SAMPLES_PER_CHAR = 200

    @staticmethod
    def create_dirs():
        """创建所有必要的目录"""
        Config.init_char_maps()

        dirs = [
            Config.BASE_DIR,
            Config.SYNTH_CHARS_DIR,
            Config.MODELS_DIR,
            Config.TEST_IMAGES_DIR,
            Config.RESULTS_DIR,
            Config.REAL_TRAIN_DIR,
        ]

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ 创建目录: {dir_path}")

        # 为每种字符类型创建子目录
        # 1. 创建省份汉字目录（使用拼音）
        for pinyin in Config.PROVINCES.values():
            (Config.SYNTH_CHARS_DIR / "provinces" / pinyin).mkdir(parents=True, exist_ok=True)
            (Config.REAL_TRAIN_DIR / "provinces" / pinyin).mkdir(parents=True, exist_ok=True)

        # 2. 创建字母目录（直接使用字母）
        for char in Config.LETTERS:
            (Config.SYNTH_CHARS_DIR / "letters" / char).mkdir(parents=True, exist_ok=True)
            (Config.REAL_TRAIN_DIR / "letters" / char).mkdir(parents=True, exist_ok=True)

        # 3. 创建数字目录（直接使用数字）
        for char in Config.DIGITS:
            (Config.SYNTH_CHARS_DIR / "digits" / char).mkdir(parents=True, exist_ok=True)
            (Config.REAL_TRAIN_DIR / "digits" / char).mkdir(parents=True, exist_ok=True)

        print("✓ 所有目录创建完成")


# ============ 第三部分：字体验证器 ============
class FontValidator:
    """验证字体文件是否能正确渲染中文"""

    @staticmethod
    def test_font_chinese(font_path, test_char="京"):
        """测试字体是否能正确显示汉字"""
        try:
            font = ImageFont.truetype(font_path, 40)
            img = Image.new('L', (50, 50), color=0)
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), test_char, fill=255, font=font)
            img_array = np.array(img)
            white_pixels = np.sum(img_array > 0)
            return white_pixels > 20

        except Exception as e:
            print(f"字体测试失败 {font_path}: {e}")
            return False


# ============ 第四部分：增强版车牌净化系统（回归v7.0字符分割） ============
class EnhancedPlatePurification:
    """
    增强版车牌净化处理类
    功能：从车牌图片中提取单个字符，包含完整可视化处理流程
    修改：字符分割回归v7.0经典算法，去除复杂预处理
    """

    def __init__(self):
        self.current_image = None
        self.original_image = None
        self.plate_region = None
        self.white_bg_plate = None
        self.cropped_chars = []
        self.char_segments = []
        self.plate_position = None

        # 颜色阈值范围（蓝色和黄色车牌）
        self.blue_lower = [100, 150, 100]
        self.blue_upper = [130, 255, 255]
        self.yellow_lower = [15, 100, 100]
        self.yellow_upper = [35, 255, 255]

        # 新增：用于可视化处理的中间结果
        self.gray_image = None
        self.edges_image = None
        self.color_mask_image = None
        self.histogram_image = None
        self.horizontal_proj = None
        self.vertical_proj = None

        # 传统处理方法结果（用于对比）
        self.traditional_plate = None
        self.traditional_binary = None

    def load_image(self, image_path):
        """加载图像"""
        self.current_image = image_path
        self.original_image = cv2.imread(image_path)
        return self.original_image is not None

    def traditional_edge_detection(self, progress_callback=None):
        """传统边缘检测方法（用于对比）"""
        if self.original_image is None:
            return False

        try:
            # 转换为灰度图
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            self.gray_image = gray.copy()

            # 高斯模糊去噪（5x5高斯核）
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Canny边缘检测（阈值50-150）
            edges = cv2.Canny(blurred, 50, 150)
            self.edges_image = edges.copy()

            # 形态学操作连接边缘
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            if progress_callback:
                progress_callback("传统边缘检测完成")

            return True

        except Exception as e:
            if progress_callback:
                progress_callback(f"传统边缘检测失败: {e}")
            return False

    def create_color_mask(self, progress_callback=None):
        """创建颜色掩码可视化（颜色特征方法）"""
        if self.original_image is None:
            return False

        try:
            image = self.original_image.copy()

            # 调整大小（如果图像太大）
            h, w = image.shape[:2]
            if w > 800:
                scale = 800 / w
                new_w = 800
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h))

            # 转换为HSV颜色空间（更容易识别颜色）
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # 蓝色车牌掩码（找出蓝色区域）
            blue_lower = np.array([self.blue_lower[0], 150, self.blue_lower[2]])
            blue_upper = np.array([self.blue_upper[0], self.blue_upper[1], self.blue_upper[2]])
            blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

            # 黄色车牌掩码（找出黄色区域）
            yellow_lower = np.array(self.yellow_lower)
            yellow_upper = np.array(self.yellow_upper)
            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

            # 合并蓝色和黄色掩码
            color_mask = cv2.bitwise_or(blue_mask, yellow_mask)

            # 创建彩色可视化（用于显示）
            mask_visual = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
            mask_visual[color_mask > 0] = (0, 255, 0)

            # 叠加在原图上（70%原图 + 30%掩码）
            result = cv2.addWeighted(image, 0.7, mask_visual, 0.3, 0)
            self.color_mask_image = result

            if progress_callback:
                progress_callback("颜色掩码创建完成")

            return True

        except Exception as e:
            if progress_callback:
                progress_callback(f"颜色掩码创建失败: {e}")
            return False

    def compact_plate_detection(self, progress_callback=None):
        """检测车牌区域 - 通过颜色识别车牌（颜色特征方法）"""
        if self.original_image is None:
            return False

        try:
            image = self.original_image.copy()

            # 调整图像大小（如果太大）
            h, w = image.shape[:2]
            if w > 800:
                scale = 800 / w
                new_w = 800
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h))

            # 转换为HSV颜色空间（更容易识别颜色）
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # 蓝色车牌掩码（找出蓝色区域）
            blue_lower = np.array([self.blue_lower[0], 150, self.blue_lower[2]])
            blue_upper = np.array([self.blue_upper[0], self.blue_upper[1], self.blue_upper[2]])
            blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

            # 黄色车牌掩码（找出黄色区域）
            yellow_lower = np.array(self.yellow_lower)
            yellow_upper = np.array(self.yellow_upper)
            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

            # 合并蓝色和黄色掩码
            color_mask = cv2.bitwise_or(blue_mask, yellow_mask)

            # 形态学处理（去除噪声）
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
            mask_cleaned = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
            mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

            # 查找轮廓（找到所有可能的车牌区域）
            contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                if progress_callback:
                    progress_callback("未检测到车牌区域")
                return False

            # 找到最佳轮廓（最像车牌的）
            best_contour = None
            best_score = -1

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500:
                    continue

                # 获取轮廓的外接矩形
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                aspect_ratio = w_rect / h_rect if h_rect > 0 else 0

                # 计算得分（车牌通常是长方形，宽高比约3.5:1）
                aspect_score = 1.0 - min(abs(aspect_ratio - 3.5) / 2.0, 1.0)
                rect_area = w_rect * h_rect
                compactness = area / rect_area if rect_area > 0 else 0
                score = aspect_score * 0.6 + compactness * 0.4

                if score > best_score:
                    best_score = score
                    best_contour = contour

            if best_contour is None:
                return False

            # 获取车牌区域
            x, y, w_rect, h_rect = cv2.boundingRect(best_contour)

            # 稍微扩大边界框（确保包含整个车牌）
            expand_x = max(1, int(w_rect * 0.02))
            expand_y = max(1, int(h_rect * 0.02))

            x = max(0, x - expand_x)
            y = max(0, y - expand_y)
            w_rect = min(image.shape[1] - x, w_rect + 2 * expand_x)
            h_rect = min(image.shape[0] - y, h_rect + 2 * expand_y)

            # 提取车牌区域
            self.plate_region = image[y:y + h_rect, x:x + w_rect]
            self.plate_position = (x, y, w_rect, h_rect)

            if progress_callback:
                progress_callback(f"车牌检测成功: {w_rect}×{h_rect}")

            return True

        except Exception as e:
            if progress_callback:
                progress_callback(f"车牌检测失败: {e}")
            return False

    def white_background_conversion(self, progress_callback=None):
        """转换为黑底白字 - 统一字符颜色"""
        if self.plate_region is None:
            return False

        try:
            plate_img = self.plate_region.copy()

            # 转换为灰度图（去掉颜色信息）
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

            # 增强对比度（CLAHE方法）
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # 二值化（Otsu方法，自动选择阈值）
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 确保是黑底白字
            white_pixels = np.sum(binary == 255)
            black_pixels = np.sum(binary == 0)

            if white_pixels > black_pixels:
                binary = cv2.bitwise_not(binary)

            # 形态学处理（去除小噪声）
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            self.white_bg_plate = binary

            if progress_callback:
                progress_callback("黑底白字转换完成")

            return True

        except Exception as e:
            if progress_callback:
                progress_callback(f"黑底白字转换失败: {e}")
            return False

    def compute_histograms(self, progress_callback=None):
        """计算并可视化水平/垂直投影直方图"""
        if self.white_bg_plate is None:
            return False

        try:
            binary = self.white_bg_plate.copy()
            height, width = binary.shape

            # 计算投影
            self.horizontal_proj = np.sum(binary == 255, axis=1)
            self.vertical_proj = np.sum(binary == 255, axis=0)

            # 创建直方图可视化图像
            hist_height = 200
            hist_width = width

            # 垂直投影直方图
            vertical_hist = np.zeros((hist_height, width), dtype=np.uint8)
            if np.max(self.vertical_proj) > 0:
                normalized_vertical = (self.vertical_proj / np.max(self.vertical_proj) * hist_height).astype(int)
                for x in range(width):
                    h = normalized_vertical[x]
                    if h > 0:
                        vertical_hist[hist_height - h:, x] = 255

            # 水平投影直方图
            horizontal_hist = np.zeros((height, hist_height), dtype=np.uint8)
            if np.max(self.horizontal_proj) > 0:
                normalized_horizontal = (self.horizontal_proj / np.max(self.horizontal_proj) * hist_height).astype(int)
                for y in range(height):
                    w = normalized_horizontal[y]
                    if w > 0:
                        horizontal_hist[y, hist_height - w:] = 255

            # 合并直方图
            combined_hist = np.zeros((height + hist_height, width + hist_height), dtype=np.uint8)
            combined_hist[:height, :width] = binary
            combined_hist[:height, width:] = horizontal_hist
            combined_hist[height:, :width] = vertical_hist
            combined_hist[height:, width:] = 0

            # 添加分隔线
            cv2.line(combined_hist, (width, 0), (width, height + hist_height), 128, 1)
            cv2.line(combined_hist, (0, height), (width + hist_height, height), 128, 1)

            # 添加标签
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(combined_hist, "二值图像", (10, 20), font, 0.5, 128, 1)
            cv2.putText(combined_hist, "水平投影", (width + 10, 20), font, 0.5, 128, 1)
            cv2.putText(combined_hist, "垂直投影", (10, height + 20), font, 0.5, 128, 1)

            self.histogram_image = combined_hist

            if progress_callback:
                progress_callback("直方图计算完成")

            return True

        except Exception as e:
            if progress_callback:
                progress_callback(f"直方图计算失败: {e}")
            return False

    def char_segmentation(self, progress_callback=None):
        """字符分割 - 使用v7.0经典算法"""
        if self.white_bg_plate is None:
            return False

        try:
            binary = self.white_bg_plate.copy()
            height, width = binary.shape

            # 垂直投影（统计每列白色像素数）
            vertical_proj = np.sum(binary == 255, axis=0)

            # 查找波峰（字符区域）
            wave_peaks = self.find_waves(vertical_proj)

            if not wave_peaks:
                if progress_callback:
                    progress_callback("未找到字符波峰")
                return False

            # 过滤波峰（去除太宽或太窄的）
            filtered_peaks = []
            min_char_width = 10
            max_char_width = width // 4

            for start, end in wave_peaks:
                char_width = end - start
                if min_char_width <= char_width <= max_char_width:
                    filtered_peaks.append((start, end))

            # 如果字符数不是7个，进行调整（标准车牌7个字符）
            if len(filtered_peaks) != 7:
                if len(filtered_peaks) > 7:
                    # 选择宽度最接近平均值的7个
                    widths = [end - start for start, end in filtered_peaks]
                    avg_width = np.mean(widths)

                    # 计算每个宽度与平均宽度的偏差
                    deviations = [abs(width - avg_width) for width in widths]
                    # 选择最接近平均宽度的7个
                    sorted_indices = np.argsort(deviations)[:7]
                    filtered_peaks = [filtered_peaks[i] for i in sorted_indices]
                    filtered_peaks.sort(key=lambda x: x[0])

            self.cropped_chars = []
            self.char_segments = filtered_peaks

            # 处理每个字符（v7.0经典算法）
            for start, end in filtered_peaks:
                char_img = binary[:, start:end]

                # 调整到标准尺寸
                target_height = Config.CHAR_HEIGHT
                target_width = Config.CHAR_WIDTH

                char_height, char_width = char_img.shape
                # 计算缩放比例（保持比例）
                scale = min(target_height / char_height, target_width / char_width)
                new_height = int(char_height * scale)
                new_width = int(char_width * scale)

                # 缩放字符
                resized_char = cv2.resize(char_img, (new_width, new_height))

                # 添加填充（居中）
                padded_char = np.zeros((target_height + 2 * Config.PADDING,
                                        target_width + 2 * Config.PADDING), dtype=np.uint8)

                # 计算居中位置
                y_offset = (target_height - new_height) // 2 + Config.PADDING
                x_offset = (target_width - new_width) // 2 + Config.PADDING

                # 将字符放在画布中央
                padded_char[y_offset:y_offset + new_height,
                x_offset:x_offset + new_width] = resized_char

                self.cropped_chars.append(padded_char)

            if progress_callback:
                progress_callback(f"字符分割完成: {len(self.cropped_chars)} 个字符")

            return True

        except Exception as e:
            if progress_callback:
                progress_callback(f"字符分割失败: {e}")
            return False

    def find_waves(self, histogram, threshold_ratio=0.1):
        """查找波峰 - 找到字符的起始和结束位置"""
        threshold = np.max(histogram) * threshold_ratio

        up_point = -1
        is_peak = False
        wave_peaks = []

        for i, value in enumerate(histogram):
            if value > threshold and not is_peak:
                up_point = i
                is_peak = True
            elif value <= threshold and is_peak:
                if i - up_point > 2:
                    wave_peaks.append((up_point, i))
                is_peak = False

        if is_peak and up_point != -1:
            wave_peaks.append((up_point, len(histogram) - 1))

        return wave_peaks

    def process_image_complete(self, image_path, progress_callback=None):
        """完整的图像处理流程（包含所有可视化步骤）"""
        success = True
        steps_completed = []

        if progress_callback:
            progress_callback("开始完整图像处理流程...")
            progress_callback("=" * 50)

        # 1. 加载图像
        if not self.load_image(image_path):
            if progress_callback:
                progress_callback("✗ 图像加载失败")
            return False, steps_completed

        steps_completed.append("1. 图像加载 ✓")
        if progress_callback:
            progress_callback("✓ 图像加载成功")

        # 2. 传统边缘检测（用于对比）
        if self.traditional_edge_detection(progress_callback):
            steps_completed.append("2. 传统边缘检测 ✓")
        else:
            steps_completed.append("2. 传统边缘检测 ✗")

        # 3. 颜色特征处理
        if self.create_color_mask(progress_callback):
            steps_completed.append("3. 颜色掩码创建 ✓")
        else:
            steps_completed.append("3. 颜色掩码创建 ✗")

        # 4. 车牌检测（颜色特征方法）
        if not self.compact_plate_detection(progress_callback):
            success = False
            steps_completed.append("4. 车牌检测 ✗")
        else:
            steps_completed.append("4. 车牌检测 ✓")

        # 5. 转换为黑底白字
        if success and not self.white_background_conversion(progress_callback):
            success = False
            steps_completed.append("5. 二值化处理 ✗")
        else:
            steps_completed.append("5. 二值化处理 ✓")

        # 6. 计算直方图
        if success and not self.compute_histograms(progress_callback):
            steps_completed.append("6. 直方图计算 ✗")
        else:
            steps_completed.append("6. 直方图计算 ✓")

        # 7. 字符分割（回归v7.0算法）
        if success and not self.char_segmentation(progress_callback):
            success = False
            steps_completed.append("7. 字符分割 ✗")
        else:
            steps_completed.append("7. 字符分割 ✓")

        # 8. 显示处理步骤摘要
        if progress_callback:
            progress_callback("=" * 50)
            progress_callback("【处理步骤完成】")
            for step in steps_completed:
                progress_callback(step)

            progress_callback("=" * 50)
            progress_callback("✓ 处理流程完成")

            # 对比说明
            progress_callback("【方法对比说明】")
            progress_callback("• 传统方法：基于边缘检测，对光照敏感")
            progress_callback("• 颜色特征：基于HSV颜色空间，更稳定")
            progress_callback("• 字符分割：使用v7.0经典算法，准确可靠")
            progress_callback("• 本系统采用颜色特征+v7.0分割方法进行后续处理")

        return success, steps_completed

    def save_chars(self, output_dir):
        """保存分割的字符"""
        if not self.cropped_chars:
            return False

        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # 保存每个字符
            for i, char_img in enumerate(self.cropped_chars):
                char_path = output_path / f"char_{i + 1:02d}.png"
                cv2.imwrite(str(char_path), char_img)

            # 创建合并图像（所有字符排成一行，方便查看）
            if len(self.cropped_chars) > 0:
                max_height = max([char.shape[0] for char in self.cropped_chars])
                total_width = sum([char.shape[1] for char in self.cropped_chars])

                # 创建合并画布
                merged_img = np.zeros((max_height, total_width + 10 * len(self.cropped_chars)), dtype=np.uint8)

                # 将每个字符拼接到画布上
                x_offset = 5
                for char_img in self.cropped_chars:
                    h, w = char_img.shape
                    y_offset = (max_height - h) // 2
                    merged_img[y_offset:y_offset + h, x_offset:x_offset + w] = char_img
                    x_offset += w + 10

                # 保存合并图像
                merged_path = output_path / "merged_chars.png"
                cv2.imwrite(str(merged_path), merged_img)

            return True

        except Exception as e:
            print(f"保存字符失败: {e}")
            return False


# ============ 第五部分：修复版字符生成器 ============
class FixedCharGenerator:
    """
    修复版字符生成器 - 使用simhei.ttf生成清晰的字符模板
    专门修复了字体渲染问题
    """

    def __init__(self):
        self.config = Config
        self.font_path = r"C:\Windows\Fonts\simhei.ttf"

        # 验证字体文件是否存在
        if not Path(self.font_path).exists():
            print(f"错误: 字体文件不存在 {self.font_path}")
            print("请确保simhei.ttf存在于C:\\Windows\\Fonts\\目录")
            raise FileNotFoundError(f"字体文件不存在 {self.font_path}")

        print(f"✓ 使用字体: {self.font_path}")

    def generate_char_image(self, char, variation=0):
        """
        生成字符图像 - 修复版
        步骤：大尺寸渲染 → 添加变化 → 缩小到标准尺寸
        """
        try:
            # 使用大尺寸画布渲染，保证清晰度
            BIG_WIDTH = 120
            BIG_HEIGHT = 180

            # 最终目标尺寸（加上填充）
            TARGET_WIDTH = self.config.CHAR_WIDTH + 2 * self.config.PADDING
            TARGET_HEIGHT = self.config.CHAR_HEIGHT + 2 * self.config.PADDING

            # 创建黑色背景的大尺寸图像
            img = Image.new('L', (BIG_WIDTH, BIG_HEIGHT), color=0)
            draw = ImageDraw.Draw(img)

            # 根据字符类型设置字体大小
            if char in self.config.DIGITS:
                font_size = 70
            elif char in self.config.LETTERS:
                font_size = 65
            else:
                font_size = 60

            try:
                font = ImageFont.truetype(self.font_path, font_size)
            except Exception as e:
                print(f"字体加载失败，使用默认字体: {e}")
                font = ImageFont.load_default()

            # 计算文字位置（居中）
            bbox = draw.textbbox((0, 0), char, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # 计算居中位置
            x = (BIG_WIDTH - text_width) // 2 - bbox[0]
            y = (BIG_HEIGHT - text_height) // 2 - bbox[1]

            # 添加随机偏移（增加样本多样性）
            if variation > 0:
                x += random.randint(-3, 3)
                y += random.randint(-3, 3)

            # 绘制字符（白色）
            draw.text((x, y), char, fill=255, font=font)

            # 转换为numpy数组进行后处理
            img_array = np.array(img)

            # 添加一些变化，模拟真实环境
            # 1. 轻微模糊（模拟印刷效果）
            if random.random() > 0.7:
                img = Image.fromarray(img_array)
                img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
                img_array = np.array(img)

            # 2. 添加轻微噪声（模拟图像噪声）
            if random.random() > 0.5:
                noise = np.random.normal(0, 5, img_array.shape).astype(np.int16)
                img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # 3. 调整对比度（增强字符）
            img = Image.fromarray(img_array)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.3)

            # 4. 调整亮度
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.1)

            # 缩放到目标尺寸（使用高质量的重采样算法）
            img = img.resize((TARGET_WIDTH, TARGET_HEIGHT),
                             Image.Resampling.LANCZOS)

            return img

        except Exception as e:
            print(f"生成字符 {char} 失败: {e}")
            return self.create_fallback_image(char)

    def create_fallback_image(self, char):
        """创建备用图像（当主方法失败时使用）"""
        width = Config.CHAR_WIDTH + 2 * Config.PADDING
        height = Config.CHAR_HEIGHT + 2 * Config.PADDING

        img = Image.new('L', (width, height), color=0)
        draw = ImageDraw.Draw(img)

        # 绘制一个白色边框
        draw.rectangle([5, 5, width - 5, height - 5], outline=255, width=2)

        # 在中间绘制字符
        try:
            font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), char, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (width - text_width) // 2 - bbox[0]
            y = (height - text_height) // 2 - bbox[1]
            draw.text((x, y), char, fill=255, font=font)
        except:
            draw.text((20, 30), "?", fill=255)

        return img

    def generate_all_chars(self, samples_per_char=200, progress_callback=None):
        """生成所有字符的合成数据"""
        if progress_callback:
            progress_callback("开始生成合成字符数据...")

        total_generated = 0
        start_time = time.time()

        # 1. 生成省份汉字
        if progress_callback:
            progress_callback("生成省份汉字...")

        for char, pinyin in self.config.PROVINCES.items():
            char_dir = self.config.SYNTH_CHARS_DIR / "provinces" / pinyin

            for i in range(samples_per_char):
                img = self.generate_char_image(char, variation=i)
                img_path = char_dir / f"{pinyin}_{i:04d}.png"
                img.save(img_path, 'PNG')
                total_generated += 1

        # 2. 生成字母
        if progress_callback:
            progress_callback("生成字母...")

        for char in self.config.LETTERS:
            char_dir = self.config.SYNTH_CHARS_DIR / "letters" / char

            for i in range(samples_per_char):
                img = self.generate_char_image(char, variation=i + 1000)
                img_path = char_dir / f"{char}_{i:04d}.png"
                img.save(img_path, 'PNG')
                total_generated += 1

        # 3. 生成数字
        if progress_callback:
            progress_callback("生成数字...")

        for char in self.config.DIGITS:
            char_dir = self.config.SYNTH_CHARS_DIR / "digits" / char

            for i in range(samples_per_char):
                img = self.generate_char_image(char, variation=i + 2000)
                img_path = char_dir / f"{char}_{i:04d}.png"
                img.save(img_path, 'PNG')
                total_generated += 1

        # 计算总耗时
        total_time = time.time() - start_time

        if progress_callback:
            progress_callback(f"\n✓ 合成数据生成完成!")
            progress_callback(f"总数量: {total_generated} 张图片")
            progress_callback(f"总用时: {total_time:.1f} 秒")
            progress_callback(f"每个字符: {samples_per_char} 张样本")
            progress_callback(f"使用字体: {Path(self.font_path).name}")

        return total_generated


# ============ 第六部分：字符数据集类 ============
class CharDataset(Dataset):
    """PyTorch数据集类"""

    def __init__(self, data_dir, transform=None, progress_callback=None, use_real_data=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.use_real_data = use_real_data

        self.load_samples(progress_callback)

        if progress_callback:
            progress_callback(f"数据集: {len(self.samples)} 张图片")

    def load_samples(self, progress_callback=None):
        """加载所有样本到内存"""
        all_chars = []

        # 合并所有字符类型
        all_chars.extend(Config.PROVINCES.keys())
        all_chars.extend(Config.LETTERS)
        all_chars.extend(Config.DIGITS)

        # 遍历每个字符
        for char in all_chars:
            # 获取字符的数字标签
            label = Config.CHAR_TO_LABEL.get(char)
            if label is None:
                continue

            # 优先使用真实数据目录（如果设置）
            if self.use_real_data:
                # 1. 先尝试从真实数据目录加载
                real_dirs = self.get_char_dirs(char, Config.REAL_TRAIN_DIR)
                for char_dir in real_dirs:
                    if char_dir.exists():
                        self.add_samples_from_dir(char_dir, label)

            # 2. 再加载合成数据（确保有足够数据）
            synth_dirs = self.get_char_dirs(char, Config.SYNTH_CHARS_DIR)
            for char_dir in synth_dirs:
                if char_dir.exists():
                    self.add_samples_from_dir(char_dir, label)

    def get_char_dirs(self, char, base_dir):
        """根据字符类型获取对应的目录"""
        dirs = []

        if char in Config.PROVINCES:
            pinyin = Config.PROVINCES[char]
            dirs.append(base_dir / "provinces" / pinyin)
        if char in Config.LETTERS:
            dirs.append(base_dir / "letters" / char)
        if char in Config.DIGITS:
            dirs.append(base_dir / "digits" / char)

        return dirs

    def add_samples_from_dir(self, char_dir, label):
        """从目录添加所有图片样本"""
        img_files = list(char_dir.glob("*.png"))
        for img_file in img_files:
            self.samples.append((str(img_file), label))

    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, idx):
        """获取第idx个样本"""
        img_path, label = self.samples[idx]

        # 加载图片并转换为灰度图
        img = Image.open(img_path).convert('L')

        # 确保是黑底白字（训练时统一格式）
        img_array = np.array(img)
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
            img = Image.fromarray(img_array)

        # 应用图像变换（缩放、归一化等）
        if self.transform:
            img = self.transform(img)

        return img, label


# ============ 第七部分：CNN神经网络模型 ============
class CharCNN(nn.Module):
    """字符识别CNN模型"""

    def __init__(self, num_classes):
        super(CharCNN, self).__init__()

        # 卷积层部分（特征提取）
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 计算全连接层输入尺寸
        self.fc_input_size = 128 * 10 * 5

        # 全连接层部分（分类）
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """前向传播（数据流动方向）"""
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# ============ 第八部分：模型训练器 ============
class ModelTrainer:
    """训练CNN模型"""

    def __init__(self, model_type='cnn'):
        self.config = Config
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

    def train_cnn(self, progress_callback=None, use_real_data=True):
        """训练CNN模型"""
        if progress_callback:
            progress_callback("开始训练CNN模型...")

        # 定义图像变换（必须与识别时一致！）
        transform = transforms.Compose([
            transforms.Resize((Config.CHAR_HEIGHT, Config.CHAR_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # 创建数据集（优先使用真实数据）
        dataset = CharDataset(Config.SYNTH_CHARS_DIR, transform, progress_callback, use_real_data)

        if len(dataset) == 0:
            if progress_callback:
                progress_callback("错误: 没有训练数据")
            return False, 0

        # 划分训练集（80%）和验证集（20%）
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # 创建数据加载器（批量读取数据）
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE,
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE,
                                shuffle=False, num_workers=0)

        # 创建模型
        num_classes = len(Config.CHAR_TO_LABEL)
        model = CharCNN(num_classes=num_classes).to(self.device)

        if progress_callback:
            progress_callback(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
            progress_callback(f"训练数据: {len(train_dataset)} 张图片")
            progress_callback(f"验证数据: {len(val_dataset)} 张图片")
            progress_callback(f"使用真实数据: {use_real_data}")

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

        # 学习率调度器（当验证准确率不再提升时降低学习率）
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

        # 训练循环
        best_accuracy = 0
        patience_counter = 0

        for epoch in range(Config.CNN_EPOCHS):
            epoch_start = time.time()

            # ---------- 训练阶段 ----------
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()

            train_accuracy = 100 * train_correct / train_total if train_total > 0 else 0

            # ---------- 验证阶段 ----------
            model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()

            val_accuracy = 100 * val_correct / val_total if val_total > 0 else 0

            # 调整学习率（根据验证准确率）
            scheduler.step(val_accuracy)

            epoch_time = time.time() - epoch_start

            if progress_callback:
                progress_callback(f"Epoch {epoch + 1:2d}/{Config.CNN_EPOCHS} | "
                                  f"训练Acc: {train_accuracy:6.2f}% | "
                                  f"验证Acc: {val_accuracy:6.2f}% | "
                                  f"时间: {epoch_time:.1f}s")

            # ---------- 早停检查 ----------
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                patience_counter = 0

                # 保存最佳模型
                model_path = Config.MODELS_DIR / "best_cnn_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_accuracy': train_accuracy,
                    'val_accuracy': val_accuracy,
                    'char_to_label': Config.CHAR_TO_LABEL,
                    'label_to_char': Config.LABEL_TO_CHAR,
                    'num_classes': num_classes,
                }, model_path)

                if progress_callback:
                    progress_callback(f"✓ 保存最佳模型: {val_accuracy:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    if progress_callback:
                        progress_callback(f"早停触发")
                    break

        if progress_callback:
            progress_callback(f"训练完成! 最佳准确率: {best_accuracy:.2f}%")

        return True, best_accuracy


# ============ 第九部分：分层字符识别器 ============
class HierarchicalRecognizer:
    """
    分层字符识别器
    根据车牌字符位置规则识别：
    位置1: 一定是汉字
    位置2: 一定是字母
    位置3-7: 字母或数字
    """

    def __init__(self):
        self.config = Config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 图像变换（必须与训练时一致！）
        self.transform = transforms.Compose([
            transforms.Resize((Config.CHAR_HEIGHT, Config.CHAR_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.full_model = None

    def load_models(self, progress_callback=None):
        """加载模型"""
        full_model_path = Config.MODELS_DIR / "best_cnn_model.pth"

        if full_model_path.exists():
            try:
                checkpoint = torch.load(full_model_path, map_location=self.device)
                num_classes = checkpoint.get('num_classes', len(Config.CHAR_TO_LABEL))
                self.full_model = CharCNN(num_classes=num_classes).to(self.device)
                self.full_model.load_state_dict(checkpoint['model_state_dict'])
                self.full_model.eval()

                if progress_callback:
                    progress_callback(f"模型加载成功")
                    val_acc = checkpoint.get('val_accuracy', 0)
                    progress_callback(f"验证准确率: {val_acc:.2f}%")

                return True

            except Exception as e:
                if progress_callback:
                    progress_callback(f"模型加载失败: {e}")

        return False

    def recognize_char_hierarchical(self, char_img, position):
        """
        根据位置分层识别字符
        char_img: 字符图像
        position: 字符位置（1-7）
        """
        if self.full_model is None:
            return None

        try:
            # 预处理图像（与训练时一致）
            if isinstance(char_img, np.ndarray):
                char_img = Image.fromarray(char_img)

            # 确保黑底白字
            img_array = np.array(char_img)
            if np.mean(img_array) > 127:
                img_array = 255 - img_array
                char_img = Image.fromarray(img_array)

            # 应用变换
            img_tensor = self.transform(char_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.full_model(img_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]

                # 根据位置过滤结果
                filtered_results = self.filter_by_position(probabilities, position)

                return filtered_results

        except Exception as e:
            print(f"识别失败: {e}")
            return None

    def filter_by_position(self, probabilities, position):
        """根据字符位置过滤结果"""
        # 根据位置确定允许的字符类型
        if position == 1:
            allowed_chars = list(Config.PROVINCES.keys())
        elif position == 2:
            allowed_chars = Config.LETTERS
        else:
            allowed_chars = Config.LETTERS + Config.DIGITS

        # 收集允许字符的结果
        allowed_results = []
        for char in allowed_chars:
            label = Config.CHAR_TO_LABEL.get(char)
            if label is not None:
                prob = probabilities[label].item()
                allowed_results.append({
                    'char': char,
                    'confidence': prob,
                    'label': label
                })

        # 按置信度排序（从高到低）
        allowed_results.sort(key=lambda x: x['confidence'], reverse=True)

        # 返回前5个结果
        return allowed_results[:5]


# ============ 第十部分：模板匹配识别器 ============
class TemplateMatcher:
    """模板匹配识别器 - 传统方法"""

    def __init__(self, config=None):
        if config:
            self.config = config
        else:
            self.config = Config

        self.templates = {}
        self.char_list = []
        self.load_templates()

        print(f"✓ 模板匹配器初始化完成，加载了 {len(self.templates)} 个字符模板")

    def load_templates(self):
        """从合成数据目录加载所有字符作为模板"""
        print("正在加载字符模板...")

        # 三种字符类型：汉字、字母、数字
        char_types = [
            ("provinces", self.config.PROVINCES),
            ("letters", {c: c for c in self.config.LETTERS}),
            ("digits", {c: c for c in self.config.DIGITS})
        ]

        loaded_count = 0

        for type_name, char_dict in char_types:
            type_dir = self.config.SYNTH_CHARS_DIR / type_name

            if not type_dir.exists():
                print(f"警告：目录不存在 {type_dir}")
                continue

            # 遍历每个字符目录
            for char_dir in type_dir.iterdir():
                if char_dir.is_dir():
                    # 获取字符名称
                    if type_name == "provinces":
                        char = self.find_chinese_from_pinyin(char_dir.name)
                    else:
                        char = char_dir.name

                    if not char:
                        continue

                    # 加载第一个模板图像
                    template_files = list(char_dir.glob("*.png"))
                    if template_files:
                        template_path = template_files[0]
                        template_img = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)

                        if template_img is None:
                            print(f"警告：无法读取模板 {template_path}")
                            continue

                        # 确保黑底白字（与CNN训练时一致）
                        if np.mean(template_img) > 127:
                            template_img = 255 - template_img

                        # 缩放到标准尺寸（去掉padding）
                        target_height = self.config.CHAR_HEIGHT
                        target_width = self.config.CHAR_WIDTH
                        template_img = cv2.resize(template_img, (target_width, target_height))

                        # 归一化到0-1范围
                        template_img = template_img.astype(np.float32) / 255.0

                        # 保存模板
                        self.templates[char] = template_img
                        self.char_list.append(char)
                        loaded_count += 1

        print(f"  成功加载 {loaded_count} 个字符模板")

    def find_chinese_from_pinyin(self, pinyin):
        """根据拼音找到对应的汉字"""
        for chinese, py in self.config.PROVINCES.items():
            if py == pinyin:
                return chinese
        return None

    def preprocess_char(self, char_img):
        """预处理字符图像（与模板格式一致）"""
        # 确保是灰度图
        if len(char_img.shape) == 3:
            gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = char_img.copy()

        # 确保黑底白字
        if np.mean(gray) > 127:
            gray = 255 - gray

        # 缩放到标准尺寸
        target_height = self.config.CHAR_HEIGHT
        target_width = self.config.CHAR_WIDTH
        resized = cv2.resize(gray, (target_width, target_height))

        # 归一化到0-1范围
        resized = resized.astype(np.float32) / 255.0

        return resized

    def match_char(self, char_img, position=None):
        """
        模板匹配识别单个字符
        """
        # 1. 预处理图像
        processed_img = self.preprocess_char(char_img)

        # 2. 根据位置过滤候选字符（分层识别规则）
        if position is not None:
            if position == 1:
                allowed_chars = list(self.config.PROVINCES.keys())
            elif position == 2:
                allowed_chars = self.config.LETTERS
            else:
                allowed_chars = self.config.LETTERS + self.config.DIGITS
        else:
            allowed_chars = self.char_list

        best_match = None
        best_score = -1
        all_scores = []

        # 3. 对每个允许的字符进行模板匹配
        for char in allowed_chars:
            if char not in self.templates:
                continue

            template = self.templates[char]

            # 使用归一化相关系数匹配
            result = cv2.matchTemplate(
                processed_img.reshape(processed_img.shape[0], processed_img.shape[1], 1),
                template.reshape(template.shape[0], template.shape[1], 1),
                cv2.TM_CCOEFF_NORMED
            )

            score = result[0][0]
            all_scores.append({
                'char': char,
                'score': float(score)
            })

            # 更新最佳匹配
            if score > best_score:
                best_score = score
                best_match = char

        # 4. 如果没有找到匹配，返回未知字符
        if best_match is None:
            return {
                'char': '?',
                'confidence': 0.0,
                'method': 'template_matching',
                'all_scores': []
            }

        # 5. 对分数进行排序，获取Top5备选结果
        all_scores.sort(key=lambda x: x['score'], reverse=True)
        top5 = all_scores[:5]

        # 6. 转换置信度（确保在0-1范围内）
        confidence = max(0.0, min(1.0, best_score))

        return {
            'char': best_match,
            'confidence': confidence,
            'method': 'template_matching',
            'top5': top5
        }

    def recognize_plate(self, char_images):
        """识别整个车牌（7个字符）"""
        if len(char_images) != 7:
            print(f"警告：期望7个字符，得到{len(char_images)}个")

        plate_number = ""
        confidences = []

        # 逐个识别字符
        for i, char_img in enumerate(char_images):
            position = i + 1
            result = self.match_char(char_img, position)

            plate_number += result['char']
            confidences.append(result['confidence'])

        # 计算平均置信度
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return plate_number, avg_confidence


# ============ 第十一部分：SVM+HOG识别器 ============
class SVMRecognizer:
    """SVM+HOG字符识别器 - 机器学习方法"""

    def __init__(self, config=None):
        if config:
            self.config = config
        else:
            self.config = Config

        # HOG参数设置（与字符尺寸匹配）
        self.winSize = (self.config.CHAR_WIDTH, self.config.CHAR_HEIGHT)
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

        # 模型和编码器
        self.svm_model = None
        self.label_encoder = LabelEncoder()

        # 模型文件路径
        self.model_path = self.config.MODELS_DIR / "svm_model.pkl"
        self.encoder_path = self.config.MODELS_DIR / "label_encoder.pkl"

        print(f"✓ SVM识别器初始化完成")
        print(f"  HOG特征维度: {self.get_hog_feature_dimension()}")

    def get_hog_feature_dimension(self):
        """计算HOG特征维度"""
        cells_per_block = (self.blockSize[0] // self.cellSize[0]) * \
                          (self.blockSize[1] // self.cellSize[1])
        n_cells = ((self.winSize[0] - self.blockSize[0]) // self.blockStride[0] + 1) * \
                  ((self.winSize[1] - self.blockSize[1]) // self.blockStride[1] + 1)
        return n_cells * cells_per_block * self.nbins

    def extract_hog_features(self, image):
        """提取图像的HOG特征"""
        # 1. 确保图像尺寸正确
        if image.shape[:2] != (self.config.CHAR_HEIGHT, self.config.CHAR_WIDTH):
            image = cv2.resize(image, (self.config.CHAR_WIDTH, self.config.CHAR_HEIGHT))

        # 2. 确保是灰度图
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 3. 确保黑底白字（与CNN训练一致）
        if np.mean(image) > 127:
            image = 255 - image

        # 4. 归一化到0-1范围
        image = image.astype(np.float32) / 255.0

        # 5. 计算HOG特征
        hog_features = self.hog.compute(image)

        # 6. 展平特征向量
        return hog_features.flatten()

    def load_training_data(self, max_samples_per_char=50):
        """从合成数据和真实数据加载训练数据"""
        print("开始加载SVM训练数据...")

        features_list = []
        labels_list = []

        # 获取所有字符
        all_chars = list(self.config.CHAR_TO_LABEL.keys())
        char_count = len(all_chars)

        for char_idx, char in enumerate(all_chars):
            # 显示进度
            if char_idx % 10 == 0:
                print(f"  加载进度: {char_idx + 1}/{char_count}")

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
            for img in char_samples[:max_samples_per_char]:
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
        """根据字符类型获取对应的目录"""
        dirs = []

        if char in self.config.PROVINCES:
            pinyin = self.config.PROVINCES[char]
            dirs.append(base_dir / "provinces" / pinyin)
        if char in self.config.LETTERS:
            dirs.append(base_dir / "letters" / char)
        if char in self.config.DIGITS:
            dirs.append(base_dir / "digits" / char)

        return dirs

    def train(self, use_cached=True):
        """训练SVM模型"""
        # 检查是否有缓存模型
        if use_cached and self.model_path.exists() and self.encoder_path.exists():
            try:
                self.load_model()
                print("✓ 加载缓存的SVM模型")
                return True
            except:
                print("⚠ 缓存模型加载失败，重新训练")

        print("开始训练SVM模型...")

        # 1. 加载训练数据
        X, y = self.load_training_data(max_samples_per_char=50)

        if len(X) == 0:
            print("✗ 错误：没有训练数据")
            return False

        # 2. 编码标签（字符 -> 数字）
        y_encoded = self.label_encoder.fit_transform(y)

        # 3. 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        print(f"  训练集: {len(X_train)} 样本")
        print(f"  测试集: {len(X_test)} 样本")

        # 4. 训练SVM
        print("  训练SVM分类器...")

        self.svm_model = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=42,
            verbose=False
        )

        self.svm_model.fit(X_train, y_train)

        # 5. 评估模型
        y_pred = self.svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"✓ SVM训练完成")
        print(f"  测试准确率: {accuracy:.2%}")
        print(f"  类别数: {len(self.label_encoder.classes_)}")

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
            encoder_data = {
                'classes': self.label_encoder.classes_,
                'fitted': True
            }
            joblib.dump(encoder_data, self.encoder_path)

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
        """识别单个字符"""
        # 如果模型未加载，尝试加载
        if self.svm_model is None:
            if not self.load_model():
                return {'char': '?', 'confidence': 0.0, 'method': 'svm'}

        try:
            # 1. 提取HOG特征
            features = self.extract_hog_features(char_img)
        except Exception as e:
            print(f"SVM特征提取失败: {e}")
            return {'char': '?', 'confidence': 0.0, 'method': 'svm'}

        # 2. 根据位置过滤允许的字符类别（分层识别）
        if position is not None:
            if position == 1:
                allowed_chars = list(self.config.PROVINCES.keys())
            elif position == 2:
                allowed_chars = self.config.LETTERS
            else:
                allowed_chars = self.config.LETTERS + self.config.DIGITS
        else:
            allowed_chars = list(self.config.CHAR_TO_LABEL.keys())

        # 3. 获取这些字符在编码器中的索引
        allowed_indices = []
        for char in allowed_chars:
            try:
                idx = np.where(self.label_encoder.classes_ == char)[0][0]
                allowed_indices.append(idx)
            except:
                continue

        if not allowed_indices:
            return {'char': '?', 'confidence': 0.0, 'method': 'svm'}

        # 4. 预测概率
        try:
            probabilities = self.svm_model.predict_proba([features])[0]
        except:
            # 如果概率预测失败，使用普通预测
            pred_label = self.svm_model.predict([features])[0]
            pred_char = self.label_encoder.inverse_transform([pred_label])[0]
            return {
                'char': pred_char,
                'confidence': 0.8,
                'method': 'svm'
            }

        # 5. 只考虑允许的类别
        best_idx = -1
        best_prob = -1

        for idx in allowed_indices:
            if idx < len(probabilities) and probabilities[idx] > best_prob:
                best_prob = probabilities[idx]
                best_idx = idx

        if best_idx == -1:
            return {'char': '?', 'confidence': 0.0, 'method': 'svm'}

        # 6. 解码字符
        try:
            best_char = self.label_encoder.inverse_transform([best_idx])[0]
        except:
            best_char = '?'

        # 7. 获取Top5备选
        top5 = []
        for idx in allowed_indices:
            if idx < len(probabilities):
                prob = probabilities[idx]
                if len(top5) < 5:
                    top5.append((idx, prob))
                    top5.sort(key=lambda x: x[1], reverse=True)
                elif prob > top5[-1][1]:
                    top5[-1] = (idx, prob)
                    top5.sort(key=lambda x: x[1], reverse=True)

        # 转换top5格式
        top5_results = []
        for idx, prob in top5:
            try:
                char = self.label_encoder.inverse_transform([idx])[0]
                top5_results.append({'char': char, 'score': float(prob)})
            except:
                pass

        return {
            'char': best_char,
            'confidence': float(best_prob),
            'method': 'svm',
            'top5': top5_results
        }

    def recognize_plate(self, char_images):
        """识别整个车牌"""
        if len(char_images) != 7:
            print(f"警告：期望7个字符，得到{len(char_images)}个")

        plate_number = ""
        confidences = []

        # 逐个识别字符
        for i, char_img in enumerate(char_images):
            position = i + 1
            result = self.predict_char(char_img, position)

            plate_number += result['char']
            confidences.append(result['confidence'])

        # 计算平均置信度
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return plate_number, avg_confidence


# ============ 第十二部分：性能评估器 ============
class PerformanceEvaluator:
    """性能评估器 - 自动计算准确率并生成报告"""

    def __init__(self, config=None):
        if config:
            self.config = config
        else:
            self.config = Config

        # 评估结果存储
        self.evaluation_results = {
            'plate_detection': {'total': 0, 'correct': 0, 'accuracy': 0.0},
            'char_segmentation': {'total': 0, 'correct': 0, 'accuracy': 0.0},
            'char_recognition': {
                'cnn': {'total': 0, 'correct': 0, 'accuracy': 0.0},
                'template': {'total': 0, 'correct': 0, 'accuracy': 0.0},
                'svm': {'total': 0, 'correct': 0, 'accuracy': 0.0}
            },
            'overall': {
                'cnn': {'total': 0, 'correct': 0, 'accuracy': 0.0},
                'template': {'total': 0, 'correct': 0, 'accuracy': 0.0},
                'svm': {'total': 0, 'correct': 0, 'accuracy': 0.0}
            }
        }

        # 测试数据存储
        self.test_cases = []
        self.report_dir = self.config.RESULTS_DIR / "performance_reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)

        print("✓ 性能评估器初始化完成")

    def evaluate_plate_detection(self, image_path, has_plate, detected):
        """评估车牌定位性能"""
        self.evaluation_results['plate_detection']['total'] += 1

        # 判断是否正确
        if has_plate == detected:
            self.evaluation_results['plate_detection']['correct'] += 1

        # 计算准确率
        total = self.evaluation_results['plate_detection']['total']
        correct = self.evaluation_results['plate_detection']['correct']
        if total > 0:
            accuracy = correct / total
            self.evaluation_results['plate_detection']['accuracy'] = accuracy

        # 记录测试案例
        test_case = {
            'image': Path(image_path).name,
            'module': 'plate_detection',
            'has_plate': has_plate,
            'detected': detected,
            'correct': has_plate == detected,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.test_cases.append(test_case)

        return has_plate == detected

    def evaluate_char_segmentation(self, image_path, expected_chars, segmented_chars):
        """评估字符分割性能"""
        self.evaluation_results['char_segmentation']['total'] += 1

        # 判断是否正确（分割出7个字符）
        is_correct = segmented_chars == expected_chars

        if is_correct:
            self.evaluation_results['char_segmentation']['correct'] += 1

        # 计算准确率
        total = self.evaluation_results['char_segmentation']['total']
        correct = self.evaluation_results['char_segmentation']['correct']
        if total > 0:
            accuracy = correct / total
            self.evaluation_results['char_segmentation']['accuracy'] = accuracy

        # 记录测试案例
        test_case = {
            'image': Path(image_path).name,
            'module': 'char_segmentation',
            'expected_chars': expected_chars,
            'segmented_chars': segmented_chars,
            'correct': is_correct,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.test_cases.append(test_case)

        return is_correct

    def evaluate_char_recognition(self, image_path, ground_truth,
                                  cnn_result, template_result, svm_result):
        """评估字符识别性能"""
        # 确保车牌号长度一致
        if len(ground_truth) != 7:
            print(f"警告：真实车牌号长度不是7位: {ground_truth}")
            return False, False, False

        # 评估每个方法
        cnn_correct = self._evaluate_single_recognition(
            ground_truth, cnn_result, 'cnn')
        template_correct = self._evaluate_single_recognition(
            ground_truth, template_result, 'template')
        svm_correct = self._evaluate_single_recognition(
            ground_truth, svm_result, 'svm')

        # 评估整体识别（整个车牌正确）
        cnn_overall = ground_truth == cnn_result
        template_overall = ground_truth == template_result
        svm_overall = ground_truth == svm_result

        # 更新整体统计
        self._update_overall_stats('cnn', cnn_overall)
        self._update_overall_stats('template', template_overall)
        self._update_overall_stats('svm', svm_overall)

        # 记录测试案例
        test_case = {
            'image': Path(image_path).name,
            'module': 'char_recognition',
            'ground_truth': ground_truth,
            'cnn_result': cnn_result,
            'cnn_correct': cnn_overall,
            'template_result': template_result,
            'template_correct': template_overall,
            'svm_result': svm_result,
            'svm_correct': svm_overall,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.test_cases.append(test_case)

        return cnn_overall, template_overall, svm_overall

    def _evaluate_single_recognition(self, ground_truth, predicted_result, method):
        """评估单个识别方法的性能"""
        if len(predicted_result) != 7:
            return False

        correct_count = 0
        total_chars = 7

        # 逐个字符比较
        for i in range(total_chars):
            if i < len(predicted_result) and predicted_result[i] == ground_truth[i]:
                correct_count += 1

        # 更新统计
        self.evaluation_results['char_recognition'][method]['total'] += total_chars
        self.evaluation_results['char_recognition'][method]['correct'] += correct_count

        # 计算准确率
        total = self.evaluation_results['char_recognition'][method]['total']
        correct = self.evaluation_results['char_recognition'][method]['correct']
        if total > 0:
            accuracy = correct / total
            self.evaluation_results['char_recognition'][method]['accuracy'] = accuracy

        return correct_count == total_chars

    def _update_overall_stats(self, method, is_correct):
        """更新整体识别统计"""
        self.evaluation_results['overall'][method]['total'] += 1
        if is_correct:
            self.evaluation_results['overall'][method]['correct'] += 1

        total = self.evaluation_results['overall'][method]['total']
        correct = self.evaluation_results['overall'][method]['correct']
        if total > 0:
            accuracy = correct / total
            self.evaluation_results['overall'][method]['accuracy'] = accuracy

    def generate_report(self, report_name=None):
        """生成性能评估报告"""
        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"performance_report_{timestamp}"

        report_path = self.report_dir / f"{report_name}.txt"

        # 生成报告内容
        report_content = self._create_report_content()

        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"✓ 性能评估报告已生成: {report_path}")
        return report_path

    def _create_report_content(self):
        """创建报告内容"""
        content = []
        content.append("=" * 70)
        content.append("车牌识别系统性能评估报告")
        content.append("=" * 70)
        content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"测试案例总数: {len(self.test_cases)}")
        content.append("")

        # 1. 车牌定位性能
        det = self.evaluation_results['plate_detection']
        content.append("1. 车牌定位性能")
        content.append("-" * 40)
        content.append(f"   测试总数: {det['total']}")
        content.append(f"   正确数: {det['correct']}")
        content.append(f"   准确率: {det['accuracy']:.2%}")
        content.append("")

        # 2. 字符分割性能
        seg = self.evaluation_results['char_segmentation']
        content.append("2. 字符分割性能")
        content.append("-" * 40)
        content.append(f"   测试总数: {seg['total']}")
        content.append(f"   正确数: {seg['correct']}")
        content.append(f"   准确率: {seg['accuracy']:.2%}")
        content.append("")

        # 3. 字符识别性能（逐个字符）
        recog = self.evaluation_results['char_recognition']
        content.append("3. 字符识别性能（逐个字符）")
        content.append("-" * 40)

        for method in ['cnn', 'template', 'svm']:
            m = recog[method]
            content.append(f"   {method.upper()}方法:")
            content.append(f"     测试字符总数: {m['total']}")
            content.append(f"     正确字符数: {m['correct']}")
            content.append(f"     字符准确率: {m['accuracy']:.2%}")

        content.append("")

        # 4. 整体识别性能（整个车牌）
        overall = self.evaluation_results['overall']
        content.append("4. 整体识别性能（整个车牌）")
        content.append("-" * 40)

        for method in ['cnn', 'template', 'svm']:
            m = overall[method]
            content.append(f"   {method.upper()}方法:")
            content.append(f"     测试车牌总数: {m['total']}")
            content.append(f"     正确车牌数: {m['correct']}")
            content.append(f"     车牌准确率: {m['accuracy']:.2%}")

        content.append("")

        # 5. 对比分析
        content.append("5. 方法对比分析")
        content.append("-" * 40)

        # 找出最佳方法
        best_method = max(['cnn', 'template', 'svm'],
                          key=lambda x: overall[x]['accuracy'])

        content.append(f"   最佳识别方法: {best_method.upper()}")
        content.append(f"   最佳准确率: {overall[best_method]['accuracy']:.2%}")
        content.append("")

        # 6. 技术指标符合情况
        content.append("6. 技术指标符合情况")
        content.append("-" * 40)

        requirements = {
            '车牌定位': ('≥85%', det['accuracy'] >= 0.85),
            '字符分割': ('≥85%', seg['accuracy'] >= 0.85),
            '字符识别': ('≥80%', recog['cnn']['accuracy'] >= 0.80),
            '系统整体': ('≥70%', overall['cnn']['accuracy'] >= 0.70)
        }

        for module, (req, passed) in requirements.items():
            status = "✓ 符合" if passed else "✗ 未达到"
            content.append(f"   {module}: 要求{req} {status}")

        content.append("")
        content.append("=" * 70)
        content.append("报告结束")
        content.append("=" * 70)

        return "\n".join(content)

    def get_summary(self):
        """获取性能摘要（用于GUI显示）"""
        det = self.evaluation_results['plate_detection']
        seg = self.evaluation_results['char_segmentation']
        overall = self.evaluation_results['overall']

        summary = f"""
性能摘要：
1. 车牌定位准确率: {det['accuracy']:.2%}
2. 字符分割准确率: {seg['accuracy']:.2%}
3. 整体识别准确率:
   - CNN: {overall['cnn']['accuracy']:.2%}
   - 模板匹配: {overall['template']['accuracy']:.2%}
   - SVM: {overall['svm']['accuracy']:.2%}
"""
        return summary

    def save_test_cases(self):
        """保存详细的测试案例到CSV文件"""
        if not self.test_cases:
            print("警告：没有测试案例数据")
            return

        csv_path = self.report_dir / "test_cases.csv"

        # 转换为DataFrame
        df = pd.DataFrame(self.test_cases)

        # 保存到CSV
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✓ 测试案例已保存到: {csv_path}")

        return csv_path


# ============ 第十三部分：手动数据收集器 ============
class ManualDataCollector:
    """手动收集真实车牌字符数据"""

    def __init__(self, parent_gui):
        self.parent = parent_gui
        self.collection_window = None
        self.current_chars = []
        self.char_labels = []
        self.tk_images = []

    def start_collection(self, char_images):
        """开始收集数据"""
        if not char_images:
            messagebox.showwarning("警告", "没有找到字符！")
            return

        self.current_chars = char_images
        self.char_labels = ['?'] * len(char_images)
        self.tk_images = []

        self.show_collection_window()

    def show_collection_window(self):
        """显示数据收集窗口"""
        # 如果已有窗口，先关闭
        if self.collection_window and self.collection_window.winfo_exists():
            self.collection_window.destroy()

        # 创建新窗口
        self.collection_window = tk.Toplevel(self.parent.root)
        self.collection_window.title("手动收集训练数据 - 关键步骤！")
        self.collection_window.geometry("900x500")

        # 添加说明
        info_frame = tk.Frame(self.collection_window)
        info_frame.pack(fill=tk.X, padx=10, pady=10)

        info_text = ("重要：请为每个字符输入正确标签！\n"
                     "规则：第1个=汉字，第2个=字母，后面=字母或数字\n"
                     "例如：皖A12345 → 第1=皖，第2=A，第3=1，第4=2，第5=3，第6=4，第7=5")

        tk.Label(info_frame, text=info_text, font=("Arial", 10),
                 fg="red", justify=tk.LEFT).pack()

        # 创建字符显示区域
        chars_frame = tk.Frame(self.collection_window)
        chars_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 显示所有字符
        for i, char_img in enumerate(self.current_chars):
            char_frame = tk.Frame(chars_frame)
            char_frame.grid(row=0, column=i, padx=5, pady=5)

            # 显示字符序号
            tk.Label(char_frame, text=f"字符 {i + 1}", font=("Arial", 10, "bold")).pack()

            # 显示字符图像
            pil_img = Image.fromarray(char_img)
            tk_img = ImageTk.PhotoImage(pil_img)
            self.tk_images.append(tk_img)

            img_label = tk.Label(char_frame, image=tk_img)
            img_label.pack()

            # 标签输入框
            label_var = tk.StringVar(value="?")
            entry = tk.Entry(char_frame, textvariable=label_var,
                             width=3, font=("Arial", 12), justify=tk.CENTER)
            entry.pack(pady=5)

            # 绑定变量
            self.char_labels[i] = label_var

        # 按钮区域
        button_frame = tk.Frame(self.collection_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        # 保存按钮
        save_btn = tk.Button(button_frame, text="💾 保存所有字符",
                             command=self.save_all_chars,
                             bg="#27ae60", fg="white",
                             font=("Arial", 12), height=2)
        save_btn.pack(side=tk.LEFT, padx=5)

        # 取消按钮
        cancel_btn = tk.Button(button_frame, text="取消",
                               command=self.collection_window.destroy,
                               bg="#e74c3c", fg="white",
                               font=("Arial", 12), height=2)
        cancel_btn.pack(side=tk.RIGHT, padx=5)

        # 添加快速输入按钮（常用字符）
        quick_frame = tk.Frame(self.collection_window)
        quick_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(quick_frame, text="快速输入:", font=("Arial", 10)).pack(side=tk.LEFT)

        # 常用汉字按钮
        common_chinese = ['京', '沪', '浙', '苏', '皖', '粤']
        for char in common_chinese:
            btn = tk.Button(quick_frame, text=char, width=3,
                            command=lambda c=char: self.quick_input(c, 0))
            btn.pack(side=tk.LEFT, padx=2)

        # 常用字母按钮
        tk.Label(quick_frame, text="  字母:", font=("Arial", 10)).pack(side=tk.LEFT)
        common_letters = ['A', 'B', 'C', 'D', 'E', 'F']
        for char in common_letters:
            btn = tk.Button(quick_frame, text=char, width=3,
                            command=lambda c=char: self.quick_input(c, 1))
            btn.pack(side=tk.LEFT, padx=2)

        # 数字按钮
        tk.Label(quick_frame, text="  数字:", font=("Arial", 10)).pack(side=tk.LEFT)
        for i in range(10):
            btn = tk.Button(quick_frame, text=str(i), width=3,
                            command=lambda c=str(i): self.quick_input(c, 2))
            btn.pack(side=tk.LEFT, padx=1)

    def quick_input(self, char, char_type):
        """快速输入字符（按钮点击）"""
        messagebox.showinfo("提示", f"点击了 {char}，请在对应输入框输入")

    def save_all_chars(self):
        """保存所有字符到训练集"""
        saved_count = 0

        for i, (char_img, label_var) in enumerate(zip(self.current_chars, self.char_labels)):
            label = label_var.get().strip()

            if label == '?' or label == '':
                continue

            # 确保标签是ASCII字符（无中文）
            ascii_label = label
            if not label.isascii():
                if label in Config.PROVINCES:
                    ascii_label = Config.PROVINCES.get(label, label)

            # 根据字符类型确定保存位置
            if label in Config.PROVINCES:
                pinyin = Config.PROVINCES.get(label, "UNKNOWN")
                save_dir = Config.REAL_TRAIN_DIR / "provinces" / pinyin
                char_type = "汉字"
            elif label in Config.LETTERS:
                save_dir = Config.REAL_TRAIN_DIR / "letters" / label
                char_type = "字母"
                ascii_label = label
            elif label in Config.DIGITS:
                save_dir = Config.REAL_TRAIN_DIR / "digits" / label
                char_type = "数字"
                ascii_label = label
            else:
                self.parent.log(f"⚠ 未知字符 '{label}'，跳过")
                continue

            # 创建目录（如果不存在） - 确保路径是ASCII
            save_dir.mkdir(parents=True, exist_ok=True)

            # 生成安全的文件名（ASCII字符）
            existing_files = list(save_dir.glob("*.png"))
            next_num = len(existing_files)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"real_{ascii_label}_{timestamp}_{next_num:04d}.png"

            # 保存图像
            save_path = save_dir / filename
            cv2.imwrite(str(save_path), char_img)

            saved_count += 1
            self.parent.log(f"✓ 保存字符 {i + 1}: '{label}' ({char_type}) 为 {filename}")

        # 关闭窗口
        self.collection_window.destroy()

        # 显示结果
        messagebox.showinfo("成功", f"保存了 {saved_count} 个字符到训练集！\n"
                                    f"现在可以重新训练模型了。")

        # 提示重新训练
        self.parent.log("⚠ 重要：请点击'训练CNN模型'按钮重新训练！")


# ============ 第十四部分：主GUI类 - 增强版 ============
class EnhancedPlateRecognitionGUI:
    """车牌识别系统主界面 - 回归v7.0处理流程版"""

    def __init__(self, root):
        self.root = root

        # 初始化各个组件
        self.purifier = EnhancedPlatePurification()
        self.cnn_recognizer = HierarchicalRecognizer()
        self.template_matcher = TemplateMatcher()
        self.svm_recognizer = SVMRecognizer()
        self.evaluator = PerformanceEvaluator()
        self.data_collector = ManualDataCollector(self)

        # 状态变量
        self.current_char_images = []
        self.current_image = None
        self.ground_truth = ""

        # 识别结果存储
        self.cnn_result = ""
        self.template_result = ""
        self.svm_result = ""

        # 设置GUI界面
        self.setup_gui()

        # 初始日志
        self.log("车牌识别系统 v8.1 - 回归v7.0处理流程版")
        self.log("字符分割回归v7.0经典算法，保留完整评估功能")
        self.log("=" * 60)

        # 检查模型文件
        self.check_models()

    def setup_gui(self):
        """设置图形用户界面 - 增强版"""
        self.root.title("车牌识别系统 v8.1 - 回归v7.0处理流程版")
        self.root.geometry("1600x1000")

        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 标题
        title_label = ttk.Label(main_frame, text="车牌识别系统 v8.1 - 回归v7.0处理流程版",
                                font=("Arial", 20, "bold"), foreground="#2c3e50")
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # ========== 左侧控制面板 ==========
        control_frame = ttk.LabelFrame(main_frame, text="系统控制", padding="15", width=400)
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.N, tk.S), padx=(0, 10))
        control_frame.grid_propagate(False)

        # 进度显示
        ttk.Label(control_frame, text="当前进度:").pack(anchor=tk.W)

        self.progress_var = tk.StringVar(value="就绪")
        progress_label = ttk.Label(control_frame, textvariable=self.progress_var,
                                   foreground="blue", font=("Arial", 10, "bold"))
        progress_label.pack(anchor=tk.W, pady=(5, 0))

        self.progress_bar = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, pady=(0, 15))

        # 按钮区域
        button_frame = tk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=(0, 20))

        # 按钮配置列表
        button_configs = [
            ("📁 创建目录", self.create_directories, "#2c3e50", "创建所有必要的文件夹"),
            ("🔄 生成合成数据", self.generate_synthetic_data, "#27ae60", "用simhei.ttf生成字符模板"),
            ("📸 选择车牌图像", self.select_plate_image, "#d35400", "选择要识别的车牌图片"),
            ("🔧 处理车牌图像", self.process_plate_image, "#8e44ad", "检测、分割车牌字符（完整可视化）"),
            ("📸 保存处理截图", self.save_processing_screenshots, "#9b59b6", "保存所有处理步骤截图用于实验报告"),
            ("💾 收集训练数据", self.collect_training_data, "#e67e22", "重要！保存真实字符到训练集"),
            ("⚙️ 训练CNN模型", self.train_cnn_model, "#2980b9", "使用收集的数据训练模型"),
            ("⚙️ 训练SVM模型", self.train_svm_model, "#8e44ad", "训练SVM识别模型"),
            ("🔍 CNN识别", self.recognize_chars, "#c0392b", "使用CNN识别分割的字符"),
            ("🔍 三种方法对比", self.compare_all_methods, "#c0392b", "对比CNN/模板/SVM三种方法"),
            ("📊 性能评估", self.run_performance_evaluation, "#34495e", "运行性能评估测试"),
            ("💾 保存结果", self.save_results, "#34495e", "保存识别结果到文件"),
        ]

        # 创建按钮
        for text, command, color, tooltip in button_configs:
            btn = tk.Button(button_frame, text=text, command=command,
                            bg=color, fg="white", font=("Arial", 10),
                            relief=tk.RAISED, height=2)
            btn.pack(fill=tk.X, pady=3)

            # 简单提示
            btn.bind("<Enter>", lambda e, t=tooltip: self.update_status(t))
            btn.bind("<Leave>", lambda e: self.update_status("就绪"))

        # ========== 数据收集状态 ==========
        collection_frame = ttk.LabelFrame(control_frame, text="数据收集状态", padding="10")
        collection_frame.pack(fill=tk.X, pady=(10, 0))

        self.collection_status = tk.StringVar(value="等待收集真实数据...")
        status_label = ttk.Label(collection_frame, textvariable=self.collection_status,
                                 font=("Arial", 9), foreground="orange")
        status_label.pack(anchor=tk.W)

        # ========== 文件信息 ==========
        info_frame = ttk.LabelFrame(control_frame, text="文件信息", padding="10")
        info_frame.pack(fill=tk.X, pady=(10, 0))

        self.file_label = ttk.Label(info_frame, text="未选择文件",
                                    font=("Arial", 10), wraplength=350)
        self.file_label.pack(anchor=tk.W)

        self.char_count_label = ttk.Label(info_frame, text="字符数: 0")
        self.char_count_label.pack(anchor=tk.W)

        # ========== 右侧显示区域 ==========
        display_frame = ttk.LabelFrame(main_frame, text="显示区域", padding="15")
        display_frame.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 选项卡控件
        self.notebook = ttk.Notebook(display_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 定义所有选项卡
        tabs = [
            ("original", "原始图像"),
            ("gray", "灰度图像"),
            ("edges", "传统边缘检测"),
            ("color_mask", "颜色特征掩码"),
            ("plate_region", "车牌区域"),
            ("binary", "二值化处理"),
            ("histogram", "投影直方图"),
            ("segmentation", "字符分割"),
            ("chars", "字符预览"),
            ("comparison", "方法对比"),
            ("performance", "性能评估"),
        ]

        self.tabs = {}

        for tab_id, tab_name in tabs:
            frame = tk.Frame(self.notebook, bg="white")
            self.notebook.add(frame, text=tab_name)

            canvas = tk.Canvas(frame, bg="white", highlightthickness=0)
            canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            self.tabs[tab_id] = {
                "frame": frame,
                "canvas": canvas,
                "image_ref": None
            }

        # ========== 结果显示区域 ==========
        result_frame = ttk.Frame(display_frame)
        result_frame.pack(fill=tk.X, pady=(10, 0))

        # 1. CNN结果
        self.cnn_frame = ttk.LabelFrame(result_frame, text="CNN识别结果", padding="10")
        self.cnn_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.cnn_result_var = tk.StringVar(value="等待识别...")
        cnn_label = ttk.Label(self.cnn_frame, textvariable=self.cnn_result_var,
                              font=("Arial", 16, "bold"), foreground="#2c3e50")
        cnn_label.pack(pady=5)

        self.cnn_confidence_var = tk.StringVar(value="置信度: 0.00%")
        cnn_confidence_label = ttk.Label(self.cnn_frame, textvariable=self.cnn_confidence_var,
                                         font=("Arial", 11), foreground="#27ae60")
        cnn_confidence_label.pack()

        # 2. 模板匹配结果
        self.template_frame = ttk.LabelFrame(result_frame, text="模板匹配结果", padding="10")
        self.template_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.template_result_var = tk.StringVar(value="等待识别...")
        template_label = ttk.Label(self.template_frame, textvariable=self.template_result_var,
                                   font=("Arial", 16, "bold"), foreground="#8e44ad")
        template_label.pack(pady=5)

        self.template_confidence_var = tk.StringVar(value="置信度: 0.00%")
        template_confidence_label = ttk.Label(self.template_frame,
                                              textvariable=self.template_confidence_var,
                                              font=("Arial", 11), foreground="#9b59b6")
        template_confidence_label.pack()

        # 3. SVM结果
        self.svm_frame = ttk.LabelFrame(result_frame, text="SVM识别结果", padding="10")
        self.svm_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        self.svm_result_var = tk.StringVar(value="等待识别...")
        svm_label = ttk.Label(self.svm_frame, textvariable=self.svm_result_var,
                              font=("Arial", 16, "bold"), foreground="#d35400")
        svm_label.pack(pady=5)

        self.svm_confidence_var = tk.StringVar(value="置信度: 0.00%")
        svm_confidence_label = ttk.Label(self.svm_frame, textvariable=self.svm_confidence_var,
                                         font=("Arial", 11), foreground="#e67e22")
        svm_confidence_label.pack()

        # ========== 日志区域 ==========
        log_frame = ttk.LabelFrame(display_frame, text="系统日志", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.log_text = ScrolledText(log_frame, width=80, height=10,
                                     font=("Consolas", 9), wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # ========== 状态栏 ==========
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                               relief=tk.SUNKEN, padding=(5, 2))
        status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))

        # ========== 布局配置 ==========
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

    # ============ GUI功能方法 ============

    def create_directories(self):
        """创建所有目录"""
        self.start_progress("创建目录...")

        def create():
            try:
                Config.create_dirs()
                self.log("✓ 目录创建完成")
                messagebox.showinfo("成功", "目录结构创建完成!")
            except Exception as e:
                self.log(f"✗ 创建目录失败: {e}")
            finally:
                self.stop_progress()

        threading.Thread(target=create, daemon=True).start()

    def generate_synthetic_data(self):
        """生成合成训练数据"""
        self.start_progress("生成合成数据...")

        def generate():
            try:
                generator = FixedCharGenerator()
                total = generator.generate_all_chars(
                    samples_per_char=Config.SAMPLES_PER_CHAR,
                    progress_callback=self.log
                )

                if total > 0:
                    self.log(f"✓ 合成数据生成完成: {total}张图片")
                    messagebox.showinfo("成功", f"合成数据生成完成!\n共{total}张字符图片")
                else:
                    self.log("✗ 合成数据生成失败")
            except Exception as e:
                self.log(f"✗ 生成失败: {e}")
                import traceback
                self.log(traceback.format_exc())
            finally:
                self.stop_progress()

        threading.Thread(target=generate, daemon=True).start()

    def collect_training_data(self):
        """收集训练数据 - 关键功能！"""
        if not self.current_char_images:
            messagebox.showwarning("警告", "请先处理车牌图像!")
            return

        self.log("开始收集训练数据...")
        self.log("重要：请准确标注每个字符！")

        # 弹出数据收集窗口
        self.data_collector.start_collection(self.current_char_images)

    def train_cnn_model(self):
        """训练CNN模型"""
        # 检查是否有数据
        synth_dir = Config.SYNTH_CHARS_DIR
        real_dir = Config.REAL_TRAIN_DIR

        # 检查是否有真实数据
        has_real_data = any(real_dir.rglob("*.png"))

        if not any(synth_dir.rglob("*.png")) and not has_real_data:
            messagebox.showwarning("警告", "请先生成合成数据或收集真实数据!")
            return

        self.start_progress("训练CNN模型...")

        def train():
            try:
                trainer = ModelTrainer('cnn')

                # 如果有真实数据，使用真实数据训练
                use_real_data = has_real_data

                success, accuracy = trainer.train_cnn(
                    progress_callback=self.log,
                    use_real_data=use_real_data
                )

                if success:
                    self.log(f"✓ CNN训练完成! 准确率: {accuracy:.2f}%")
                    if use_real_data:
                        self.log("✓ 本次训练使用了真实数据，效果应该更好！")

                    messagebox.showinfo("成功", f"CNN训练完成!\n验证准确率: {accuracy:.2f}%")
                else:
                    self.log("✗ CNN训练失败")
            except Exception as e:
                self.log(f"✗ 训练错误: {e}")
                import traceback
                self.log(traceback.format_exc())
            finally:
                self.stop_progress()

        threading.Thread(target=train, daemon=True).start()

    def train_svm_model(self):
        """训练SVM模型"""
        self.start_progress("训练SVM模型...")

        def train():
            try:
                success = self.svm_recognizer.train()

                if success:
                    self.log("✓ SVM模型训练完成")
                    messagebox.showinfo("成功", "SVM模型训练完成！")
                else:
                    self.log("✗ SVM模型训练失败")
            except Exception as e:
                self.log(f"✗ SVM训练错误: {e}")
                import traceback
                self.log(traceback.format_exc())
            finally:
                self.stop_progress()

        threading.Thread(target=train, daemon=True).start()

    def select_plate_image(self):
        """选择车牌图像"""
        file_path = filedialog.askopenfilename(
            title="选择车牌图像",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
        )

        if file_path:
            self.current_image = file_path
            filename = Path(file_path).name

            # 显示文件名（缩短）
            display_name = filename[:30] + "..." if len(filename) > 30 else filename
            self.file_label.config(text=display_name)
            self.log(f"选择图像: {filename}")

            # 加载并显示图像
            img = cv2.imread(file_path)
            self.display_image(img, "original", "原始图像")

    def process_plate_image(self):
        """处理车牌图像（增强版，显示所有处理步骤）"""
        if not self.current_image:
            messagebox.showwarning("警告", "请先选择图像!")
            return

        self.start_progress("处理车牌图像（显示所有步骤）...")

        def process():
            try:
                self.log("=" * 60)
                self.log("开始完整图像处理流程")
                self.log("字符分割回归v7.0经典算法")
                self.log("=" * 60)

                # 使用增强的处理流程
                success, steps_completed = self.purifier.process_image_complete(
                    self.current_image, self.log
                )

                if success:
                    # 显示所有处理步骤结果
                    if self.purifier.gray_image is not None:
                        self.display_image(self.purifier.gray_image, "gray", "灰度图像")

                    if self.purifier.edges_image is not None:
                        self.display_image(self.purifier.edges_image, "edges", "传统边缘检测")

                    if self.purifier.color_mask_image is not None:
                        self.display_image(self.purifier.color_mask_image, "color_mask", "颜色特征掩码")

                    if self.purifier.plate_region is not None:
                        self.display_image(self.purifier.plate_region, "plate_region", "车牌区域检测")

                    if self.purifier.white_bg_plate is not None:
                        self.display_image(self.purifier.white_bg_plate, "binary", "二值化处理")

                    if self.purifier.histogram_image is not None:
                        self.display_image(self.purifier.histogram_image, "histogram", "投影直方图分析")

                    # 显示字符分割结果
                    if self.purifier.cropped_chars:
                        # 在二值图像上绘制分割线
                        if self.purifier.white_bg_plate is not None:
                            segmentation_img = cv2.cvtColor(self.purifier.white_bg_plate, cv2.COLOR_GRAY2BGR)
                            height, width = segmentation_img.shape[:2]

                            # 绘制分割线
                            for i, (start, end) in enumerate(self.purifier.char_segments):
                                color = (0, 255, 0)
                                # 绘制矩形框（高度减少30%）
                                short_height = int(height * 0.75)
                                y_start = int((height - short_height) / 2)
                                cv2.rectangle(segmentation_img, (start, y_start), (end, y_start + short_height), color, 2)

                                # 添加字符序号
                                cv2.putText(segmentation_img, str(i + 1),
                                            (start + (end - start) // 2 - 5, 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                            self.display_image(segmentation_img, "segmentation",
                                               f"字符分割 ({len(self.purifier.cropped_chars)}个)")

                        # 显示字符预览（使用v7.0简单方法）
                        if len(self.purifier.cropped_chars) <= 7:
                            char_height = Config.CHAR_HEIGHT + 2 * Config.PADDING
                            char_width = Config.CHAR_WIDTH + 2 * Config.PADDING

                            # 创建预览画布
                            preview_img = np.zeros((char_height, char_width * len(self.purifier.cropped_chars)),
                                                   dtype=np.uint8)

                            # 将所有字符拼接在一起（简单方法）
                            for i, char_img in enumerate(self.purifier.cropped_chars):
                                x_offset = i * char_width
                                preview_img[:, x_offset:x_offset + char_width] = char_img

                                # 添加字符序号
                                cv2.putText(preview_img, str(i + 1),
                                            (x_offset + 5, 8),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, 128, 1)

                            self.display_image(preview_img, "chars",
                                               f"字符预览 ({len(self.purifier.cropped_chars)}个字符)")

                        # 更新字符数量显示
                        if hasattr(self, 'char_count_label'):
                            self.char_count_label.config(text=f"字符数: {len(self.purifier.cropped_chars)}")

                        # 保存字符图像
                        self.current_char_images = self.purifier.cropped_chars

                        # 更新状态
                        if hasattr(self, 'collection_status'):
                            self.collection_status.set(f"有{len(self.current_char_images)}个字符可收集！")

                    self.log("✓ 所有处理步骤可视化完成")

                else:
                    self.log("✗ 处理流程失败")

                # 提示可以保存截图
                self.log("提示：可以切换到不同选项卡查看处理结果")
                self.log("使用截图工具保存图片用于实验报告")

            except Exception as e:
                self.log(f"✗ 处理错误: {e}")
                import traceback
                self.log(traceback.format_exc())
            finally:
                self.stop_progress()

        threading.Thread(target=process, daemon=True).start()

    def save_processing_screenshots(self):
        """保存所有处理步骤的截图（新增功能）"""
        if not self.current_image:
            messagebox.showwarning("警告", "请先处理车牌图像!")
            return

        # 选择保存目录
        save_dir = filedialog.askdirectory(title="选择截图保存目录",
                                           initialdir=str(Config.RESULTS_DIR))
        if not save_dir:
            return

        try:
            save_dir = Path(save_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 创建子目录
            screenshots_dir = save_dir / f"processing_screenshots_{timestamp}"
            screenshots_dir.mkdir(parents=True, exist_ok=True)

            saved_files = []

            # 保存各个处理步骤
            steps_info = [
                ("original", self.purifier.original_image, "原始图像"),
                ("gray", self.purifier.gray_image, "灰度图像"),
                ("edges", self.purifier.edges_image, "传统边缘检测"),
                ("color_mask", self.purifier.color_mask_image, "颜色特征掩码"),
                ("plate_region", self.purifier.plate_region, "车牌区域检测"),
                ("binary", self.purifier.white_bg_plate, "二值化处理"),
                ("histogram", self.purifier.histogram_image, "投影直方图"),
            ]

            for name, image, description in steps_info:
                if image is not None:
                    file_path = screenshots_dir / f"{name}_{description}.png"
                    if len(image.shape) == 2:
                        cv2.imwrite(str(file_path), image)
                    else:
                        if image.shape[2] == 3:
                            cv2.imwrite(str(file_path), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        else:
                            cv2.imwrite(str(file_path), image)
                    saved_files.append(description)

            # 保存字符分割结果
            if self.purifier.cropped_chars:
                # 保存所有字符
                chars_dir = screenshots_dir / "characters"
                chars_dir.mkdir(exist_ok=True)

                for i, char_img in enumerate(self.purifier.cropped_chars):
                    char_path = chars_dir / f"char_{i + 1:02d}.png"
                    cv2.imwrite(str(char_path), char_img)

                saved_files.append(f"字符分割({len(self.purifier.cropped_chars)}个)")

            # 创建说明文件
            readme_path = screenshots_dir / "README.txt"
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write("车牌识别系统处理步骤截图\n")
                f.write("=" * 50 + "\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"原始文件: {Path(self.current_image).name}\n\n")
                f.write("包含的处理步骤:\n")
                for step in saved_files:
                    f.write(f"  - {step}\n")
                f.write("\n可用于课程设计实验报告\n")

            self.log(f"✓ 处理截图已保存到: {screenshots_dir}")
            self.log(f"  保存了 {len(saved_files)} 个处理步骤")

            # 显示成功消息
            messagebox.showinfo("成功",
                                f"处理截图已保存!\n\n"
                                f"保存位置: {screenshots_dir}\n"
                                f"包含 {len(saved_files)} 个处理步骤\n\n"
                                f"可用于实验报告对比分析")

        except Exception as e:
            self.log(f"✗ 保存截图失败: {e}")
            messagebox.showerror("错误", f"保存截图失败: {e}")

    def recognize_chars(self):
        """使用CNN识别字符（兼容旧版本）"""
        if not self.current_char_images:
            messagebox.showwarning("警告", "请先处理车牌图像!")
            return

        # 检查模型是否已加载
        if not hasattr(self.cnn_recognizer, 'full_model') or self.cnn_recognizer.full_model is None:
            if not self.cnn_recognizer.load_models(self.log):
                messagebox.showwarning("警告", "请先训练CNN模型!")
                return

        self.start_progress("CNN识别字符中...")

        def recognize():
            try:
                self.log("开始CNN识别字符...")

                plate_number = ""
                confidences = []

                # 逐个识别字符
                for i, char_img in enumerate(self.current_char_images):
                    position = i + 1

                    results = self.cnn_recognizer.recognize_char_hierarchical(char_img, position)

                    if results and len(results) > 0:
                        best_char = results[0]['char']
                        confidence = results[0]['confidence']

                        plate_number += best_char
                        confidences.append(confidence)

                        self.log(f"字符{position}: '{best_char}' ({confidence:.2%})")
                    else:
                        plate_number += "?"
                        confidences.append(0)
                        self.log(f"字符{position}: 识别失败")

                # 显示结果
                self.cnn_result_var.set(f"车牌: {plate_number}")

                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    self.cnn_confidence_var.set(f"置信度: {avg_confidence:.2%}")

                    self.log(f"✓ CNN识别完成: {plate_number}")
                    self.log(f"平均置信度: {avg_confidence:.2%}")

                # 保存结果
                self.cnn_result = plate_number

            except Exception as e:
                self.log(f"✗ CNN识别错误: {e}")
                import traceback
                self.log(traceback.format_exc())
            finally:
                self.stop_progress()

        threading.Thread(target=recognize, daemon=True).start()

    def compare_all_methods(self):
        """对比三种识别方法"""
        if not self.current_char_images:
            messagebox.showwarning("警告", "请先处理车牌图像!")
            return

        self.start_progress("三种方法对比中...")

        def compare():
            try:
                self.log("=" * 50)
                self.log("开始三种方法对比识别...")

                plate_number_cnn = ""
                plate_number_template = ""
                plate_number_svm = ""

                confidences_cnn = []
                confidences_template = []
                confidences_svm = []

                # 逐个字符识别（7个字符）
                for i, char_img in enumerate(self.current_char_images):
                    position = i + 1

                    # 1. CNN识别
                    cnn_results = self.cnn_recognizer.recognize_char_hierarchical(char_img, position)
                    if cnn_results and len(cnn_results) > 0:
                        plate_number_cnn += cnn_results[0]['char']
                        confidences_cnn.append(cnn_results[0]['confidence'])
                        self.log(
                            f"字符{position} CNN: '{cnn_results[0]['char']}' ({cnn_results[0]['confidence']:.2%})")
                    else:
                        plate_number_cnn += "?"
                        confidences_cnn.append(0)

                    # 2. 模板匹配识别
                    template_result = self.template_matcher.match_char(char_img, position)
                    plate_number_template += template_result['char']
                    confidences_template.append(template_result['confidence'])
                    self.log(
                        f"字符{position} 模板: '{template_result['char']}' ({template_result['confidence']:.2%})")

                    # 3. SVM识别
                    svm_result = self.svm_recognizer.predict_char(char_img, position)
                    plate_number_svm += svm_result['char']
                    confidences_svm.append(svm_result['confidence'])
                    self.log(f"字符{position} SVM: '{svm_result['char']}' ({svm_result['confidence']:.2%})")

                # 保存结果
                self.cnn_result = plate_number_cnn
                self.template_result = plate_number_template
                self.svm_result = plate_number_svm

                # 计算平均置信度
                avg_cnn = np.mean(confidences_cnn) if confidences_cnn else 0
                avg_template = np.mean(confidences_template) if confidences_template else 0
                avg_svm = np.mean(confidences_svm) if confidences_svm else 0

                # 更新显示
                self.cnn_result_var.set(f"CNN: {plate_number_cnn}")
                self.cnn_confidence_var.set(f"置信度: {avg_cnn:.2%}")

                self.template_result_var.set(f"模板: {plate_number_template}")
                self.template_confidence_var.set(f"置信度: {avg_template:.2%}")

                self.svm_result_var.set(f"SVM: {plate_number_svm}")
                self.svm_confidence_var.set(f"置信度: {avg_svm:.2%}")

                # 切换到对比选项卡
                self.notebook.select(9)

                # 显示对比分析
                self.log("=" * 50)
                self.log("【三种方法对比结果】")
                self.log(f"CNN识别结果: {plate_number_cnn} (置信度: {avg_cnn:.2%})")
                self.log(f"模板匹配结果: {plate_number_template} (置信度: {avg_template:.2%})")
                self.log(f"SVM识别结果: {plate_number_svm} (置信度: {avg_svm:.2%})")

                # 判断哪种方法最好（基于置信度）
                methods = {
                    'CNN': avg_cnn,
                    '模板匹配': avg_template,
                    'SVM': avg_svm
                }
                best_method = max(methods, key=methods.get)

                self.log(f"最佳方法: {best_method} (置信度: {methods[best_method]:.2%})")
                self.log("=" * 50)

                # 询问是否要进行性能评估
                self.ask_for_performance_evaluation(plate_number_cnn,
                                                    plate_number_template,
                                                    plate_number_svm)

            except Exception as e:
                self.log(f"✗ 对比识别失败: {e}")
                import traceback
                self.log(traceback.format_exc())
            finally:
                self.stop_progress()

        threading.Thread(target=compare, daemon=True).start()

    def ask_for_performance_evaluation(self, cnn_result, template_result, svm_result):
        """询问用户是否进行性能评估"""
        response = messagebox.askyesno("性能评估",
                                       "识别完成！是否输入真实车牌号进行性能评估？\n\n"
                                       f"CNN结果: {cnn_result}\n"
                                       f"模板匹配: {template_result}\n"
                                       f"SVM结果: {svm_result}")

        if response:
            self.ask_ground_truth(cnn_result, template_result, svm_result)

    def ask_ground_truth(self, cnn_result, template_result, svm_result):
        """询问真实车牌号"""
        # 创建输入对话框
        dialog = tk.Toplevel(self.root)
        dialog.title("输入真实车牌号")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="请输入真实车牌号:",
                 font=("Arial", 12)).pack(pady=20)

        entry_var = tk.StringVar()
        entry = tk.Entry(dialog, textvariable=entry_var,
                         font=("Arial", 14), justify=tk.CENTER)
        entry.pack(pady=10)
        entry.focus()

        def submit():
            ground_truth = entry_var.get().strip().upper()
            if len(ground_truth) == 7:
                dialog.destroy()
                self.run_single_evaluation(ground_truth, cnn_result,
                                           template_result, svm_result)
            else:
                messagebox.showerror("错误", "车牌号必须是7位！")

        tk.Button(dialog, text="提交", command=submit,
                  bg="#27ae60", fg="white",
                  font=("Arial", 12), width=10).pack(pady=20)

        # 回车键提交
        dialog.bind('<Return>', lambda e: submit())

    def run_single_evaluation(self, ground_truth, cnn_result, template_result, svm_result):
        """运行单次性能评估"""
        self.log(f"真实车牌号: {ground_truth}")

        # 评估识别性能
        cnn_correct, template_correct, svm_correct = self.evaluator.evaluate_char_recognition(
            self.current_image, ground_truth, cnn_result, template_result, svm_result
        )

        # 显示评估结果
        self.log("【单次性能评估结果】")
        self.log(f"CNN: {'✓ 正确' if cnn_correct else '✗ 错误'}")
        self.log(f"模板匹配: {'✓ 正确' if template_correct else '✗ 错误'}")
        self.log(f"SVM: {'✓ 正确' if svm_correct else '✗ 错误'}")

        # 询问是否生成完整报告
        response = messagebox.askyesno("性能报告",
                                       "评估完成！是否生成详细性能报告？")

        if response:
            self.generate_performance_report()

    def run_performance_evaluation(self):
        """运行完整性能评估测试"""
        if not self.current_image:
            messagebox.showwarning("警告", "请先选择和处理图像!")
            return

        self.start_progress("运行性能评估...")

        def evaluate():
            try:
                self.log("开始性能评估测试...")

                # 评估车牌定位（假设图像中有车牌）
                self.evaluator.evaluate_plate_detection(
                    self.current_image,
                    has_plate=True,
                    detected=self.purifier.plate_region is not None
                )

                # 评估字符分割（应该是7个字符）
                char_count = len(self.current_char_images) if self.current_char_images else 0
                self.evaluator.evaluate_char_segmentation(
                    self.current_image,
                    expected_chars=7,
                    segmented_chars=char_count
                )

                self.log("✓ 定位和分割评估完成")
                self.log("  请使用'三种方法对比'进行识别评估")

            except Exception as e:
                self.log(f"✗ 性能评估失败: {e}")
            finally:
                self.stop_progress()

        threading.Thread(target=evaluate, daemon=True).start()

    def generate_performance_report(self):
        """生成性能评估报告"""
        try:
            report_path = self.evaluator.generate_report()

            # 显示报告内容
            with open(report_path, 'r', encoding='utf-8') as f:
                report_content = f.read()

            # 在GUI中显示报告（简化版）
            self.log("=" * 50)
            self.log("【性能评估报告摘要】")

            # 只显示关键信息
            lines = report_content.split('\n')
            for line in lines:
                if any(keyword in line for keyword in ['准确率', '最佳', '符合']):
                    self.log(line)

            self.log("=" * 50)

            # 保存测试案例
            self.evaluator.save_test_cases()

            # 显示成功消息
            messagebox.showinfo("成功", f"性能评估报告已生成:\n{report_path}")

            # 切换到性能评估选项卡
            self.notebook.select(10)

        except Exception as e:
            self.log(f"✗ 生成报告失败: {e}")

    def save_results(self):
        """保存识别结果"""
        if not hasattr(self.purifier, 'cropped_chars') or not self.purifier.cropped_chars:
            messagebox.showwarning("警告", "没有可保存的结果!")
            return

        # 选择保存目录
        save_dir = filedialog.askdirectory(title="选择保存目录",
                                           initialdir=str(Config.RESULTS_DIR))
        if not save_dir:
            return

        try:
            # 保存分割的字符图像
            self.purifier.save_chars(save_dir)

            # 保存识别结果文本
            result_file = Path(save_dir) / "recognition_result.txt"
            with open(result_file, 'w', encoding='utf-8') as f:
                f.write("车牌识别结果\n")
                f.write("=" * 50 + "\n")
                f.write(f"识别时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"文件路径: {self.current_image}\n\n")

                # CNN结果
                cnn_result = self.cnn_result_var.get().replace("车牌: ", "")
                cnn_confidence = self.cnn_confidence_var.get().replace("置信度: ", "")
                if cnn_result != "等待识别...":
                    f.write("CNN识别结果:\n")
                    f.write(f"  车牌号码: {cnn_result}\n")
                    f.write(f"  置信度: {cnn_confidence}\n\n")

                f.write(f"字符数量: {len(self.purifier.cropped_chars)}\n")
                f.write("=" * 50 + "\n")

            self.log(f"✓ 结果保存到: {save_dir}")
            messagebox.showinfo("成功", f"结果已保存到:\n{save_dir}")

        except Exception as e:
            self.log(f"✗ 保存失败: {e}")
            messagebox.showerror("错误", f"保存失败: {e}")

    # ============ 其他辅助方法 ============

    def check_models(self):
        """检查模型文件是否存在 - 增强版"""
        # CNN模型
        cnn_path = Config.MODELS_DIR / "best_cnn_model.pth"
        if cnn_path.exists():
            self.log("✓ CNN模型已存在")
        else:
            self.log("⚠ CNN模型不存在，请先训练")

        # SVM模型
        svm_path = Config.MODELS_DIR / "svm_model.pkl"
        if svm_path.exists():
            self.log("✓ SVM模型已存在")
        else:
            self.log("⚠ SVM模型不存在，请先训练")

        # 模板匹配器总是可用的
        self.log("✓ 模板匹配器已加载")

    def start_progress(self, message):
        """开始显示进度条"""
        self.progress_var.set(message)
        self.progress_bar.start()
        self.root.update_idletasks()

    def stop_progress(self):
        """停止进度条"""
        self.progress_bar.stop()
        self.progress_var.set("就绪")
        self.root.update_idletasks()

    def log(self, message):
        """添加日志信息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def update_status(self, message):
        """更新状态栏"""
        self.status_var.set(message)
        self.root.update_idletasks()

    def display_image(self, image, tab_id, title=""):
        """在指定选项卡显示图像"""
        if image is None:
            return

        canvas_info = self.tabs[tab_id]
        canvas = canvas_info["canvas"]

        canvas.delete("all")

        # 获取画布尺寸
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        # 如果画布尺寸太小，使用默认值
        if canvas_width <= 10 or canvas_height <= 10:
            canvas_width = 500
            canvas_height = 300

        # 确保图像是RGB格式
        if len(image.shape) == 2:
            display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            display_img = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        # 计算缩放比例（保持比例）
        h, w = display_img.shape[:2]
        scale = min(canvas_width / w, canvas_height / h) * 0.9
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 缩放图像
        img_resized = cv2.resize(display_img, (new_w, new_h))

        # 转换为Tkinter可显示的格式
        pil_img = Image.fromarray(img_resized)
        tk_img = ImageTk.PhotoImage(pil_img)

        # 居中显示
        x_center = canvas_width // 2
        y_center = canvas_height // 2
        canvas.create_image(x_center, y_center, image=tk_img, anchor=tk.CENTER)

        # 添加标题
        if title:
            canvas.create_text(x_center, 20, text=title,
                               font=("Arial", 12, "bold"), fill="black")

        # 保存图像引用（防止被垃圾回收）
        canvas_info["image_ref"] = tk_img


# ============ 第十五部分：主程序入口 ============
def main():
    """主程序入口"""
    print("=" * 70)
    print("车牌识别系统 v8.1 - 回归v7.0处理流程版")
    print("功能：完整处理流程可视化 + CNN + 模板匹配 + SVM 三种方法对比")
    print("字符分割：回归v7.0经典算法，提高准确率")
    print("性能：自动评估准确率，生成详细报告")
    print("=" * 70)

    # 创建目录
    Config.create_dirs()

    # 启动GUI
    root = tk.Tk()
    app = EnhancedPlateRecognitionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    """
    程序入口点

    使用步骤：
    1. 运行程序
    2. 点击"创建目录"按钮
    3. 点击"生成合成数据"按钮
    4. 点击"选择车牌图像"选择图片
    5. 点击"处理车牌图像"查看完整处理流程
    6. 点击"保存处理截图"保存所有步骤图片
    7. 点击"收集训练数据"手动标注字符
    8. 点击"训练CNN模型"和"训练SVM模型"
    9. 点击"三种方法对比"查看识别结果
    10. 点击"性能评估"生成准确率报告

    注意：所有路径都使用英文，避免中文路径问题
    """
    main()