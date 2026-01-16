import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def super_match(big_path, small_path):
    """超级匹配：多方法融合 + 高亮显示"""
    # 读取图像
    big = cv2.imread(big_path)
    small = cv2.imread(small_path)

    # 方法1：特征点匹配
    try:
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(small, None)
        kp2, des2 = sift.detectAndCompute(big, None)

        if des1 is not None and des2 is not None and len(des1) > 10 and len(des2) > 10:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(des1, des2, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

            if len(good_matches) > 10:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if M is not None:
                    h, w = small.shape[:2]
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)

                    xs = [p[0][0] for p in dst]
                    ys = [p[0][1] for p in dst]
                    x, y = int(min(xs)), int(min(ys))
                    w, h = int(max(xs) - min(xs)), int(max(ys) - min(ys))

                    conf = min(len(good_matches) / 50.0, 1.0)
                    method = "SIFT特征匹配"
    except:
        method = "模板匹配"

    # 方法2：灰度模板匹配
    if 'method' not in locals() or method == "模板匹配":
        big_gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        scales = [0.8, 0.9, 1.0, 1.1, 1.2]
        best_conf = -1

        for scale in scales:
            scaled_w = int(small_gray.shape[1] * scale)
            scaled_h = int(small_gray.shape[0] * scale)

            if scaled_w < 20 or scaled_h < 20:
                continue

            scaled_small = cv2.resize(small_gray, (scaled_w, scaled_h))
            result = cv2.matchTemplate(big_gray, scaled_small, cv2.TM_CCOEFF_NORMED)
            _, conf, _, (x, y) = cv2.minMaxLoc(result)

            if conf > best_conf:
                best_conf = conf
                best_x, best_y = x, y
                best_w, best_h = scaled_w, scaled_h

        x, y, w, h, conf = best_x, best_y, best_w, best_h, best_conf
        method = "多尺度模板匹配"

    # 输出结果
    print(f"🔍 {method}")
    print(f"📍 位置: ({x}, {y})")
    print(f"📐 尺寸: {w} × {h}")
    print(f"🎯 置信度: {conf:.3f}")
    print("-" * 40)

    # ========== 生成三种结果图 ==========

    # 1. 带框标注的完整结果图
    result_img = big.copy()
    cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(result_img, f"{conf:.2f}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 2. 高亮显示图：匹配区域保持明亮，其他区域变暗（50%亮度）
    highlight_img = big.copy()
    # 创建暗色背景
    dark_bg = (big * 0.3).astype(np.uint8)  # 30%亮度
    # 将暗色背景复制到高亮图
    highlight_img = dark_bg.copy()
    # 将匹配区域恢复为原亮度
    highlight_img[y:y + h, x:x + w] = big[y:y + h, x:x + w]
    # 在高亮图上也画框（可选）
    cv2.rectangle(highlight_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 3. 融合显示图：半透明高亮效果
    blend_img = big.copy()
    # 创建高亮区域的掩码
    mask = np.zeros((big.shape[0], big.shape[1]), dtype=np.uint8)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)  # 填充白色矩形

    # 对非匹配区域应用暗化
    for c in range(3):
        blend_img[:, :, c] = np.where(mask == 255,
                                      blend_img[:, :, c],  # 匹配区域保持原值
                                      blend_img[:, :, c] * 0.4)  # 其他区域变暗到40%

    # ========== 显示结果 ==========
    plt.figure(figsize=(15, 10))

    # 小图
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    plt.title("待匹配的小图")
    plt.axis('off')

    # 大图原图
    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))
    plt.title("大图（原始）")
    plt.axis('off')

    # 带框标注结果
    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(f"标注结果 - {method}")
    plt.axis('off')

    # 高亮显示（区域明亮，其他变暗）
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(highlight_img, cv2.COLOR_BGR2RGB))
    plt.title("高亮显示效果")
    plt.axis('off')

    # 半透明融合效果
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(blend_img, cv2.COLOR_BGR2RGB))
    plt.title("融合显示效果")
    plt.axis('off')

    # 信息显示
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.5,
             f"匹配结果信息\n\n"
             f"方法: {method}\n"
             f"位置: ({x}, {y})\n"
             f"尺寸: {w} × {h}\n"
             f"置信度: {conf:.3f}\n\n"
             f"图像说明:\n"
             f"• 标注结果: 绿色框标记位置\n"
             f"• 高亮显示: 匹配区域保持明亮\n"
             f"• 融合效果: 渐变暗化背景",
             fontsize=12, verticalalignment='center')
    plt.axis('off')
    plt.title("匹配信息")

    plt.tight_layout()
    plt.show()

    # ========== 保存结果 ==========
    cv2.imwrite('1_标注结果.jpg', result_img)
    cv2.imwrite('2_高亮显示.jpg', highlight_img)
    cv2.imwrite('3_融合效果.jpg', blend_img)

    print("💾 结果已保存:")
    print("  1_标注结果.jpg - 带框标注的完整图")
    print("  2_高亮显示.jpg - 匹配区域明亮，其他变暗")
    print("  3_融合效果.jpg - 渐变暗化背景效果")

    return x, y, w, h, conf, method


# 使用示例
if __name__ == "__main__":
    # 修改这里使用你的图片路径
    big_path = "./test_images/da6.jpg"
    small_path = "./test_images/xiao65.png"




    result = super_match(big_path, small_path)