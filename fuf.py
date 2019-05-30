#! /usr/bin/env python
# -*- coding: utf-8 -*-
# .__author__. = "Lily Yu"
# .DATE.: 2019/5/21
""" Frequently used functions"""

# import the necessary packages
import cv2
import numpy as np
import matplotlib.pyplot as plt


### Image Processing图像处理 ###
def findBestMatch(template, src, ptInSrc=(0, 0), width=640, height=360):
    """
    从输入图像src上截取一块ROI，然后去寻找&匹配模板，返回最佳匹配结果 - 匹配度与位置
    匹配方法为cv2.TM_CCOEFF_NORMED归一化相关系数匹配法
    :param template: 待寻找&匹配的模板；类型为numpy.ndarray
    :param src: 输入图像src，在src图片上的某个ROI寻找&匹配模板；类型为numpy.ndarray
    :param ptInSrc: src图像上的一点
    :param width: ROI的宽
    :param height: ROI的高
    :return: 返回ROI上的最佳匹配结果 - 匹配度与位置
    """
    # # Step# 0. 裁切原图src获取ROI
    # Crop image[startY:endY, startX:endX]
    roiImg = src[ptInSrc[1]:ptInSrc[1] + height, ptInSrc[0]:ptInSrc[0] + width]

    # # Step# 1.模板匹配
    # res.w = img_rgb.W - template.w + 1; res.h = img_rgb.H - template.h + 1
    res = cv2.matchTemplate(roiImg, template, cv2.TM_CCOEFF_NORMED)  # type(res): <class 'numpy.ndarray'>

    # # Step# 2.从匹配结果中寻找最佳匹配：匹配值与匹配位置
    # 返回的是一个tuple(minVal, maxVal, minLoc, maxLoc)
    # Sample: (-0.33576542139053345, 0.5124101042747498, (787, 935), (329, 138))
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # # Step# 3.max_loc为ROI上的位置，需要将其转换为原图src上的位置，就要加上roiImg左上角点在原图中的坐标(x, y)
    # max_loc.x: max_loc[0], max_loc.y: max_loc[1]
    bestMacthLoc = [0, 0]  # 原图上的位置
    bestMacthLoc[0] = max_loc[0] + ptInSrc[0]  # x轴坐标
    bestMacthLoc[1] = max_loc[1] + ptInSrc[1]  # y轴坐标
    print("max_loc.x:{}, max_loc.y:{}".format(max_loc[0], max_loc[1]))

    # # Step# 4. 返回最佳匹配结果 - 匹配度与位置
    return max_val, bestMacthLoc


def checkTemplate(templateImg, srcImg):
    """
    将输入图片井字格地切成九块ROI，每个区块的大小为(w_one_third,h_one_third)，在每块ROI上寻找最佳匹配
    :param templateImg: 待匹配的模板图片位置，e.g. 'F:/images/template.jpg'
    :param srcImg: 输入图片位置，e.g. 'F:/images/g.jpg'
    :return: Pass 返回(True,匹配的九个结果list)； Fail返回(False, [])
    Note: 会将结果写在"template_result.txt"中
    # matchResult为List类型，每个元素都是tuple，tuple[0]为最佳匹配的相似度，tuple[1]为最佳匹配的位置
    # [(0.9884182810783386, [331, 131]), (0.9904218912124634, [857, 135]),
    # (0.9929701685905457, [1385, 138]), (0.990262508392334, [327, 481]),
    # (0.9998119473457336, [855, 484]), (0.9894646406173706, [1384, 488]),
    #  (0.9916950464248657, [324, 831]), (0.9905069470405579, [852, 835]),
    # (0.9920342564582825, [1382, 838])]
    """

    # 1. 读入灰度图片
    src_gray = cv2.imread(srcImg, 0)  # 目标图，待搜索的图片
    template = cv2.imread(templateImg, 0)  # 模板图
    # w_templ, h_templ = template.shape[::-1]  # w，h要倒序取出
    h_templ, w_templ = template.shape  # shape返回（rows, cols)

    src_h, src_w = src_gray.shape
    w_one_third = int(src_w / 3)  # 原始图宽度的三分之一, x
    h_one_third = int(src_h / 3)  # 原始图高度的三分之一, y

    # TODO: 单独写一个函数
    # 统计白色点的概率
    _T, thresh = cv2.threshold(src_gray, 200, 255, cv2.THRESH_BINARY)
    srcArea = src_h * src_w
    srcNonZero = cv2.countNonZero(thresh)
    srcNonZeroRate = srcNonZero / srcArea  # 0.04122636959876543
    print("srcNonZeroRate:", srcNonZeroRate)

    # 2. 将原始图src_gray拆分为九个区块，每个区块的大小为(w_one_third,h_one_third)
    # 点集仅保留每个区块左上角leftTop在源图的坐标
    # 点集的顺序为从左至右，从上至下
    ninePoints = []
    for y in range(0, src_h - 1, h_one_third):
        for x in range(0, src_w - 1, w_one_third):
            ninePoints.append((x, y))

    # 3. 对每块ROI寻找最佳匹配，并在源图上标记出来
    # 创建画布
    drawMat = cv2.cvtColor(src_gray, cv2.COLOR_GRAY2BGR)

    matchResult = []  # 最终匹配结果
    for i, pt in enumerate(ninePoints):
        print(i, pt)
        best = findBestMatch(template, src_gray, ptInSrc=pt, width=w_one_third, height=h_one_third)
        cv2.rectangle(drawMat, tuple(best[1]), (best[1][0] + w_templ, best[1][1] + h_templ), (0, 0, 255), 2)
        matchResult.append(best)
    print("result type:{}, {}".format(type(matchResult), matchResult))
    cv2.imwrite("template_result.png", drawMat)

    # 4. 将每块ROI上的最佳匹配写入到"GLogo_result.txt"
    with open("template_result.txt", 'w') as wObj:
        for oneItem in matchResult:
            # print(item)
            wObj.write(str(round(oneItem[0], 3)))  # 相似度，保留三位小数
            wObj.write(",(")
            wObj.write(str(oneItem[1][0]))  # 位置x
            wObj.write(",")
            wObj.write(str(oneItem[1][0]))  # 位置y
            wObj.write(")\n")
    return True, matchResult


### Image histogram图像直方图相关 ###
def showGrayHist(imgPath):
    """
    展示图片的灰度直方图
    :param imgPath: 输入图片的路径
    :return: void
    """
    srcImg = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)  # 读入灰度图片
    cv2.imshow("src", srcImg)  # 显示输入图片的灰度图

    # 得到矩阵的高和宽
    rows, cols = srcImg.shape

    # 将二维的图像矩阵，变为一维的矩阵，便于计算灰度直方图
    pixelSequence = srcImg.reshape([rows * cols, ])
    numberBins = 256  # 组数

    # 计算灰度直方图
    histogram, bins, patch = plt.hist(pixelSequence, bins=numberBins,
                                      facecolor='black', histtype='bar')

    # 设置坐标轴的标签
    plt.xlabel("gray Level")
    plt.ylabel("number of pixels")

    # 设置坐标轴范围
    y_maxValue = np.max(histogram)
    plt.axis([0, 255, 0, y_maxValue])

    plt.show()


def calcAndShowGrayHist(imgPath):
    """
    先计算图片的灰度直方图然后展示
    :param imgPath: 输入图片的路径
    :return: void
    """
    srcImg = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)  # 读入灰度图片
    cv2.imshow("src", srcImg)  # 显示输入图片的灰度图
    # 计算灰度直方图
    grayHist = calcGrayHist(srcImg)
    # 画灰度直方图
    x_range = range(256)
    plt.plot(x_range, grayHist, 'r', linewidth=2, c='black')

    # 设置坐标轴范围
    y_maxValue = np.max(grayHist)
    plt.axis([0, 255, 0, y_maxValue])

    # 设置坐标轴的标签
    plt.xlabel("gray Level")
    plt.ylabel("number of pixels")
    plt.show()


def calcGrayHist(image):
    """
    计算图片的灰度直方图
    :param image: 输入图片的np.ndarrary类型
    :return: 图片的灰度直方图
    """
    rows, cols = image.shape
    # 存储灰度直方图
    grayHist = np.zeros([256], np.uint64)
    for r in range(rows):
        for c in range(cols):
            grayHist[image[r, c]] += 1
    return grayHist


def showColorHist(imgPath):
    """
    展示图片的彩色直方图
    :param imgPath: 输入图片的路径。务必确保输入图片为三通道的彩色图片。
    :return: void
    """
    # 将图像从磁盘加载并显示它
    srcImg = cv2.imread(imgPath)  # 读取图片，彩色格式即flag=cv2.IMREAD_COLOR
    cv2.imshow("Original", srcImg)
    if len(srcImg.shape) != 3:
        print("Please input 3-channel picture!")

    # 拆分图片
    chans = cv2.split(srcImg)  # B, G, R
    colors = ("b", "g", "r")

    # 用matplotlib PyPlot画图，set up our PyPlot figure
    plt.figure()
    plt.title("Flattened Color Histogram")  # 设定图的标题
    plt.xlabel("Bins")  # 设定x轴标签
    plt.ylabel("# of pixels")  # 设定y轴标签

    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)  # 画曲线，并设定曲线的颜色
        plt.xlim([0, 256])  # 设定x轴的上下限

    plt.show()  # 显示画出的图片


def show2DColorHist(imgPath):
    """
    展示图片的彩色2D直方图, Green and Blue, Green and Red, Blue and Red并排显示
    :param imgPath: 输入图片的路径。务必确保输入图片为三通道的彩色图片。
    :return: void
    """
    # 将图像从磁盘加载并显示它
    srcImg = cv2.imread(imgPath)  # 读取图片，彩色格式即flag=cv2.IMREAD_COLOR
    cv2.imshow("Original", srcImg)
    if len(srcImg.shape) != 3:
        print("Please input 3-channel picture!")

    # 拆分图片
    chans = cv2.split(srcImg)  # B, G, R
    colors = ("b", "g", "r")

    # 画2D直方图
    fig = plt.figure()
    plt.title("2D Color Histogram")  # 设定图的标题
    plt.axis('off')  # 隐藏坐标轴x, y

    # 2D直方图由于计算量的关系，只展示32*32Bin
    # Green and Blue
    ax = fig.add_subplot(131)
    hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None, [32, 32], [0, 256, 0, 256])
    p = ax.imshow(hist, interpolation="nearest")
    ax.set_title("Green and Blue")  # Green and Blue
    plt.colorbar(p)

    # Green and Red
    ax = fig.add_subplot(132)
    hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None, [32, 32], [0, 256, 0, 256])
    p = ax.imshow(hist, interpolation="nearest")
    ax.set_title("Green and Red")  # Green and Red
    plt.colorbar(p)

    # Blue and Red
    ax = fig.add_subplot(133)
    hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None, [32, 32], [0, 256, 0, 256])
    p = ax.imshow(hist, interpolation="nearest")
    ax.set_title("Blue and Red")  # Blue and Red
    plt.colorbar(p)

    print("2D histogram shape: {}, with {} values".format(hist.shape, hist.flatten().shape[0]))
    plt.show()  # 显示画出的图片


if __name__ == '__main__':
    checkTemplate('F:/images/gTemplate.jpg', 'F:/images/8.bmp')
    showGrayHist("F:/images/lena.jpg")  # 显示灰色直方图
    showColorHist("F:/images/lena.jpg")  # 显示彩色直方图
    show2DColorHist("F:/images/lena.jpg")  # 显示彩色2D直方图
