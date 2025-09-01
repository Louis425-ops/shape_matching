#!/bin/bash

echo "🔧 螺母检测系统快速测试脚本"
echo "================================"

# 检查程序是否存在
if [ ! -f "./nut_detector" ]; then
    echo "❌ nut_detector 程序不存在，开始编译..."
    g++ -I. -I./MIPP/ -fopenmp -march=native -O3 -std=c++14 \
        line2Dup.cpp nut_detection.cpp -o nut_detector \
        `pkg-config --cflags --libs opencv4`
    
    if [ $? -eq 0 ]; then
        echo "✅ 编译成功！"
    else
        echo "❌ 编译失败，请检查依赖"
        exit 1
    fi
else
    echo "✅ 程序已存在"
fi

# 函数：检查文件是否存在
check_file() {
    if [ ! -f "$1" ]; then
        echo "❌ 文件不存在: $1"
        echo "💡 请准备你的图片文件，然后重新运行此脚本"
        echo ""
        echo "需要准备的文件:"
        echo "  1. 模板图片 (单个螺母的清晰图片)"
        echo "  2. 测试图片 (包含多个螺母的图片)"
        echo ""
        echo "然后运行:"
        echo "  ./quick_test.sh <模板图片路径> <测试图片路径>"
        exit 1
    fi
}

# 获取参数
if [ $# -eq 2 ]; then
    TEMPLATE_IMG="$1"
    TEST_IMG="$2"
    OUTPUT_IMG="detection_result.jpg"
else
    # 使用默认的测试图片
    echo "📋 使用方式:"
    echo "  ./quick_test.sh <模板图片> <测试图片>"
    echo ""
    echo "💡 你也可以使用项目自带的测试图片："
    
    # 检查是否有现成的测试图片
    if [ -f "./test/case0/templ/circle.png" ]; then
        echo "  使用圆形检测测试: ./quick_test.sh test/case0/templ/circle.png test/case0/1.jpg"
    fi
    
    if [ -f "./test/case1/train.png" ]; then
        echo "  使用机械零件测试: ./quick_test.sh test/case1/train.png test/case1/test.png"
    fi
    
    exit 1
fi

# 检查输入文件
check_file "$TEMPLATE_IMG"
check_file "$TEST_IMG"

echo ""
echo "🎯 开始螺母检测测试"
echo "模板图片: $TEMPLATE_IMG"
echo "测试图片: $TEST_IMG"
echo "输出图片: $OUTPUT_IMG"
echo ""

# 第一步：训练模板
echo "📚 步骤 1/2: 训练模板..."
echo "================================"
./nut_detector "$TEMPLATE_IMG" "$TEST_IMG" "$OUTPUT_IMG" train

if [ $? -ne 0 ]; then
    echo "❌ 模板训练失败"
    exit 1
fi

echo ""
echo "🔍 步骤 2/2: 执行检测..."
echo "================================"
./nut_detector "$TEMPLATE_IMG" "$TEST_IMG" "$OUTPUT_IMG" test

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 检测完成！"
    echo "📁 结果文件:"
    echo "  - 检测结果图片: $OUTPUT_IMG"
    echo "  - 模板文件: nut_nut_templ.yaml"
    echo "  - 信息文件: nut_info.yaml"
    echo ""
    echo "💡 你可以尝试不同的参数："
    echo "  ./nut_detector $TEMPLATE_IMG $TEST_IMG result2.jpg test 85 0.2"
    echo "  (相似度85%, NMS阈值0.2)"
else
    echo "❌ 检测失败"
    exit 1
fi