#include "line2Dup.h"
#include <memory>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <iomanip>
using namespace std;
using namespace cv;

class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }
    void out(std::string message = ""){
        double t = elapsed();
        std::cout << message << " elapsed time: " << t << "s" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

// NMS函数用于去除重复检测
namespace cv_dnn {
template <typename T>
static inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2)
{
    return pair1.first > pair2.first;
}

inline void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
                      std::vector<std::pair<float, int> >& score_index_vec)
{
    for (size_t i = 0; i < scores.size(); ++i)
    {
        if (scores[i] > threshold)
        {
            score_index_vec.push_back(std::make_pair(scores[i], i));
        }
    }
    std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
                     SortScorePairDescend<int>);
    if (top_k > 0 && top_k < (int)score_index_vec.size())
    {
        score_index_vec.resize(top_k);
    }
}

template <typename BoxType>
inline void NMSFast_(const std::vector<BoxType>& bboxes,
      const std::vector<float>& scores, const float score_threshold,
      const float nms_threshold, const float eta, const int top_k,
      std::vector<int>& indices, float (*computeOverlap)(const BoxType&, const BoxType&))
{
    CV_Assert(bboxes.size() == scores.size());
    std::vector<std::pair<float, int> > score_index_vec;
    GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);

    float adaptive_threshold = nms_threshold;
    indices.clear();
    for (size_t i = 0; i < score_index_vec.size(); ++i) {
        const int idx = score_index_vec[i].second;
        bool keep = true;
        for (int k = 0; k < (int)indices.size() && keep; ++k) {
            const int kept_idx = indices[k];
            float overlap = computeOverlap(bboxes[idx], bboxes[kept_idx]);
            keep = overlap <= adaptive_threshold;
        }
        if (keep)
            indices.push_back(idx);
        if (keep && eta < 1 && adaptive_threshold > 0.5) {
          adaptive_threshold *= eta;
        }
    }
}

template<typename _Tp> static inline
double jaccardDistance__(const Rect_<_Tp>& a, const Rect_<_Tp>& b) {
    _Tp Aa = a.area();
    _Tp Ab = b.area();

    if ((Aa + Ab) <= std::numeric_limits<_Tp>::epsilon()) {
        return 0.0;
    }

    double Aab = (a & b).area();
    return 1.0 - Aab / (Aa + Ab - Aab);
}

template <typename T>
static inline float rectOverlap(const T& a, const T& b)
{
    return 1.f - static_cast<float>(jaccardDistance__(a, b));
}

void NMSBoxes(const std::vector<Rect>& bboxes, const std::vector<float>& scores,
                          const float score_threshold, const float nms_threshold,
                          std::vector<int>& indices, const float eta=1, const int top_k=0)
{
    NMSFast_(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices, rectOverlap);
}
}

/**
 * 螺母检测函数
 * @param template_path 单个螺母模板图片路径
 * @param test_image_path 包含多个螺母的测试图片路径
 * @param output_path 结果输出图片路径
 * @param mode "train" 表示训练模板，"test" 表示检测
 * @param similarity_threshold 相似度阈值 (0-100)
 * @param nms_threshold NMS重叠阈值 (0-1)
 */
void detect_nuts(const string& template_path, 
                 const string& test_image_path,
                 const string& output_path,
                 const string& mode = "test",
                 float similarity_threshold = 80.0f,
                 float nms_threshold = 0.3f)
{
    // 设置特征数量，根据螺母复杂度调整
    int num_feature = 128;
    
    // 2层金字塔
    line2Dup::Detector detector(num_feature, {4, 8});
    
    string class_id = "nut";
    
    if(mode == "train") {
        cout << "=== 开始训练螺母模板 ===" << endl;
        
        // 读取螺母模板图片
        Mat template_img = imread(template_path);
        if(template_img.empty()) {
            cerr << "错误: 无法读取模板图片 " << template_path << endl;
            return;
        }
        
        cout << "模板图片尺寸: " << template_img.cols << "x" << template_img.rows << endl;
        
        // 掩码
        Mat mask = Mat(template_img.size(), CV_8UC1, Scalar(255));
        
        // 生成不同缩放的模板
        shape_based_matching::shapeInfo_producer shapes(template_img, mask);
        shapes.scale_range = {0.8f, 1.2f};  // 缩放范围 80%-120%
        shapes.scale_step = 0.1f;           // 缩放步长
        shapes.angle_range = {0, 360};      // 旋转角度范围
        shapes.angle_step = 30;             // 旋转步长
        
        Timer timer;
        shapes.produce_infos();
        timer.out("生成模板变换信息");
        
        cout << "总共生成 " << shapes.infos.size() << " 个模板变换" << endl;
        
        std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
        
        int success_count = 0;
        for(size_t i = 0; i < shapes.infos.size(); i++) {
            auto& info = shapes.infos[i];
            
            // 添加模板
            int templ_id = detector.addTemplate(shapes.src_of(info), class_id, 
                                               shapes.mask_of(info), num_feature);
            
            if(templ_id != -1) {
                infos_have_templ.push_back(info);
                success_count++;
                if(success_count % 10 == 0) {
                    cout << "已训练 " << success_count << " 个模板..." << endl;
                }
            }
        }
        
        cout << "成功训练 " << success_count << " 个模板" << endl;
        
        // 保存模板文件
        detector.writeClasses("nut_%s_templ.yaml");
        shapes.save_infos(infos_have_templ, "nut_info.yaml");
        
        timer.out("模板训练完成，总用时");
        
    } else if(mode == "test") {
        cout << "=== 开始检测螺母 ===" << endl;
        
        // 加载训练好的模板
        std::vector<std::string> ids;
        ids.push_back(class_id);
        detector.readClasses(ids, "nut_%s_templ.yaml");
        
        // 加载模板信息
        auto infos = shape_based_matching::shapeInfo_producer::load_infos("nut_info.yaml");
        cout << "加载了 " << infos.size() << " 个模板信息" << endl;
        
        // 读取测试图片
        Mat test_img = imread(test_image_path);
        if(test_img.empty()) {
            cerr << "错误: 无法读取测试图片 " << test_image_path << endl;
            return;
        }
        
        cout << "测试图片尺寸: " << test_img.cols << "x" << test_img.rows << endl;
        
        // 确保图片尺寸是32的倍数（算法要求）
        int stride = 32;
        int n = test_img.rows / stride;
        int m = test_img.cols / stride;
        Rect roi(0, 0, stride * m, stride * n);
        Mat img = test_img(roi).clone();
        
        Timer timer;
        
        // 执行检测
        auto matches = detector.match(img, similarity_threshold, ids);
        timer.out("检测完成，用时");
        
        cout << "原始检测结果: " << matches.size() << " 个匹配" << endl;
        
        if(matches.empty()) {
            cout << "未检测到螺母，尝试降低相似度阈值" << endl;
            return;
        }
        
        // 准备NMS处理
        vector<Rect> boxes;
        vector<float> scores;
        
        for(auto& match : matches) {
            auto templ = detector.getTemplates(class_id, match.template_id);
            
            Rect box;
            box.x = match.x;
            box.y = match.y;
            box.width = templ[0].width;
            box.height = templ[0].height;
            
            boxes.push_back(box);
            scores.push_back(match.similarity);
        }
        
        // 执行NMS去除重复检测
        vector<int> nms_indices;
        cv_dnn::NMSBoxes(boxes, scores, similarity_threshold, nms_threshold, nms_indices);
        
        cout << "NMS处理后: " << nms_indices.size() << " 个有效检测" << endl;
        
        // 可视化结果
        Mat result_img = img.clone();
        
        cout << "\n=== 检测结果详情 ===" << endl;
        cout << setw(5) << "序号" << setw(10) << "X坐标" << setw(10) << "Y坐标" 
             << setw(10) << "相似度" << setw(12) << "模板ID" << endl;
        cout << string(50, '-') << endl;
        
        for(size_t i = 0; i < nms_indices.size(); i++) {
            int idx = nms_indices[i];
            auto& match = matches[idx];
            auto& box = boxes[idx];

            Scalar color(rand() % 255, rand() % 255, rand() % 255);

            rectangle(result_img, box, color, 3);

            string label = "Nut " + to_string(i+1) + ": " + to_string((int)match.similarity) + "%";
            putText(result_img, label, Point(box.x, box.y - 5), 
                   FONT_HERSHEY_SIMPLEX, 0.7, color, 2);

            cout << setw(5) << (i+1) << setw(10) << match.x << setw(10) << match.y 
                 << setw(9) << fixed << setprecision(1) << match.similarity << "%" 
                 << setw(12) << match.template_id << endl;
        }
        
        cout << "\n=== 检测统计 ===" << endl;
        cout << "检测到螺母数量: " << nms_indices.size() << " 个" << endl;
        
        if(!nms_indices.empty()) {
            float avg_confidence = 0;
            for(int idx : nms_indices) {
                avg_confidence += matches[idx].similarity;
            }
            avg_confidence /= nms_indices.size();
            cout << "平均置信度: " << fixed << setprecision(1) << avg_confidence << "%" << endl;
        }

        imwrite(output_path, result_img);
        cout << "结果已保存到: " << output_path << endl;

        namedWindow("Detection Results", WINDOW_NORMAL);
        imshow("Detection Results", result_img);
        cout << "按任意键关闭显示窗口..." << endl;
        waitKey(0);
        destroyAllWindows();
    }
}

/**
 * 多模板投票检测函数
 * @param template_path1-3 三个螺母模板图片路径
 * @param test_image_path 包含多个螺母的测试图片路径
 * @param output_path 结果输出图片路径
 * @param mode "train" 表示训练模板，"test" 表示检测
 * @param similarity_threshold 相似度阈值 (0-100)
 * @param nms_threshold NMS重叠阈值 (0-1)
 */

void detect_nuts_multi_template(const string& template_path1,
                               const string& template_path2, 
                               const string& template_path3,
                               const string& test_image_path,
                               const string& output_path,
                               const string& mode = "test",
                               float similarity_threshold = 80.0f,
                               float nms_threshold = 0.3f)
{
    int num_feature = 128;
    string class_id = "nut";
    
    if(mode == "train") {
        cout << "=== 开始训练三个螺母模板 ===" << endl;
        
        vector<string> template_paths = {template_path1, template_path2, template_path3};
        vector<string> templ_suffixes = {"_templ1.yaml", "_templ2.yaml", "_templ3.yaml"};
        vector<string> info_suffixes = {"_info1.yaml", "_info2.yaml", "_info3.yaml"};
        
        for(int i = 0; i < 3; i++) {
            cout << "\n--- 训练模板 " << (i+1) << " ---" << endl;
            
            line2Dup::Detector detector(num_feature, {16, 32});
            
            Mat template_img = imread(template_paths[i]);
            if(template_img.empty()) {
                cerr << "错误: 无法读取模板图片 " << template_paths[i] << endl;
                continue;
            }
            
            cout << "模板" << (i+1) << "图片尺寸: " << template_img.cols << "x" << template_img.rows << endl;
            
            Mat mask = Mat(template_img.size(), CV_8UC1, Scalar(255));
            
            shape_based_matching::shapeInfo_producer shapes(template_img, mask);
            shapes.scale_range = {0.8f, 1.2f};
            shapes.scale_step = 0.1f;
            shapes.angle_range = {0, 360};
            shapes.angle_step = 30;
            
            Timer timer;
            shapes.produce_infos();
            timer.out("生成模板" + to_string(i+1) + "变换信息");
            
            cout << "模板" << (i+1) << "总共生成 " << shapes.infos.size() << " 个变换" << endl;
            
            std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
            
            int success_count = 0;
            for(size_t j = 0; j < shapes.infos.size(); j++) {
                auto& info = shapes.infos[j];
                
                int templ_id = detector.addTemplate(shapes.src_of(info), class_id, 
                                                   shapes.mask_of(info), num_feature);
                
                if(templ_id != -1) {
                    infos_have_templ.push_back(info);
                    success_count++;
                }
            }
            
            cout << "模板" << (i+1) << "成功训练 " << success_count << " 个变换" << endl;
            
            detector.writeClasses(("nut" + templ_suffixes[i]).c_str());
            shapes.save_infos(infos_have_templ, "nut" + info_suffixes[i]);
        }
        
        cout << "\n=== 三个模板训练完成 ===" << endl;
        
    } else if(mode == "test") {
        cout << "=== 开始三模板投票检测 ===" << endl;
        
        vector<line2Dup::Detector> detectors(3);
        vector<string> templ_suffixes = {"_templ1.yaml", "_templ2.yaml", "_templ3.yaml"};
        vector<string> info_suffixes = {"_info1.yaml", "_info2.yaml", "_info3.yaml"};
        
        // 加载三个检测器
        std::vector<std::string> ids = {class_id};
        for(int i = 0; i < 3; i++) {
            detectors[i] = line2Dup::Detector(num_feature, {4, 8});
            detectors[i].readClasses(ids, ("nut" + templ_suffixes[i]).c_str());
            cout << "加载检测器 " << (i+1) << " 完成" << endl;
        }
        
        // 读取测试图片
        Mat test_img = imread(test_image_path);
        if(test_img.empty()) {
            cerr << "错误: 无法读取测试图片 " << test_image_path << endl;
            return;
        }
        
        cout << "测试图片尺寸: " << test_img.cols << "x" << test_img.rows << endl;
        
        // 确保图片尺寸是32的倍数
        int stride = 32;
        int n = test_img.rows / stride;
        int m = test_img.cols / stride;
        Rect roi(0, 0, stride * m, stride * n);
        Mat img = test_img(roi).clone();
        
        Timer timer;
        
        // 执行三次检测
        vector<vector<line2Dup::Match>> all_matches(3);
        vector<vector<Rect>> all_boxes(3);
        vector<vector<float>> all_scores(3);
        
        for(int i = 0; i < 3; i++) {
            cout << "\n--- 执行检测器 " << (i+1) << " ---" << endl;
            
            auto matches = detectors[i].match(img, similarity_threshold, ids);
            cout << "检测器" << (i+1) << "原始检测结果: " << matches.size() << " 个匹配" << endl;
            
            vector<Rect> boxes;
            vector<float> scores;
            
            for(auto& match : matches) {
                auto templ = detectors[i].getTemplates(class_id, match.template_id);
                
                Rect box;
                box.x = match.x;
                box.y = match.y;  
                box.width = templ[0].width;
                box.height = templ[0].height;
                
                boxes.push_back(box);
                scores.push_back(match.similarity);
            }
            
            // 对单个检测器结果先进行NMS
            vector<int> nms_indices;
            cv_dnn::NMSBoxes(boxes, scores, similarity_threshold, nms_threshold, nms_indices);
            
            // 保存NMS后的结果
            for(int idx : nms_indices) {
                all_matches[i].push_back(matches[idx]);
                all_boxes[i].push_back(boxes[idx]);
                all_scores[i].push_back(scores[idx]);
            }
            
            cout << "检测器" << (i+1) << "NMS后: " << all_matches[i].size() << " 个检测" << endl;
        }
        
        timer.out("三次检测完成，总用时");
        
        // 实现投票机制
        cout << "\n=== 开始投票合并 ===" << endl;
        
        vector<Rect> final_boxes;
        vector<float> final_scores;
        vector<int> vote_counts;
        
        /*
         * 修改前的问题分析：
         * 1. 使用x=-1标记"已使用"的方式不可靠，容易产生错误坐标
         * 2. 双重遍历逻辑导致同一检测可能被多次处理
         * 3. 合并坐标计算方式容易产生虚假位置
         * 
         * 修改后的解决方案：
         * 1. 使用独立的used标记数组，避免修改原始坐标
         * 2. 改用单次遍历所有检测，避免双重处理
         * 3. 使用加权平均合并坐标，提高准确性
         */
        
        // 创建used标记数组，避免修改原始坐标数据
        vector<vector<bool>> used(3);
        for(int i = 0; i < 3; i++) {
            used[i].resize(all_boxes[i].size(), false);
        }
        
        // 遍历每个检测器的检测结果
        for(int i = 0; i < 3; i++) {
            for(size_t j = 0; j < all_boxes[i].size(); j++) {
                // 跳过已经被处理过的检测
                if(used[i][j]) continue;
                
                Rect& box1 = all_boxes[i][j];
                float score1 = all_scores[i][j];
                
                // 初始化投票信息
                vector<int> voter_detectors = {i};          // 投票的检测器ID
                vector<size_t> voter_indices = {j};         // 对应的检测索引
                vector<float> voter_scores = {score1};      // 对应的置信度
                vector<Rect> voter_boxes = {box1};          // 对应的检测框
                
                // 查找与当前检测重叠的其他检测
                for(int k = 0; k < 3; k++) {
                    if(k == i) continue; // 跳过自己
                    
                    for(size_t l = 0; l < all_boxes[k].size(); l++) {
                        if(used[k][l]) continue; // 跳过已使用的
                        
                        Rect& box2 = all_boxes[k][l];
                        float score2 = all_scores[k][l];
                        
                        // 计算重叠度
                        float overlap = cv_dnn::rectOverlap(box1, box2);
                        
                        if(overlap > 0.3) { // 重叠度阈值
                            voter_detectors.push_back(k);
                            voter_indices.push_back(l);
                            voter_scores.push_back(score2);
                            voter_boxes.push_back(box2);
                            
                            // 立即标记为已使用，避免重复处理
                            used[k][l] = true;
                        }
                    }
                }
                
                // 标记当前检测为已使用
                used[i][j] = true;
                
                // 如果得到2票或以上，进行合并并加入最终结果
                if(voter_detectors.size() >= 2) {
                    // 使用加权平均计算合并后的检测框
                    float total_weight = 0;
                    float weighted_x = 0, weighted_y = 0;
                    float weighted_width = 0, weighted_height = 0;
                    float total_score = 0;
                    
                    for(size_t v = 0; v < voter_detectors.size(); v++) {
                        float weight = voter_scores[v] / 100.0f; // 将置信度作为权重
                        total_weight += weight;
                        
                        weighted_x += voter_boxes[v].x * weight;
                        weighted_y += voter_boxes[v].y * weight;
                        weighted_width += voter_boxes[v].width * weight;
                        weighted_height += voter_boxes[v].height * weight;
                        total_score += voter_scores[v];
                    }
                    
                    // 计算加权平均后的检测框
                    Rect merged_box;
                    merged_box.x = (int)(weighted_x / total_weight);
                    merged_box.y = (int)(weighted_y / total_weight);
                    merged_box.width = (int)(weighted_width / total_weight);
                    merged_box.height = (int)(weighted_height / total_weight);
                    
                    final_boxes.push_back(merged_box);
                    final_scores.push_back(total_score / voter_detectors.size());
                    vote_counts.push_back((int)voter_detectors.size());
                }
            }
        }
        
        cout << "投票合并后: " << final_boxes.size() << " 个有效检测" << endl;
        
        // 对最终结果再次应用NMS
        vector<int> final_nms_indices;
        cv_dnn::NMSBoxes(final_boxes, final_scores, similarity_threshold, nms_threshold, final_nms_indices);
        
        cout << "最终NMS后: " << final_nms_indices.size() << " 个检测结果" << endl;
        
        // 可视化结果
        Mat result_img = img.clone();
        
        cout << "\n=== 三模板投票检测结果 ===" << endl;
        cout << setw(5) << "序号" << setw(10) << "X坐标" << setw(10) << "Y坐标" 
             << setw(10) << "相似度" << setw(8) << "票数" << endl;
        cout << string(45, '-') << endl;
        
        for(size_t i = 0; i < final_nms_indices.size(); i++) {
            int idx = final_nms_indices[i];
            Rect& box = final_boxes[idx];
            float score = final_scores[idx];
            int votes = vote_counts[idx];
            
            // 根据票数设置颜色：3票=绿色，2票=蓝色
            Scalar color = (votes >= 3) ? Scalar(0, 255, 0) : Scalar(255, 0, 0);
            
            rectangle(result_img, box, color, 3);
            
            string label = "Nut " + to_string(i+1) + ": " + to_string((int)score) + "% (" + to_string(votes) + " votes)";
            putText(result_img, label, Point(box.x, box.y - 5), 
                   FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
            
            cout << setw(5) << (i+1) << setw(10) << box.x << setw(10) << box.y 
                 << setw(9) << fixed << setprecision(1) << score << "%" 
                 << setw(8) << votes << endl;
        }
        
        cout << "\n=== 投票检测统计 ===" << endl;
        cout << "有效检测数量: " << final_nms_indices.size() << " 个" << endl;
        
        int three_vote_count = 0;
        int two_vote_count = 0;
        for(int idx : final_nms_indices) {
            if(vote_counts[idx] >= 3) three_vote_count++;
            else two_vote_count++;
        }
        cout << "三票检测: " << three_vote_count << " 个" << endl;
        cout << "二票检测: " << two_vote_count << " 个" << endl;
        
        if(!final_nms_indices.empty()) {
            float avg_confidence = 0;
            for(int idx : final_nms_indices) {
                avg_confidence += final_scores[idx];
            }
            avg_confidence /= final_nms_indices.size();
            cout << "平均置信度: " << fixed << setprecision(1) << avg_confidence << "%" << endl;
        }
        
        // 保存结果图片
        imwrite(output_path, result_img);
        cout << "结果已保存到: " << output_path << endl;
        
        // 显示结果
        namedWindow("Multi-Template Detection Results", WINDOW_NORMAL);
        imshow("Multi-Template Detection Results", result_img);
        cout << "按任意键关闭显示窗口..." << endl;
        waitKey(0);
        destroyAllWindows();
    }
}



int main(int argc, char* argv[]) {
    // 检查是否是新的多模板模式（6个或以上参数）
    if(argc == 9) {
        // 多模板模式: ./nut_detector <模板1> <模板2> <模板3> <测试图片> <输出图片> test [相似度阈值] [NMS阈值]
        string template_path1 = argv[1];
        string template_path2 = argv[2];
        string template_path3 = argv[3];
        string test_image_path = argv[4];
        string output_path = argv[5];
        string mode = (argc > 6) ? argv[6] : "test";
        float similarity_threshold = (argc > 7) ? atof(argv[7]) : 80.0f;
        float nms_threshold = (argc > 8) ? atof(argv[8]) : 0.3f;
        
        cout << "螺母检测系统 (三模板投票模式)" << endl;
        cout << "================================" << endl;
        cout << "模板图片1: " << template_path1 << endl;
        cout << "模板图片2: " << template_path2 << endl;
        cout << "模板图片3: " << template_path3 << endl;
        cout << "测试图片: " << test_image_path << endl;
        cout << "输出图片: " << output_path << endl;
        cout << "运行模式: " << mode << endl;
        if(mode == "test") {
            cout << "相似度阈值: " << similarity_threshold << "%" << endl;
            cout << "NMS阈值: " << nms_threshold << endl;
        }
        cout << "================================" << endl << endl;
        
        detect_nuts_multi_template(template_path1, template_path2, template_path3, 
                                 test_image_path, output_path, mode, 
                                 similarity_threshold, nms_threshold);
    }
    else if(argc == 7) {
        // 原有单模板模式: ./nut_detector <模板图片> <测试图片> <输出图片> [test] [相似度阈值] [NMS阈值]
        string template_path = argv[1];
        string test_image_path = argv[2];
        string output_path = argv[3];
        string mode = (argc > 4) ? argv[4] : "test";
        float similarity_threshold = (argc > 5) ? atof(argv[5]) : 80.0f;
        float nms_threshold = (argc > 6) ? atof(argv[6]) : 0.3f;
        
        cout << "螺母检测系统 (单模板模式)" << endl;
        cout << "========================" << endl;
        cout << "模板图片: " << template_path << endl;
        cout << "测试图片: " << test_image_path << endl;
        cout << "输出图片: " << output_path << endl;
        cout << "运行模式: " << mode << endl;
        if(mode == "test") {
            cout << "相似度阈值: " << similarity_threshold << "%" << endl;
            cout << "NMS阈值: " << nms_threshold << endl;
        }
        cout << "========================" << endl << endl;
        
        detect_nuts(template_path, test_image_path, output_path, mode, similarity_threshold, nms_threshold);
    }
    else {
        cout << "用法: " << endl;
        cout << "单模板模式:" << endl;
        cout << "  训练: ./nut_detector <模板图片> <测试图片> <输出图片> train" << endl;
        cout << "  检测: ./nut_detector <模板图片> <测试图片> <输出图片> [test] [相似度阈值] [NMS阈值]" << endl;
        cout << endl;
        cout << "三模板投票模式:" << endl;
        cout << "  训练: ./nut_detector <模板1> <模板2> <模板3> <测试图片> <输出图片> train [相似度阈值] [NMS阈值]" << endl;
        cout << "  检测: ./nut_detector <模板1> <模板2> <模板3> <测试图片> <输出图片> test [相似度阈值] [NMS阈值]" << endl;
        cout << endl;
        cout << "示例:" << endl;
        cout << "单模板: ./nut_detector nut1.jpg test.jpg result.jpg test 80 0.3" << endl;
        cout << "三模板: ./nut_detector nut1.jpg nut2.jpg nut3.jpg test.jpg result.jpg test 80 0.3" << endl;
        return -1;
    }
    
    return 0;
}
