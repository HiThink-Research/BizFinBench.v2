import json
import re
from utils import JsonPaser

def evaluation(input_path, **kwargs):
    corrects = []
    total_scores = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(l) for l in f]
    
    out = []
    for d in data:
        try:
            # 获取标准答案
            choices = d.get("choices", [])
            correct_answer = ""
            if choices and len(choices) > 0:
                message = choices[0].get("message", {})
                content = message.get("content", [])
                if isinstance(content, list) and len(content) > 0:
                    correct_answer = content[0].get("text", "").strip()
                else:
                    correct_answer = str(content).strip()
            
            # 获取模型预测结果
            predict_result_str = d.get("predict_result", "")
            
            # 使用JsonPaser解析预测结果
            j_paser = JsonPaser()
            predict_data = j_paser.extract_json_from_text(predict_result_str)
            
            # 提取预测的答案
            predicted_answer = ""
            if predict_data:
                # 尝试从不同可能的键中获取答案
                if "answer" in predict_data:
                    predicted_answer = str(predict_data["answer"]).strip()
                elif "排序结果" in predict_data:  # 中文键名
                    predicted_answer = str(predict_data["排序结果"]).strip()
                elif "result" in predict_data:
                    predicted_answer = str(predict_data["result"]).strip()
            
            # 如果直接提取失败，尝试从字符串中匹配
            if not predicted_answer and "answer" in predict_result_str:
                # 简单的正则匹配，提取answer字段的值
                # import re  # ← 删除这行重复的导入
                match = re.search(r'"answer"\s*:\s*"([^"]+)"', predict_result_str)
                if match:
                    predicted_answer = match.group(1).strip()
            
            # 比较答案（忽略空格和格式差异）
            # 标准化答案格式：移除所有空格，统一为逗号分隔
            correct_normalized = re.sub(r'\s+', '', correct_answer)
            predicted_normalized = re.sub(r'\s+', '', predicted_answer)
            
            # 检查是否完全一致
            if correct_normalized == predicted_normalized:
                score = 1.0
                d['eval_result'] = {"result": "True"}
            else:
                score = 0.0
                d['eval_result'] = {
                    "result": "False", 
                    "correct_answer": correct_answer,
                    "predicted_answer": predicted_answer
                }
            
            d['score'] = score
            corrects.append(score)
            total_scores.append(1)
            
            # 添加调试信息
            d['debug_info'] = {
                "correct_normalized": correct_normalized,
                "predicted_normalized": predicted_normalized
            }

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            d['eval_result'] = {"result": f"error: {str(e)}"}
            d['score'] = 0
            total_scores.append(1)
            corrects.append(0)

        out.append(d)

    # 写回结果
    with open(input_path, 'w', encoding='utf-8') as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + '\n')

    # 计算准确率
    total_correct = sum(corrects)
    total_possible = sum(total_scores)
    overall_score = total_correct / total_possible if total_possible > 0 else 0

    return {"acc": overall_score}