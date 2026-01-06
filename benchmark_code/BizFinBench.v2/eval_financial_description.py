import json
import re
import numpy as np
from rouge import Rouge

def evaluation(input_path, **kwargs):
    """
    收紧容错机制的评估函数（兼容 <think>...</think> 思考块）
    """
    total_TP = total_FP = total_FN = 0
    rouge_scores = []
    rouge = Rouge()

    def extract_json_array(text: str):
        """严格解析JSON数组，失败返回None（自动丢弃 <think>...</think>）"""
        if not text:
            return None

        # 0) 丢弃所有 <think>...</think> 块（大小写不敏感）
        #   例如："<think>思考内容</think>\n\n[ {...} ]" -> "\n\n[ {...} ]"
        reg_list = [r'<think>[\s\S]*?</think>', r'[\s\S]*think>']
        for reg in reg_list:
            text = re.sub(reg, '', text, flags=re.IGNORECASE)
            text = text.strip()
        
        if not text:
            return None
        
        # 1) 移除代码围栏，但要求围栏格式必须规范
        if text.startswith('```'):
            # 必须匹配完整的围栏结构
            fence_match = re.match(r'^```(?:json)?\s*\n([\s\S]*?)\n\s*```$', text)
            if fence_match:
                text = fence_match.group(1).strip()
            else:
                # 围栏不完整，直接返回None
                return None
        
        # 2) 严格JSON解析，必须是数组格式
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                # 验证数组元素必须是字典
                if all(isinstance(item, dict) for item in parsed):
                    return parsed
        except:
            pass
            
        return None

    def to_id_list(answer_field):
        """严格提取ID列表，只接受数字ID"""
        out = []
        
        def process_value(v):
            if isinstance(v, (int, float)):
                return [str(int(v))]
            elif isinstance(v, str):
                # 尝试解析JSON，失败则提取纯数字
                try:
                    parsed = json.loads(v)
                    return process_value(parsed)
                except:
                    # 只提取连续数字
                    numbers = re.findall(r'\b\d+\b', v)
                    return numbers if numbers else []
            elif isinstance(v, (list, tuple)):
                result = []
                for item in v:
                    result.extend(process_value(item))
                return result
            else:
                return []
        
        numbers = process_value(answer_field)
        # 去重并保持顺序
        seen = set()
        unique_numbers = []
        for num in numbers:
            if num not in seen:
                seen.add(num)
                unique_numbers.append(num)
                
        return unique_numbers

    def parse_prediction(sample):
        """严格解析预测结果（兼容 predict_result / assistant 消息，以及 <think> 块）"""
        pred_ids, id2cot = set(), {}
        
        # 优先从 predict_result 解析
        predict_result = sample.get("predict_result", "")
        if predict_result:
            plist = extract_json_array(predict_result)
            if plist:
                for obj in plist:
                    if not isinstance(obj, dict):
                        continue
                    # 严格匹配字段名（大小写敏感）
                    answer_val = obj.get("answer") or obj.get("Answer")
                    cot_val = obj.get("cot") or obj.get("Cot") or obj.get("COT")
                    
                    if answer_val is not None:
                        ids = to_id_list(answer_val)
                        for _id in ids:
                            pred_ids.add(_id)
                            if cot_val and str(cot_val).strip():
                                id2cot[_id] = str(cot_val).strip()
        
        # 如果 predict_result 没有有效结果，尝试从 assistant 消息解析
        if not pred_ids:
            for msg in sample.get("messages", []):
                if msg.get("role") == "assistant":
                    for item in msg.get("content", []):
                        if item.get("type") == "text":
                            text_content = item.get("text", "").strip()
                            if text_content:
                                plist = extract_json_array(text_content)
                                if plist:
                                    for obj in plist:
                                        if not isinstance(obj, dict):
                                            continue
                                        answer_val = obj.get("answer") or obj.get("Answer")
                                        cot_val = obj.get("cot") or obj.get("Cot") or obj.get("COT")
                                        
                                        if answer_val is not None:
                                            ids = to_id_list(answer_val)
                                            for _id in ids:
                                                pred_ids.add(_id)
                                                if cot_val and str(cot_val).strip():
                                                    id2cot[_id] = str(cot_val).strip()
                                    break  # 找到一个有效结果就停止
                    if pred_ids:
                        break
        
        return pred_ids, id2cot

    def parse_ground_truth(sample):
        """严格解析真实标签（同样兼容可能出现的 <think> 块）"""
        truth_ids, id2cot = set(), {}
        
        # 从标准位置解析
        choices = sample.get("choices", [])
        if not choices:
            return truth_ids, id2cot
            
        message = choices[0].get("message", {})
        content_list = message.get("content", [])
        if not content_list:
            return truth_ids, id2cot
            
        text_content = content_list[0].get("text", "").strip()
        if not text_content:
            return truth_ids, id2cot
        
        glist = extract_json_array(text_content)
        if not glist:
            return truth_ids, id2cot
            
        for obj in glist:
            if not isinstance(obj, dict):
                continue
                
            answer_val = obj.get("answer") or obj.get("Answer")
            cot_val = obj.get("cot") or obj.get("Cot") or obj.get("COT")
            
            if answer_val is not None:
                ids = to_id_list(answer_val)
                for _id in ids:
                    truth_ids.add(_id)
                    if cot_val and str(cot_val).strip():
                        id2cot[_id] = str(cot_val).strip()
        
        return truth_ids, id2cot

    # 统计解析失败的情况
    parse_failures = 0
    total_samples = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            total_samples += 1
            try:
                data = json.loads(line.strip())
            except:
                parse_failures += 1
                continue

            pred_ids, id2cot_pred = parse_prediction(data)
            truth_ids, id2cot_truth = parse_ground_truth(data)

            # 如果预测结果完全无法解析，同时也没有 ground truth，跳过该样本
            if not pred_ids and not truth_ids:
                parse_failures += 1
                continue

            # ---- F1计算 ----
            TP = len(pred_ids & truth_ids)
            FP = len(pred_ids - truth_ids)
            FN = len(truth_ids - pred_ids)

            total_TP += TP
            total_FP += FP
            total_FN += FN

            # ---- ROUGE-L计算 ----
            for _id in (pred_ids & truth_ids):
                pc = id2cot_pred.get(_id)
                tc = id2cot_truth.get(_id)
                if pc and tc and pc.strip() and tc.strip():
                    try:
                        scores = rouge.get_scores(pc, tc)
                        rouge_scores.append(scores[0]['rouge-l']['f'])
                    except:
                        pass

    # 计算指标
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) else 0.0
    recall    = total_TP / (total_TP + total_FN) if (total_TP + total_FN) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    avg_rouge = float(np.mean(rouge_scores)) if rouge_scores else 0.0
    
    # 返回评估结果和解析统计
    return {
        "acc": f1, 
        "cot_quality": avg_rouge,
        "stats": {
            "total_samples": total_samples,
            "parse_failures": parse_failures,
            "success_rate": (total_samples - parse_failures) / total_samples if total_samples else 0
        }
    }