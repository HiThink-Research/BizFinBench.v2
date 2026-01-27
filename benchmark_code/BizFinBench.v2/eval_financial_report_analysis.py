import json
import re

def evaluation(input_path, **kwargs):
    corrects = []
    total_scores = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(l) for l in f]
    
    out = []
    for d in data:
        try:
            choices = d.get("choices", [])
            correct_answer = ""
            if choices and len(choices) > 0:
                message = choices[0].get("message", {})
                content = message.get("content", [])
                if isinstance(content, list) and len(content) > 0:
                    correct_answer = content[0].get("text", "")
                else:
                    correct_answer = content    
            
            predict_result_str = d.get("predict_result", "")
            if predict_result_str:
                predict_result_str = re.sub("[\s\S]*think>", "", predict_result_str).strip()
                predict_result_list = re.findall("boxed\{.*?\}", predict_result_str)
                if predict_result_list:
                    predicted_answers = re.search("boxed\{(.*)\}", predict_result_list[-1]).group(1)
                    predicted_answers = re.sub(r"[,，\s]", "", predicted_answers)
                else:
                    predicted_answers = ""
            else:
                predicted_answers = ""
            


            correct_answers = d['choices'][0]['message']['content'][0]['text']
            correct_answers = re.sub(r"[,，\s]", "", correct_answers)
            
            if predicted_answers == correct_answers:
                score = 1.0
                d['eval_result'] = {"result": "True"}
            else:
                score = 0.0
                d['eval_result'] = {"result": "False"}
            
            
            d['score'] = score
            
            corrects.append(score)
            total_scores.append(1)

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            d['eval_result'] = {"result": f"error: {str(e)}"}
            d['score'] = 0
            total_scores.append(1)

        out.append(d)

    with open(input_path, 'w', encoding='utf-8') as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + '\n')

    total_correct = sum(corrects)
    total_possible = sum(total_scores)
    overall_score = total_correct / total_possible if total_possible > 0 else 0

    return {"acc": overall_score}  
