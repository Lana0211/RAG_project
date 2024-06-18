import json
import os

import faiss
import numpy as np
import torch
from transformers import BertModel, BertTokenizer

# 加載BERT模型和分詞器
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 加載病歷數據
with open("patient_info.json", "r") as f:
    patient_records = json.load(f)

# 檢查是否是字典或列表，如果是列表則取第一個元素
if isinstance(patient_records, list):
    if len(patient_records) > 0:
        patient_record = patient_records[0]
    else:
        raise ValueError("The patient records list is empty")
elif isinstance(patient_records, dict):
    patient_record = patient_records
else:
    raise TypeError("The patient records should be a dictionary or a list of dictionaries")

# 將病歷數據轉換為向量
def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# 構建FAISS向量資料庫
d = 768  # 向量維度（BERT輸出的向量維度）
index = faiss.IndexFlatL2(d)

# 存儲病患ID和向量的對應關係
patient_id_to_vector = {}

# 檢查是否存在'medical_history'鍵並且裡面包含'conditions'鍵
if 'medical_history' in patient_record and 'conditions' in patient_record['medical_history']:
    conditions_text = ', '.join([f"{cond['name']} diagnosed on {cond['diagnosis_date']}, treated with {cond['treatment']}" for cond in patient_record['medical_history']['conditions']])
else:
    conditions_text = ""

if 'medical_history' in patient_record and 'surgeries' in patient_record['medical_history']:
    surgeries_text = ', '.join([f"{surg['name']} on {surg['date']}, {surg['details']}" for surg in patient_record['medical_history']['surgeries']])
else:
    surgeries_text = ""

if 'medical_history' in patient_record and 'allergies' in patient_record['medical_history']:
    allergies_text = ', '.join([f"{allergy['name']} causing {allergy['reaction']}" for allergy in patient_record['medical_history']['allergies']])
else:
    allergies_text = ""

if 'medical_history' in patient_record and 'family_history' in patient_record['medical_history']:
    family_history_text = ', '.join([f"{key}: {value}" for key, value in patient_record['medical_history']['family_history'].items()])
else:
    family_history_text = ""

if 'medications' in patient_record:
    medications_text = ', '.join([f"{med['name']} {med['dosage']} {med['frequency']}" for med in patient_record['medications']])
else:
    medications_text = ""

text = (
    f"Patient ID: {patient_record['patient_id']}. Name: {patient_record['name']}. Age: {patient_record['age']}. Gender: {patient_record['gender']}. "
    f"Medical history: {conditions_text}. "
    f"Surgeries: {surgeries_text}. "
    f"Allergies: {allergies_text}. "
    f"Family history: {family_history_text}. "
    f"Social history: Smoking: {patient_record['medical_history']['social_history']['smoking']}, Alcohol: {patient_record['medical_history']['social_history']['alcohol']}, Exercise: {patient_record['medical_history']['social_history']['exercise']}. "
    f"Medications: {medications_text}. "
    f"Recent vitals: Heart rate: {patient_record['recent_vitals']['heart_rate']}, Blood pressure: {patient_record['recent_vitals']['blood_pressure']}, Blood glucose: {patient_record['recent_vitals']['blood_glucose']}, Oxygen saturation: {patient_record['recent_vitals']['oxygen_saturation']}."
)

# 將文本轉換為向量
vector = encode_text(text)

# 添加向量到FAISS索引
index.add(vector)

# 存儲病患ID和向量
patient_id_to_vector[patient_record['patient_id']] = vector.tolist()  # 轉換為列表以便json.dump

# 創建數據文件夾
os.makedirs("data", exist_ok=True)

# 保存FAISS索引
faiss.write_index(index, "data/faiss_index")

# 保存病患ID對應的向量字典
with open("data/patient_id_to_vector.json", "w") as f:
    json.dump(patient_id_to_vector, f)
