import json

import faiss
import requests
from langchain import Chain
from langchain.prompts import PromptTemplate
from langchain.retrievers import FAISSRetriever
from langchain.vectorstores import FAISS

# 加載FAISS索引和向量字典
index = faiss.read_index("data/faiss_index")
with open("data/patient_id_to_vector.json", "r") as f:
    patient_id_to_vector = json.load(f)

# 設置FAISS向量資料庫
vector_store = FAISS(index=index, patient_id_to_vector=patient_id_to_vector)

# 設置LangChain檢索器
retriever = FAISSRetriever(vector_store=vector_store)

# 設置生成EPL的模板
epl_template = PromptTemplate(
    input_variables=["patient_info"],
    template="""
    @Name('AgeGenderSpecificHighHeartRate')
    insert into HighHeartRateEvent
    select patientId, heartRate, timestamp
    from HeartRateStream
    where {patient_info['age']} >= 18 and {patient_info['gender']} = 'male' and heartRate > 100
    
    @Name('DrugAllergyEvent')
    insert into DrugAllergyAlert
    select patientId, drugName, symptoms, timestamp
    from DrugAdministrationStream
    where patientId = '{patient_info['patient_id']}' and drugName in ({','.join([f"'{allergy}'" for allergy in patient_info['allergies']])})
    
    @Name('CardiovascularRiskEvent')
    insert into CardiovascularRiskAlert
    select patientId, heartRate, systolic, diastolic, timestamp
    from HeartRateStream as h, BloodPressureStream as b
    where h.patientId = b.patientId and h.patientId = '{patient_info['patient_id']}' and (heartRate > 110 or systolic > 150 or diastolic > 90)
    
    @Name('AgeGenderSpecificLowOxygenLevel')
    insert into LowOxygenLevelEvent
    select patientId, oxygenLevel, timestamp
    from OxygenLevelStream
    where {patient_info['age']} >= 65 and oxygenLevel < 92
    """
)

# 設置LangChain
chain = Chain(retriever=retriever, prompt_template=epl_template)

# 取得病患資訊並生成EPL
for patient_id in patient_id_to_vector.keys():
    # 構建查詢以獲取患者信息
    query = f"Retrieve patient information for patient ID {patient_id}"
    
    # 使用LangChain檢索病患資訊
    patient_info = chain.run(query)
    
    # 生成EPL語句
    epl_code = epl_template.render(patient_info=patient_info)

    # 使用Requests將EPL部署到Esper
    esper_api_url = "http://localhost:5000/deploy"
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(esper_api_url, json={"epl_code": epl_code}, headers=headers)

    if response.status_code == 200:
        print(f"EPL successfully deployed to Esper for patient ID {patient_id}")
    else:
        print(f"Failed to deploy EPL to Esper for patient ID {patient_id}")
        print(response.text)
