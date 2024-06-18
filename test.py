import json

import faiss
from langchain.prompts import PromptTemplate
from langchain.retrievers import FAISSRetriever
from langchain.vectorstores import FAISS

# 加载FAISS索引和向量字典
index = faiss.read_index("data/faiss_index")
with open("data/patient_id_to_vector.json", "r") as f:
    patient_id_to_vector = json.load(f)

# 设置FAISS向量资料库
vector_store = FAISS(index=index, patient_id_to_vector=patient_id_to_vector)

# 设置LangChain检索器
retriever = FAISSRetriever(vector_store=vector_store)

# 设置生成EPL的模板
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

# 设置LangChain
chain = Chain(retriever=retriever, prompt_template=epl_template)

# 生成所有病患的EPL并保存到文件
with open("generated_epl_statements.txt", "w") as file:
    for patient_id in patient_id_to_vector.keys():
        # 构建查询以获取患者信息
        query = f"Retrieve patient information for patient ID {patient_id}"
        
        # 使用LangChain检索病患信息
        patient_info = chain.run(query)
        
        # 生成EPL语句
        epl_code = epl_template.render(patient_info=patient_info)
        
        # 将EPL语句写入文件
        file.write(f"EPL for patient ID {patient_id}:\n{epl_code}\n\n")

print("EPL statements have been generated and saved to 'generated_epl_statements.txt'")
