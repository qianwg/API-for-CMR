from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import requests
from openai import OpenAI

app = Flask(__name__)

# 加载模型和预处理管道
# model = joblib.load('V2-logistic_best_model.pkl')
DataStep7 = pd.read_feather('V2-DataStep7.feather')

# Define X and y
X = DataStep7.drop(columns=['Cluster'])
y = DataStep7['Cluster']

# 创建 LabelEncoder 对象
label_encoder = LabelEncoder()

# 将 y 中的标签转换为数值
y_encoded = label_encoder.fit_transform(y)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define the numeric and categorical columns
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),           # Normalize numeric columns
        ('cat', OneHotEncoder(), categorical_cols)  # One-hot encode categorical columns
    ])

# Create the complete pipeline with zero-variance feature removal
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('variance_threshold', VarianceThreshold())           # Remove zero-variance columns
])

# Fit the pipeline to the training data
pipeline.fit(X_train)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 从表单获取数据
        data = {
            'Basal_metabolic_rate': float(request.form['bmr']),
            'Standing_height': float(request.form['height']),
            'Sex': request.form['sex'],
            'Testosterone': float(request.form['testosterone']),
            'Weight': float(request.form['weight']),
            'Waist_circumference': float(request.form['waist']),
            'FVC': float(request.form['fvc']),
            'FEV1': float(request.form['fev1']),
            'Creatinine': float(request.form['creatinine']),
            'Urate': float(request.form['urate'])
        }
        age = {'Age': float(request.form['age'])}
        # 将数据转换为 DataFrame
        input_df = pd.DataFrame([data])
        print(input_df)

        # processed_data = preprocessor.transform(input_df)
        # Transform the training and test sets
        processed_data = pipeline.transform(input_df)
        # # 预处理数据
        # processed_data = preprocessor.fit_transform(input_df)
        print(processed_data)
        print('transform complete')
        model = joblib.load('V2-rf_best_model.pkl')
        # 预测概率
        prediction_prob = model.predict_proba(processed_data)[:, 1]

        # 计算类别 0 的概率
        prob_class0 = 1 - prediction_prob[0]
        
        # 返回结果
        result = {
            'prob_class0': round(prob_class0, 4),   # 四舍五入
            'prob_class1': round(prediction_prob[0], 4)  # 类别 1 的概率
        }


        if prediction_prob > 0.7:
            prompt = "Your cardiac function may be compromised, posing a high risk for stroke"
        elif 0.5 <= prediction_prob <= 0.7:
            prompt = "Your cardiac function is generally poor, which places you at an increased risk for stroke."
        else:
            prompt = "Your cardiac function is good, resulting in a lower risk of stroke"

        import json
        
        client = OpenAI(api_key="sk-b7bce0c106314df0af233d8f7d59b2d1", base_url="https://api.deepseek.com")
        print("1--------")
        #print(response.choices[0].message.content)

        #url = "https://api.deepseek.com/chat/completions"
        #model="deepseek-chat"

        patient_description = (
            f"The patient's physical examination findings are: Age {age['Age']}, Sex {data['Sex']}，Basal metabolic_rate {data['Basal_metabolic_rate']} KJ，"
            f"Height {data['Standing_height']} cm，Weight {data['Weight']} kg，Waist circumference {data['Waist_circumference']} cm，"
            f"Testosterone {data['Testosterone']} nmol/L，FVC {data['FVC']} L，FEV1 {data['FEV1']} L，"
            f"Creatinine {data['Creatinine']} μmoI/L，Urate {data['Urate']} μmoI/L。"
        )
        full_message = (
            f"The health status of the inquirer is:：{patient_description}，after our machine learning model computation,{prompt}"
            "Please analyze the patient's health status, combining our predictive results, and provide medical recommendations for the heart, stroke, and overall health. Please answer the question in English. Do not add any questions; only provide conclusions and recommendations."
        )       

        messages=[
            {"role": "system", "content": "As a medical expert, you are skilled at providing specific medical information tailored to a patient's health status."},
            {"role": "user", "content": full_message}
        ]
        
        response_raw = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )
        response = response_raw.choices[0].message.content
        # payload = {
        #     "model": model,
        #     "messages": messages,
        #     "stream": False,  # 启用流式处理
        #     "max_tokens": 16384,#max_tokens必须小于等于16384
        #     "stop": ["null"],
        #     "temperature": 0.7,
        #     "top_p": 0.7,
        #     "top_k": 50,
        #     "frequency_penalty": 0.5,
        #     "n": 1,
        #     "response_format": {"type": "text"},
        # }

        # headers = {
        #     "Authorization": "Bearer sk-b7bce0c106314df0af233d8f7d59b2d1",
        #     "Content-Type": "application/json"
        # }       

        # response = requests.post(url, json=payload, headers=headers, stream=True)
        print(response)
        print("2--------")

        # final_response = ""

        # if response.status_code == 200:
        #     first_reasoning_content_output = True
        #     first_content_output = True

        #     for chunk in response.iter_lines():
        #         if chunk:  # 过滤掉 keep-alive 新行
        #             chunk_str = chunk.decode('utf-8').strip()
            
        #             try:
        #                 if chunk_str.startswith('data:'):
        #                     chunk_str = chunk_str[6:].strip()  # 去除 "data:" 前缀
        #                 if chunk_str == "[DONE]":  # 结束信号
        #                     print("\n\n============[DONE]============\n")
        #                     break
                
        #                 # 解析 JSON
        #                 chunk_json = json.loads(chunk_str)
        #                 if 'choices' in chunk_json and isinstance(chunk_json['choices'], list) and len(chunk_json['choices']) > 0:
        #                     choice = chunk_json['choices'][0]
        #                     delta = choice.get('delta', {})

        #                     # 获取生成内容
        #                     content = delta.get('content')
                    
        #                     # 如果有内容，则拼接
        #                     if content:
        #                         if first_content_output:
        #                             print("\n\n==============================\nResult:")
        #                             first_content_output = False
        #                         print(content, end='', flush=True)
        #                         final_response += content  # ✅ 追加到最终响应变量

        #             except json.JSONDecodeError as e:
        #                 print(f"JSON 解码错误: {e}", flush=True)

        # ✅ 最终完整响应
        # print("\n\n最终完整响应:")
        # print(final_response)
        print("3-----")
        return render_template('index.html', result=result,
                               final_response = response)

    # GET 请求时显示空表单
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)