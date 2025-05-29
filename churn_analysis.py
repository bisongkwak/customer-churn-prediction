#졸업 프로젝트 코드
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. 데이터 불러오기
df = pd.read_csv("/Users/bisong/Desktop/4학년 2학기/졸업프로젝트/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df.head())
print("데이터 크기:", df.shape)

# 2. 전처리

# TotalCharges 컬럼은 문자열로 되어 있으며, 공백 포함 값이 있으므로 숫자형으로 변환
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# 결측치 확인 및 중간값으로 대체
print("결측치 개수:\n", df.isnull().sum())
df.loc[:, 'TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
# 가입 기간이 0인 이상치 제거
df = df[df['tenure'] > 0]
print("정제 후 데이터 크기:", df.shape)

# 타깃 변수 Churn을 0, 1로 인코딩
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 분석에 불필요한 고객 ID 제거
df.drop('customerID', axis=1, inplace=True)

# 범주형 변수 원 핫 인코딩
df_encoded = pd.get_dummies(df)
print("인코딩 후 데이터 크기:", df_encoded.shape)
print(df_encoded.head())

# 3. 모델 학습 및 평가
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("정확도:", accuracy_score(y_test, y_pred))
print("\n혼동 행렬:\n", confusion_matrix(y_test, y_pred))
print("\n분류 리포트:\n", classification_report(y_test, y_pred))

# 4. 변수 중요도 시각화
importances = model.feature_importances_
features = X.columns
indices = importances.argsort()[::-1]

plt.figure(figsize=(12, 8))
sns.barplot(x=importances[indices][:15], y=features[indices][:15])
plt.title("Top 15 Feature Importances for Churn Prediction")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# 5. KMeans 군집 분석 + 군집별 이탈률 출력

# 군집 분석 대상 변수만 선택
cluster_data = df_encoded[['tenure', 'MonthlyCharges', 'TotalCharges']]

# 표준화
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

# 클러스터링 수행
kmeans = KMeans(n_clusters=4, random_state=42)
df_encoded['Cluster'] = kmeans.fit_predict(scaled_data)

# 군집별 이탈률 확인
print("\n군집별 평균 이탈률:")
print(df_encoded.groupby('Cluster')['Churn'].mean())

# 시각화 
plt.figure(figsize=(10, 6))
plt.scatter(df_encoded['TotalCharges'], df_encoded['MonthlyCharges'], c=df_encoded['Cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('TotalCharges')
plt.ylabel('MonthlyCharges')
plt.title('Customer Segments by KMeans Clustering')
plt.colorbar(label='Cluster')
plt.show()