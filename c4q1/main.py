import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os


def load_data():
    """데이터 로드"""
    train_path = os.path.join('data', 'train.csv')
    test_path = os.path.join('data', 'test.csv')
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df


def preprocess_data(train_df, test_df):
    """데이터 전처리"""
    # Cabin 정보 분리 (Deck/Num/Side)
    def split_cabin(df):
        df['Cabin_Deck'] = df['Cabin'].str.split('/').str[0]
        df['Cabin_Num'] = df['Cabin'].str.split('/').str[1]
        df['Cabin_Side'] = df['Cabin'].str.split('/').str[2]
        df['Cabin_Num'] = pd.to_numeric(df['Cabin_Num'], errors='coerce')
        return df
    
    train_df = split_cabin(train_df)
    test_df = split_cabin(test_df)
    
    # Name에서 성 추출 (같은 그룹 식별용)
    train_df['LastName'] = train_df['Name'].str.split().str[-1]
    test_df['LastName'] = test_df['Name'].str.split().str[-1]
    
    # 그룹 크기 계산 (같은 PassengerId 접두사)
    train_df['GroupId'] = train_df['PassengerId'].str.split('_').str[0]
    test_df['GroupId'] = test_df['PassengerId'].str.split('_').str[0]
    
    train_group_size = train_df.groupby('GroupId').size().to_dict()
    test_group_size = test_df.groupby('GroupId').size().to_dict()
    
    train_df['GroupSize'] = train_df['GroupId'].map(train_group_size)
    test_df['GroupSize'] = test_df['GroupId'].map(test_group_size)
    
    # 사용할 특징 선택
    feature_cols = [
        'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
        'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
        'Cabin_Deck', 'Cabin_Num', 'Cabin_Side', 'GroupSize'
    ]
    
    # 결측치 처리
    # 숫자형 변수는 중앙값으로 채우기
    numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Cabin_Num']
    for col in numeric_cols:
        train_median = train_df[col].median()
        train_df[col] = train_df[col].fillna(train_median)
        test_df[col] = test_df[col].fillna(train_median)
    
    # 카테고리컬 변수는 최빈값으로 채우기
    categorical_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Cabin_Deck', 'Cabin_Side']
    for col in categorical_cols:
        train_mode = train_df[col].mode()[0] if len(train_df[col].mode()) > 0 else 'Unknown'
        train_df[col] = train_df[col].fillna(train_mode)
        test_df[col] = test_df[col].fillna(train_mode)
    
    # 불리언 변수 변환
    bool_cols = ['CryoSleep', 'VIP']
    for col in bool_cols:
        train_df[col] = train_df[col].astype(int)
        test_df[col] = test_df[col].astype(int)
    
    # 카테고리컬 변수 인코딩
    le_dict = {}
    for col in ['HomePlanet', 'Destination', 'Cabin_Deck', 'Cabin_Side']:
        le = LabelEncoder()
        # train과 test를 합쳐서 인코딩 (새로운 카테고리 대비)
        combined = pd.concat([train_df[col], test_df[col]])
        le.fit(combined)
        train_df[col + '_encoded'] = le.transform(train_df[col])
        test_df[col + '_encoded'] = le.transform(test_df[col])
        le_dict[col] = le
    
    # 최종 특징 선택
    final_features = [
        'HomePlanet_encoded', 'CryoSleep', 'Destination_encoded', 'Age', 'VIP',
        'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
        'Cabin_Deck_encoded', 'Cabin_Num', 'Cabin_Side_encoded', 'GroupSize'
    ]
    
    X_train = train_df[final_features]
    y_train = train_df['Transported'].astype(int)
    X_test = test_df[final_features]
    
    return X_train, y_train, X_test, test_df['PassengerId']


def train_model(X_train, y_train):
    """모델 학습"""
    # 랜덤 포레스트 모델 사용
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model


def make_predictions(model, X_test):
    """예측 수행"""
    predictions = model.predict(X_test)
    return predictions


def save_submission(passenger_ids, predictions, filename='submission.csv'):
    """제출 파일 저장"""
    submission_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Transported': predictions.astype(bool)
    })
    submission_df.to_csv(filename, index=False)
    print(f"제출 파일이 {filename}에 저장되었습니다.")
    return submission_df


def main():
    """메인 함수"""
    print("데이터 로드 중...")
    train_df, test_df = load_data()
    print(f"Train 데이터 크기: {train_df.shape}")
    print(f"Test 데이터 크기: {test_df.shape}")
    
    print("\n데이터 전처리 중...")
    X_train, y_train, X_test, passenger_ids = preprocess_data(train_df, test_df)
    print(f"학습 특징 수: {X_train.shape[1]}")
    
    print("\n모델 학습 중...")
    model = train_model(X_train, y_train)
    
    print("\n예측 수행 중...")
    predictions = make_predictions(model, X_test)
    
    print("\n제출 파일 생성 중...")
    submission = save_submission(passenger_ids, predictions)
    
    print(f"\n예측 결과 요약:")
    print(f"Transported=True: {predictions.sum()}명")
    print(f"Transported=False: {(predictions == 0).sum()}명")
    
    return model, submission


if __name__ == "__main__":
    model, submission = main()
