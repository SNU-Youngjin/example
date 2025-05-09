import pandas as pd

class FeatureEngineer:
    """
    CSV 파일을 읽어 주요 피처를 생성하고 결과를 저장하는 클래스
    """
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.df = None

    def load_data(self) -> None:
        """CSV 파일을 읽고 날짜 컬럼을 파싱하여 정렬"""
        df = pd.read_csv(self.input_path)
        # 첫 번째 컬럼을 date로 표준화
        date_col = df.columns[0]
        df = df.rename(columns={date_col: 'date'})
        df['date'] = pd.to_datetime(df['date'])
        # 날짜 기준 오름차순 정렬
        df = df.sort_values('date').reset_index(drop=True)
        self.df = df

    def engineer_features(self) -> None:
        """기존 데이터에 다양한 테크니컬 피처 추가 후 NaN 제거"""
        df = self.df.copy()
        # 전일 대비 수익률
        df['return_1m'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
        # 이동평균 및 이동표준편차
        for ma in [5, 10, 20]:
            df[f'ma_{ma}'] = df['close'].rolling(window=ma).mean()
        for std in [5, 10]:
            df[f'std_{std}'] = df['close'].rolling(window=std).std()
        # 거래량 기반 피처
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        # 거래대금 추정
        df['turnover_est'] = df['volume'] * df['close']
        # 가격 범위 및 시가/종가 차이
        df['high_low_range'] = df['high'] - df['low']
        df['close_vs_open'] = df['close'] - df['open']
        # NaN 값이 생긴 행 제거
        df = df.dropna().reset_index(drop=True)
        self.df = df

    def save(self, output_path: str) -> None:
        """피처 엔지니어링 결과를 CSV로 저장"""
        self.df.to_csv(output_path, index=False)
        print(f"Feature engineered CSV saved to: {output_path}")