from feature_engineering import FeatureEngineer
from transformer_pipeline import adf_test, TimeSeriesDataset, TimeSeriesMultiTransformer
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

def main():
    # 1) 피처 엔지니어링
    fe = FeatureEngineer('005930_daily_3d_to_yesterday.csv')
    fe.load_data()
    fe.engineer_features()
    fe.save('5m8d_test.csv')

    # 2) 데이터 로드 및 NaN 제거
    df = pd.read_csv('5m8d_test.csv', parse_dates=['date'], index_col='date').dropna()

    # 3) 다중 타깃 생성
    df['close_1m']  = df['close'].shift(-1)
    df['close_5m']  = df['close'].shift(-5)
    df['close_60m'] = df['close'].shift(-60)
    df['date_only'] = df.index.date
    df['close_eod'] = df.groupby('date_only')['close'].transform('last')
    df = df.drop(columns='date_only').dropna()

    # 4) 특성·타깃 배열 생성
    features = [
        'return_1m', 'ma_5', 'ma_10', 'ma_20',
        'std_5', 'std_10', 'volume_ma_5', 'volume_ma_20',
        'volume_ratio', 'turnover_est', 'high_low_range', 'close_vs_open'
    ]
    X = df[features].values
    y = df[['close_1m', 'close_5m', 'close_60m', 'close_eod']].values

    # 5) 스케일링
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # 6) Dataset & DataLoader
    seq_len = 60
    dataset = TimeSeriesDataset(X_scaled, y_scaled, seq_len=seq_len)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False)

    # ✅ device 설정 (MPS → CUDA → CPU)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")


    # 7) 모델·옵티마이저 초기화
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    model = TimeSeriesMultiTransformer(input_dim=len(features)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 8) 학습 루프
    epochs = 5
    loss_history = []
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for Xb, yb in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(Xb)                    # (batch, 4)
            loss  = F.mse_loss(preds, yb)        # 모든 타깃 MSE
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch}/{epochs} - Avg Loss: {avg_loss:.6f}")

    # 9) 손실 곡선 시각화
    plt.figure(figsize=(8,4))
    plt.plot(range(1, epochs+1), loss_history, marker='o')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.show()

    # 10) 전체 데이터에 대한 예측 수집
    model.eval()
    all_preds, all_truths = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            ps = model(Xb.to(device)).cpu().numpy()
            ts = yb.numpy()
            all_preds.append(ps)
            all_truths.append(ts)
    preds_all  = np.vstack(all_preds)   # shape (n_samples, 4)
    truths_all = np.vstack(all_truths)  # same shape

    # 11) 타임스탬프 매핑
    n_samples = len(dataset)
    end_idxs   = np.arange(seq_len - 1, seq_len - 1 + n_samples)
    times_all  = df.index[end_idxs]

    # 12) 스케일 역변환
    preds_inv  = scaler_y.inverse_transform(preds_all)
    truths_inv = scaler_y.inverse_transform(truths_all)

    # 13) 4가지 예측 결과 전체 시각화
    horizons = ['1min', '5min', '60min', 'EOD']
    fig, axes = plt.subplots(2, 2, figsize=(14,10), sharex=True)
    for i, ax in enumerate(axes.flatten()):
        ax.plot(times_all, truths_inv[:, i], label='Actual', linewidth=1)
        ax.plot(times_all, preds_inv[:, i], linestyle='--', label='Predicted')
        ax.set_title(f'{horizons[i]} Ahead')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
    for ax in axes[1]:
        ax.set_xlabel('Time')
    plt.tight_layout()
    plt.show()

    # 14) 정상성 검정
    adf_test(df['close'], '종가 원시')


if __name__ == '__main__':
    main()
