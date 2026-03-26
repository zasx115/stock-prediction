#!/usr/bin/env python3
# ============================================
# 파일명: src/run_hybrid_new_backtest.py
# 설명: 하이브리드_New 전략 백테스트 CLI 실행기 (Walk-Forward Sliding Window)
#
# ============================================
# [개요]
# ============================================
#
#   GitHub Actions 워크플로우(.github/workflows/backtest_hybrid_new.yml)의 진입점.
#   CLI 인자로 백테스트 파라미터를 받아 결과 텍스트와 PNG 그래프를 출력.
#
#   기존 하이브리드 전략(run_hybrid_backtest.py)과 동일한 Walk-Forward 구조를 사용하되,
#   과적합 방지에 초점을 맞춘 Hybrid_New 최적 파라미터를 적용.
#
# ============================================
# [하이브리드 vs 하이브리드_New 파라미터 비교]
# ============================================
#
#   | 항목              | 하이브리드 (기존)    | 하이브리드_New       |
#   |-------------------|---------------------|---------------------|
#   | XGB max_depth     | 4                   | 3 (얕은 트리)        |
#   | XGB subsample     | 0.7                 | 0.5 (강한 배깅)      |
#   | XGB colsample     | 0.7                 | 0.5 (강한 피처 샘플링)|
#   | 라벨 타겟         | 5D-5% (+5%)         | 5D-3% (+3%, 현실적)  |
#   | 가중치            | M50% + AI50%        | M50% + AI50%        |
#   | WF 스텝           | 6개월               | 6개월               |
#
# ============================================
# [Walk-Forward Sliding Window 알고리즘]
# ============================================
#
#   학습 윈도우 크기(train_end - train_start = 5년)를 고정하고,
#   매 스텝마다 train_start와 train_end를 wf_step_months(6개월)씩 슬라이딩.
#   테스트 윈도우는 항상 wf_step_months(6개월).
#
#   기본값 예시 (train_start=2015-01-01, train_end=2020-01-01, wf_step_months=6):
#
#     Fold 1: Train [2015-01-01 ~ 2020-01-01] → Test [2020-01-01 ~ 2020-07-01]
#     Fold 2: Train [2015-07-01 ~ 2020-07-01] → Test [2020-07-01 ~ 2021-01-01]
#     Fold 3: Train [2016-01-01 ~ 2021-01-01] → Test [2021-01-01 ~ 2021-07-01]
#     ...backtest_end까지 반복
#
#     |---- 5년 학습 (고정 크기) ----|--- 6M 테스트 ---|
#     |============================|================|
#              ← 6M 슬라이딩 →
#
# ============================================
# [실행 흐름]
# ============================================
#
#   1. parse_args()로 CLI 인자 파싱
#   2. generate_walk_forward_folds()로 폴드 목록 생성
#   3. get_sp500_list() + get_backtest_data()로 전체 기간 데이터 다운로드
#   4. create_features(target_return=0.03)로 5D-3% 라벨 피처 생성
#   5. 각 폴드별 반복:
#      a. 학습/테스트 데이터 슬라이싱 (날짜 범위)
#      b. AIStrategy(params=XGB_PARAMS_OPTIMIZED) 직접 생성 및 학습
#         ※ HybridStrategy.prepare() 대신 직접 주입하여 최적 파라미터 적용
#      c. HybridStrategy에 AI 전략 주입 + 모멘텀 전략 준비
#      d. run_ai_backtest()로 폴드별 백테스트 실행
#   6. stitch_portfolios()로 폴드별 포트폴리오를 하나의 연속 시계열로 연결
#   7. calculate_ai_metrics()로 전체 기간 메트릭 계산
#   8. print_results()로 텍스트 출력
#   9. plot_results()로 4-subplot 그래프 저장 (PNG)
#
# ============================================
# [AI 전략 주입 방식 (핵심)]
# ============================================
#
#   HybridStrategy.prepare()를 사용하면 내부에서 기본 XGB_PARAMS로 AIStrategy를
#   생성하므로, 최적 파라미터(XGB_PARAMS_OPTIMIZED)를 적용할 수 없음.
#   따라서 run_single_experiment() 패턴을 차용하여 다음과 같이 직접 주입:
#
#     ai = AIStrategy(params=XGB_PARAMS_OPTIMIZED)  # 최적 파라미터로 직접 생성
#     ai.train(train_features, feature_cols)         # 학습
#     strategy = HybridStrategy(...)                 # 빈 전략 생성
#     strategy.ai_strategy = ai                      # AI 전략 직접 주입
#     strategy.momentum_strategy = CustomStrategy()  # 모멘텀 전략 별도 준비
#     strategy.is_prepared = True                    # 준비 완료 플래그
#
# ============================================
# [포트폴리오 체이닝 (stitch_portfolios)]
# ============================================
#
#   각 폴드는 initial_capital에서 시작하므로, 이전 폴드의 최종 가치를 기준으로 스케일링:
#
#     Fold 1: [10000 → V1]           (배율 1.0)
#     Fold 2: [10000 → V2] × V1/C   = [V1 → V2*(V1/C)]
#     Fold 3: [10000 → V3] × V2/C   = [V2 → V3*(V2/C)]
#
# ============================================
# [GitHub Actions 연동]
# ============================================
#
#   워크플로우: .github/workflows/backtest_hybrid_new.yml
#   트리거: workflow_dispatch (수동 실행)
#   입력값: train_start, train_end, backtest_end, wf_step_months 등 → CLI 인자로 전달
#   출력물: backtest_result_hybrid_new.png → Artifacts 업로드 (30일 보관)
#
# ============================================
# [사용법]
# ============================================
#
#   # 기본값으로 실행 (2015~오늘, WF 6개월, M50%+AI50%)
#   python run_hybrid_new_backtest.py
#
#   # 커스텀 파라미터
#   python run_hybrid_new_backtest.py \
#     --train_start 2015-01-01 \
#     --train_end 2020-01-01 \
#     --backtest_end 2024-12-31 \
#     --wf_step_months 6 \
#     --momentum_weight 0.50 \
#     --ai_weight 0.50 \
#     --initial_capital 10000 \
#     --commission 0.001 \
#     --slippage 0.001 \
#     --output backtest_result_hybrid_new.png
#
# ============================================
# [의존 관계]
# ============================================
#
#   ← data.py             : get_backtest_data()     - S&P 500 + SPY OHLCV 데이터 다운로드
#                            get_sp500_list()        - S&P 500 종목 목록 (Wikipedia 스크래핑)
#   ← ai_data.py          : create_features()       - 기술적 지표 + 라벨 생성 (5D-3%)
#                            get_feature_columns()   - AI 학습용 피처 컬럼 목록
#                            TARGET_RETURN_OPTIMIZED - 최적 라벨 타겟 (0.03 = +3%)
#   ← hybrid_strategy.py  : HybridStrategy          - 모멘텀+AI 하이브리드 전략 클래스
#   ← ai_strategy.py      : AIStrategy              - XGBoost 이진 분류 래퍼
#                            XGB_PARAMS_OPTIMIZED    - 최적 XGBoost 하이퍼파라미터
#   ← ai_backtest.py      : run_ai_backtest()       - AI 전략 백테스트 엔진
#                            calculate_ai_metrics()  - 성과 메트릭 계산
#   ← strategy.py         : prepare_price_data()    - 원시 데이터 → 피벗 가격 데이터
#                            CustomStrategy          - 3-팩터 모멘텀 전략 클래스
#                            filter_tuesday()        - 화요일 필터 (주간 리밸런싱)
#   ← run_model_experiment.py : generate_walk_forward_folds() - WF 폴드 목록 생성
#                                stitch_portfolios()          - 폴드별 포트폴리오 체이닝
# ============================================

import sys
import os
import argparse
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(__file__))

from data import get_backtest_data, get_sp500_list
from ai_data import create_features, get_feature_columns, TARGET_RETURN_OPTIMIZED
from hybrid_strategy import HybridStrategy
from ai_strategy import AIStrategy, XGB_PARAMS_OPTIMIZED
from ai_backtest import run_ai_backtest, calculate_ai_metrics
from strategy import prepare_price_data, CustomStrategy, filter_tuesday
from run_model_experiment import generate_walk_forward_folds, stitch_portfolios


# ============================================
# [1] CLI 인자 파싱
# ============================================

def parse_args():
    """
    CLI 인자 파싱.

    GitHub Actions workflow_dispatch 입력값이 CLI 인자로 전달됨.
    모든 인자에 기본값이 설정되어 있어 인자 없이도 실행 가능.

    Returns:
        argparse.Namespace: 파싱된 인자 객체
            - train_start (str): AI 학습 시작일 (기본 '2015-01-01')
            - train_end (str): AI 학습 종료일 = 첫 번째 테스트 시작일 (기본 '2020-01-01')
            - backtest_start (str): 하위 호환용, train_end와 동일 (기본 '2020-01-01')
            - backtest_end (str): 백테스트 종료일, 빈값이면 오늘 (기본 '')
            - wf_step_months (int): Walk-Forward 테스트 윈도우 크기 (기본 6개월)
            - initial_capital (float): 초기 자본금 USD (기본 10000)
            - momentum_weight (float): 모멘텀 가중치 0.0~1.0 (기본 0.50)
            - ai_weight (float): AI 가중치 0.0~1.0 (기본 0.50)
            - commission (float): 수수료율 (기본 0.001 = 0.1%)
            - slippage (float): 슬리피지율 (기본 0.001 = 0.1%)
            - output (str): 그래프 저장 경로 (기본 'backtest_result_hybrid_new.png')
    """
    parser = argparse.ArgumentParser(description='하이브리드_New 전략 백테스트 (Walk-Forward)')
    parser.add_argument('--train_start', type=str, default='2015-01-01',
                        help='AI 학습 시작일 (YYYY-MM-DD)')
    parser.add_argument('--train_end', type=str, default='2020-01-01',
                        help='AI 학습 종료일 = 첫 번째 테스트 시작일 (YYYY-MM-DD)')
    parser.add_argument('--backtest_start', type=str, default='2020-01-01',
                        help='(하위 호환용, train_end와 동일하게 처리)')
    parser.add_argument('--backtest_end', type=str, default='',
                        help='백테스트 종료일 (YYYY-MM-DD, 빈값=오늘)')
    parser.add_argument('--wf_step_months', type=int, default=6,
                        help='Walk-Forward 테스트 윈도우 크기 (개월, 기본 6)')
    parser.add_argument('--initial_capital', type=float, default=10000,
                        help='초기 자본금 (USD)')
    parser.add_argument('--momentum_weight', type=float, default=0.50,
                        help='모멘텀 가중치 (기본 0.50)')
    parser.add_argument('--ai_weight', type=float, default=0.50,
                        help='AI 가중치 (기본 0.50)')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='수수료율 (기본 0.001 = 0.1%%)')
    parser.add_argument('--slippage', type=float, default=0.001,
                        help='슬리피지율 (기본 0.001 = 0.1%%)')
    parser.add_argument('--output', type=str, default='backtest_result_hybrid_new.png',
                        help='그래프 저장 경로')
    return parser.parse_args()


# ============================================
# [2] 결과 출력
# ============================================

def print_results(metrics, args, backtest_end, folds):
    """
    백테스트 결과를 텍스트로 출력.

    출력 섹션:
        - 기간 설정: 학습/백테스트 기간, WF 스텝, 폴드 수, 자본금, 가중치
        - Hybrid_New 파라미터: XGBoost 설정, 라벨 정의
        - 수익률: 총 수익률, SPY 수익률, Alpha, CAGR
        - 리스크: Sharpe, Calmar, MDD, Volatility
        - 거래: 총 거래 수, 승률, 손절 횟수, 수수료/슬리피지

    Args:
        metrics (dict): calculate_ai_metrics()의 반환값
            - total_return (float): 총 수익률 (소수, 예: 0.35 = +35%)
            - spy_return (float): SPY 수익률 (소수)
            - alpha (float): 전략 수익률 - SPY 수익률 (소수)
            - cagr (float): 연평균 복합 성장률 (소수)
            - sharpe_ratio (float): 샤프 비율 (무위험이자율 0 가정)
            - mdd (float): 최대 낙폭 (음수, 예: -0.25 = -25%)
            - volatility (float): 연간 변동성 (소수)
            - total_trades (int): 총 거래 횟수
            - win_rate (float): 승률 (소수, 예: 0.61 = 61%)
            - stop_loss_count (int): 손절 횟수
            - total_commission (float): 총 수수료 (USD)
            - total_slippage (float): 총 슬리피지 (USD)
        args (argparse.Namespace): CLI 인자
        backtest_end (str): 백테스트 종료일 (YYYY-MM-DD)
        folds (list[dict]): Walk-Forward 폴드 목록
    """
    calmar = abs(metrics['cagr'] / metrics['mdd']) if metrics['mdd'] != 0 else 0
    net_cost = metrics['total_commission'] + metrics.get('total_slippage', 0)

    print()
    print("=" * 55)
    print("   Hybrid_New Backtest Result (Walk-Forward)")
    print("=" * 55)
    print(f"[기간 설정]")
    print(f"학습 초기      : {args.train_start} ~ {args.train_end}")
    print(f"백테스트       : {args.train_end} ~ {backtest_end}")
    print(f"WF 스텝        : {args.wf_step_months}개월  /  폴드 수: {len(folds)}개")
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
    print(f"Weights        : Momentum {args.momentum_weight*100:.0f}% / AI {args.ai_weight*100:.0f}%")
    print(f"Commission     : {args.commission*100:.2f}%")
    print(f"Slippage       : {args.slippage*100:.2f}%")
    print()
    print(f"[Hybrid_New 파라미터]")
    print(f"XGBoost        : max_depth=3, subsample=0.5, colsample=0.5")
    print(f"라벨           : 5D-{TARGET_RETURN_OPTIMIZED*100:.0f}%")
    print()
    print("[수익률]")
    print(f"Total Return   : {metrics['total_return']*100:+.2f}%")
    print(f"SPY Return     : {metrics['spy_return']*100:+.2f}%")
    print(f"Alpha          : {metrics['alpha']*100:+.2f}%")
    print(f"CAGR           : {metrics['cagr']*100:.2f}%")
    print()
    print("[리스크]")
    print(f"Sharpe Ratio   : {metrics['sharpe_ratio']:.2f}")
    print(f"Calmar Ratio   : {calmar:.2f}")
    print(f"Max Drawdown   : {metrics['mdd']*100:.2f}%")
    print(f"Volatility     : {metrics['volatility']*100:.2f}%")
    print()
    print("[거래]")
    print(f"Total Trades   : {metrics['total_trades']}")
    print(f"Win Rate       : {metrics['win_rate']*100:.2f}%")
    print(f"Stop Loss      : {metrics['stop_loss_count']}")
    print(f"Total Commission: ${metrics['total_commission']:,.2f}")
    print(f"Total Slippage  : ${metrics.get('total_slippage', 0):,.2f}")
    print(f"Net Cost        : ${net_cost:,.2f}")
    print("=" * 55)


# ============================================
# [3] 그래프 생성
# ============================================

def plot_results(portfolio_df, trades_df, spy_raw, output_path, args, backtest_end, folds):
    """
    4-subplot 그래프 생성 및 PNG 저장.

    Subplot 구성:
        1. 누적 수익률 vs SPY + Walk-Forward 폴드 경계선 (수직 점선)
           - Hybrid_New 포트폴리오 (파란색 실선)
           - SPY 벤치마크 (주황색 점선)
           - 손절 시점 (빨간 원)
           - 폴드 경계 (회색 수직 점선)
        2. 월별 수익률 바 차트 (양수: 파란색, 음수: 빨간색)
        3. Drawdown 차트 (빨간색 영역)
        4. 폴드별 수익률 바 차트 (각 WF 폴드의 구간 수익률)

    Args:
        portfolio_df (pd.DataFrame): stitch_portfolios()의 결과
            - date (datetime): 거래일
            - value (float): 포트폴리오 가치 (USD)
        trades_df (pd.DataFrame): 전체 거래 내역
            - date (datetime): 거래일
            - action (str): 'BUY', 'SELL', 'STOP_LOSS'
        spy_raw (pd.DataFrame): 전체 원시 데이터 (SPY 포함)
            - date, symbol, close 등
        output_path (str): PNG 저장 경로
        args (argparse.Namespace): CLI 인자
        backtest_end (str): 백테스트 종료일 (YYYY-MM-DD)
        folds (list[dict]): Walk-Forward 폴드 목록
            - 각 폴드: {'train_start', 'train_end', 'test_start', 'test_end'}
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 20))
    title = (
        f"Hybrid_New Strategy Backtest (Walk-Forward)\n"
        f"Train Init: {args.train_start}~{args.train_end}  "
        f"| Backtest: {args.train_end}~{backtest_end}  "
        f"| WF Step: {args.wf_step_months}M  "
        f"| Momentum {args.momentum_weight*100:.0f}% / AI {args.ai_weight*100:.0f}%\n"
        f"XGB: depth=3, sample=0.5  |  Label: 5D-{TARGET_RETURN_OPTIMIZED*100:.0f}%"
    )
    fig.suptitle(title, fontsize=12, fontweight='bold')

    portfolio_df = portfolio_df.copy()
    portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
    portfolio_df['normalized'] = portfolio_df['value'] / portfolio_df['value'].iloc[0] * 100

    # 폴드 경계 날짜 (두 번째 폴드부터 test_start)
    fold_boundaries = [pd.Timestamp(f['test_start']) for f in folds[1:]]

    # ----- Subplot 1: 누적 수익률 + SPY + 폴드 경계 -----
    ax1 = axes[0]
    ax1.plot(portfolio_df['date'], portfolio_df['normalized'],
             label='Hybrid_New Portfolio', linewidth=2, color='steelblue')

    if 'SPY' in spy_raw['symbol'].unique():
        spy = spy_raw[spy_raw['symbol'] == 'SPY'].sort_values('date').copy()
        spy['date'] = pd.to_datetime(spy['date'])
        spy = spy[spy['date'] >= portfolio_df['date'].iloc[0]]
        if not spy.empty:
            spy['normalized'] = spy['close'] / spy['close'].iloc[0] * 100
            ax1.plot(spy['date'], spy['normalized'],
                     label='SPY', linewidth=2, linestyle='--', color='orange', alpha=0.8)

    for boundary in fold_boundaries:
        ax1.axvline(x=boundary, color='gray', linewidth=0.8, linestyle=':', alpha=0.7)

    if not trades_df.empty:
        trades_plot = trades_df.copy()
        trades_plot['date'] = pd.to_datetime(trades_plot['date'])
        port_idx = portfolio_df.set_index('date')['normalized']

        stop_trades = trades_plot[trades_plot['action'] == 'STOP_LOSS']
        if not stop_trades.empty:
            stop_values = port_idx.reindex(stop_trades['date'].values).values
            ax1.scatter(stop_trades['date'].values, stop_values,
                        color='crimson', marker='o', s=50, alpha=0.9,
                        label='Stop Loss', zorder=6)

    ax1.set_title(f'Cumulative Return vs SPY | WF 폴드 경계 ({len(folds)}개)', fontsize=12)
    ax1.set_ylabel('Value (base=100)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # ----- Subplot 2: 월별 수익률 바 차트 -----
    ax2 = axes[1]
    portfolio_df['month'] = portfolio_df['date'].dt.to_period('M')
    monthly = portfolio_df.groupby('month')['value'].last()
    monthly_returns = monthly.pct_change().dropna()

    colors = ['steelblue' if r >= 0 else 'crimson' for r in monthly_returns]
    x = range(len(monthly_returns))
    ax2.bar(x, monthly_returns * 100, color=colors, alpha=0.8)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_title('Monthly Returns (%)', fontsize=12)
    ax2.set_ylabel('Return (%)')
    ax2.grid(True, alpha=0.3, axis='y')

    step = max(1, len(monthly_returns) // 12)
    ax2.set_xticks(list(x)[::step])
    ax2.set_xticklabels([str(m) for m in monthly_returns.index[::step]],
                        rotation=45, ha='right', fontsize=8)

    # ----- Subplot 3: Drawdown -----
    ax3 = axes[2]
    peak = portfolio_df['value'].cummax()
    drawdown = (portfolio_df['value'] - peak) / peak * 100
    ax3.fill_between(portfolio_df['date'], drawdown, 0, color='crimson', alpha=0.4)
    ax3.plot(portfolio_df['date'], drawdown, color='crimson', linewidth=1)
    ax3.set_title('Drawdown (%)', fontsize=12)
    ax3.set_ylabel('Drawdown (%)')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # ----- Subplot 4: 폴드별 수익률 -----
    ax4 = axes[3]
    fold_returns = []
    fold_labels = []
    for i, f in enumerate(folds):
        fold_labels.append(f"F{i+1}\n{f['test_start'][:7]}")
    for i, f in enumerate(folds):
        ts = pd.Timestamp(f['test_start'])
        te = pd.Timestamp(f['test_end'])
        fold_port = portfolio_df[
            (portfolio_df['date'] >= ts) & (portfolio_df['date'] <= te)
        ]
        if len(fold_port) >= 2:
            ret = (fold_port['value'].iloc[-1] / fold_port['value'].iloc[0] - 1) * 100
            fold_returns.append(ret)
        else:
            fold_returns.append(0)

    bar_colors = ['steelblue' if r >= 0 else 'crimson' for r in fold_returns]
    bars = ax4.bar(fold_labels, fold_returns, color=bar_colors, alpha=0.8)
    ax4.axhline(y=0, color='black', linewidth=0.8)
    for bar, v in zip(bars, fold_returns):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{v:+.1f}%', ha='center',
                 va='bottom' if v >= 0 else 'top', fontsize=8)
    ax4.set_title('Return by Walk-Forward Fold (%)', fontsize=12)
    ax4.set_ylabel('Return (%)')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nGraph saved: {output_path}")


# ============================================
# [4] 메인 실행
# ============================================

def main():
    """
    하이브리드_New 전략 Walk-Forward 백테스트 메인 함수.

    실행 흐름:
        [1] CLI 인자 파싱 및 WF 폴드 생성
        [2] S&P 500 종목 목록 로딩
        [3] 전체 기간 데이터 다운로드 (train_start ~ backtest_end)
        [4] 피처 생성 (5D-3% 라벨, TARGET_RETURN_OPTIMIZED=0.03)
        [5] 각 폴드별 반복:
            - 학습/테스트 데이터 슬라이싱
            - XGB_PARAMS_OPTIMIZED로 AIStrategy 직접 생성·학습·주입
            - 모멘텀 전략 준비 (7개월 lookback)
            - run_ai_backtest()로 백테스트 실행
        [6] stitch_portfolios()로 폴드별 결과 체이닝
        [7] 전체 기간 메트릭 계산 및 출력
        [8] 4-subplot 그래프 PNG 저장
    """
    args = parse_args()

    backtest_end = args.backtest_end if args.backtest_end else datetime.now().strftime('%Y-%m-%d')

    # 가중치 합 검증 (momentum_weight + ai_weight = 1.0 이어야 함)
    total_weight = args.momentum_weight + args.ai_weight
    if abs(total_weight - 1.0) > 0.01:
        print(f"경고: momentum_weight({args.momentum_weight}) + "
              f"ai_weight({args.ai_weight}) = {total_weight:.2f}")

    # ----- Walk-Forward 폴드 생성 -----
    # generate_walk_forward_folds():
    #   학습 윈도우 크기(5년)를 고정하고, 매 스텝마다 6개월씩 슬라이딩
    #   반환: [{'train_start', 'train_end', 'test_start', 'test_end'}, ...]
    folds = generate_walk_forward_folds(
        args.train_start, args.train_end, backtest_end, args.wf_step_months
    )

    print("=" * 60)
    print("[하이브리드_New 백테스트 설정 (Walk-Forward)]")
    print(f"학습 초기   : {args.train_start} ~ {args.train_end}")
    print(f"백테스트    : {args.train_end} ~ {backtest_end}")
    print(f"WF 스텝     : {args.wf_step_months}개월  /  폴드 수: {len(folds)}개")
    print(f"가중치      : Momentum {args.momentum_weight*100:.0f}% / AI {args.ai_weight*100:.0f}%")
    print(f"XGBoost     : max_depth=3, subsample=0.5, colsample=0.5")
    print(f"라벨        : 5D-{TARGET_RETURN_OPTIMIZED*100:.0f}%")
    print(f"수수료/슬립 : {args.commission*100:.2f}% / {args.slippage*100:.2f}%")
    print("=" * 60)

    if not folds:
        print("생성된 폴드가 없습니다. train_end와 backtest_end를 확인하세요.")
        return

    for i, f in enumerate(folds):
        print(f"  Fold {i+1:2d}: Train {f['train_start']}~{f['train_end']}  "
              f"| Test {f['test_start']}~{f['test_end']}")
    print()

    # ----- [1] 종목 목록 -----
    # Wikipedia에서 S&P 500 종목 스크래핑, 실패 시 config.py 백업 리스트 사용
    # SPY는 벤치마크 비교용으로 항상 포함
    print("[1] S&P 500 종목 목록 로딩...")
    try:
        sp500 = get_sp500_list()
        symbols = sp500['symbol'].tolist() + ['SPY']
    except Exception:
        from config import SP500_BACKUP
        symbols = SP500_BACKUP + ['SPY']
    print(f"종목 수: {len(symbols)}")

    # ----- [2] 전체 기간 데이터 한 번에 다운로드 -----
    # 폴드별로 분할 다운로드하지 않고, 전체 기간을 한 번에 다운로드하여 효율성 확보
    # all_raw 컬럼: date, symbol, open, high, low, close, volume
    print(f"\n[2] 전체 데이터 다운로드: {args.train_start} ~ {backtest_end}")
    all_raw = get_backtest_data(symbols, start_date=args.train_start, end_date=backtest_end)
    all_raw['date'] = pd.to_datetime(all_raw['date'])
    print(f"전체 데이터 rows: {len(all_raw):,}")

    # ----- [3] 전체 피처 한 번에 생성 (3% 라벨) -----
    # ※ 하이브리드_New 핵심 차이: target_return=0.03 (기존 하이브리드는 0.05)
    # create_features()가 생성하는 피처:
    #   - 수익률: ret_5d, ret_10d, ret_20d, ret_60d, ret_120d
    #   - 이동평균 비율: ma_ratio_5d, ma_ratio_20d, ma_ratio_60d, ma_ratio_120d
    #   - 변동성: vol_5d, vol_20d, vol_60d
    #   - RSI(14), MACD(12/26/9), 볼린저밴드(%B), 거래량 비율
    #   - 라벨: label = 1 if 5거래일 후 수익률 >= 0.03 else 0
    print("\n[3] 피처 생성 중 (5D-3% 라벨)...")
    all_features = create_features(all_raw, target_return=TARGET_RETURN_OPTIMIZED)
    all_features['date'] = pd.to_datetime(all_features['date'])
    feature_cols = get_feature_columns(all_features)
    print(f"피처 수: {len(feature_cols)}  /  전체 샘플: {len(all_features):,}")

    # ----- [4] Walk-Forward 폴드별 실행 -----
    # 각 폴드에서 수행하는 작업:
    #   1) 학습/테스트 피처 데이터 날짜 범위로 슬라이싱
    #   2) 모멘텀 계산용 가격 데이터 준비 (테스트 시작일 - 7개월 lookback)
    #   3) XGB_PARAMS_OPTIMIZED로 AIStrategy 생성·학습·주입
    #   4) CustomStrategy로 모멘텀 전략 준비 (화요일 필터)
    #   5) run_ai_backtest()로 해당 폴드 구간 백테스트 실행
    #
    # ※ 7개월 lookback 이유:
    #   모멘텀 스코어 = ret_1m × 3.5 + ret_3m × 2.5 + ret_6m × 1.5
    #   6개월 수익률 계산에 최소 6개월 과거 데이터 필요 + 1개월 여유 = 7개월
    fold_portfolios = []
    fold_trades = []

    for fold_idx, fold in enumerate(folds):
        fold_train_start = pd.Timestamp(fold['train_start'])
        fold_train_end = pd.Timestamp(fold['train_end'])
        fold_test_end = pd.Timestamp(fold['test_end'])

        print(f"\n{'='*60}")
        print(f"[Fold {fold_idx+1}/{len(folds)}]  "
              f"Train: {fold['train_start']}~{fold['train_end']}  "
              f"| Test: {fold['test_start']}~{fold['test_end']}")
        print(f"{'='*60}")

        # 폴드별 데이터 슬라이싱 (전체 피처에서 날짜 범위로 추출)
        # 학습: [fold_train_start, fold_train_end] (양 끝 포함)
        # 테스트: (fold_train_end, fold_test_end] (학습 종료일 미포함, 테스트 종료일 포함)
        train_features_fold = all_features[
            (all_features['date'] >= fold_train_start) &
            (all_features['date'] <= fold_train_end)
        ]
        test_features_fold = all_features[
            (all_features['date'] > fold_train_end) &
            (all_features['date'] <= fold_test_end)
        ]

        if len(test_features_fold) == 0:
            print(f"  경고: 테스트 데이터 없음, Fold {fold_idx+1} 스킵")
            continue

        print(f"학습 샘플: {len(train_features_fold):,}  /  테스트 샘플: {len(test_features_fold):,}")

        # 모멘텀 점수 계산에 과거 7개월 데이터 필요
        # (6개월 수익률 계산 + 1개월 여유분)
        momentum_lookback = fold_train_end - pd.DateOffset(months=7)
        price_raw_fold = all_raw[
            (all_raw['date'] >= momentum_lookback) &
            (all_raw['date'] <= fold_test_end)
        ]
        # prepare_price_data(): 원시 long-format → 피벗 wide-format (index=date, columns=symbol, values=close)
        price_df_fold = prepare_price_data(price_raw_fold)

        # ────────────────────────────────────────────
        # HybridStrategy 학습 및 백테스트
        # ※ 핵심: XGB_PARAMS_OPTIMIZED를 사용하기 위해 AI 전략을 직접 주입
        #   - HybridStrategy.prepare()는 내부에서 기본 XGB_PARAMS를 사용
        #   - 따라서 prepare()를 우회하고 AI를 직접 생성·학습·주입
        # ────────────────────────────────────────────
        try:
            # [Step 1] AIStrategy를 최적 파라미터로 직접 생성 및 학습
            # XGB_PARAMS_OPTIMIZED = {max_depth: 3, subsample: 0.5, colsample: 0.5, ...}
            ai = AIStrategy(params=XGB_PARAMS_OPTIMIZED)
            ai.train(train_features_fold, feature_cols)

            # [Step 2] HybridStrategy 초기화 (prepare() 호출하지 않음)
            strategy = HybridStrategy(
                weight_momentum=args.momentum_weight,
                weight_ai=args.ai_weight,
            )
            # AI 전략 직접 주입 (prepare() 우회)
            strategy.feature_cols = feature_cols
            strategy.ai_strategy = ai

            # [Step 3] 모멘텀 전략 준비
            # CustomStrategy.prepare()가 반환하는 값:
            #   - score_df: 날짜×종목 모멘텀 점수 (3-팩터 가중합)
            #   - correlation_df: 날짜×종목 SPY 상관관계 (60일 롤링)
            #   - ret_1m: 날짜×종목 1개월 수익률 (마켓 필터용)
            strategy.momentum_strategy = CustomStrategy()
            tuesday_df = filter_tuesday(price_df_fold)
            strategy.score_df, strategy.correlation_df, strategy.ret_1m = \
                strategy.momentum_strategy.prepare(price_df_fold, tuesday_df)
            strategy.is_prepared = True

            # [Step 4] 백테스트 실행
            # run_ai_backtest()가 반환하는 값:
            #   - portfolio: DataFrame (date, value) - 일별 포트폴리오 가치
            #   - trades: DataFrame (date, symbol, action, price, ...) - 거래 내역
            #   - metrics: dict - 성과 메트릭 (total_return, cagr, sharpe, mdd 등)
            result = run_ai_backtest(
                strategy=strategy,
                test_df=test_features_fold,
                feature_cols=feature_cols,
                initial_capital=args.initial_capital,
                commission=args.commission,
                slippage=args.slippage,
            )

            fold_portfolios.append(result['portfolio'])
            fold_trades.append(result['trades'])

            m = result['metrics']
            print(f"  → 수익률: {m['total_return']*100:+.1f}%  "
                  f"CAGR: {m['cagr']*100:.1f}%  "
                  f"Sharpe: {m['sharpe_ratio']:.2f}  "
                  f"MDD: {m['mdd']*100:.1f}%")
        except Exception as e:
            print(f"  ✗ Fold {fold_idx+1} 실패: {e}")

    if not fold_portfolios:
        print("유효한 폴드 결과가 없습니다. 종료합니다.")
        return

    # ----- [5] 폴드 결과 집계 -----
    # stitch_portfolios(): 각 폴드의 포트폴리오를 하나의 연속 시계열로 연결
    # 각 폴드는 initial_capital에서 독립 시작하므로, 이전 폴드 최종 가치로 스케일링
    #   Fold 1: [10000 → V1] (배율 1.0)
    #   Fold 2: [10000 → V2] × (V1 / 10000)
    #   Fold 3: [10000 → V3] × (V2 / 10000) ...
    print(f"\n\n[집계] Walk-Forward 결과 통합 중...")

    stitched_portfolio = stitch_portfolios(fold_portfolios, args.initial_capital)

    # 거래 내역 합산 (모든 폴드의 거래를 하나의 DataFrame으로)
    trade_dfs = [t for t in fold_trades if not t.empty]
    all_trades_df = pd.concat(trade_dfs, ignore_index=True) if trade_dfs else pd.DataFrame()

    # SPY 데이터 (벤치마크 비교용, 백테스트 기간만 추출)
    spy_raw = all_raw[
        (all_raw['symbol'] == 'SPY') &
        (all_raw['date'] > pd.Timestamp(args.train_end))
    ][['date', 'close']].rename(columns={'close': 'spy_close'})

    # 전체 기간 메트릭 계산
    # calculate_ai_metrics() 반환값: total_return, cagr, sharpe_ratio, mdd,
    #   volatility, alpha, spy_return, total_trades, win_rate, stop_loss_count,
    #   total_commission, total_slippage 등
    metrics = calculate_ai_metrics(
        stitched_portfolio, all_trades_df, spy_raw,
        args.initial_capital, args.slippage
    )

    # ----- [6] 결과 출력 -----
    print_results(metrics, args, backtest_end, folds)

    # ----- [7] 그래프 저장 -----
    # 4-subplot PNG: 누적수익률, 월별수익률, Drawdown, 폴드별수익률
    plot_results(stitched_portfolio, all_trades_df, all_raw, args.output, args, backtest_end, folds)


if __name__ == '__main__':
    main()
