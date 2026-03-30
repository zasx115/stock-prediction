#!/usr/bin/env python3
# ============================================
# 파일명: src/run_ma_filter_experiment.py
# 설명: 이동평균 추세 필터 실험 (기존 파일 수정 없음)
#
# 목적:
#   기존 3가지 전략(모멘텀, 하이브리드, 하이브리드New)에
#   추가 필터를 적용한 변형을 비교 백테스트.
#   결과가 좋으면 기존 파일을 수정하는 방향으로 활용.
#
# 추가 필터 (기존 시장모멘텀 + SPY상관관계 필터 이후 적용):
#   1. 시장 MA 필터: SPY 또는 QQQ 종가 > 200일 이동평균 → 매수
#   2. 정배열 필터: 종목의 MA20 > MA60 > MA200 → 해당 종목 포함
#
# 실험 구성 (6가지 + SPY 벤치마크):
#   ① 모멘텀 (기존)
#   ② 모멘텀 + MA 필터
#   ③ 하이브리드 (기존)
#   ④ 하이브리드 + MA 필터
#   ⑤ 하이브리드New (기존)
#   ⑥ 하이브리드New + MA 필터
#
# 사용법:
#   python run_ma_filter_experiment.py
#   python run_ma_filter_experiment.py \
#     --train_start 2015-01-01 \
#     --train_end 2020-01-01 \
#     --backtest_end 2024-12-31 \
#     --wf_step_months 6 \
#     --initial_capital 10000 \
#     --output ma_filter_result.png
#
# 의존 관계 (기존 파일을 읽기만 함):
#   ← data.py         (get_backtest_data, download_stock_data, get_sp500_list)
#   ← ai_data.py      (create_features, get_feature_columns, TARGET_RETURN_OPTIMIZED)
#   ← strategy.py     (CustomStrategy, prepare_price_data, filter_tuesday)
#   ← hybrid_strategy.py (HybridStrategy)
#   ← ai_strategy.py  (XGB_PARAMS, XGB_PARAMS_OPTIMIZED)
#   ← ai_backtest.py  (run_ai_backtest, calculate_ai_metrics)
#   ← run_model_experiment.py (generate_walk_forward_folds, stitch_portfolios)
# ============================================

import sys
import os
import argparse
from datetime import datetime
from copy import deepcopy

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(__file__))

# 기존 모듈 임포트 (수정 없이 그대로 사용)
from data import get_backtest_data, download_stock_data, get_sp500_list
from ai_data import create_features, get_feature_columns, TARGET_RETURN_OPTIMIZED
from strategy import CustomStrategy, prepare_price_data, filter_tuesday
from hybrid_strategy import HybridStrategy
from ai_strategy import XGB_PARAMS, XGB_PARAMS_OPTIMIZED
from ai_backtest import run_ai_backtest, calculate_ai_metrics
from run_model_experiment import generate_walk_forward_folds, stitch_portfolios
try:
    from config import SP500_BACKUP
except ImportError:
    SP500_BACKUP = []


# ============================================
# [1] 파라미터
# ============================================

def parse_args():
    p = argparse.ArgumentParser(description='MA 추세 필터 실험')
    p.add_argument('--train_start',     default='2015-01-01')
    p.add_argument('--train_end',       default='2020-01-01')
    p.add_argument('--backtest_end',    default='')
    p.add_argument('--wf_step_months',  type=int,   default=6)
    p.add_argument('--initial_capital', type=float, default=10000)
    p.add_argument('--commission',      type=float, default=0.001)
    p.add_argument('--slippage',        type=float, default=0.001)
    p.add_argument('--output',          default='ma_filter_result.png')
    return p.parse_args()


# ============================================
# [2] MA 필터 헬퍼 함수
# ============================================

def build_ma_tables(price_df):
    """
    전 종목에 대한 MA20 / MA60 / MA200 테이블 계산.
    price_df: 피벗된 일별 종가 (날짜 × 종목)
    반환: {'ma20': df, 'ma60': df, 'ma200': df}
    """
    return {
        'ma20':  price_df.rolling(20,  min_periods=20).mean(),
        'ma60':  price_df.rolling(60,  min_periods=60).mean(),
        'ma200': price_df.rolling(200, min_periods=200).mean(),
    }


def check_market_ma_filter(date_ts, market_price_df, market_ma200_df):
    """
    필터 1: SPY 또는 QQQ 종가 > 200일 MA  →  True(매수 가능)

    둘 중 하나라도 200일선 위에 있으면 통과.
    데이터가 없으면 통과(fallback)로 처리.
    """
    for ticker in ['SPY', 'QQQ']:
        if (market_price_df is None
                or market_ma200_df is None
                or ticker not in market_price_df.columns
                or ticker not in market_ma200_df.columns):
            return True  # 데이터 없으면 fallback: 통과
        if date_ts not in market_price_df.index or date_ts not in market_ma200_df.index:
            return True
        price = market_price_df.loc[date_ts, ticker]
        ma    = market_ma200_df.loc[date_ts, ticker]
        if pd.isna(price) or pd.isna(ma):
            continue
        if price > ma:
            return True
    return False


def check_jungbaeyeol(symbol, date_ts, ma_tables):
    """
    필터 2: 종목의 MA20 > MA60 > MA200 (정배열)  →  True(포함)

    데이터 부족 시 통과(fallback)로 처리.
    """
    if ma_tables is None:
        return True
    try:
        ma20  = ma_tables['ma20'].loc[date_ts, symbol]
        ma60  = ma_tables['ma60'].loc[date_ts, symbol]
        ma200 = ma_tables['ma200'].loc[date_ts, symbol]
        if any(pd.isna(v) for v in [ma20, ma60, ma200]):
            return True  # 데이터 부족 → fallback
        return float(ma20) > float(ma60) > float(ma200)
    except (KeyError, TypeError):
        return True


def apply_extra_filters(result, date_ts,
                        market_price_df, market_ma200_df, ma_tables):
    """
    기존 전략이 반환한 result에 추가 필터(필터1 + 필터2)를 순서대로 적용.

    result: {'picks': [...], 'scores': [...], 'allocations': [...]} 또는 None
    반환  : 동일 형식 또는 None
    """
    if result is None:
        return None

    # ── 필터 1: 시장 MA (전역 필터) ─────────────────────────────
    if not check_market_ma_filter(date_ts, market_price_df, market_ma200_df):
        return None   # 시장 전체가 200일선 아래 → 매수 중단

    # ── 필터 2: 정배열 (종목별 필터) ────────────────────────────
    filtered_picks  = []
    filtered_scores = []
    for sym, sc in zip(result['picks'], result['scores']):
        if check_jungbaeyeol(sym, date_ts, ma_tables):
            filtered_picks.append(sym)
            filtered_scores.append(sc)

    if not filtered_picks:
        return None

    n = len(filtered_picks)
    if n >= 3:
        allocs = [0.4, 0.3, 0.3]
    elif n == 2:
        allocs = [0.5, 0.5]
    else:
        allocs = [1.0]

    return {
        'picks':       filtered_picks,
        'scores':      filtered_scores,
        'allocations': allocs[:n],
    }


# ============================================
# [3] MomentumAdapter
#     CustomStrategy를 run_ai_backtest() 인터페이스로 래핑
# ============================================

class MomentumAdapter:
    """
    CustomStrategy.select_stocks(score_df, corr_df, date, ret_1m) 인터페이스를
    run_ai_backtest()가 요구하는 select_stocks(df, feature_cols, date) 인터페이스로 변환.

    use_extra_filters=True 이면 추가 MA 필터 적용.
    """

    def __init__(self, use_extra_filters=False):
        self.use_extra_filters = use_extra_filters
        self._strategy         = CustomStrategy()
        self._score_df         = None
        self._corr_df          = None
        self._ret_1m           = None
        # MA 필터용 캐시
        self._ma_tables        = None
        self._market_price_df  = None
        self._market_ma200_df  = None

    def prepare(self, all_raw, market_raw=None):
        """
        all_raw   : get_backtest_data() 결과 (long format)
        market_raw: SPY + QQQ long format (MA 계산 기간 포함)
                    None 이면 시장 MA 필터 비활성(fallback=통과)
        """
        price_df   = prepare_price_data(all_raw)
        tuesday_df = filter_tuesday(price_df)
        self._score_df, self._corr_df, self._ret_1m = \
            self._strategy.prepare(price_df, tuesday_df)

        if self.use_extra_filters:
            # 정배열 필터용 MA 계산 (전 종목)
            self._ma_tables = build_ma_tables(price_df)

            # 시장 MA 필터용 (SPY + QQQ)
            if market_raw is not None and len(market_raw) > 0:
                mkt_pivot            = prepare_price_data(market_raw)
                self._market_price_df = mkt_pivot
                self._market_ma200_df = mkt_pivot.rolling(200, min_periods=200).mean()

    def select_stocks(self, df, feature_cols=None, date=None):
        """run_ai_backtest() 에서 호출되는 메서드"""
        result = self._strategy.select_stocks(
            self._score_df, self._corr_df, date, self._ret_1m
        )
        if not self.use_extra_filters:
            return result
        return apply_extra_filters(
            result, pd.Timestamp(date),
            self._market_price_df, self._market_ma200_df,
            self._ma_tables,
        )


# ============================================
# [4] HybridWithMAFilter
#     HybridStrategy에 추가 필터를 씌우는 래퍼
# ============================================

class HybridWithMAFilter:
    """
    HybridStrategy를 그대로 사용하고, select_stocks() 결과에만 추가 필터 적용.
    use_extra_filters=False 이면 기존 HybridStrategy와 동일하게 동작.
    """

    def __init__(self, hybrid_strategy, use_extra_filters=False):
        self._strategy        = hybrid_strategy
        self.use_extra_filters = use_extra_filters
        self._ma_tables        = None
        self._market_price_df  = None
        self._market_ma200_df  = None

    def set_ma_data(self, price_df, market_raw=None):
        """
        Walk-Forward 폴드별로 호출.
        price_df  : 해당 폴드의 일별 가격 피벗
        market_raw: SPY + QQQ long format
        """
        if not self.use_extra_filters:
            return
        self._ma_tables = build_ma_tables(price_df)
        if market_raw is not None and len(market_raw) > 0:
            mkt_pivot             = prepare_price_data(market_raw)
            self._market_price_df  = mkt_pivot
            self._market_ma200_df  = mkt_pivot.rolling(200, min_periods=200).mean()

    def select_stocks(self, df, feature_cols=None, date=None):
        result = self._strategy.select_stocks(df, feature_cols, date)
        if not self.use_extra_filters:
            return result
        return apply_extra_filters(
            result, pd.Timestamp(date),
            self._market_price_df, self._market_ma200_df,
            self._ma_tables,
        )


# ============================================
# [5] 모멘텀 단일 기간 백테스트 실행
# ============================================

def run_momentum_experiment(all_raw, market_raw, backtest_start,
                             use_extra_filters, initial_capital,
                             commission, slippage, label):
    """
    모멘텀 전략 단일 기간 백테스트.
    Walk-Forward 없이 전 기간을 한 번에 시뮬레이션.
    """
    print(f"\n{'='*60}")
    print(f"[모멘텀] {label}")
    print(f"{'='*60}")

    adapter = MomentumAdapter(use_extra_filters=use_extra_filters)
    adapter.prepare(all_raw, market_raw if use_extra_filters else None)

    # 실제 백테스트 구간만 test_df로 전달
    test_raw = all_raw[all_raw['date'] >= pd.Timestamp(backtest_start)].copy()

    if len(test_raw) == 0:
        print("  경고: 테스트 데이터 없음")
        return None

    result = run_ai_backtest(
        strategy=adapter,
        test_df=test_raw,
        feature_cols=[],
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage,
    )

    # SPY 수익률 패치 (run_ai_backtest 내부에서 test_df에 SPY가 있으면 자동 계산됨)
    spy_raw = all_raw[
        (all_raw['symbol'] == 'SPY') &
        (all_raw['date'] >= pd.Timestamp(backtest_start))
    ][['date', 'close']].rename(columns={'close': 'spy_close'})

    metrics = calculate_ai_metrics(
        result['portfolio'], result['trades'], spy_raw,
        initial_capital, slippage,
    )
    return {
        'label':     label,
        'portfolio': result['portfolio'],
        'trades':    result['trades'],
        'metrics':   metrics,
    }


# ============================================
# [6] 하이브리드 Walk-Forward 백테스트 실행
# ============================================

def run_hybrid_experiment(all_raw, all_features, feature_cols, folds,
                          market_raw, use_extra_filters,
                          weight_momentum, weight_ai, xgb_params,
                          target_return, initial_capital,
                          commission, slippage, label):
    """
    하이브리드(또는 하이브리드New) Walk-Forward 백테스트.
    각 폴드에서 HybridStrategy를 재학습하고,
    use_extra_filters=True 면 HybridWithMAFilter로 래핑.
    """
    print(f"\n{'='*60}")
    print(f"[하이브리드] {label}  |  M:{weight_momentum*100:.0f}% A:{weight_ai*100:.0f}%")
    print(f"{'='*60}")

    fold_portfolios = []
    fold_trades     = []

    for fold_idx, fold in enumerate(folds):
        fold_train_start = pd.Timestamp(fold['train_start'])
        fold_train_end   = pd.Timestamp(fold['train_end'])
        fold_test_end    = pd.Timestamp(fold['test_end'])

        print(f"  Fold {fold_idx+1}/{len(folds)} "
              f"Train:{fold['train_start']}~{fold['train_end']} "
              f"Test:{fold['test_start']}~{fold['test_end']}")

        # ── 피처 슬라이싱 ──────────────────────────────────────
        train_feat = all_features[
            (all_features['date'] >= fold_train_start) &
            (all_features['date'] <= fold_train_end)
        ]
        test_feat = all_features[
            (all_features['date'] > fold_train_end) &
            (all_features['date'] <= fold_test_end)
        ]

        if len(test_feat) == 0:
            print(f"    경고: 테스트 데이터 없음, 스킵")
            continue

        # ── 모멘텀용 가격 피벗 (7개월 lookback) ────────────────
        momentum_lookback = fold_train_end - pd.DateOffset(months=7)
        price_raw_fold    = all_raw[
            (all_raw['date'] >= momentum_lookback) &
            (all_raw['date'] <= fold_test_end)
        ]
        price_df_fold = prepare_price_data(price_raw_fold)

        # ── MA 필터용 가격 피벗 (12개월 lookback, MA200 충분히 확보) ──
        ma_lookback   = fold_train_end - pd.DateOffset(months=12)
        ma_price_raw  = all_raw[
            (all_raw['date'] >= ma_lookback) &
            (all_raw['date'] <= fold_test_end)
        ]
        ma_price_df   = prepare_price_data(ma_price_raw)

        try:
            # 기존 HybridStrategy 생성 및 학습
            hybrid = HybridStrategy(
                weight_momentum=weight_momentum,
                weight_ai=weight_ai,
            )
            # target_return 이 기본값(0.05)과 다를 경우
            # create_features() 시 이미 해당 target_return으로 피처가 생성되어 있으므로
            # HybridStrategy 내부 AIStrategy 학습에서 label 컬럼을 그대로 사용
            hybrid.prepare(train_feat, price_df_fold, feature_cols)

            # 필터 래퍼 적용 여부
            if use_extra_filters:
                strategy = HybridWithMAFilter(hybrid, use_extra_filters=True)
                strategy.set_ma_data(
                    ma_price_df,
                    market_raw if market_raw is not None else None,
                )
            else:
                strategy = hybrid

            result = run_ai_backtest(
                strategy=strategy,
                test_df=test_feat,
                feature_cols=feature_cols,
                initial_capital=initial_capital,
                commission=commission,
                slippage=slippage,
            )
            fold_portfolios.append(result['portfolio'])
            fold_trades.append(result['trades'])

            m = result['metrics']
            print(f"    → 수익률:{m['total_return']*100:+.1f}%  "
                  f"Sharpe:{m['sharpe_ratio']:.2f}  "
                  f"MDD:{m['mdd']*100:.1f}%")

        except Exception as e:
            print(f"    ✗ Fold {fold_idx+1} 실패: {e}")
            import traceback; traceback.print_exc()

    if not fold_portfolios:
        print("  유효한 폴드 없음")
        return None

    stitched   = stitch_portfolios(fold_portfolios, initial_capital)
    trade_dfs  = [t for t in fold_trades if not t.empty]
    all_trades = pd.concat(trade_dfs, ignore_index=True) if trade_dfs else pd.DataFrame()

    # SPY 수익률 (원시 데이터에서 추출)
    test_start = folds[0]['test_start']
    spy_raw = all_raw[
        (all_raw['symbol'] == 'SPY') &
        (all_raw['date'] >= pd.Timestamp(test_start))
    ][['date', 'close']].rename(columns={'close': 'spy_close'})

    metrics = calculate_ai_metrics(
        stitched, all_trades, spy_raw,
        initial_capital, slippage,
    )
    return {
        'label':     label,
        'portfolio': stitched,
        'trades':    all_trades,
        'metrics':   metrics,
    }


# ============================================
# [7] 결과 출력
# ============================================

def print_results_table(results, spy_total_return):
    """6가지 전략 비교 테이블 출력"""
    print("\n")
    print("=" * 80)
    print("   MA 추세 필터 실험 결과 비교")
    print("=" * 80)
    header = f"{'전략':<22} {'수익률':>8} {'CAGR':>7} {'Sharpe':>7} {'MDD':>8} {'알파':>8} {'승률':>7}"
    print(header)
    print("-" * 80)
    for r in results:
        if r is None:
            continue
        m = r['metrics']
        print(f"{r['label']:<22} "
              f"{m['total_return']*100:>+7.1f}% "
              f"{m['cagr']*100:>6.1f}% "
              f"{m['sharpe_ratio']:>7.2f} "
              f"{m['mdd']*100:>7.1f}% "
              f"{m['alpha']*100:>+7.1f}% "
              f"{m['win_rate']*100:>6.1f}%")
    print("-" * 80)
    print(f"{'SPY 벤치마크':<22} {spy_total_return*100:>+7.1f}%")
    print("=" * 80)


# ============================================
# [8] 차트 생성
# ============================================

def plot_comparison(results, spy_series, output_path):
    """
    3행 × 2열 서브플롯:
      각 전략 쌍(기존 vs MA필터)의 누적수익률 + SPY 비교
    """
    pairs = [
        ('모멘텀 (기존)',       '모멘텀 + MA필터'),
        ('하이브리드 (기존)',   '하이브리드 + MA필터'),
        ('하이브리드New (기존)','하이브리드New + MA필터'),
    ]

    result_map = {r['label']: r for r in results if r is not None}

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('MA 추세 필터 실험 — 전략별 누적 수익률 비교\n'
                 '(좌: 기존 전략  /  우: MA 필터 추가)',
                 fontsize=14, fontweight='bold')

    colors = {
        'original': 'steelblue',
        'filtered': 'darkorange',
        'spy':      'gray',
    }

    for row_idx, (orig_label, filt_label) in enumerate(pairs):
        for col_idx, label in enumerate([orig_label, filt_label]):
            ax = axes[row_idx][col_idx]

            if label in result_map:
                port = result_map[label]['portfolio'].copy()
                port['date'] = pd.to_datetime(port['date'])
                port = port.sort_values('date')
                norm = port['value'] / port['value'].iloc[0] * 100

                color = colors['filtered'] if col_idx == 1 else colors['original']
                ax.plot(port['date'], norm, label=label,
                        linewidth=2, color=color)

                # SPY 오버레이
                if spy_series is not None and len(spy_series) > 0:
                    spy_aligned = spy_series[spy_series.index >= port['date'].iloc[0]]
                    if len(spy_aligned) > 0:
                        spy_norm = spy_aligned / spy_aligned.iloc[0] * 100
                        ax.plot(spy_norm.index, spy_norm, label='SPY',
                                linewidth=1.5, linestyle='--',
                                color=colors['spy'], alpha=0.7)

                m = result_map[label]['metrics']
                subtitle = (f"수익률:{m['total_return']*100:+.1f}%  "
                            f"Sharpe:{m['sharpe_ratio']:.2f}  "
                            f"MDD:{m['mdd']*100:.1f}%")
                ax.set_title(f"{label}\n{subtitle}", fontsize=9)
            else:
                ax.set_title(f"{label}\n(데이터 없음)", fontsize=9)
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=12, color='gray')

            ax.set_ylabel('Value (base=100)')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n차트 저장: {output_path}")
    plt.close()


# ============================================
# [9] 메인
# ============================================

def main():
    args = parse_args()
    backtest_end = args.backtest_end or datetime.now().strftime('%Y-%m-%d')

    print("=" * 60)
    print("  MA 추세 필터 실험")
    print("=" * 60)
    print(f"  학습 초기    : {args.train_start} ~ {args.train_end}")
    print(f"  백테스트     : {args.train_end} ~ {backtest_end}")
    print(f"  WF 스텝      : {args.wf_step_months}개월")
    print(f"  초기 자본금  : ${args.initial_capital:,.0f}")
    print(f"  수수료/슬립  : {args.commission*100:.2f}% / {args.slippage*100:.2f}%")
    print("=" * 60)

    # ── [1] 종목 목록 ─────────────────────────────────────────
    print("\n[1] S&P 500 종목 목록 로딩...")
    try:
        sp500   = get_sp500_list()
        symbols = sp500['symbol'].tolist() + ['SPY']
    except Exception:
        symbols = SP500_BACKUP + ['SPY']
    if 'SPY' not in symbols:
        symbols.append('SPY')
    print(f"    종목 수: {len(symbols)}")

    # ── [2] 전체 원시 데이터 다운로드 ─────────────────────────
    print(f"\n[2] 전체 데이터 다운로드: {args.train_start} ~ {backtest_end}")
    all_raw = get_backtest_data(symbols, start_date=args.train_start,
                                end_date=backtest_end)
    all_raw['date'] = pd.to_datetime(all_raw['date'])
    print(f"    rows: {len(all_raw):,}")

    # ── [3] 시장 MA 필터용: SPY + QQQ 다운로드 ────────────────
    print(f"\n[3] 시장 MA 필터용 데이터 다운로드 (SPY + QQQ)...")
    try:
        market_raw = download_stock_data(['SPY', 'QQQ'],
                                         start_date=args.train_start,
                                         end_date=backtest_end)
        market_raw['date'] = pd.to_datetime(market_raw['date'])
        print(f"    rows: {len(market_raw):,}  "
              f"(tickers: {market_raw['symbol'].unique().tolist()})")
    except Exception as e:
        print(f"    경고: 다운로드 실패({e}), 시장 MA 필터 비활성화(fallback)")
        market_raw = None

    # ── [4] 피처 생성 (하이브리드 전략용) ─────────────────────
    print("\n[4] 피처 생성 중 (하이브리드용)...")
    # 기본 target_return=0.05 (Hybrid)
    all_features_base = create_features(all_raw)
    all_features_base['date'] = pd.to_datetime(all_features_base['date'])
    feature_cols = get_feature_columns(all_features_base)
    print(f"    피처 수: {len(feature_cols)}  /  샘플: {len(all_features_base):,}")

    # Hybrid_New 전용 target_return=0.03
    print("    Hybrid_New용 피처 생성 중...")
    all_features_new = create_features(all_raw,
                                        target_return=TARGET_RETURN_OPTIMIZED)
    all_features_new['date'] = pd.to_datetime(all_features_new['date'])

    # ── [5] Walk-Forward 폴드 생성 ────────────────────────────
    folds = generate_walk_forward_folds(
        args.train_start, args.train_end, backtest_end, args.wf_step_months
    )
    print(f"\n[5] Walk-Forward 폴드: {len(folds)}개")
    for i, f in enumerate(folds):
        print(f"    Fold {i+1:2d}: {f['train_start']}~{f['train_end']} "
              f"→ Test {f['test_start']}~{f['test_end']}")

    backtest_start = args.train_end   # 모멘텀 단일 기간 시작

    # ── [6] 6가지 실험 실행 ───────────────────────────────────
    results = []
    run_kwargs = dict(
        initial_capital=args.initial_capital,
        commission=args.commission,
        slippage=args.slippage,
    )

    # ① 모멘텀 (기존)
    results.append(run_momentum_experiment(
        all_raw, market_raw, backtest_start,
        use_extra_filters=False,
        label='모멘텀 (기존)',
        **run_kwargs,
    ))

    # ② 모멘텀 + MA 필터
    results.append(run_momentum_experiment(
        all_raw, market_raw, backtest_start,
        use_extra_filters=True,
        label='모멘텀 + MA필터',
        **run_kwargs,
    ))

    # ③ 하이브리드 (기존)
    results.append(run_hybrid_experiment(
        all_raw, all_features_base, feature_cols, folds,
        market_raw=None,
        use_extra_filters=False,
        weight_momentum=0.35, weight_ai=0.65,
        xgb_params=XGB_PARAMS,
        target_return=0.05,
        label='하이브리드 (기존)',
        **run_kwargs,
    ))

    # ④ 하이브리드 + MA 필터
    results.append(run_hybrid_experiment(
        all_raw, all_features_base, feature_cols, folds,
        market_raw=market_raw,
        use_extra_filters=True,
        weight_momentum=0.35, weight_ai=0.65,
        xgb_params=XGB_PARAMS,
        target_return=0.05,
        label='하이브리드 + MA필터',
        **run_kwargs,
    ))

    # ⑤ 하이브리드New (기존)
    results.append(run_hybrid_experiment(
        all_raw, all_features_new, feature_cols, folds,
        market_raw=None,
        use_extra_filters=False,
        weight_momentum=0.50, weight_ai=0.50,
        xgb_params=XGB_PARAMS_OPTIMIZED,
        target_return=TARGET_RETURN_OPTIMIZED,
        label='하이브리드New (기존)',
        **run_kwargs,
    ))

    # ⑥ 하이브리드New + MA 필터
    results.append(run_hybrid_experiment(
        all_raw, all_features_new, feature_cols, folds,
        market_raw=market_raw,
        use_extra_filters=True,
        weight_momentum=0.50, weight_ai=0.50,
        xgb_params=XGB_PARAMS_OPTIMIZED,
        target_return=TARGET_RETURN_OPTIMIZED,
        label='하이브리드New + MA필터',
        **run_kwargs,
    ))

    # ── [7] SPY 벤치마크 수익률 계산 ──────────────────────────
    spy_df = all_raw[
        (all_raw['symbol'] == 'SPY') &
        (all_raw['date'] >= pd.Timestamp(backtest_start))
    ].sort_values('date')

    spy_total_return = 0.0
    spy_series       = None
    if len(spy_df) >= 2:
        spy_total_return = (spy_df['close'].iloc[-1] / spy_df['close'].iloc[0]) - 1
        spy_series = spy_df.set_index('date')['close']

    # ── [8] 결과 출력 ─────────────────────────────────────────
    print_results_table(results, spy_total_return)

    # ── [9] 차트 저장 ──────────────────────────────────────────
    plot_comparison(results, spy_series, args.output)

    print("\n실험 완료!")


if __name__ == '__main__':
    main()
