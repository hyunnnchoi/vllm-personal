#!/usr/bin/env python3
"""
HOL Blocking Analysis Script

이 스크립트는 vLLM 스케줄러에서 생성된 HOL blocking 로그를 분석합니다.
"""

import pandas as pd
import argparse
import os
from pathlib import Path
import sys


def analyze_hol_blocking(csv_path: str):
    """HOL blocking CSV 파일을 분석하고 주요 메트릭을 출력합니다."""
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    print(f"Analyzing HOL blocking log: {csv_path}")
    print("=" * 80)
    
    # CSV 파일 읽기
    df = pd.read_csv(csv_path)
    
    print(f"\n총 이벤트 수: {len(df)}")
    print(f"고유 요청 수: {df['request_id'].nunique()}")
    print(f"스케줄링 단계 수: {df['scheduling_step'].nunique()}")
    
    # 이벤트 타입별 통계
    print("\n=== 이벤트 타입별 통계 ===")
    event_counts = df['event_type'].value_counts()
    for event_type, count in event_counts.items():
        print(f"{event_type}: {count}")
    
    # 블로킹 이벤트 분석
    blocking_df = df[df['event_type'] == 'blocking']
    if len(blocking_df) > 0:
        print("\n=== 블로킹 이유별 통계 ===")
        blocking_reasons = blocking_df['blocking_reason'].value_counts()
        for reason, count in blocking_reasons.items():
            print(f"{reason}: {count}")
        
        # HOL blocking 비율 계산
        hol_blocked = len(blocking_df[blocking_df['blocking_reason'] == 'blocked_by_hol'])
        direct_blocked = len(blocking_df[blocking_df['blocking_reason'] == 'kv_cache_full'])
        
        if direct_blocked > 0:
            print(f"\n=== HOL Blocking 분석 ===")
            print(f"직접 블로킹 (KV cache full): {direct_blocked}")
            print(f"간접 블로킹 (HOL에 의한): {hol_blocked}")
            print(f"평균 간접/직접 블로킹 비율: {hol_blocked/direct_blocked:.2f}")
        
        # 대기 시간 분석
        print("\n=== 대기 시간 통계 (ms) ===")
        finished_df = df[df['event_type'] == 'finished']
        if len(finished_df) > 0:
            wait_times = pd.to_numeric(finished_df['wait_duration_ms'], errors='coerce')
            print(f"평균: {wait_times.mean():.2f}")
            print(f"중앙값: {wait_times.median():.2f}")
            print(f"최소: {wait_times.min():.2f}")
            print(f"최대: {wait_times.max():.2f}")
            print(f"표준편차: {wait_times.std():.2f}")
    
    # 요청별 블로킹 횟수
    print("\n=== 요청별 블로킹 횟수 (Top 10) ===")
    blocking_per_req = blocking_df.groupby('request_id').size().sort_values(ascending=False).head(10)
    for req_id, count in blocking_per_req.items():
        print(f"{req_id}: {count}회")
    
    # 스케줄링 단계별 통계
    print("\n=== 스케줄링 단계별 평균 대기 요청 수 ===")
    avg_waiting = df.groupby('scheduling_step')['num_waiting_requests'].mean()
    print(f"평균: {avg_waiting.mean():.2f}")
    print(f"최대: {avg_waiting.max():.2f}")
    print(f"최소: {avg_waiting.min():.2f}")
    
    # 토큰 예산 분석
    print("\n=== 토큰 예산 통계 ===")
    token_budget = df[df['token_budget_remaining'] > 0]['token_budget_remaining']
    if len(token_budget) > 0:
        print(f"평균 남은 토큰 예산: {token_budget.mean():.2f}")
        print(f"중앙값: {token_budget.median():.2f}")
    
    print("\n" + "=" * 80)
    print("분석 완료!")
    
    # 상세 분석 파일 생성
    output_dir = os.path.dirname(csv_path)
    summary_path = os.path.join(output_dir, "hol_blocking_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("HOL Blocking Analysis Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"총 이벤트 수: {len(df)}\n")
        f.write(f"고유 요청 수: {df['request_id'].nunique()}\n")
        f.write(f"스케줄링 단계 수: {df['scheduling_step'].nunique()}\n\n")
        
        f.write("이벤트 타입별 통계:\n")
        for event_type, count in event_counts.items():
            f.write(f"  {event_type}: {count}\n")
        
        if len(blocking_df) > 0:
            f.write("\n블로킹 이유별 통계:\n")
            for reason, count in blocking_reasons.items():
                f.write(f"  {reason}: {count}\n")
    
    print(f"\n요약 파일 생성: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze HOL blocking logs from vLLM scheduler"
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        help="Path to HOL blocking CSV file (or directory containing logs)"
    )
    
    args = parser.parse_args()
    
    # CSV 파일 찾기
    if args.csv_file:
        path = Path(args.csv_file)
        if path.is_dir():
            # 디렉토리인 경우 가장 최근 파일 찾기
            csv_files = sorted(path.glob("hol_blocking_*.csv"))
            if not csv_files:
                print(f"Error: No HOL blocking CSV files found in {path}")
                sys.exit(1)
            csv_path = str(csv_files[-1])
        else:
            csv_path = str(path)
    else:
        # 기본 경로에서 찾기
        default_dir = Path("./hol_blocking_logs")
        if not default_dir.exists():
            print("Error: No CSV file specified and default directory not found.")
            print("Usage: python analyze_hol_blocking.py <csv_file>")
            sys.exit(1)
        
        csv_files = sorted(default_dir.glob("hol_blocking_*.csv"))
        if not csv_files:
            print(f"Error: No HOL blocking CSV files found in {default_dir}")
            sys.exit(1)
        csv_path = str(csv_files[-1])
    
    analyze_hol_blocking(csv_path)


if __name__ == "__main__":
    main()

