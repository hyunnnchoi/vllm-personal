# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# [NOTE, hyunnnchoi, 2025.12.07] ISRTF scheduling metrics tracker

"""
ISRTF Scheduling Metrics Tracker

Tracks and analyzes:
1. Prediction accuracy (predicted vs actual output tokens)
2. Kendall's tau correlation (predicted order vs actual order)
3. Scheduling quality metrics
"""

import csv
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import kendalltau

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    request_id: str
    arrival_time: float
    completion_time: Optional[float] = None
    
    # Prediction metrics
    initial_prediction: float = float('inf')
    final_prediction: float = float('inf')
    actual_output_tokens: int = 0
    prediction_history: List[Tuple[int, float]] = field(default_factory=list)
    
    # Scheduling metrics
    queue_wait_time: float = 0.0
    first_scheduled_time: Optional[float] = None
    time_to_first_token: Optional[float] = None
    
    def __post_init__(self):
        if not self.prediction_history:
            self.prediction_history = []
    
    @property
    def prediction_error(self) -> float:
        """Absolute error of final prediction."""
        if self.final_prediction == float('inf'):
            return float('inf')
        return abs(self.final_prediction - self.actual_output_tokens)
    
    @property
    def prediction_error_rate(self) -> float:
        """Relative error rate of final prediction."""
        if self.actual_output_tokens == 0:
            return float('inf')
        return self.prediction_error / self.actual_output_tokens
    
    @property
    def total_latency(self) -> Optional[float]:
        """Total latency from arrival to completion."""
        if self.completion_time is None:
            return None
        return self.completion_time - self.arrival_time


class ISRTFMetricsTracker:
    """
    Tracks and analyzes ISRTF scheduling metrics.
    
    Features:
    - Per-request prediction accuracy
    - Kendall's tau correlation between predicted and actual order
    - Scheduling quality metrics (latency, queue wait time, etc.)
    - CSV/JSON export for detailed analysis
    """
    
    def __init__(
        self,
        log_dir: str = "/tmp/isrtf_metrics",
        enable_detailed_logging: bool = True,
        kendall_tau_window: int = 100,
    ):
        """
        Args:
            log_dir: Directory to save metric logs
            enable_detailed_logging: Enable per-request detailed logging
            kendall_tau_window: Window size for Kendall's tau calculation
        """
        self.log_dir = log_dir
        self.enable_detailed_logging = enable_detailed_logging
        self.kendall_tau_window = kendall_tau_window
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Request tracking
        self.active_requests: Dict[str, RequestMetrics] = {}
        self.completed_requests: List[RequestMetrics] = []
        
        # Kendall's tau tracking
        self.kendall_tau_history: List[Tuple[float, float, int]] = []  # (timestamp, tau, sample_size)
        
        # Open CSV files
        self._init_csv_files()
        
        logger.info(f"[ISRTF Metrics] Initialized tracker at {self.log_dir}")
    
    def _init_csv_files(self):
        """Initialize CSV files for logging."""
        # Per-request metrics CSV
        self.request_csv_path = os.path.join(self.log_dir, "request_metrics.csv")
        self.request_csv = open(self.request_csv_path, 'w', newline='')
        self.request_writer = csv.writer(self.request_csv)
        self.request_writer.writerow([
            'request_id',
            'arrival_time',
            'completion_time',
            'total_latency',
            'queue_wait_time',
            'time_to_first_token',
            'initial_prediction',
            'final_prediction',
            'actual_output_tokens',
            'prediction_error',
            'prediction_error_rate',
            'num_prediction_updates'
        ])
        
        # Kendall's tau CSV
        self.kendall_csv_path = os.path.join(self.log_dir, "kendall_tau.csv")
        self.kendall_csv = open(self.kendall_csv_path, 'w', newline='')
        self.kendall_writer = csv.writer(self.kendall_csv)
        self.kendall_writer.writerow([
            'timestamp',
            'kendall_tau',
            'p_value',
            'sample_size',
            'mean_prediction_error',
            'median_prediction_error'
        ])
        
        # Prediction history CSV (detailed)
        self.prediction_csv_path = os.path.join(self.log_dir, "prediction_history.csv")
        self.prediction_csv = open(self.prediction_csv_path, 'w', newline='')
        self.prediction_writer = csv.writer(self.prediction_csv)
        self.prediction_writer.writerow([
            'request_id',
            'num_output_tokens',
            'predicted_remaining',
            'timestamp'
        ])
        
        logger.info(f"[ISRTF Metrics] CSV files initialized:")
        logger.info(f"  - Request metrics: {self.request_csv_path}")
        logger.info(f"  - Kendall's tau: {self.kendall_csv_path}")
        logger.info(f"  - Prediction history: {self.prediction_csv_path}")
    
    def register_request(
        self,
        request_id: str,
        arrival_time: float,
        initial_prediction: float
    ):
        """Register a new request."""
        metrics = RequestMetrics(
            request_id=request_id,
            arrival_time=arrival_time,
            initial_prediction=initial_prediction,
            final_prediction=initial_prediction
        )
        metrics.prediction_history.append((0, initial_prediction))
        self.active_requests[request_id] = metrics
        
        logger.debug(
            f"[ISRTF Metrics] Registered request {request_id[:8]}... "
            f"initial_prediction={initial_prediction:.1f}"
        )
    
    def update_prediction(
        self,
        request_id: str,
        num_output_tokens: int,
        predicted_remaining: float
    ):
        """Update prediction for a request."""
        if request_id not in self.active_requests:
            return
        
        metrics = self.active_requests[request_id]
        metrics.final_prediction = predicted_remaining
        metrics.prediction_history.append((num_output_tokens, predicted_remaining))
        
        # Log to prediction history CSV
        if self.enable_detailed_logging:
            self.prediction_writer.writerow([
                request_id,
                num_output_tokens,
                predicted_remaining,
                time.time()
            ])
            self.prediction_csv.flush()
        
        logger.debug(
            f"[ISRTF Metrics] Updated prediction for {request_id[:8]}... "
            f"at {num_output_tokens} tokens: {predicted_remaining:.1f}"
        )
    
    def mark_first_scheduled(self, request_id: str, scheduled_time: float):
        """Mark when a request is first scheduled."""
        if request_id not in self.active_requests:
            return
        
        metrics = self.active_requests[request_id]
        if metrics.first_scheduled_time is None:
            metrics.first_scheduled_time = scheduled_time
            metrics.queue_wait_time = scheduled_time - metrics.arrival_time
    
    def mark_first_token(self, request_id: str, first_token_time: float):
        """Mark when first token is generated."""
        if request_id not in self.active_requests:
            return
        
        metrics = self.active_requests[request_id]
        if metrics.time_to_first_token is None:
            metrics.time_to_first_token = first_token_time - metrics.arrival_time
    
    def complete_request(
        self,
        request_id: str,
        completion_time: float,
        actual_output_tokens: int
    ):
        """Mark a request as completed and log metrics."""
        if request_id not in self.active_requests:
            logger.warning(f"[ISRTF Metrics] Unknown request {request_id[:8]}... completed")
            return
        
        metrics = self.active_requests[request_id]
        metrics.completion_time = completion_time
        metrics.actual_output_tokens = actual_output_tokens
        
        # Log to request CSV
        self.request_writer.writerow([
            request_id,
            metrics.arrival_time,
            metrics.completion_time,
            metrics.total_latency,
            metrics.queue_wait_time,
            metrics.time_to_first_token,
            metrics.initial_prediction,
            metrics.final_prediction,
            metrics.actual_output_tokens,
            metrics.prediction_error,
            metrics.prediction_error_rate,
            len(metrics.prediction_history)
        ])
        self.request_csv.flush()
        
        # Move to completed
        self.completed_requests.append(metrics)
        del self.active_requests[request_id]
        
        logger.info(
            f"[ISRTF Metrics] Request {request_id[:8]}... completed: "
            f"predicted={metrics.final_prediction:.1f}, "
            f"actual={metrics.actual_output_tokens}, "
            f"error={metrics.prediction_error:.1f} "
            f"({metrics.prediction_error_rate*100:.1f}%)"
        )
        
        # Calculate Kendall's tau periodically
        if len(self.completed_requests) % self.kendall_tau_window == 0:
            self._calculate_and_log_kendall_tau()
    
    def _calculate_and_log_kendall_tau(self):
        """
        Calculate Kendall's tau correlation for recent requests.
        
        Kendall's tau measures the ordinal association between two rankings:
        - τ = +1: Perfect agreement (predicted order = actual order)
        - τ = -1: Perfect disagreement (predicted order = reverse of actual)
        - τ = 0: No correlation (random prediction)
        
        For ISRTF scheduling:
        - High positive τ means good prediction (shorter jobs predicted accurately)
        - τ close to 1 means ISRTF is effectively implementing SJF
        """
        if len(self.completed_requests) < 2:
            return
        
        # Get recent completed requests
        recent_requests = self.completed_requests[-self.kendall_tau_window:]
        
        # Extract predictions and actual values
        predictions = [r.final_prediction for r in recent_requests]
        actuals = [r.actual_output_tokens for r in recent_requests]
        
        # Calculate Kendall's tau
        try:
            tau, p_value = kendalltau(predictions, actuals)
            
            # Calculate prediction errors
            errors = [r.prediction_error for r in recent_requests]
            mean_error = np.mean(errors)
            median_error = np.median(errors)
            
            # Log to CSV
            self.kendall_writer.writerow([
                time.time(),
                tau,
                p_value,
                len(recent_requests),
                mean_error,
                median_error
            ])
            self.kendall_csv.flush()
            
            logger.info(
                f"[ISRTF Metrics] Kendall's tau (n={len(recent_requests)}): "
                f"τ={tau:.4f} (p={p_value:.4f}), "
                f"mean_error={mean_error:.1f}, "
                f"median_error={median_error:.1f}"
            )
            
            self.kendall_tau_history.append((time.time(), tau, len(recent_requests)))
            
        except Exception as e:
            logger.warning(f"[ISRTF Metrics] Failed to calculate Kendall's tau: {e}")
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics for all completed requests."""
        if not self.completed_requests:
            return {}
        
        predictions = [r.final_prediction for r in self.completed_requests]
        actuals = [r.actual_output_tokens for r in self.completed_requests]
        errors = [r.prediction_error for r in self.completed_requests]
        error_rates = [r.prediction_error_rate for r in self.completed_requests 
                       if r.prediction_error_rate != float('inf')]
        latencies = [r.total_latency for r in self.completed_requests 
                     if r.total_latency is not None]
        
        summary = {
            'total_requests': len(self.completed_requests),
            'prediction_accuracy': {
                'mean_error': float(np.mean(errors)),
                'median_error': float(np.median(errors)),
                'std_error': float(np.std(errors)),
                'mean_error_rate': float(np.mean(error_rates)) if error_rates else None,
                'median_error_rate': float(np.median(error_rates)) if error_rates else None,
            },
            'latency': {
                'mean': float(np.mean(latencies)) if latencies else None,
                'median': float(np.median(latencies)) if latencies else None,
                'p95': float(np.percentile(latencies, 95)) if latencies else None,
                'p99': float(np.percentile(latencies, 99)) if latencies else None,
            }
        }
        
        # Calculate overall Kendall's tau
        if len(predictions) >= 2:
            try:
                tau, p_value = kendalltau(predictions, actuals)
                summary['kendall_tau'] = {
                    'tau': float(tau),
                    'p_value': float(p_value),
                    'sample_size': len(predictions)
                }
            except:
                pass
        
        return summary
    
    def save_summary(self):
        """Save summary statistics to JSON."""
        summary = self.get_summary_statistics()
        
        summary_path = os.path.join(self.log_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, indent=2, fp=f)
        
        logger.info(f"[ISRTF Metrics] Summary saved to {summary_path}")
        logger.info(f"[ISRTF Metrics] Summary: {json.dumps(summary, indent=2)}")
    
    def close(self):
        """Close CSV files and save final summary."""
        # Calculate final Kendall's tau
        if self.completed_requests:
            self._calculate_and_log_kendall_tau()
        
        # Save summary
        self.save_summary()
        
        # Close files
        self.request_csv.close()
        self.kendall_csv.close()
        self.prediction_csv.close()
        
        logger.info(f"[ISRTF Metrics] Tracker closed. Total requests: {len(self.completed_requests)}")
    
    def __del__(self):
        """Ensure files are closed on deletion."""
        try:
            self.close()
        except:
            pass

