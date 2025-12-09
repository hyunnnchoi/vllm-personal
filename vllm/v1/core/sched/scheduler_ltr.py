# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [NOTE, hyunnnchoi, 2025.12.09] Learning-to-Rank Scheduler
# This is a specialized scheduler that uses LTR predictor for request prioritization.
# It inherits from the base Scheduler but overrides key methods to implement LTR logic.

from __future__ import annotations

import os
from typing import Optional

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.sched.predictor import LTRPredictor
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)


class LTRScheduler(Scheduler):
    """
    Learning-to-Rank Scheduler.
    
    This scheduler uses an LTR predictor model to assign priority scores to requests
    based on their prompts. Higher scores indicate higher priority.
    
    Key differences from base Scheduler:
    - Initializes LTRPredictor for scoring requests
    - Computes LTR scores on request arrival
    - Uses score-based prioritization in the waiting queue
    
    Environment variables:
    - VLLM_LTR_PREDICTOR_PATH: Path to predictor model (required)
    - VLLM_LTR_STARVATION_THRESHOLD: Tokens before boosting low-priority reqs (default: 256)
    - VLLM_LTR_PRIORITY_QUANTUM: Token increment for starvation prevention (default: 32)
    """
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        # Initialize base scheduler (FCFS/Priority/ISRTF logic)
        # But we'll override the policy to LTR
        super().__init__(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )
        
        # Override policy to LTR
        self.policy = SchedulingPolicy.LTR
        
        # [NOTE, hyunnnchoi, 2025.12.09] Initialize LTR predictor
        predictor_path = os.environ.get("VLLM_LTR_PREDICTOR_PATH")
        if predictor_path is None:
            raise ValueError(
                "VLLM_LTR_PREDICTOR_PATH environment variable must be set "
                "when using LTR scheduling policy"
            )
        
        logger.info("[LTR Scheduler] Initializing with predictor from %s", 
                   predictor_path)
        
        self.ltr_predictor = LTRPredictor(
            target_model=vllm_config.model_config.model,
            predictor_model_path=predictor_path
        )
        
        # Starvation prevention parameters
        self.starvation_threshold = int(os.environ.get(
            "VLLM_LTR_STARVATION_THRESHOLD", "256"
        ))
        self.priority_quantum = int(os.environ.get(
            "VLLM_LTR_PRIORITY_QUANTUM", "32"
        ))
        
        logger.info(
            "[LTR Scheduler] Starvation threshold: %d tokens, "
            "Priority quantum: %d tokens",
            self.starvation_threshold, self.priority_quantum
        )
    
    def add_request(self, request: Request) -> None:
        """
        Add a request to the scheduler.
        
        Overrides base method to compute LTR score before adding to queue.
        """
        # [NOTE, hyunnnchoi, 2025.12.09] Compute LTR score
        try:
            score = self.ltr_predictor.get_score(request.prompt_token_ids)
            # Store score as request attribute
            request.ltr_score = score
            logger.debug(
                "[LTR] Request %s score: %.3f (prompt len: %d)",
                request.request_id[:8], score, len(request.prompt_token_ids)
            )
        except Exception as e:
            logger.warning(
                "[LTR] Failed to compute score for request %s: %s. Using 0.0",
                request.request_id[:8], e
            )
            request.ltr_score = 0.0
        
        # Initialize starvation tracking
        request.ltr_starvation_tokens = 0
        request.ltr_promote = False
        
        # Call parent to add to queue and track
        super().add_request(request)
    
    def _update_starvation_tracking(self, request: Request, 
                                   num_scheduled_tokens: int) -> None:
        """
        Update starvation tracking for a request.
        
        If a waiting request hasn't been scheduled for too long, boost its priority.
        """
        if hasattr(request, 'ltr_starvation_tokens'):
            request.ltr_starvation_tokens += num_scheduled_tokens
            
            # Check if we should promote this request
            if (request.ltr_starvation_tokens >= self.starvation_threshold 
                and not getattr(request, 'ltr_promote', False)):
                request.ltr_promote = True
                logger.info(
                    "[LTR] Promoting starved request %s (waited %d tokens)",
                    request.request_id[:8], request.ltr_starvation_tokens
                )
    
    # Note: For a minimal implementation, we don't override schedule().
    # The base Scheduler's schedule() will work with LTRRequestQueue,
    # which internally uses the ltr_score attribute we set in add_request().
    #
    # If you want to implement custom LTR scheduling logic (like jaehoon's version),
    # you would override schedule() here. But that would require replicating
    # a lot of complex logic from the base scheduler.
    #
    # For now, we rely on LTRRequestQueue's score-based ordering.

