import asyncio
import logging
import torch
import numpy as np
from typing import Dict, Any, List

from core_v2.engine_signals import (
    DecisionBatch,
    EntrySignal,
    ExitSignal,
    PositionDecision,
)
from training.mamba_engine.mamba_network import MambaPhysicsEncoder
from training.mamba_engine.llm_agent import GemmaOllamaAgent

logger = logging.getLogger('mamba_decider')

class MambaDecider:
    """
    Live inference decider for the Mamba model.
    Maintains a 6-bar rolling window of V2 features and evaluates them.
    Optionally queries a local Gemma LLM for trade verification.
    """
    def __init__(self, checkpoint_path: str, llm_mode: bool = False, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.llm_mode = llm_mode
        self.seq_len = 6
        self.n_features = 385  # Assuming V2 feature size. 

        # Initialize the model
        logger.info(f"Loading MambaPhysicsEncoder from {checkpoint_path} on {self.device}...")
        try:
            self.model = MambaPhysicsEncoder(
                d_model=128,
                d_state=16,
                d_conv=4,
                expand=2,
                num_layers=2,
                input_dim=self.n_features,
                num_classes=4
            ).to(self.device)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            logger.info("Mamba model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Mamba model: {e}")
            raise

        # Initialize the rolling buffer
        self.buffer = []

        # Initialize LLM Agent if enabled
        if self.llm_mode:
            logger.info("Initializing GemmaOllamaAgent for LLM verification...")
            self.llm_agent = GemmaOllamaAgent()
        else:
            self.llm_agent = None

    async def evaluate_async(self, state: Dict[str, Any]) -> DecisionBatch:
        """
        Asynchronous evaluation loop.
        Appends new features to buffer, runs Mamba inference, and optionally queries LLM.
        """
        features = state['features_79d']  # Actually V2 features (size 385)
        self.n_features = len(features)
        positions = state['positions']
        price = state['price']
        
        # 1. Update sequence buffer
        self.buffer.append(features)
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)

        # Base counter updates (Mamba has no explicit manual counters yet, 
        # but we must emit PositionDecision for every open position to prevent errors)
        pos_decs = []
        for pos in positions.all_positions:
            pos_decs.append(PositionDecision(contract_id=pos.contract_id))

        batch = DecisionBatch(position_decisions=pos_decs)

        # Not enough history yet
        if len(self.buffer) < self.seq_len:
            return batch

        # 2. Run Mamba Forward Pass
        # Convert to tensor shape (1, seq_len, num_features)
        x_seq = np.array(self.buffer, dtype=np.float32)
        x_tensor = torch.tensor(x_seq).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _ = self.model(x_tensor)
            # Class definition: 0: NOTHING, 1: BUY, 2: SELL, 3: FLATTEN
            pred_class = torch.argmax(logits, dim=-1).item()

        # Translate classes to Mamba Suggested Action
        mamba_action_str = "HOLD"
        if pred_class == 1:
            mamba_action_str = "LONG"
        elif pred_class == 2:
            mamba_action_str = "SHORT"
        elif pred_class == 3:
            mamba_action_str = "SCRATCH"

        # 3. LLM Verification
        final_action = mamba_action_str
        if self.llm_mode and mamba_action_str in ["LONG", "SHORT", "SCRATCH"]:
            # Provide simple context to the LLM (in reality, could provide more stats)
            market_context = f"Price: {price:.2f}"
            state_vector = features[:4]  # Example subset

            # Query LLM asynchronously using asyncio.to_thread
            try:
                llm_action, reasoning = await asyncio.to_thread(
                    self.llm_agent.prompt_model,
                    mamba_state_vector=state_vector,
                    mamba_suggested_action=mamba_action_str,
                    market_context_text=market_context
                )
                logger.info(f"LLM Reasoning: {reasoning}")
                final_action = llm_action
            except Exception as e:
                logger.error(f"Error querying LLM: {e}")
                final_action = mamba_action_str

        # 4. Map final action to engine signals
        is_flat = positions.is_flat
        primary = positions.primary

        if final_action == "SCRATCH":
            if not is_flat:
                batch.negative_exit = ExitSignal(
                    contract_id=primary.contract_id,
                    reason="mamba_flatten"
                )
        
        elif final_action == "LONG":
            if is_flat:
                # Enter Long
                batch.entry = EntrySignal(tier="M1", direction="long", cnn_flipped=False)
            elif primary.direction == "short":
                # Flip: Exit short, enter long
                batch.negative_exit = ExitSignal(
                    contract_id=primary.contract_id,
                    reason="mamba_flip_to_long"
                )
                batch.entry = EntrySignal(tier="M1", direction="long", cnn_flipped=False)
                
        elif final_action == "SHORT":
            if is_flat:
                # Enter Short
                batch.entry = EntrySignal(tier="M1", direction="short", cnn_flipped=False)
            elif primary.direction == "long":
                # Flip: Exit long, enter short
                batch.negative_exit = ExitSignal(
                    contract_id=primary.contract_id,
                    reason="mamba_flip_to_short"
                )
                batch.entry = EntrySignal(tier="M1", direction="short", cnn_flipped=False)

        return batch
