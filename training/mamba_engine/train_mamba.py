import os
import gc
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm

from mamba_network import MambaPhysicsEncoder
from llm_agent import GemmaOllamaAgent
from dataset import BayesianAtlasDataset, get_all_available_dates

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Action space mapping
ACTIONS = {0: "HOLD", 1: "LONG", 2: "SHORT", 3: "SCRATCH"}
CHECKPOINT_PATH = "mamba_checkpoint.pth"

def parse_args():
    parser = argparse.ArgumentParser(description="Batched Mamba+LLM Training Pipeline")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY_MM_DD). If None, starts from beginning or resume state.")
    parser.add_argument("--num-days", type=int, default=1, help="Number of days to train in this batch.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint.")
    parser.add_argument("--llm-mode", type=str, choices=["off", "end-of-day", "on-error"], default="off", help="When to query the LLM Cortex.")
    parser.add_argument("--features-dir", type=str, default=r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\DATA\ATLAS\FEATURES_5s_v2", help="Path to features directory")
    return parser.parse_args()

def save_checkpoint(model, optimizer, current_date_idx, date_str):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'date_idx': current_date_idx,
        'date_str': date_str
    }
    torch.save(state, CHECKPOINT_PATH)
    logger.info(f"Checkpoint saved at {CHECKPOINT_PATH} for {date_str}")

def load_checkpoint(model, optimizer):
    if os.path.exists(CHECKPOINT_PATH):
        state = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        logger.info(f"Resumed from checkpoint. Last processed date: {state['date_str']} (idx {state['date_idx']})")
        return state['date_idx']
    else:
        logger.warning(f"No checkpoint found at {CHECKPOINT_PATH}. Starting from scratch.")
        return -1

def check_vram_limit(threshold_ratio=0.9):
    """
    Checks if CUDA VRAM usage exceeds the threshold.
    Returns True if we should abort the batch.
    """
    if torch.cuda.is_available():
        reserved = torch.cuda.memory_reserved(0)
        total = torch.cuda.get_device_properties(0).total_memory
        ratio = reserved / max(1, total)
        if ratio > threshold_ratio:
            logger.warning(f"VRAM WATCHDOG: Usage at {ratio*100:.1f}%. Aborting further batch processing to prevent OOM.")
            return True
    return False

def train_autonomous_engine(args):
    """
    Main training and validation loop for the Mamba + LLM architecture.
    """
    base_dir = args.features_dir
    
    # 1. Discover Dates
    all_dates = get_all_available_dates(base_dir)
    if not all_dates:
        logger.error("No dates found in dataset. Exiting.")
        return
        
    start_idx = 0
    
    # Determine the starting dimension by peeking at the first dataset
    # We load just one sequence to get num_features
    peek_dataset = BayesianAtlasDataset(features_dir=base_dir, date_str=all_dates[0], seq_len=2)
    input_dim = peek_dataset.num_features
    del peek_dataset
    gc.collect()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initializing Mamba Physics Encoder on {device} with input_dim={input_dim}...")
    model = MambaPhysicsEncoder(input_dim=input_dim, d_model=128, num_classes=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    if args.resume:
        last_idx = load_checkpoint(model, optimizer)
        start_idx = last_idx + 1
        if start_idx >= len(all_dates):
            logger.info("All available dates have been processed. Exiting.")
            return

    if args.start_date:
        if args.start_date in all_dates:
            start_idx = all_dates.index(args.start_date)
        else:
            logger.error(f"Start date {args.start_date} not found in dataset. Exiting.")
            return
            
    end_idx = min(start_idx + args.num_days, len(all_dates))
    batch_dates = all_dates[start_idx:end_idx]
    
    logger.info(f"Training Batch: {len(batch_dates)} days, starting from {batch_dates[0]}")
    
    llm_agent = None
    if args.llm_mode != "off":
        logger.info(f"Initializing Gemma 4 LLM Agent (Ollama) in {args.llm_mode} mode...")
        llm_agent = GemmaOllamaAgent(model_name="gemma")
    
    # Loop over the requested batch of days
    for current_idx, date_str in enumerate(batch_dates, start=start_idx):
        if check_vram_limit():
            break
            
        logger.info(f"--- Loading Data for {date_str} ({current_idx - start_idx + 1}/{len(batch_dates)}) ---")
        try:
            dataset = BayesianAtlasDataset(features_dir=base_dir, date_str=date_str, seq_len=100)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        except Exception as e:
            logger.error(f"Failed to load {date_str}: {e}. Skipping.")
            continue
            
        epoch_loss = 0.0
        steps = 0
        
        model.train()
        for step, (X_batch, y_batch) in enumerate(tqdm(dataloader, desc=f"Training Mamba [{date_str}]")):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            action_logits, regime_state = model(X_batch)
            loss = criterion(action_logits, y_batch)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            steps += 1
            
            # On-Error LLM Mode
            if args.llm_mode == "on-error" and loss.item() > 2.0: # Arbitrary high loss threshold for smoke test
                logger.warning(f"High loss ({loss.item():.4f}) detected. Triggering LLM Cortex...")
                if llm_agent:
                    mamba_action_idx = torch.argmax(action_logits[0]).item()
                    state_vector = regime_state[0].detach()
                    market_context = f"Date {date_str}, Step {step}. Unexpected loss spike."
                    final_action, reasoning = llm_agent.prompt_model(
                        mamba_state_vector=state_vector,
                        mamba_suggested_action=ACTIONS[mamba_action_idx],
                        market_context_text=market_context
                    )
                    logger.info(f"LLM Reasoning: {reasoning}")
        
        avg_loss = epoch_loss / max(1, steps)
        logger.info(f"Completed {date_str}. Avg Loss: {avg_loss:.4f}")
        
        # End-of-Day LLM Mode
        if args.llm_mode == "end-of-day" and llm_agent:
            logger.info("Triggering End-of-Day LLM Cortex Summary...")
            market_context = f"End of day {date_str} summary. Average Loss: {avg_loss:.4f}."
            final_action, reasoning = llm_agent.prompt_model(
                mamba_state_vector=regime_state[0].detach(),
                mamba_suggested_action="HOLD",
                market_context_text=market_context
            )
            logger.info(f"EOD LLM Reasoning: {reasoning}")
            
        # Checkpoint and memory cleanup
        save_checkpoint(model, optimizer, current_idx, date_str)
        
        del dataset
        del dataloader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    args = parse_args()
    train_autonomous_engine(args)
