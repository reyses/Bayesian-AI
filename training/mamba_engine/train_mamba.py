import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm

from mamba_network import MambaPhysicsEncoder
from llm_agent import GemmaOllamaAgent
from dataset import BayesianAtlasDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Action space mapping
ACTIONS = {0: "HOLD", 1: "LONG", 2: "SHORT", 3: "SCRATCH"}

def train_autonomous_engine():
    """
    Main training and validation loop for the Mamba + LLM architecture.
    """
    base_dir = r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\DATA\ATLAS\FEATURES_5s_v2"
    date_str = "2024_01_02"
    
    logger.info(f"Loading Real Data from {base_dir} for {date_str}...")
    dataset = BayesianAtlasDataset(features_dir=base_dir, date_str=date_str, seq_len=100)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    input_dim = dataset.num_features
    logger.info(f"Initializing Mamba Physics Encoder with input_dim={input_dim}...")
    
    # We set num_classes=4 for our 4 basic actions
    model = MambaPhysicsEncoder(input_dim=input_dim, d_model=128, num_classes=4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    logger.info("Initializing Gemma 4 LLM Agent (Ollama)...")
    llm_agent = GemmaOllamaAgent(model_name="gemma")
    
    epochs = 1  # Just a smoke test
    
    for epoch in range(epochs):
        logger.info(f"--- Epoch {epoch+1}/{epochs} ---")
        
        # Iterating over the 15,000+ sequences in the day
        for step, (X_batch, y_batch) in enumerate(tqdm(dataloader, desc="Training Mamba")):
            optimizer.zero_grad()
            
            # 1. Forward Pass
            action_logits, regime_state = model(X_batch)
            loss = criterion(action_logits, y_batch)
            
            # 2. Backpropagation
            loss.backward()
            optimizer.step()
            
            # 3. Autonomous Reasoning Loop (Smoke test: do this on the 10th step then break)
            if step == 10:
                logger.info("Triggering LLM Reasoning Cortex...")
                mamba_action_idx = torch.argmax(action_logits[0]).item()
                mamba_suggested_action = ACTIONS[mamba_action_idx]
                
                state_vector = regime_state[0].detach()
                market_context = f"Epoch {epoch+1}, Step {step}. Testing real data pipeline."
                
                # Query the LLM
                final_action, reasoning = llm_agent.prompt_model(
                    mamba_state_vector=state_vector,
                    mamba_suggested_action=mamba_suggested_action,
                    market_context_text=market_context
                )
                
                logger.info(f"Mamba Suggestion : {mamba_suggested_action}")
                logger.info(f"LLM Final Action : {final_action}")
                logger.info(f"LLM Reasoning    : {reasoning}")
                
                logger.info("Smoke test successful! Breaking loop.")
                break
                
if __name__ == "__main__":
    train_autonomous_engine()
