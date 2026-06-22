import requests
import json
import logging

logger = logging.getLogger(__name__)

class GemmaOllamaAgent:
    """
    Interfaces with a local Gemma 4 model via Ollama.
    Acts as the 'Cortex' or reasoning layer on top of the Mamba physics engine.
    """
    def __init__(self, model_name="gemma", host="http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        self.api_url = f"{self.host}/api/generate"
        
        # This is where the 'Wet Memory' rules live.
        self.system_prompt = (
            "You are an autonomous trading reasoning agent.\n"
            "You receive physical state vectors from a Mamba sequence model.\n"
            "Your job is to override or approve the model's trades based on strict structural rules.\n"
            "RULES:\n"
            "1. If Mamba outputs a LONG action, but the 10m Cubic Slope is negative and we are below the 1H Mean, you must OVERRIDE and output SCRATCH or HOLD.\n"
            "2. Always output a JSON response with two keys: 'action' (LONG, SHORT, SCRATCH, HOLD) and 'reasoning' (a short explanation)."
        )

    def prompt_model(self, mamba_state_vector, mamba_suggested_action, market_context_text):
        """
        Sends the state to Gemma and asks for a decision.
        """
        prompt = (
            f"Current Market Context: {market_context_text}\n"
            f"Mamba Suggested Action: {mamba_suggested_action}\n"
            f"Mamba Regime State Vector (first 4 dims): {mamba_state_vector[:4].tolist()}\n"
            "Given the RULES, what is your final action? Respond ONLY in valid JSON."
        )
        
        payload = {
            "model": self.model_name,
            "system": self.system_prompt,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=5)
            response.raise_for_status()
            
            result_text = response.json().get("response", "{}")
            result_json = json.loads(result_text)
            
            action = result_json.get("action", "HOLD")
            reasoning = result_json.get("reasoning", "No reasoning provided.")
            
            return action, reasoning
            
        except Exception as e:
            logger.error(f"Ollama Agent Failed: {e}. Falling back to Mamba's raw action.")
            return mamba_suggested_action, f"Ollama failed, defaulting to Mamba: {e}"

    def update_rules(self, new_rule: str):
        """
        Dynamic 'Wet Memory' injection. 
        Adds a new rule to the LLM's system prompt in real-time.
        """
        self.system_prompt += f"\nNEW RULE: {new_rule}"
        logger.info(f"Wet Memory Updated with new rule: {new_rule}")
