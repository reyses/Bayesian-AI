import random
from environment import NMPEnvironment

class ParallelWorlds:
    """
    Defines the behavioral constraints for the 4 parallel Multi-Agent RL explorers.
    """
    WORLD_1_FULL = 1
    WORLD_2_ENTRY = 2
    WORLD_3_EXIT = 3
    WORLD_4_CHAOS = 4

    @staticmethod
    def _get_random_action(is_flat: bool) -> int:
        """ 
        0=Buy/Hold, 1=Sell/Exit, 2=Pass
        """
        if is_flat:
            return random.choice([0, 1, 2])
        return random.choice([0, 1])

    @staticmethod
    def execute_constrained_action(agent_id: int, current_state, is_flat: bool, is_long: bool, rl_action: int) -> int:
        """
        Routes the action based on the agent's specific mathematical constraints.
        """
        if agent_id == ParallelWorlds.WORLD_1_FULL:
            # Full RL control
            return rl_action

        elif agent_id == ParallelWorlds.WORLD_2_ENTRY:
            # Entry Specialist: RL Entry, NMP Exit
            if is_flat:
                return rl_action
            else:
                force_exit = NMPEnvironment.check_theoretical_exit(current_state, is_long)
                return 1 if force_exit else 0

        elif agent_id == ParallelWorlds.WORLD_3_EXIT:
            # Exit Specialist: Random Entry, RL Exit
            if is_flat:
                return ParallelWorlds._get_random_action(True)
            else:
                return rl_action

        elif agent_id == ParallelWorlds.WORLD_4_CHAOS:
            # Chaos Explorer: Pure random structural noise
            return ParallelWorlds._get_random_action(is_flat)

        return 2 # Default Pass
