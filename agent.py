# from lux.utils import direction_to
import numpy as np
from DQN import MultiAgentDQN
import sys

class Agent():
    def __init__(self, player: str, env_cfg):
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg

        # Initialize the Multi-Agent DQN
        n_agents = self.env_cfg["max_units"]  # Number of agents (units)
        input_dims = (self.env_cfg["max_units"] * 3) + (12)  # Compute input size
        n_actions = 6  # Movement + Sapping actions

        self.multi_agent_dqn = MultiAgentDQN(n_agents=n_agents, input_dims=(input_dims,), n_actions=n_actions, player = self.team_id, opponent_player = self.opp_team_id)

        
        # Tracking previous state information
        self.prev_unit_coordinates = None
        self.prev_unit_energies = None
        self.prev_map_energy = None
        self.prev_relic_positions = None

        # Exploration strategy
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_locations = dict()

    def compute_reward(self, unit_positions, prev_unit_positions, 
                   unit_energies, prev_unit_energies, 
                   map_energy, prev_map_energy, 
                   relic_positions, sapping_cost=5, stuck_penalty=-5):
        """Computes reward for each unit to encourage movement, efficient sapping, and strategic positioning."""

        reward = np.zeros(self.env_cfg["max_units"])  # Initialize reward for all units

        for unit_id in range(self.env_cfg["max_units"]):
            curr_pos = unit_positions[unit_id]
            prev_pos = prev_unit_positions[unit_id]
            curr_energy = unit_energies[unit_id]
            prev_energy = prev_unit_energies[unit_id]
            
            ### ** Movement Reward (Avoid Getting Stuck)**
            if np.array_equal(curr_pos, prev_pos):  # If the unit didn't move
                reward[unit_id] += stuck_penalty  # Penalize staying in the same place
            else:
                reward[unit_id] += 2  # Small reward for moving

            ### ** Sapping Reward & Avoiding Unnecessary Sapping**
            x, y = curr_pos
            if prev_energy - curr_energy >= sapping_cost:  # If energy was spent on sapping
                if map_energy[x, y] < prev_map_energy[x, y]:  # Successful sapping
                    reward[unit_id] += 20  # High reward for successful sapping
                    
                    # **Bonus for teamwork in sapping**  
                    nearby_units = sum(1 for pos in unit_positions if np.linalg.norm(pos - curr_pos) <= 2)
                    if nearby_units > 1:
                        reward[unit_id] += 5 * nearby_units  # Encourage coordinated sapping
                else:  # **Failed sapping (wasted energy)**
                    reward[unit_id] -= 10  # Penalize unnecessary sapping

            ### **⚡ Energy Management (Avoid Wasting Energy)**
            if curr_energy < prev_energy and not (prev_energy - curr_energy >= sapping_cost):
                reward[unit_id] -= 2  # Penalize energy loss without meaningful action

            ### ** Survival Reward**
            if curr_energy > 0 and prev_energy == 0:
                reward[unit_id] += 10  # Bonus for recovering from 0 energy

            ### ** Exploration & Relic Seeking Reward**
            if len(relic_positions) > 0:
                closest_relic = min(relic_positions, key=lambda r: np.linalg.norm(r - curr_pos))
                distance = np.linalg.norm(closest_relic - curr_pos)
                reward[unit_id] += max(0, 10 - distance)  # Closer to relic = higher reward

        return reward


    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """Selects actions for each unit and updates the replay buffer."""

        # **Step 1: Extract current state**
        # team_wins = obs['team_wins'][self.team_id] + obs['team_wins'][self.opp_team_id]
        # if team_wins%10 == 0 and team_wins != 0:
        #     self.multi_agent_dqn._sample_models()
        
        unit_mask = np.array(obs["units_mask"][self.team_id]) 
        unit_positions = np.array(obs["units"]["position"][self.team_id]) 
        unit_energies = np.array(obs["units"]["energy"][self.team_id]) 
        map_energy = np.array(obs["map_features"]["energy"])  
        relic_positions = np.array(obs["relic_nodes"])  
        
        # **Step 2: Handle first step initialization**
        if self.prev_unit_coordinates is None:
            self.prev_unit_coordinates = unit_positions.copy()
            self.prev_unit_energies = unit_energies.copy()
            self.prev_map_energy = map_energy.copy()
            self.prev_relic_positions = relic_positions.copy()
            self.prev_actions = np.zeros(self.env_cfg["max_units"], dtype=int)  # Initialize actions
            # self.prev_actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)  # Initialize actions

            return np.zeros((self.env_cfg["max_units"], 3), dtype=int)  # No actions on first step

        # **Step 3: Compute rewards**
        rewards = self.compute_reward(
            unit_positions, self.prev_unit_coordinates,
            unit_energies, self.prev_unit_energies,
            map_energy, self.prev_map_energy,
            relic_positions
        )

        # **Step 4: Prepare Current State Representation**
        unit_representation = np.concatenate((unit_positions.flatten(), unit_energies.flatten()))    
        relic_representation = relic_positions.flatten()

        # **Ensure relic representation has exactly 12 elements**
        if relic_representation.shape[0] < 12:
            padding = np.zeros(12 - relic_representation.shape[0])  
            relic_representation = np.concatenate((relic_representation, padding)) 

        state_representation = np.concatenate((unit_representation, relic_representation))

        # **Step 5: Prepare Previous State Representation**
        prev_unit_representation = np.concatenate((self.prev_unit_coordinates.flatten(), self.prev_unit_energies.flatten()))
        prev_relic_representation = self.prev_relic_positions.flatten()

        # **Ensure prev_relic_representation is always 12 elements**
        if prev_relic_representation.shape[0] < 12:
            padding = np.zeros(12 - prev_relic_representation.shape[0])  
            prev_relic_representation = np.concatenate((prev_relic_representation, padding)) 

        prev_state_representation = np.concatenate((prev_unit_representation, prev_relic_representation))

        # **Step 6: Select actions using Multi-Agent DQN**
        actions, x_coords, y_coords = self.multi_agent_dqn.choose_actions([state_representation] * self.env_cfg["max_units"])

        # **Step 7: Store experience in replay buffer**
        # Store all experiences at once
        available_unit_ids = np.where(unit_mask)[0]  # Extract indices of available units
        num_available_units = len(available_unit_ids)  # Get the count
        # sys.stderr.write(f"raw actions - {actions}\n")
        # sys.stderr.write(f"memory Input - {list(self.prev_actions[:len(available_unit_ids)])}\n")
        self.multi_agent_dqn.store_experience(
            [prev_state_representation] * len(available_unit_ids),  # Previous states
            [state_representation] * len(available_unit_ids),  # Current states
            list(self.prev_actions[:len(available_unit_ids)]),  # Previous actions
            list(rewards[:len(available_unit_ids)]),  # Rewards
            [False] * len(available_unit_ids),  # Done flags
            num_available_units
        )

        with open("/DATA/Shashank/LuxAis3/logs/rewards_log.txt", "a") as log_file:
            log_file.write(f"Step {step}: {list(rewards)}\n")
            log_file.flush()  # Ensure immediate writing

        # **Step 8: Format actions as a list**
        formatted_actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        for i, unit_id in enumerate(np.where(unit_mask)[0]):
            formatted_actions[unit_id] = [actions[i], x_coords[i], y_coords[i]]

        # **Step 9: Update previous state and actions for the next iteration**
        self.prev_unit_coordinates = unit_positions.copy()
        self.prev_unit_energies = unit_energies.copy()
        self.prev_map_energy = map_energy.copy()
        self.prev_relic_positions = relic_positions.copy()
        self.prev_actions = actions.copy()

        return formatted_actions
