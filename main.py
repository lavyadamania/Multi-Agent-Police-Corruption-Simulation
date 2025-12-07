import os
import random
import time
from database.db_manager import DBManager
from agents.corrupt_cop import CorruptCop
from agents.honest_cop import HonestCop
from agents.police_chief import PoliceChief
from agents.ia_detective import IADetective
from environment.game_world import GameWorld
from visualization.plotter import Plotter
from config import (
    DB_PATH, PROJECT_ROOT, NUM_EPISODES, INSPECTION_FREQUENCY,
    NUM_CORRUPT_COPS, NUM_HONEST_COPS, INITIAL_CORRUPTION_MIN,
    INITIAL_CORRUPTION_MAX, INITIAL_INTEGRITY_HONEST,
    REWARD_SUCCESS_FACTOR, REWARD_CAUGHT, REWARD_REJECTED,
    WITNESS_BRIBE_COST, REWARD_KILLED, BRAIN_DIR,
    ALERT_DECAY_RATE
)
import json

def main():
    print("======================================================================")
    print("🎮 CORRUPTION MARL - COMPLETE SYSTEM")
    print("👑 Police Chief + IA + Multi-Agent Hierarchy")
    
    # 1. Initialize Database
    schema_path = os.path.join(PROJECT_ROOT, 'database', 'schema.sql')
    db = DBManager(DB_PATH, schema_path)
    
    # Load Global Training Stats
    state_file = os.path.join(PROJECT_ROOT, 'training_state.json')
    start_episode = 0
    
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                data = json.load(f)
                start_episode = data.get('total_episodes', 0)
                print(f"🔄 RESUMING simulation from Episode {start_episode}...")
        except:
            print("⚠️ Error loading state file. Starting fresh.")
            start_episode = 0


    
    agents_map = {} # id -> agent_obj
    corrupt_ids = []
    honest_ids = []
    
    # Define personality types for use in creation and respawn
    personalities = ['greedy', 'cautious', 'paranoid']
    chief = None
    ia = None

    if start_episode > 0:
        # === RESUME MODE ===
        print("📥 Loading Agents from Database...")
        # Fetch active cops
        rows = db.fetch_all("SELECT cop_id, name, cop_type, rank, personality, corruption_score, loyalty_score, times_bribed, times_caught, total_money_earned FROM cops WHERE status='active'")
        
        for row in rows:
            cid, name, c_type, rank, pers, c_score, l_score, t_bribed, t_caught, money = row
            
            if c_type == 'chief':
                agent = PoliceChief(cid)
                agents_map[cid] = agent
                chief = agent # Capture for main loop usage
            elif c_type == 'detective':
                agent = IADetective(cid)
                agents_map[cid] = agent
                ia = agent # Capture for main loop usage
            elif c_type == 'corrupt':
                agent = CorruptCop(cid, name, pers, c_score)
                # Restore Stats
                agent.loyalty_score = l_score
                agent.times_bribed = t_bribed
                agent.times_caught = t_caught
                agent.total_money_earned = money
                agents_map[cid] = agent
                corrupt_ids.append(cid)
            elif c_type == 'honest':
                agent = HonestCop(cid, name, l_score) # Init with integrity/loyalty
                agent.times_bribed = t_bribed
                agent.times_caught = t_caught
                agent.total_money_earned = money
                agents_map[cid] = agent
                honest_ids.append(cid)
        
        print(f"✅ Loaded {len(agents_map)} Active Agents.")
        
    else:
        # === FRESH START MODE ===
        print("✨ Starting FRESH Simulation (Clearing DB)...")
        db.execute_query("DELETE FROM cops")
        db.execute_query("DELETE FROM bribe_history")
        db.execute_query("DELETE FROM investigations")
        db.execute_query("DELETE FROM orders")
        db.execute_query("DELETE FROM episode_stats")

        # 2. Create Agents
        print("Creating Leadership:")
        # Chief
        chief = PoliceChief(0)
        agents_map[0] = chief
        db.execute_query(
            "INSERT INTO cops (cop_id, name, cop_type, rank, personality) VALUES (?, ?, ?, ?, ?)",
            (0, chief.name, chief.cop_type, chief.rank, "strict")
        )

        # IA
        ia = IADetective(1)
        agents_map[1] = ia
        db.execute_query(
            "INSERT INTO cops (cop_id, name, cop_type, rank, personality) VALUES (?, ?, ?, ?, ?)",
            (1, ia.name, ia.cop_type, ia.rank, "analytical")
        )

        print("\nCreating Corrupt Officers:")
        # personalities defined above
        
        for i in range(NUM_CORRUPT_COPS):
            cid = 2 + i
            p_type = personalities[i % len(personalities)]
            c_score = random.uniform(INITIAL_CORRUPTION_MIN, INITIAL_CORRUPTION_MAX)
            
            agent = CorruptCop(cid, f"Officer_{cid}", p_type, c_score)
            agents_map[cid] = agent
            corrupt_ids.append(cid)
            
            db.execute_query(
                "INSERT INTO cops (cop_id, name, cop_type, rank, personality, corruption_score) VALUES (?, ?, ?, ?, ?, ?)",
                (cid, agent.name, agent.cop_type, agent.rank, agent.personality, agent.corruption_score)
            )

        print("\nCreating Honest Officers:")
        for i in range(NUM_HONEST_COPS):
            hid = 2 + NUM_CORRUPT_COPS + i
            i_score = random.uniform(INITIAL_INTEGRITY_HONEST-5, INITIAL_INTEGRITY_HONEST+5)
            
            agent = HonestCop(hid, f"Officer_{hid}", i_score)
            agents_map[hid] = agent
            honest_ids.append(hid)
            
            db.execute_query(
                "INSERT INTO cops (cop_id, name, cop_type, rank, personality, loyalty_score) VALUES (?, ?, ?, ?, ?, ?)",
                (hid, agent.name, agent.cop_type, agent.rank, agent.personality, agent.integrity_score)
            )

        print(f"\n✓ World initialized: {len(agents_map)} agents total")

    print(f"Starting/Resuming Simulation: {NUM_EPISODES} episodes (Target: {start_episode + NUM_EPISODES})...\n")
    
    global_episodes = start_episode



    # 3. Run Simulation
    env = GameWorld()
    
    # Global Alert State
    global_alert_level = 0.0
    
    target_episode = start_episode + NUM_EPISODES

    for episode in range(1, NUM_EPISODES + 1):
        current_global_ep = global_episodes + episode
        
        # Decay Alert Level (Cool down)
        global_alert_level = max(0.0, global_alert_level - ALERT_DECAY_RATE)
        
        # Dynamic Constable List
        all_constables = corrupt_ids + honest_ids
        
        if not all_constables:
            print("Everyone died. Simulation Over.")
            break

        # Pick random constable
        active_cop_id = random.choice(all_constables)
        cop_agent = agents_map[active_cop_id]
        
        # Generate Scenario (Pass Alert Level)
        state = env.generate_scenario()
        state['alert_level'] = global_alert_level # Inject global state
        
        # Agent Decides
        decision = cop_agent.decide_bribe(state)
        
        # Handle Complex Actions
        modified_state = state.copy()
        if decision == 'CLEAN_AND_ACCEPT':
            # Creative Move: Bribe the witnesses to leave!
            modified_state['witnesses'] = 0
            # Outcome is resolved based on modified state (No witnesses)
            outcome = env.resolve_outcome('ACCEPT', modified_state)
        else:
            outcome = env.resolve_outcome(decision, state)
        
        # Update Logic & Learning
        if cop_agent.cop_type == 'corrupt':
            cop_agent.update_stats(outcome, state['offer'])
            
            # Q-Learning Reward Calculation
            reward = 0
            if outcome == 'success':
                reward = state['offer'] * REWARD_SUCCESS_FACTOR
            elif outcome == 'caught':
                reward = REWARD_CAUGHT * state['severity'] 
            elif outcome == 'rejected':
                reward = REWARD_REJECTED
            elif outcome == 'killed':
                reward = REWARD_KILLED
            
            if decision == 'CLEAN_AND_ACCEPT':
                reward -= 5
            
            # LEARN
            cop_agent.learn(reward, next_state_raw=None) 
            
            # Handle Death & Respawn
            if outcome == 'killed':
                print(f"💀 Officer_{active_cop_id} was KILLED in the line of duty! (Legacy Transfer Initiated...)")
                
                # Check for Witness Backup Call
                if state['witnesses'] > 0:
                    print(f"📞 BANDIT DOWN! Witnesses called backup! GLOBAL HIGH ALERT triggered! 🚨")
                    global_alert_level = 1.0 # Max Heat
                
                # 1. Save the Dead Brain
                cop_agent.save_brain()
                dead_brain_path = os.path.join(BRAIN_DIR, f"cop_{active_cop_id}.pth")
                
                # 2. Recruitment
                next_cop_id = max(agents_map.keys()) + 1
                new_p = random.choice(personalities)
                new_c_score = random.uniform(INITIAL_CORRUPTION_MIN, INITIAL_CORRUPTION_MAX)
                
                new_agent = CorruptCop(next_cop_id, f"Officer_{next_cop_id}", new_p, new_c_score)
                
                # 3. Brain Transplant
                new_agent.inherit_brain(dead_brain_path)
                
                # 4. System Update
                agents_map[next_cop_id] = new_agent
                
                if active_cop_id in corrupt_ids:
                    corrupt_ids.remove(active_cop_id)
                    corrupt_ids.append(next_cop_id)
                elif active_cop_id in honest_ids:
                    honest_ids.remove(active_cop_id)
                    honest_ids.append(next_cop_id)

                # Update DB
                db.execute_query("UPDATE cops SET status='killed' WHERE cop_id=?", (active_cop_id,))
                
                db.execute_query(
                    "INSERT INTO cops (cop_id, name, cop_type, rank, personality, corruption_score) VALUES (?, ?, ?, ?, ?, ?)",
                    (next_cop_id, new_agent.name, new_agent.cop_type, new_agent.rank, new_agent.personality, new_agent.corruption_score)
                )
                
                print(f"♻️ RECRUITMENT: {new_agent.name} replaced Officer_{active_cop_id}. Brain/Memory Inherited! 🧠")
                continue

            # DB Update for stats (Alive agents only)
            db.execute_query(
                "UPDATE cops SET corruption_score=?, paranoia_level=?, loyalty_score=?, times_bribed=?, times_caught=?, total_money_earned=? WHERE cop_id=?",
                (cop_agent.corruption_score, cop_agent.paranoia_level, cop_agent.loyalty_score, 
                 cop_agent.times_bribed, cop_agent.times_caught, cop_agent.total_money_earned, active_cop_id)
            )
        else:
            # Honest cop logic update
            if state['witnesses'] > 0:
                 cop_agent.witness_corruption()
            
            db.execute_query(
                "UPDATE cops SET reports_filed=? WHERE cop_id=?",
                (cop_agent.reports_filed, active_cop_id)
            )

        # Log Transaction
        db.execute_query(
            """INSERT INTO bribe_history 
            (cop_id, episode_number, player_offer, witness_count, ia_nearby, location_risk, decision, outcome) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (active_cop_id, episode, state['offer'], state['witnesses'], state['ia_nearby'], state['location_risk'], decision, outcome)
        )

        # Print notifications
        if outcome == 'caught':
            print(f"🚨 {cop_agent.name} caught! (Ep {current_global_ep})")

        if episode % 50 == 0:
            print(f"Episode {current_global_ep}/{target_episode} (Total Lifetime: {current_global_ep})...")

        # 4. Chief Inspection
        if episode > 0 and episode % INSPECTION_FREQUENCY == 0:
            print(f"Episode {episode} Inspection...")
            
            # Fetch current stats
            cop_rows = db.fetch_all("SELECT cop_id, rank, corruption_score, times_bribed, times_caught, total_money_earned FROM cops WHERE status != 'killed'")
            cop_data = []
            for row in cop_rows:
                 cop_data.append({
                     'cop_id': row[0],
                     'rank': row[1],
                     'corruption_score': row[2],
                     'times_bribed': row[3],
                     'times_caught': row[4],
                     'total_money_earned': row[5]
                 })

            inspection_orders = chief.monitor_subordinates(cop_data)
            
            for order in inspection_orders:
                target_id = order['target_cop_id']
                if target_id not in agents_map: continue 

                if order['type'] == 'BLACKMAIL':
                    extortion_amount = order['amount']
                    chief.total_money_earned += extortion_amount
                    chief.corruption_score += 3
                    
                    db.execute_query(
                        "UPDATE cops SET total_money_earned = total_money_earned - ? WHERE cop_id=?",
                        (extortion_amount, target_id)
                    )
                    
                    print(f"💀 Chief BLACKMAILED Officer_{target_id} for ₹{extortion_amount}!")
                    
                    db.execute_query(
                        "INSERT INTO orders (chief_id, target_cop_id, order_type, order_details, status) VALUES (?, ?, ?, ?, ?)",
                        (chief.agent_id, target_id, 'BLACKMAIL', f"Extortion: {extortion_amount}", 'completed')
                    )
                    continue 

                # Legitimate Investigation Logic
                db.execute_query(
                    "INSERT INTO orders (chief_id, target_cop_id, order_type, order_details) VALUES (?, ?, ?, ?)",
                    (chief.agent_id, target_id, order['type'], order['details'])
                )
                
                target_row = db.fetch_one("SELECT corruption_score, times_caught FROM cops WHERE cop_id=?", (target_id,))
                if target_row:
                    target_stats = {'corruption_score': target_row[0], 'times_caught': target_row[1]}
                    
                    investigation_result, bribe_taken = ia.conduct_investigation(target_stats)
                    
                    if investigation_result == 'COVERUP':
                        print(f"🤝 IA Covered up case against Officer_{target_id} for ₹{bribe_taken}!")
                        db.execute_query(
                            "INSERT INTO investigations (investigator_id, target_cop_id, evidence_score, status, outcome) VALUES (?, ?, ?, ?, ?)",
                            (ia.agent_id, target_id, 0, 'closed', 'COVERUP')
                        )
                    else:
                        punishment = "NONE"
                        if investigation_result in ['STRONG', 'MODERATE']:
                            punishment = chief.decide_punishment(investigation_result, target_stats)
                            if punishment != 'WARNING':
                                print(f"⚖️ Judgment on Officer_{target_id}: {punishment}")
                        
                        db.execute_query(
                            "INSERT INTO investigations (investigator_id, target_cop_id, evidence_score, status, outcome) VALUES (?, ?, ?, ?, ?)",
                            (ia.agent_id, target_id, 0, 'closed', punishment)
                        )

    # 5. Final Statistics
    print("\nResults:")
    counts = db.fetch_all("SELECT outcome, COUNT(*) FROM bribe_history GROUP BY outcome")
    total_eps = NUM_EPISODES
    for outcome, count in counts:
        print(f"{outcome.capitalize()}: {count} ({count/total_eps*100:.1f}%)")
    
    print("\nLeadership Activity:")
    print(chief)
    print(ia)

    # Save Global Stats
    with open(state_file, 'w') as f:
        json.dump({'total_episodes': global_episodes + NUM_EPISODES}, f)

    # Save Brains
    print("\nSaving Agent Memories (Brains)...")
    for cid in corrupt_ids:
        if hasattr(agents_map[cid], 'save_brain'):
            agents_map[cid].save_brain()
    print("✅ All brains saved to /brains/")

    # Visualization
    plotter = Plotter(db)
    plotter.generate_all()
    
    db.close()

if __name__ == "__main__":
    main()
