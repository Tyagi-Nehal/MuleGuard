import pandas as pd
import numpy as np
import random
import uuid
import os
from datetime import datetime, timedelta
from faker import Faker

def generate_upi_dataset(num_normal_tx=3000, num_mule_rings=3, accounts_per_ring=30):
    print("--- Starting MuleGuard Synthetic UPI Data Engine ---")
    fake = Faker('en_IN') # Generates realistic Indian names and contexts
    
    data = []
    start_time = datetime(2026, 6, 1, 9, 0, 0)

    # --- Part 1: Generate Normal UPI Transactions (Legitimate daily spending) ---
    print(f"Generating {num_normal_tx} normal retail UPI transactions...")
    
    # 1. Create a pool of standard user VPAs (Virtual Payment Addresses)
    normal_users = [f"{fake.first_name().lower()}{random.randint(10,99)}@okaxis" for _ in range(500)]
    merchants = [f"{fake.company().lower().replace(' ', '')}@oksbi" for _ in range(50)]
    
    # 2. Loop to create individual, random transactions
    for _ in range(num_normal_tx):
        tx_id = f"TXN{str(uuid.uuid4())[:8].upper()}"
        sender = random.choice(normal_users)
        receiver = random.choice(merchants)
        
        # 3. Apply the Exponential Distribution Formula for realistic retail amounts
        amount = round(float(np.random.exponential(scale=250) + 15), 2)
        
        # 4. Scatter timestamps randomly across a 24-hour window
        timestamp = start_time + timedelta(seconds=random.randint(0, 86400))
        
        # 5. Append to data list with '0' meaning Legitimate
        data.append([tx_id, sender, receiver, amount, timestamp, 0])




# --- Part 2: Inject Coordinated Micro-Structured Mule Rings (The Fraud) ---
    print(f"Injecting {num_mule_rings} coordinated star-topology fraud rings...")
    
    for ring_idx in range(num_mule_rings):
        # 1. Create the central "Master Account" that collects the stolen cash
        mule_aggregator = f"mule_center_ring{ring_idx}_{random.randint(100,999)}@okicici"
        
        # 2. Time Crunch: Force the entire attack to happen within a tight 2-hour window
        attack_window_start = start_time + timedelta(hours=random.randint(2, 20))
        
        for i in range(accounts_per_ring):
            tx_id = f"TXN{str(uuid.uuid4())[:8].upper()}"
            
            # 3. Create the fake sender accounts ("Smurfs")
            smurf_source = f"smurf_node_{ring_idx}_{i}_{random.randint(10,99)}@paytm"
            
            # 4. THE LOOPHOLE: Keep amounts strictly under ₹500 to dodge standard bank alerts
            amount = round(random.uniform(150, 495), 2)
            timestamp = attack_window_start + timedelta(minutes=random.randint(0, 120))
            
            # 5. Append to data list with '1' meaning Fraud Ring
            data.append([tx_id, smurf_source, mule_aggregator, amount, timestamp, 1])
        
# --- Part 3: Compile, Sort, and Save the Dataset ---
    # 1. Turn our bucket list into a structured DataFrame (Spreadsheet Table)
    df = pd.DataFrame(data, columns=['tx_id', 'sender', 'receiver', 'amount', 'timestamp', 'is_fraud'])
    
    # 2. Sort all transactions by time so normal and fraud are mixed naturally
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    # 3. Save the final file to your disk
    output_path = 'data/transactions.csv'
    df.to_csv(output_path, index=False)
    
    print("\n--- Generation Phase Completed Successfully ---")
    print(f"Dataset compiled and stored at: {output_path}")
    print(f"Total simulated transaction records logged: {len(df)}")

# This tells Python to actually run our generator machine when the file is executed
if __name__ == "__main__":
    generate_upi_dataset()