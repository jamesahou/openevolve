import pickle

db_path = "/mnt/sdd1/jamesh/funsearch-ks-law/data/backups/program_db_priority_1740104178_0.pickle"
with open(db_path, "rb") as f:
    db = pickle.load(f)

for i, _island in enumerate(db["_islands"]):
    print("island #", i, "best_score", db["_best_score_per_island"][i], "best_program")
    print(db["_best_program_per_island"][i])