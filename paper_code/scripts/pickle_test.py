import pickle

# Test data
data = ["test", 123, {"a": 1}]

# Save
with open('test.pkl', 'wb') as f:
    pickle.dump(data, f)

# Load
with open('test.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

print("Original:", data)
print("Loaded:", loaded_data)
print("Test passed:", data == loaded_data)

# Cleanup
import os
os.remove('test.pkl')