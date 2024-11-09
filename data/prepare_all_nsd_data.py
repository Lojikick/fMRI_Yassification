import os

def main():
    subjects = [1, 2, 5, 7]
    
    for sub in subjects:
        command = f"python prepare_nsddata.py -sub {sub}"
        print(f"Running: {command}")
        os.system(command)

if __name__ == "__main__":
    main()