import argparse

def main():
    parser = argparse.ArgumentParser(description="Fetch person info from Wikipedia")
    parser.add_argument("--name", required=True, help="Person's name (use underscores for spaces)")
    args = parser.parse_args()
    
    print(f"Fetching info for: {args.name}")

if __name__ == "__main__":
    main()
