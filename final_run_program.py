
import argparse

def main():
    parser = argparse.ArgumentParser(description="New Pipeline Runner")
    parser.add_argument("--scan_path", required=True, help="Path to the images scan folder")
    parser.add_argument("--results_folder", required=True, help="Folder to write results")
    args = parser.parse_args()


    #WE NEED TO PUT OUR FINAL CODE HERE!!
    
    
    print(f"Scan path: {args.scan_path}")
    print(f"Results Folder: {args.results_folder}")

if __name__ == "__main__":
    main()