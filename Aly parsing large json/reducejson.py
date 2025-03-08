import json

def filter_entries(data, allowed_categories={1, 2, 3}):
    
    if isinstance(data, list):
        return [entry for entry in data if entry.get("category_id") in allowed_categories]
    elif isinstance(data, dict):
        filtered_data = {}
        for key, value in data.items():
            if isinstance(value, list):
                filtered_data[key] = [entry for entry in value if isinstance(entry, dict) and entry.get("category_id") in allowed_categories]
            else:
                filtered_data[key] = value
        return filtered_data
    return data

def main():
    input_file = 'coco.json'
    output_file = 'filtered_coco.json'
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return

    filtered_data = filter_entries(data)

    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(filtered_data, outfile, indent=4)
        print(f"Filtered JSON written to {output_file}")
    except Exception as e:
        print(f"Error writing {output_file}: {e}")

if __name__ == "__main__":
    main()