import json

# --- Category lookup table
CATEGORY_MAP = {
    'tshirt': 'clothing',
    'shirt': 'clothing',
    'jeans': 'clothing',
    'shoe': 'footwear',
    'laptop': 'electronics',
    'keyboard': 'electronics',
    'mouse': 'electronics',
    'chair': 'furniture',
    'bottle': 'utensils',
    'book': 'stationery',
    'pillow': 'bedding'
}

# --- Estimate the zone (floor, table, shelf)
def estimate_zone(y2):
    if y2 > 400:
        return "floor"
    elif y2 > 250:
        return "table"
    else:
        return "shelf"

# --- Main function
def categorize_objects():
    # Load input from phase3
    with open("../phase3_result-parsing-display/detected_objects.json", "r") as infile:
        data = json.load(infile)

    new_data = {"objects": []}

    for obj in data:
        name = obj["name"].lower()
        bbox = obj["bbox"]
        y2 = bbox[3]  # Bottom of bounding box

        category = CATEGORY_MAP.get(name, "unknown")
        zone = estimate_zone(y2)

        new_obj = {
            "name": name,
            "bbox": bbox,
            "category": category,
            "zone": zone
        }

        new_data["objects"].append(new_obj)

    # Save enriched data
    with open("categorized_objects.json", "w") as outfile:
        json.dump(new_data, outfile, indent=2)

    print("Categorization complete. Output saved to categorized_objects.json.")

# Run it
if __name__ == "__main__":
    categorize_objects()
