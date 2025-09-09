import csv
from collections import OrderedDict

def count_matching_chars(str1, str2):
    return sum(a == b for a, b in zip(str1, str2))

def read_csv_map(path):
    # Read CSV into OrderedDict: filename -> number (handle BOM header, skip empty)
    rows = OrderedDict()
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                filename = row[0].strip()
                number = row[1].strip()
                if filename:
                    rows[filename] = number
    return rows

def compare_results(result_accurate_path, result_path, output_path):
    accurate_map = read_csv_map(result_accurate_path)
    result_map = read_csv_map(result_path)

    all_keys = list(OrderedDict.fromkeys(list(accurate_map.keys()) + list(result_map.keys())))

    detailed_rows = []
    total_chars = 0
    total_matches = 0

    for key in all_keys:
        a = accurate_map.get(key)
        b = result_map.get(key)
        if a is None:
            detailed_rows.append([key, "<missing-in-accurate>", b or "<missing>", 0])
            continue
        if b is None:
            detailed_rows.append([key, a, "<missing-in-result>", 0])
            continue
        m = count_matching_chars(a, b)
        detailed_rows.append([key, a, b, m])
        total_chars += len(a)
        total_matches += m

    overall_match_rate = (total_matches / total_chars) if total_chars > 0 else 0.0

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as fout:
        writer = csv.writer(fout)
        writer.writerow(['filename', 'accurate_number', 'result_number', 'matching_chars'])
        writer.writerows(detailed_rows)
        writer.writerow([])
        writer.writerow(['overall_match_rate', overall_match_rate])
        print(f"overall_match_rate: {overall_match_rate}")

if __name__ == '__main__':
    compare_results('result_accurate.csv', 'result.csv', 'compare_output.csv')