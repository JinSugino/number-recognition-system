import csv

def count_matching_chars(str1, str2):
    return sum(a == b for a, b in zip(str1, str2))

def compare_results(result_accurate_path, result_path, output_path):
    accurate_numbers = []
    result_numbers = []

    with open(result_accurate_path, newline='', encoding='utf-8') as f1:
        reader1 = csv.reader(f1)
        next(reader1)
        for row in reader1:
            if len(row) > 1:
                accurate_numbers.append(row[1].strip())

    with open(result_path, newline='', encoding='utf-8') as f2:
        reader2 = csv.reader(f2)
        next(reader2)
        for row in reader2:
            if len(row) > 1:
                result_numbers.append(row[1].strip())

    match_counts = [count_matching_chars(a, b) for a, b in zip(accurate_numbers, result_numbers)]
    total_chars = sum(len(a) for a in accurate_numbers)
    total_matches = sum(match_counts)
    overall_match_rate = total_matches / total_chars if total_chars > 0 else 0

    with open(output_path, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout)
        writer.writerow(['index', 'accurate_number', 'result_number', 'matching_chars'])
        for i, (a, b, m) in enumerate(zip(accurate_numbers, result_numbers, match_counts)):
            writer.writerow([i+1, a, b, m])
        writer.writerow([])
        writer.writerow(['overall_match_rate', overall_match_rate])

compare_results('result_accurate.csv', 'result.csv', 'compare_output.csv')