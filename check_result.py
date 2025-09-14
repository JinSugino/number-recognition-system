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

def compare_models(result_accurate_path, sklearn_path, pytorch_path, output_path):
    """両モデルの精度を比較"""
    accurate_map = read_csv_map(result_accurate_path)
    sklearn_map = read_csv_map(sklearn_path)
    pytorch_map = read_csv_map(pytorch_path)
    
    all_keys = list(OrderedDict.fromkeys(list(accurate_map.keys()) + list(sklearn_map.keys()) + list(pytorch_map.keys())))
    
    detailed_rows = []
    sklearn_total_chars = 0
    sklearn_total_matches = 0
    pytorch_total_chars = 0
    pytorch_total_matches = 0
    
    for key in all_keys:
        accurate = accurate_map.get(key)
        sklearn = sklearn_map.get(key)
        pytorch = pytorch_map.get(key)
        
        if accurate is None:
            continue
            
        sklearn_matches = count_matching_chars(accurate, sklearn) if sklearn else 0
        pytorch_matches = count_matching_chars(accurate, pytorch) if pytorch else 0
        
        detailed_rows.append([key, accurate, sklearn or "<missing>", pytorch or "<missing>", sklearn_matches, pytorch_matches])
        
        sklearn_total_chars += len(accurate)
        sklearn_total_matches += sklearn_matches
        pytorch_total_chars += len(accurate)
        pytorch_total_matches += pytorch_matches
    
    sklearn_accuracy = (sklearn_total_matches / sklearn_total_chars) if sklearn_total_chars > 0 else 0.0
    pytorch_accuracy = (pytorch_total_matches / pytorch_total_chars) if pytorch_total_chars > 0 else 0.0
    
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as fout:
        writer = csv.writer(fout)
        writer.writerow(['filename', 'accurate', 'sklearn_result', 'pytorch_result', 'sklearn_matches', 'pytorch_matches'])
        writer.writerows(detailed_rows)
        writer.writerow([])
        writer.writerow(['sklearn_accuracy', sklearn_accuracy])
        writer.writerow(['pytorch_accuracy', pytorch_accuracy])
        writer.writerow(['improvement', pytorch_accuracy - sklearn_accuracy])
    
    print(f"scikit-learn精度: {sklearn_accuracy:.4f} ({sklearn_accuracy*100:.1f}%)")
    print(f"PyTorch精度: {pytorch_accuracy:.4f} ({pytorch_accuracy*100:.1f}%)")
    print(f"改善幅: {pytorch_accuracy - sklearn_accuracy:.4f} ({(pytorch_accuracy - sklearn_accuracy)*100:.1f}%)")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'pytorch':
        # PyTorchモデルの結果を比較
        print("=== PyTorchモデルの精度確認 ===")
        compare_results('result_accurate.csv', 'result_pytorch.csv', 'compare_output_pytorch.csv')
    elif len(sys.argv) > 1 and sys.argv[1] == 'compare':
        # 両モデルの比較
        print("=== 両モデルの精度比較 ===")
        compare_models('result_accurate.csv', 'result.csv', 'result_pytorch.csv', 'model_comparison_detailed.csv')
    else:
        # 従来のscikit-learnモデルの結果を比較
        print("=== scikit-learnモデルの精度確認 ===")
        compare_results('result_accurate.csv', 'result.csv', 'compare_output.csv')