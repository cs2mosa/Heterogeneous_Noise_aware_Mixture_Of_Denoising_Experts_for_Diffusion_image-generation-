import openpyxl

def normalize_arabic(text):
    """Normalizes common Arabic letter variations for accurate searching."""
    if not text:
        return ""
    text = str(text).strip()
    replacements = {'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ى': 'ي', 'ة': 'ه'}
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


def search_by_name(file_path, target_name):
    # read_only=True is critical for speed and low memory usage
    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    sheet = wb.active

    norm_target = normalize_arabic(target_name)

    # Stream through rows
    for row in sheet.iter_rows(values_only=True):
        # Column B (index 1) is 'arabic_name'
        arabic_name = row[1]

        if arabic_name and norm_target in normalize_arabic(arabic_name):
            wb.close()
            return {
                "seating_no": row[0],
                "arabic_name": row[1],
                "total_degree": row[2],
                "student_case_desc": row[3]
            }

    wb.close()
    return None


# --- Usage Example ---
file_name = r"C:\Users\Maha Mamdouh\Desktop\نتيجة ثانوية عامة نظام حديث.xlsx"
search_name = "محمد عبدالعزيز مرجان"

result = search_by_name(file_name, search_name)

if result:
    print("Found Student:")
    print(result)
else:
    print("Student not found.")