import re

def remove_unwanted_lines_and_emojis(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )

    for line in lines:
        if line.strip() == '# O':
            continue  # Skip lines with '# O'
        cleaned_line = emoji_pattern.sub(r'', line)  # Remove emojis
        cleaned_lines.append(cleaned_line)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)

def main():
    input_file = 'data/labeled/labeled_ner_data.conll'
    output_file = 'data/labeled/cleaned_labeled_ner_data.conll'
    remove_unwanted_lines_and_emojis(input_file, output_file)

if __name__ == "__main__":
    main()