import pandas as pd
import re

# Load the dataset
df = pd.read_csv('@sinayelj.json.csv')

# Define sets of entities
LOCATION_ENTITIES = {
    'አድራሻ', '1️ቁጥር1', 'ገርጂ', 'ኢምፔሪያል', 'ከሳሚ',
    'ህንፃ', 'ጎን', 'አልፎዝ', 'ፕላዛ', 'ግራውንድ',
    'ላይ', 'እንደገቡ', 'ያገኙናል', '2️ቁጥር2', '4ኪሎ',
    'ቅድስት', 'ስላሴ', 'ህንፃ', 'ማለትም', 'ከብልፅግና',
    'ዋናፅፈት', 'ቤት', 'ህንፃ', 'በስተ', 'ቀኝ',
    'ባለው', 'አስፓልት', '20ሜትር', 'ዝቅ', 'እንዳሉ',
    'ሀበሻ', 'ኮፊ', 'የሚገኝበት', 'ቀይ', 'ሸክላ',
    'ህንፃ', '2ተኛ', 'ፎቅ', 'ላይ', 'ያገኙናል',
    '3️ቁጥር3', 'ብስራተ', 'ገብርኤል', 'ላፍቶ',
    'ሞል', 'መግቢያው', 'ፊት', 'ለፊት',
    'የሚገኘው', 'የብስራተ', 'ገብርኤል',
    'ቤተ', 'ክርስቲያን', 'ህንፃ', 'አንደኛ',
    'ፎቅ', 'ላይ', 'ደረጃ', 'እንደወጣቹ',
    'በስተግራ', 'በኩል', 'ሱቅ', 'ቁጥር', '-09'
}

PRODUCT_ENTITIES = {
    'ለጠባብ',  'ገላግሌ','ከንፁህ',
    'የሲልከን', 'ጥሬ', 'እቃ', 'የልጆች', 'ማጠቢያ',
    'መታጠፍ', 'መዘርጋት', 'ፈር', 'ውስጡ', 'ኮተን',
    'ኦሪጅናል', 'ማቴሪያል', 'በሳይዙ', 'ትልቅ', 'ለልጆች',
    'ምቹ',  'ኮንፈርተብል', 'ማት', 'ከላይ',
    'ማጫወቻ',  'ከተፈለገ', 'ማጫወቻዉን',
    'መተኛም',  'ጨርቁም',
    'መታጠብ',  'ከለር', 'ብሉ', 'ፒንክ',
    'ነጭ',  'ፉሬዎች', 
    'ለልጆ', 'ይግዙ', 'በባትሪ',
    'የሚሰራ', 'ድምፅ',  'የልጆች',
    'ፖፖ', 'የመዋኛ', 'ገንዳ', 'ትልቅ',
    'ሳይዝ', 'የራሱ', 'መንፊያ', 'ፓንፕ',
    'ለልጆ',
    'እቃ'
}

PRICE_ENTITIES = {'ብር'}

def label_entities(tokens):
    labels = ['O'] * len(tokens)
    i = 0
    while i < len(tokens):
        # Label product entities
        if tokens[i] in PRODUCT_ENTITIES:
            labels[i] = 'B-Product'
            j = i + 1
            while j < len(tokens) and tokens[j] in PRODUCT_ENTITIES:
                labels[j] = 'I-Product'
                j += 1
            i = j
            continue  # Move to the next token after labeling
        # Label price entities
        if tokens[i].isdigit():
            # Check if the next token is "ብር"
            if i + 1 < len(tokens) and tokens[i + 1] == 'ብር':
                labels[i] = 'B-Price'
                labels[i + 1] = 'I-Price'
                i += 2
                continue  # Move to the next token after labeling

        # Label location entities
        if tokens[i] in LOCATION_ENTITIES:
            labels[i] = 'B-LOC'
            j = i + 1
            # Label subsequent tokens as I-LOC if they are also in LOCATION_ENTITIES
            while j < len(tokens) and tokens[j] in LOCATION_ENTITIES:
                labels[j] = 'I-LOC'
                j += 1
            i = j
            continue  # Move to the next token after labeling

        # If no entity is matched, move to the next token
        i += 1

    return labels

def create_conll_format(tokens, labels):
    return '\n'.join(f"{token} {label}" for token, label in zip(tokens, labels))

def process_messages(df, num_messages=50):
    conll_data = []
    for _, row in df.head(num_messages).iterrows():
        message = row['message']
        if pd.isna(message):
            continue  # Skip rows with NaN messages
        # Remove non-Amharic text and links
        message = re.sub(r'http\S+', '', message)  # Remove URLs
        message = re.sub(r'[a-zA-Z]', '', message)  # Remove English letters
        message = re.sub(r'\s+', ' ', message).strip()  # Remove extra spaces
        tokens = message.split()  # Split the message into tokens
        labels = label_entities(tokens)
        conll_data.append(create_conll_format(tokens, labels))
    return '\n\n'.join(conll_data)

def main():
    processed_data_dir = 'data/processed'
    labeled_data_dir = 'data/labeled'
    os.makedirs(labeled_data_dir, exist_ok=True)
    
    all_conll_data = []
    
    for filename in os.listdir(processed_data_dir):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(processed_data_dir, filename))
            conll_output = process_messages(df)
            all_conll_data.append(conll_output)
    
    combined_conll_output = '\n\n'.join(all_conll_data)
    
    # Remove lines with only '#' and 'O'
    combined_conll_output = '\n'.join(line for line in combined_conll_output.split('\n') if line.strip() != '# O')
    
    with open(os.path.join(labeled_data_dir, 'labeled_ner_data.conll'), 'w', encoding='utf-8') as f:
        f.write(combined_conll_output)

if __name__ == "__main__":
    main()