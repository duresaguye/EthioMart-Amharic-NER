Telegram E-commerce Data Pipeline
Overview

This project involves scraping, preprocessing, and labeling data from Ethiopian-based Telegram e-commerce channels. The collected data is prepared for Named Entity Recognition (NER) tasks, focusing on extracting entities such as products, prices, and locations in Amharic text.
Features

    Data Ingestion:
        Fetch messages from multiple Telegram channels using the Telethon API.
        Collect text, timestamps, and sender metadata in real time.

    Data Preprocessing:
        Clean and normalize Amharic text for consistency.
        Structure data into CSV format for easier analysis.

    Dataset Labeling:
        Annotate a subset of messages in CoNLL format for NER training.
        Identify and label entities: products, prices, and locations.
