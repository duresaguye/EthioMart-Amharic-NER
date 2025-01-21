Telegram-ECommerce-NER
Overview

This project aims to develop a Named Entity Recognition (NER) system for EthioMart, a centralized e-commerce platform in Ethiopia. The system will extract key business entities such as product names, prices, and locations from Amharic text, images, and documents shared across multiple Telegram channels.
Business Need

EthioMart's vision is to become the primary hub for all Telegram-based e-commerce activities in Ethiopia. By consolidating real-time data from multiple e-commerce Telegram channels, EthioMart aims to provide a seamless experience for customers to explore and interact with multiple vendors in one place.
Key Objectives

    Real-time data extraction from Telegram channels
    Fine-tuning Large Language Models (LLMs) for Amharic Named Entity Recognition
    Extraction of entities such as Product names, Prices, and Locations

Project Structure

The project is divided into the following main tasks:

    Data Ingestion and Preprocessing
    Data Labeling in CoNLL Format
    Fine-tuning NER Models
    Model Comparison and Selection
    Model Interpretability

Folder Structure
EthioMart-NER-LLM/
├── .dvc/                  # DVC configuration files and cache
├── .vscode/               # VS Code configuration files
├── data/                  # Directory for raw and processed data
├── labeled_data/          # Directory for labeled data
├── models/                # Directory for storing models
├── notebooks/             # Jupyter notebooks for experiments and analysis
├── reports/               # Generated reports
├── scripts/               # Python scripts for various tasks
│   ├── data_ingestion/    # Scripts for data ingestion
│   │   └── telegram_scrapper.py
│   ├── labeling/          # Scripts for data labeling
│   │   └── labeling.py
│   └── preprocessing/     # Scripts for data preprocessing
│       └── preprocessing.py
├── tests/                 # Unit tests
├── venv/                  # Python virtual environment
├── .dvcignore             # DVC ignore file
├── .env                   # Environment variables
├── .gitignore             # Git ignore file
├── data.dvc               # DVC data tracking file
├── LICENSE                # License file
├── README.md              # Project README file
├── requirements.txt       # Python dependencies
└── unittests.yml          # GitHub Actions workflow for running unit tests
```

Installation and Setup
Clone this repository:

git clone https://github.com/duresaguye/EthioMart-Ecommerce-Amharic-NER.git
cd Telegram-ECommerce-NER

Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required dependencies:

pip install -r requirements.txt

Data

    Source: Messages and data from Ethiopian-based e-commerce Telegram channel(SINA KIDS/ሲና ኪድስⓇ)
    scraped data: SINA KIDS
    labeled NER dataset: SINA KIDS

Models

We experiment with the following models:

    xlm-roberta-base
    distilbert-base-multilingual-cased
    bert-base-multilingual-cased
