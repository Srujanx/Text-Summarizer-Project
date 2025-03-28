
#  Text Summarization Project

A machine learning pipeline for automatic text summarization using state-of-the-art NLP models.

##  Features

- **End-to-End Pipeline**: Data ingestion → validation → transformation → training → evaluation
- **Pre-trained Models**: Support for Pegasus, BART, T5, and other transformer models
- **REST API**: FastAPI web service for training and predictions
- **Docker Support**: Containerized for easy deployment
- **Logging**: Detailed pipeline execution tracking

### Prerequisites
- Python 3.8+
- pip
-  Docker for containerization

### Setup
```bash
git clone https://github.com/Srujanx/text-summarizer.git
cd text-summarizer

# Create virtual environment 
python -m venv venv
source venv/bin/activate  Mac


# Install dependencies
pip install -r requirements.txt

# Project Structure
text-summarizer/
├── config/                 # Configuration files
├── data/                   # Raw and processed data
├── models/                 # Saved models
├── research/              # Jupyter notebooks
├── src/                    # Source code
│   ├── pipeline/           # ML pipeline stages
│   ├── logging/            # Logging configuration
│   └── components                 
├── tests/                  # Unit tests
├── Dockerfile              # Container configuration
├── requirements.txt        # Dependencies
└── README.md               # This file
## Workflows

- [x] Update `config.yaml`
- [x] Update `params.yaml`
- [x] Update `entity`
- [x] Update the configuration manager in `src/config`
- [x] Update the `components`
- [x] Update the `pipeline`
- [x] Update the `main.py`
- [x] Update the `app.py`
