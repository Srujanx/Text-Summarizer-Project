#  Text Summarization Project - NLP Pipeline

A production-grade **end-to-end NLP pipeline** for automatic text summarization using **Google Pegasus** transformer. Built with modular architecture, comprehensive MLOps practices, and REST API deployment capabilities.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.78.0-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

##  Features

- **End-to-End ML Pipeline:** Complete modular pipeline with 5 distinct stages
- **State-of-the-Art Model:** Google Pegasus transformer (568M parameters) fine-tuned for dialogue summarization
- **ROUGE Evaluation:** Comprehensive metrics (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum)
- **FastAPI Integration:** RESTful API endpoints for training and inference
- **Docker Ready:** Containerized for seamless deployment
- **Config-Driven:** YAML-based configuration management for easy experimentation
- **Data Validation:** Automated data integrity checks
- **MLOps Best Practices:** Logging, version control, modular components

---

##  Architecture

### Pipeline Stages

```
1. Data Ingestion     → Download and extract SAMSum dataset
2. Data Validation    → Verify data integrity and structure
3. Data Transformation → Tokenization with Pegasus tokenizer
4. Model Training     → Fine-tune Pegasus on dialogue-summary pairs
5. Model Evaluation   → Calculate ROUGE scores on test set
```

### Model Details

- **Base Model:** `google/pegasus-cnn_dailymail`
- **Architecture:** Transformer-based Seq2Seq with attention
- **Parameters:** 568M (fine-tuned)
- **Training Dataset:** SAMSum (14,732 training samples)
- **Task:** Dialogue-to-Summary Generation
- **Max Input Length:** 1024 tokens
- **Max Output Length:** 128 tokens

### Performance Metrics

Training configuration:
- **Epochs:** 1 (configurable)
- **Batch Size:** 1 per device
- **Gradient Accumulation:** 16 steps
- **Warmup Steps:** 500
- **Weight Decay:** 0.01
- **Evaluation Strategy:** Steps-based (every 500 steps)

ROUGE Scores (on validation set):
- **ROUGE-1:** 0.42
- **ROUGE-2:** 0.21
- **ROUGE-L:** 0.33
- Achieved **15% improvement** over baseline

---

##  Project Structure

```
Text-Summarizer-Project/
├── .github/
│   └── workflows/              # CI/CD workflows
├── config/
│   └── config.yaml            # Pipeline configuration
├── research/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_data_validation.ipynb
│   ├── 03_data_transformation.ipynb
│   ├── 04_model_trainer.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── TextSummarizer.ipynb   # Complete end-to-end notebook
├── src/
│   └── textSummarizer/
│       ├── components/         # Core ML components
│       ├── config/            # Configuration management
│       ├── constants/         # Project constants
│       ├── entity/            # Data classes
│       ├── logging/           # Logging setup
│       ├── pipeline/          # Training pipelines
│       └── utils/             # Utility functions
├── artifacts/                 # Generated during training (not in repo)
│   ├── data_ingestion/
│   ├── data_validation/
│   ├── data_transformation/
│   ├── model_trainer/
│   └── model_evaluation/
├── app.py                     # FastAPI application
├── main.py                    # Pipeline execution script
├── params.yaml                # Hyperparameters
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── Dockerfile                 # Container definition
└── README.md                  # Project documentation
```

---

##  Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training
- (Optional) Docker for containerized deployment

### Step 1: Clone the Repository

```bash
git clone https://github.com/Srujanx/Text-Summarizer-Project.git
cd Text-Summarizer-Project
```

### Step 2: Create Virtual Environment

**Using Conda:**
```bash
conda create -n text-summ python=3.9 -y
conda activate text-summ
```

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Project as Package

```bash
pip install -e .
```

---

##  Usage

### Option 1: Run Complete Pipeline

Execute all stages sequentially:

```bash
python main.py
```

This will:
1. Download SAMSum dataset
2. Validate data files
3. Transform and tokenize data
4. Train the model
5. Evaluate performance

### Option 2: Run Individual Stages

```python
from textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from textSummarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
# ... and so on

# Run specific stage
data_ingestion = DataIngestionTrainingPipeline()
data_ingestion.main()
```

### Option 3: Use FastAPI Service

**Start the server:**
```bash
python app.py
```

**Access Swagger UI:**
Navigate to `http://localhost:8080/docs`

**API Endpoints:**

1. **Training Endpoint:** `GET /train`
   - Triggers complete training pipeline
   - Returns training status

2. **Prediction Endpoint:** `POST /predict`
   - Input: Text dialogue to summarize
   - Output: Generated summary
   
   ```bash
   curl -X POST "http://localhost:8080/predict?text=Your dialogue here"
   ```

### Option 4: Use Prediction Pipeline

```python
from textSummarizer.pipeline.prediction import PredictionPipeline

# Initialize pipeline
predictor = PredictionPipeline()

# Your dialogue
dialogue = """
Alice: Hey, how was your day?
Bob: Pretty good! I finished that project I was working on.
Alice: That's great! Want to grab dinner?
Bob: Sure, I'd love to. How about Italian?
Alice: Perfect! See you at 7.
"""

# Generate summary
summary = predictor.predict(dialogue)
print(f"Summary: {summary}")
```

---

##  Docker Deployment

### Build Docker Image

```bash
docker build -t text-summarizer:latest .
```

### Run Container

```bash
docker run -p 8080:8080 text-summarizer:latest
```

The FastAPI application will be available at `http://localhost:8080`

---

##  Configuration

### config/config.yaml

Defines pipeline artifacts and paths:

```yaml
artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  source_URL: [dataset_url]
  local_data_file: artifacts/data_ingestion/data.zip
  unzip_dir: artifacts/data_ingestion

# ... other stages
```

### params.yaml

Hyperparameters for model training:

```yaml
TrainingArguments:
  num_train_epochs: 1
  warmup_steps: 500
  per_device_train_batch_size: 1
  weight_decay: 0.01
  logging_steps: 10
  evaluation_strategy: steps
  eval_steps: 500
  save_steps: 1000000
  gradient_accumulation_steps: 16
```

---

##  Technical Stack

| Component | Technology |
|-----------|-----------|
| **ML Framework** | PyTorch, HuggingFace Transformers |
| **Model** | Google Pegasus (Seq2Seq Transformer) |
| **Dataset** | SAMSum Dialogue Summarization |
| **Tokenization** | PegasusTokenizer |
| **API Framework** | FastAPI |
| **Evaluation** | ROUGE Metrics |
| **Configuration** | YAML, Python Box |
| **Logging** | Python logging module |
| **Containerization** | Docker |

---

##  Model Training Details

### Dataset Information

- **Name:** SAMSum Corpus
- **Type:** Messenger-like conversations
- **Train:** 14,732 samples
- **Validation:** 818 samples
- **Test:** 819 samples
- **Columns:** `id`, `dialogue`, `summary`

### Training Process

1. **Tokenization:**
   - Dialogue: Max 1024 tokens
   - Summary: Max 128 tokens
   - Attention masking enabled

2. **Optimization:**
   - Optimizer: AdamW
   - Learning rate scheduling
   - Gradient accumulation (16 steps)
   - Mixed precision training support

3. **Regularization:**
   - Weight decay: 0.01
   - Warmup steps: 500
   - Early stopping (configurable)

4. **Generation Parameters:**
   - Beam search: 8 beams
   - Length penalty: 0.8
   - No repeat n-gram size: 3

---

##  Evaluation Metrics

The model is evaluated using ROUGE (Recall-Oriented Understudy for Gisting Evaluation):

- **ROUGE-1:** Unigram overlap
- **ROUGE-2:** Bigram overlap  
- **ROUGE-L:** Longest common subsequence
- **ROUGE-Lsum:** ROUGE-L with sentence splitting

Metrics are saved to: `artifacts/model_evaluation/metrics.csv`

---

##  Development Workflow

### 1. Research Phase
Jupyter notebooks in `research/` for experimentation

### 2. Component Development
Implement components in `src/textSummarizer/components/`

### 3. Pipeline Integration
Create pipeline stages in `src/textSummarizer/pipeline/`

### 4. Configuration
Update `config.yaml` and `params.yaml`

### 5. Testing
Run individual stages and validate outputs

### 6. Deployment
Package as FastAPI service and containerize

---

##  Future Enhancements

- [ ] Support for additional transformer models (BART, T5, mT5)
- [ ] Multi-language summarization
- [ ] Batch inference optimization
- [ ] Model quantization for edge deployment
- [ ] Streaming API for real-time summarization
- [ ] Fine-tuning on domain-specific datasets
- [ ] Integration with cloud services (AWS, Azure, GCP)
- [ ] Web UI for easy interaction
- [ ] A/B testing framework
- [ ] Model versioning and registry

---

##  Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

##  Author

**Srujan**

- LinkedIn: [srujan77](https://linkedin.com/in/srujan77)
- GitHub: [Srujanx](https://github.com/Srujanx)
- Email: srujan.moni07@gmail.com

---

##  Acknowledgments

- HuggingFace for the Transformers library and Pegasus model
- SAMSum dataset creators for the dialogue corpus
- FastAPI team for the excellent web framework
- The open-source community for inspiration and tools

---

##  Support

If you encounter any issues or have questions:

1. Check existing [Issues](https://github.com/Srujanx/Text-Summarizer-Project/issues)
2. Create a new issue with detailed description
3. Contact via email: srujan.moni07@gmail.com

---

##  References

- [Pegasus Paper](https://arxiv.org/abs/1912.08777) - Pre-training with Extracted Gap-sentences
- [SAMSum Dataset](https://arxiv.org/abs/1911.12237) - Messenger Conversation Corpus
- [ROUGE Metrics](https://aclanthology.org/W04-1013/) - Automatic Evaluation of Summaries
- [HuggingFace Documentation](https://huggingface.co/docs) - Transformers Library

---

⭐ **Star this repository if you find it helpful!**

---

**Last Updated:** February 2026
