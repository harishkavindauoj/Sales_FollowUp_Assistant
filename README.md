# Sales Follow-Up Assistant

An AI-powered sales follow-up assistant built with **LangGraph** and **AWS Bedrock Nova** that analyzes customer purchase behavior and provides actionable recommendations for sales representatives.

## 🚀 Features

- **Customer Analysis**: RFM scoring, churn risk analysis, and priority assessment
- **AI-Powered Summaries**: Natural language summaries of customer behavior
- **Action Recommendations**: Top 3 recommended actions (call, email, bundle offers, promos)
- **Daily Follow-ups**: Prioritized list of customers to contact today
- **Parallel Processing**: RFM and churn analysis run in parallel using LangGraph
- **Cost & Performance Monitoring**: Real-time tracking of Bedrock API costs and latency
- **Circuit Breaker**: 8-second timeout protection for API calls
- **Auto-Retry**: Automatic retry for malformed responses (max 2 retries)
- **Structured Logging**: Comprehensive observability with PII redaction

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   LangGraph     │    │  AWS Bedrock    │
│   REST API      │───▶│   Workflow      │───▶│  Nova Models    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Service  │    │  Analysis Tools │    │  Cost Tracking  │
│   CSV Processing│    │  RFM & Churn    │    │  & Monitoring   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### LangGraph Workflow

```
Fetch Customer Data
         │
         ▼
    ┌─────────┐         ┌─────────────┐
    │   RFM   │◄────────┤  Parallel   │────────►┌─────────────┐
    │Analysis │         │  Execution  │         │    Churn    │
    └─────────┘         └─────────────┘         │  Analysis   │
         │                                      └─────────────┘
         ▼                                              │
   Generate Summary ◄──────────────────────────────────┘
         │
         ▼
  Generate Recommendations
         │
         ▼
   Get Top Follow-ups
         │
         ▼
   Format Final Response
```

## 📋 Requirements Met

✅ **Framework**: LangGraph for workflow orchestration  
✅ **Model Host**: AWS Bedrock Nova family  
✅ **Parallelism**: RFM & churn analysis run in parallel  
✅ **Tooling**: 2+ Python tools (CustomerAnalysisTools, RecommendationTools)  
✅ **Output Schema**: Strict JSON validation with Pydantic  
✅ **Auto-Retry**: Max 2 retries for non-conforming output  
✅ **Latency & Cost**: Comprehensive monitoring and logging  
✅ **Circuit Breaker**: 8-second timeout per LLM call  
✅ **Observability**: Structured logging with PII redaction  
✅ **FastAPI**: `/analyze` and `/top-followups` endpoints  
✅ **Determinism**: Temperature ≤ 0.3, JSON mode enforcement  
✅ **Edge Cases**: Unknown customer, empty history, malformed data  

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sales-followup-assistant
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your AWS credentials
   ```

5. **Create data files**
   ```bash
   mkdir data
   # Add orders.csv and customers.csv to data/ directory
   # Or use the provided sample data (automatically created if files missing)
   ```

## 🔧 Configuration

### Environment Variables

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1

# Application Configuration
LOG_LEVEL=INFO
BEDROCK_TIMEOUT=8
MAX_RETRIES=2
MODEL_TEMPERATURE=0.2
```

### Data Files

Place your CSV files in the `data/` directory:

**orders.csv**
```csv
customer_id,order_id,order_date,sku,qty,price
C001,SO-101,2025-08-20,CAKE-CHOC,3,12.50
```

**customers.csv**
```csv
customer_id,name,segment,territory,credit_terms
C001,Gourmet Gateway,HO.RE.CA,West,NET15
```

## 🚀 Running the Application

### Development
```bash
python run.py
```

### Production
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

### Interactive Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Endpoints

#### 1. Analyze Customer
```bash
POST /analyze
Content-Type: application/json

{
  "customer_id": "C001"
}
```

**Response:**
```json
{
  "customer_id": "C001",
  "scores": {
    "rfm_score": 85,
    "churn_risk": 0.23,
    "priority": 4
  },
  "summary": "Gourmet Gateway is a high-value HO.RE.CA customer with consistent ordering patterns...",
  "recommendations": [
    {
      "action": "call",
      "reason": "High-value customer deserves personal attention to strengthen relationship"
    },
    {
      "action": "offer_bundle",
      "reason": "High AOV customer perfect for premium product bundles"
    },
    {
      "action": "email",
      "reason": "Follow up with product updates and company news"
    }
  ],
  "top_followups_today": ["C001", "C003", "C002", "C004"]
}
```

#### 2. Get Top Follow-ups
```bash
POST /top-followups
Content-Type: application/json

{
  "date": "2025-09-17"
}
```

**Response:**
```json
{
  "date": "2025-09-17",
  "top_followups_today": ["C001", "C003", "C002", "C004"],
  "count": 4
}
```

#### 3. Additional Endpoints
- `GET /` - Health check
- `GET /health` - Detailed health status
- `GET /customers` - List all customers
- `GET /customer/{customer_id}/summary` - Basic customer info

## 🔍 Monitoring & Observability

### Structured Logging
All operations are logged with structured data:

```json
{
  "timestamp": "2025-09-17T10:30:00Z",
  "level": "INFO",
  "node": "analyze_rfm_parallel",
  "customer_id": "C001",
  "latency_seconds": 1.234,
  "tokens_used": 150,
  "cost_estimate_usd": 0.0024
}
```

### Cost Monitoring
- Real-time token usage tracking
- Cost estimation per API call
- Latency monitoring with circuit breaker
- Performance metrics logging

### Error Handling
- Comprehensive error logging
- Fallback responses for failures
- PII redaction in logs
- Graceful degradation

## 🧪 Testing

### Manual Testing
```bash
# Test customer analysis
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"customer_id": "C001"}'

# Test top follow-ups
curl -X POST "http://localhost:8000/top-followups" \
     -H "Content-Type: application/json" \
     -d '{"date": "2025-09-17"}'

# Test health
curl "http://localhost:8000/health"
```

### Example Customers
- `C001`: Gourmet Gateway (HO.RE.CA, West)
- `C002`: Snack Shack (Retail, East)
- `C003`: Daily Delights (Retail, North)
- `C004`: Leaf & Cup (Cafe, South)

## 🏗️ Project Structure

```
sales-followup-assistant/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── models.py               # Pydantic models
│   ├── agent/
│   │   ├── graph.py            # LangGraph implementation
│   │   ├── tools.py            # Agent tools
│   │   └── nodes.py            # Graph nodes
│   ├── services/
│   │   ├── bedrock.py          # AWS Bedrock service
│   │   └── data.py             # Data processing
│   └── utils/
│       └── logger.py           # Structured logging
├── data/
│   ├── orders.csv              # Order data
│   └── customers.csv           # Customer data
├── .env                        # Environment variables
├── requirements.txt            # Dependencies
└── run.py                     # Entry point
```

## 📈 Performance

- **Latency**: ~2-4 seconds per analysis (with parallel processing)
- **Cost**: ~$0.002-0.005 per customer analysis
- **Throughput**: Handles concurrent requests efficiently
- **Reliability**: Circuit breaker prevents hanging requests

## 🔒 Security

- PII redaction in logs
- Environment variable configuration
- Input validation with Pydantic
- Error handling without sensitive data exposure

## 📝 License

MIT License

---

**Built with ❤️ using LangGraph, AWS Bedrock Nova, and FastAPI**