# Deployment Guide - Data Breach Analytics Dashboard

**Project:** Data Breach Analytics System  
**Author:** T. Spivey  
**Course:** BUS 761 - Assignment 7  
**Date:** October 24, 2025

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Local Deployment](#local-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Production Deployment](#production-deployment)
6. [Configuration](#configuration)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)
9. [Maintenance](#maintenance)

---

## System Overview

### Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Data Breach Analytics System              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────┐
│   Raw Data      │
│ - Excel (105MB) │
│ - CSV files     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Assignment 4   │
│  ETL Pipeline   │
│ - dataclean.py  │
│ - dataload.py   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ databreach.db   │
│ SQLite (71 MB)  │
│ 35,378 records  │
└────────┬────────┘
         │
         ├──────────────┬──────────────┬──────────────┐
         │              │              │              │
         ▼              ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│Assignment 5  │ │Assignment 6  │ │Assignment 7  │ │   Models     │
│ EDA Package  │ │ ML Engine    │ │  Dashboard   │ │  (Trained)   │
│- Statistics  │ │- Classifiers │ │- Streamlit   │ │- Random      │
│- Analysis    │ │- Predictions │ │- 5 Pages     │ │  Forest      │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
                                          │
                                          ▼
                                  ┌──────────────┐
                                  │   Browser    │
                                  │ localhost:   │
                                  │    8501      │
                                  └──────────────┘
```

### Technology Stack

**Backend:**
- Python 3.8+
- SQLite 3.x
- pandas, numpy, scipy
- scikit-learn 1.2+

**Frontend:**
- Streamlit 1.28+
- Plotly 5.14+
- HTML/CSS (via Streamlit)

**Deployment:**
- Docker 20.10+
- Docker Compose 2.0+

**Database:**
- SQLite (file-based, no server)
- 71 MB database file
- 5 tables (1 main + 4 analytical)

---

## Prerequisites

### System Requirements

**Minimum:**
- CPU: 2 cores
- RAM: 4 GB
- Disk: 500 MB free space
- OS: Windows 10, macOS 10.14+, Linux (Ubuntu 20.04+)

**Recommended:**
- CPU: 4+ cores
- RAM: 8+ GB
- Disk: 1 GB free space
- SSD for faster database access

### Software Requirements

**Required:**
- Python 3.8 or higher
- pip (Python package manager)

**Optional:**
- Docker Desktop (for containerized deployment)
- Git (for version control)

### Installation Check

Verify Python installation:
```bash
python --version
# Should output: Python 3.8.x or higher

pip --version
# Should output: pip 21.x.x or higher
```

---

## Local Deployment

### Method 1: Quick Start (5 minutes)

**Step 1: Navigate to Project Directory**
```bash
cd C:\Users\mcobp\OneDrive\Desktop\DataBreach
```

**Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

This installs:
- pandas, numpy, scipy (data processing)
- scikit-learn (machine learning)
- streamlit, plotly (dashboard)
- matplotlib, seaborn (visualizations)

**Step 3: Verify Database**
```bash
# Check database exists
dir databreach.db  # Windows
ls -lh databreach.db  # Mac/Linux

# Should show: ~71 MB file
```

**Step 4: Run Dashboard**
```bash
streamlit run app.py
```

**Expected Output:**
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.1.x:8501
```

**Step 5: Access Dashboard**

Browser automatically opens to `http://localhost:8501`

If not, manually navigate to that URL.

**Step 6: Verify All Pages Load**

Click through sidebar:
- ✅ Home
- ✅ Data Explorer
- ✅ Statistical Analysis
- ✅ ML Predictions
- ✅ Risk Assessment

### Method 2: Virtual Environment (Recommended for Production)

**Step 1: Create Virtual Environment**
```bash
# Navigate to project
cd C:\Users\mcobp\OneDrive\Desktop\DataBreach

# Create venv
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

**Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 3: Run Dashboard**
```bash
streamlit run app.py
```

**Step 4: Deactivate When Done**
```bash
deactivate
```

### Stopping the Dashboard

**Method 1:** Press `Ctrl+C` in terminal

**Method 2:** Close terminal window

---

## Docker Deployment

### Prerequisites

**Install Docker Desktop:**
- Windows: https://docs.docker.com/desktop/install/windows-install/
- Mac: https://docs.docker.com/desktop/install/mac-install/
- Linux: https://docs.docker.com/engine/install/

**Verify Installation:**
```bash
docker --version
# Output: Docker version 24.x.x or higher

docker-compose --version
# Output: Docker Compose version 2.x.x or higher
```

### Method 1: Docker Compose (Recommended)

**Step 1: Navigate to Project**
```bash
cd C:\Users\mcobp\OneDrive\Desktop\DataBreach
```

**Step 2: Build and Run**
```bash
docker-compose up --build
```

First time: Takes 3-5 minutes to download base image and build.

Subsequent runs: Takes 10-30 seconds.

**Expected Output:**
```
[+] Building 45.2s (12/12) FINISHED
[+] Running 1/1
 ✔ Container breach-analytics-dashboard  Started
breach-analytics-dashboard  | You can now view your Streamlit app in your browser.
breach-analytics-dashboard  | Local URL: http://localhost:8501
```

**Step 3: Access Dashboard**

Navigate to `http://localhost:8501` in browser.

**Step 4: Run in Background (Detached Mode)**
```bash
docker-compose up -d
```

Dashboard runs in background.

**Step 5: View Logs**
```bash
docker-compose logs -f
```

Press `Ctrl+C` to exit logs (container keeps running).

**Step 6: Stop Container**
```bash
docker-compose down
```

### Method 2: Docker Commands (Advanced)

**Step 1: Build Image**
```bash
docker build -t breach-analytics:latest .
```

**Step 2: Run Container**
```bash
docker run -p 8501:8501 \
  -v $(pwd)/databreach.db:/app/databreach.db:ro \
  -v $(pwd)/models:/app/models:ro \
  --name breach-dashboard \
  breach-analytics:latest
```

**Step 3: Stop Container**
```bash
docker stop breach-dashboard
docker rm breach-dashboard
```

### Docker Configuration

**Dockerfile Breakdown:**
```dockerfile
FROM python:3.11-slim        # Lightweight Python image
WORKDIR /app                 # Set working directory
COPY requirements.txt .      # Copy dependencies
RUN pip install -r requirements.txt  # Install packages
COPY . .                     # Copy application code
EXPOSE 8501                  # Expose Streamlit port
CMD ["streamlit", "run", "app.py"]  # Start command
```

**docker-compose.yml Breakdown:**
```yaml
services:
  dashboard:
    build: .                 # Build from Dockerfile
    ports:
      - "8501:8501"          # Map port
    volumes:
      - ./databreach.db:/app/databreach.db:ro  # Mount database (read-only)
      - ./models:/app/models:ro                # Mount models (read-only)
    restart: unless-stopped  # Auto-restart on failure
```

### Docker Best Practices

**Security:**
- Database mounted as read-only (`:ro`)
- Models mounted as read-only
- No sensitive environment variables

**Performance:**
- Slim Python image (smaller size)
- Multi-stage build possible for optimization
- Health checks monitor container status

**Maintenance:**
- Named container for easy management
- Logs accessible via `docker-compose logs`
- Automatic restart on crash

---

## Production Deployment

### Cloud Platform Options

#### Option 1: Streamlit Cloud (Easiest)

**Steps:**
1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect GitHub repository
4. Deploy with one click

**Pros:**
- Free tier available
- Automatic HTTPS
- No infrastructure management

**Cons:**
- Database must be <100 MB (ours is 71 MB ✓)
- Limited compute resources

#### Option 2: AWS EC2

**Steps:**

1. Launch EC2 instance (t3.medium recommended)
2. SSH into instance
3. Clone repository
4. Install Docker
5. Run `docker-compose up -d`
6. Configure security group (port 8501)
7. Access via public IP

**Estimated Cost:** ~$30/month

#### Option 3: Azure App Service

**Steps:**

1. Create App Service (Python 3.11)
2. Deploy via GitHub Actions or Azure CLI
3. Configure startup command: `streamlit run app.py`
4. Access via Azure-provided URL

**Estimated Cost:** ~$15-50/month

#### Option 4: Google Cloud Run

**Steps:**

1. Push Docker image to Google Container Registry
2. Deploy to Cloud Run
3. Configure for HTTP traffic
4. Access via Cloud Run URL

**Pros:**
- Pay per use
- Auto-scaling
- Serverless

**Estimated Cost:** $5-20/month (depending on usage)

### Production Checklist

**Before Deployment:**

- [ ] Test all 5 pages locally
- [ ] Verify database integrity
- [ ] Check all requirements are in requirements.txt
- [ ] Test Docker build locally
- [ ] Review security settings (no hardcoded credentials)
- [ ] Set up monitoring/logging
- [ ] Configure automatic backups for database
- [ ] Document deployment process
- [ ] Set up CI/CD pipeline (optional)

**Security Considerations:**

- [ ] Use environment variables for sensitive config
- [ ] Enable HTTPS (via reverse proxy or platform)
- [ ] Restrict database write access
- [ ] Implement rate limiting (if public)
- [ ] Add authentication (if handling sensitive data)
- [ ] Regular security updates

---

## Configuration

### Environment Variables

Create `.env` file (optional):
```bash
# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Database
DATABASE_PATH=databreach.db

# Models
MODEL_PATH=models/random_forest_model.pkl
```

### Streamlit Configuration

Create `.streamlit/config.toml`:
```toml
[server]
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#262730"
textColor = "#fafafa"
```

### Database Configuration

**Connection String:**
```python
conn = sqlite3.connect('databreach.db')
```

**Read-Only Mode (Recommended for Production):**
```python
conn = sqlite3.connect('file:databreach.db?mode=ro', uri=True)
```

---

## Monitoring

### Health Checks

**Docker Health Check (Configured):**
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

**Manual Health Check:**
```bash
curl http://localhost:8501/_stcore/health
# Should return: {"status": "ok"}
```

### Performance Metrics

**Monitor These:**
- Response time (<2 seconds for page load)
- Memory usage (<500 MB typical)
- CPU usage (<50% typical)
- Database query time (<100ms for most queries)

**Tools:**
- Docker stats: `docker stats breach-analytics-dashboard`
- Streamlit built-in metrics (development mode)
- Cloud platform dashboards (AWS CloudWatch, Azure Monitor, etc.)

### Logging

**View Logs:**
```bash
# Docker Compose
docker-compose logs -f

# Docker
docker logs -f breach-analytics-dashboard

# Local
# Check terminal output where streamlit is running
```

**Log Levels:**
- INFO: Normal operations
- WARNING: Potential issues
- ERROR: Failed operations

---

## Troubleshooting

### Common Issues

#### Issue 1: Port Already in Use

**Error:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Find process using port 8501
# Windows:
netstat -ano | findstr :8501
taskkill /PID  /F

# Mac/Linux:
lsof -ti:8501 | xargs kill -9

# Or use different port:
streamlit run app.py --server.port 8502
```

#### Issue 2: Database Not Found

**Error:**
```
sqlite3.OperationalError: unable to open database file
```

**Solution:**
```bash
# Verify database exists
ls -lh databreach.db

# Check permissions
chmod 644 databreach.db  # Mac/Linux

# Verify path in Docker volume mount
docker-compose down
docker-compose up
```

#### Issue 3: Module Not Found

**Error:**
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or specific package
pip install streamlit plotly
```

#### Issue 4: Docker Build Fails

**Error:**
```
failed to solve with frontend dockerfile.v0
```

**Solution:**
```bash
# Clear Docker cache
docker system prune -a

# Rebuild with no cache
docker-compose build --no-cache
```

#### Issue 5: Slow Performance

**Symptoms:**
- Pages take >5 seconds to load
- Filters lag
- Visualizations don't render

**Solutions:**

1. **Reduce Data Size:**
   - Use date range filters
   - Limit sample size in scatter plots

2. **Optimize Queries:**
   - Add indexes to database
   - Cache repeated queries

3. **Increase Resources:**
   - Docker: Increase memory allocation in Docker Desktop
   - Cloud: Upgrade instance type

4. **Clear Streamlit Cache:**
   - Delete `.streamlit/cache` folder
   - Restart application

---

## Maintenance

### Regular Tasks

**Weekly:**
- [ ] Review error logs
- [ ] Check disk space
- [ ] Monitor performance metrics

**Monthly:**
- [ ] Update dependencies: `pip install --upgrade -r requirements.txt`
- [ ] Review security advisories
- [ ] Backup database
- [ ] Test disaster recovery

**Quarterly:**
- [ ] Refresh data (re-run ETL if new breach data available)
- [ ] Retrain ML models
- [ ] Performance optimization review
- [ ] User feedback review

### Updating the Application

**Update Code:**
```bash
# Pull latest changes
git pull origin main

# Reinstall dependencies (if requirements changed)
pip install -r requirements.txt

# Restart application
# Local:
Ctrl+C, then streamlit run app.py

# Docker:
docker-compose down
docker-compose up --build -d
```

### Backup Strategy

**What to Backup:**
1. Database: `databreach.db` (71 MB)
2. Trained models: `models/*.pkl`
3. Configuration files: `.streamlit/`, `.env`
4. Application code: Entire repository

**Backup Frequency:**
- Database: Daily (if updating data)
- Code: Every commit (via Git)
- Models: After each retraining

**Backup Commands:**
```bash
# Simple backup
cp databreach.db backups/databreach_$(date +%Y%m%d).db

# With compression
tar -czf backup_$(date +%Y%m%d).tar.gz databreach.db models/ .streamlit/

# Automated backup (cron job - Linux/Mac)
# Add to crontab:
0 2 * * * cd /path/to/project && ./backup.sh
```

### Disaster Recovery

**Database Corruption:**
```bash
# Restore from backup
cp backups/databreach_20251024.db databreach.db

# Or re-run ETL pipeline
python dataclean.py
python dataload.py
```

**Application Failure:**
```bash
# Docker: Restart container
docker-compose restart

# Local: Restart Streamlit
Ctrl+C
streamlit run app.py
```

---

## Performance Optimization

### Database Optimization

**Add Indexes (Optional):**
```sql
CREATE INDEX idx_breach_date ON databreach(breach_date);
CREATE INDEX idx_org_type ON databreach(organization_type);
CREATE INDEX idx_breach_type ON databreach(breach_type);
```

**Run in Python:**
```python
import sqlite3
conn = sqlite3.connect('databreach.db')
conn.execute("CREATE INDEX IF NOT EXISTS idx_breach_date ON databreach(breach_date)")
conn.commit()
conn.close()
```

### Caching Strategy

Streamlit caching is already implemented in `app.py`:
```python
@st.cache_data
def load_data():
    # Loads database only once per session
    pass
```

**Cache Efficiency:**
- First page load: 3-5 seconds
- Subsequent page loads: <1 second

### Memory Management

**Typical Memory Usage:**
- Base Streamlit: 100-150 MB
- Loaded data: 200-300 MB
- Plotly visualizations: 50-100 MB
- **Total: ~500 MB**

**Reduce Memory:**
- Sample large datasets (already implemented for scatter plots)
- Limit date ranges in queries
- Clear cache periodically

---

## Security Best Practices

### Application Security

1. **Database Read-Only:**
   - Docker mounts are read-only (`:ro`)
   - No write operations in dashboard

2. **Input Validation:**
   - Streamlit widgets validate inputs
   - Date ranges limited to reasonable values

3. **No Sensitive Data:**
   - No PII in database
   - No hardcoded credentials

### Network Security

**If Deploying Publicly:**

1. **Use HTTPS:**
   - Implement via reverse proxy (nginx, Caddy)
   - Or use platform HTTPS (Streamlit Cloud, AWS, etc.)

2. **Add Authentication:**
   - Streamlit doesn't include auth by default
   - Use OAuth, basic auth, or platform auth

3. **Rate Limiting:**
   - Prevent abuse
   - Implement at reverse proxy level

4. **Firewall Rules:**
   - Restrict access to port 8501
   - Allow only necessary IPs

---

## Support and Contact

### Documentation

- **User Guide:** `USER_GUIDE.md`
- **Model Documentation:** `documentation/model_documentation.md`
- **Performance Evaluation:** `documentation/performance_evaluation.md`

### Resources

- **Streamlit Docs:** https://docs.streamlit.io
- **Plotly Docs:** https://plotly.com/python/
- **Docker Docs:** https://docs.docker.com

### Contact

- **Developer:** T. Spivey
- **Course:** BUS 761 - Python for Business
- **Email:** ts2427@students.jagmail.edu
- **GitHub:** https://github.com/ts2427

---

## Appendix

### Project File Structure
```
DataBreach/
├── app.py                          # Main Streamlit application
├── databreach.db                   # SQLite database (71 MB)
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
├── docker-compose.yml              # Docker Compose configuration
├── README.md                       # Project overview
├── USER_GUIDE.md                   # End user documentation
├── DEPLOYMENT.md                   # This file
├── dataclean.py                    # Assignment 4: Data cleaning
├── dataload.py                     # Assignment 4: Data loading
├── eda.py                          # Assignment 5: Statistical analysis
├── models/                         # Assignment 6: Trained ML models
│   ├── random_forest_model.pkl
│   └── random_forest_model_metadata.json
├── eda_package/                    # Assignment 5: EDA utilities
│   └── data_loader.py
├── analytics_engine/               # Assignment 6: ML engine
│   ├── __init__.py
│   ├── feature_engineer.py
│   ├── model_trainer.py
│   ├── evaluator.py
│   ├── predictor.py
│   └── recommender.py
├── documentation/                  # Detailed documentation
│   ├── analysis_methodology.md
│   ├── cleaning.md
│   ├── data_dictionary.md
│   ├── eda.md
│   ├── ERD.md
│   ├── executive_summary.md
│   ├── loading.md
│   ├── model_documentation.md
│   ├── performance_evaluation.md
│   ├── setup_instructions.md
│   └── sources.md
└── .streamlit/                     # Streamlit configuration
    └── config.toml
```

### Port Reference

- **8501:** Streamlit dashboard (default)
- **8502-8510:** Alternative Streamlit ports

### Commands Quick Reference
```bash
# Local Development
streamlit run app.py
streamlit run app.py --server.port 8502  # Different port

# Docker
docker-compose up                    # Build and run
docker-compose up -d                 # Run in background
docker-compose down                  # Stop and remove
docker-compose logs -f               # View logs
docker-compose restart               # Restart services
docker system prune -a               # Clean up Docker

# Database
sqlite3 databreach.db                # Open SQLite CLI
sqlite3 databreach.db "SELECT COUNT(*) FROM databreach;"  # Query

# Python Environment
python -m venv venv                  # Create virtual environment
venv\Scripts\activate                # Activate (Windows)
source venv/bin/activate             # Activate (Mac/Linux)
pip install -r requirements.txt      # Install dependencies
pip freeze > requirements.txt        # Update requirements
deactivate                           # Deactivate virtual environment
```

---

**Document Version:** 1.0  
**Last Updated:** October 24, 2025  
**Maintained by:** T. Spivey, BUS 761