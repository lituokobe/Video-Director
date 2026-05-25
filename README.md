# AI Director Service 🎬

> Intelligent Short Video Creation Director Service | Compliance Check · Script Optimization · Smart Material Selection · Solution Recommendation

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Async-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-1.2+-orange.svg)](https://python.langchain.com/)
[![LLM](https://img.shields.io/badge/LLM-Qwen3.6--Series-purple.svg)](https://qwenlm.github.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

---

## 🎯 Product Overview

**AI Director Service** is an intelligent creation engine designed for short video marketing scenarios, dedicated to helping brands, content operation teams, and marketing platforms efficiently produce high-quality short video content.

By deeply integrating the **Qwen3.6-series Large Language Model** with industry knowledge rules, this service can automatically complete the full-chain decision-making process of "Script Compliance → Content Optimization → Material Matching → Solution Output", significantly reducing manual creation costs and improving content production efficiency and compliance safety.

### 🎁 Core Value

| Value Dimension | Specific Benefits |
|----------------|-----------------|
| ⚡ Efficiency Improvement | Single short video solution generation reduced from hours to seconds, supporting batch task parallel processing |
| 🛡️ Compliance Assurance | Built-in industry sensitive word library and expression standards, automatically intercepting risky scripts and reducing review rework rates |
| 🎨 Creative Empowerment | Based on large model understanding of event intent, intelligently matching video/templates/voiceovers/BGM to inspire creative combinations |
| 📊 Controllable Strategy | Supports business parameter configuration such as template strategy, retry mechanism, video count, flexibly adapting to different scenarios |
| 🔌 Seamless Integration | Standard RESTful API design, easily connecting to content middle platforms, marketing automation systems, or third-party platforms |

---

## ✨ Core Capabilities

### 🔍 Script Compliance Check
Automatically identifies typos, non-standard expressions, sensitive words, and industry red-line content in scripts, returning clear violation reasons and correction suggestions to help content go live with "zero risk".

### ✨ Intelligent Script Optimization
Based on Qwen3.6-series' semantic understanding and generation capabilities, performs style polishing, expression enhancement, and scenario adaptation on original scripts, making content more attractive and conversion-effective.

### 🎞️ AI Smart Material Selection
Input basic event/activity information (city, name, time, location, industry), automatically recommend matching video clips, cover templates, background music, voiceover roles, and reference scripts from massive material libraries, achieving "input requirements, output solutions".

### 🎁 One-Stop Solution Recommendation
Combining user preferences, historical data, and business strategies, outputs multiple structured short video generation solutions, including opening videos, regular clips, visual templates, audio configuration, and final scripts, directly connecting to video synthesis engines for end-to-end automated production.

---

## 🏗️ Technical Highlights

- **Large Model Driven**: Built on Qwen3.6-series for inference and generation core, balancing Chinese understanding depth with generation quality
- **Async High Performance**: FastAPI + async/await architecture, single instance supports high concurrent requests with low response latency
- **Flexible Orchestration**: LangChain enables modular AI decision-making processes, facilitating future capability expansion or strategy adjustments
- **Containerized Deployment**: Docker image delivery, supports Kubernetes orchestration, adaptable to private/hybrid cloud deployment scenarios
- **Observability Friendly**: Structured logging + health check interface, facilitating monitoring, alerting, and troubleshooting

---

## 🚀 Quick Start

### 1. Service Startup
```bash
# Docker method (recommended)
docker compose up -d

# Or local development startup
uvicorn main:app --host 0.0.0.0 --port 8014
```
### 2. Health Verification
```bash
curl http://localhost:8014/health
# Returns: {"status": "healthy", "version": "1.0.0", ...}
```
### 3. Call Business Interfaces
For complete interface definitions, request/response examples, and field descriptions, please refer to:
📄 [API Documentation](./API_doc.md)

> 💡 Recommendation: First obtain candidate materials via `/select_material`, then combine calls to `/recommend` to generate the final solution.

---

## 📦 Typical Application Scenarios

### 🎪 Exhibition/Event Marketing
- **Input**: Exhibition name, time, location, industry, promotional highlights
- **Output**: Multiple short video solutions, including opening video + highlight clips + promotional scripts + matching BGM
- **Value**: Rapidly batch-generate event preheating/on-site recap/recruitment promotion videos

### 🛍️ E-commerce Promotion Short Videos
- **Input**: Product selling points, promotional information, target audience
- **Output**: Compliance-optimized voiceover scripts + matching product display videos + rhythmic BGM
- **Value**: Improve product video production efficiency, ensure script compliance, reduce delivery risks

### 🏢 Enterprise Brand Content Production
- **Input**: Brand tone, communication themes, material library permissions
- **Output**: Series of short video solutions compliant with brand standards, supporting multi-style/multi-role voiceovers
- **Value**: Unify brand expression, scale production of high-quality content