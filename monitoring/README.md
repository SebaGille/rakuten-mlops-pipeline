# Rakuten API Monitoring Setup

This directory contains the monitoring configuration for the Rakuten Product Classification API using Prometheus and Grafana.

## 📁 Files

- **`prometheus.yml`** - Prometheus scrape configuration
- **`grafana-dashboard-rakuten.json`** - Pre-built Grafana dashboard (ready to import)
- **`grafana-provisioning-datasources.yml`** - Auto-provision Prometheus data source
- **`grafana-provisioning-dashboards.yml`** - Auto-provision dashboards
- **`README.md`** - This file

## 🚀 Quick Start

### Method 1: Manual Import (Quickest)

1. Start the monitoring stack:
```bash
docker-compose -f docker-compose.monitor.yml up -d
```

2. Open Grafana: http://localhost:3000
   - Username: `seba`
   - Password: `sebamlops`

3. Import the dashboard:
   - Click **☰** → **Dashboards** → **New** → **Import**
   - Upload: `monitoring/grafana-dashboard-rakuten.json`
   - Select Prometheus as data source
   - Click **Import**

### Method 2: Auto-Provisioning (Production Ready)

The dashboard is automatically loaded when Grafana starts!

1. Restart Grafana to apply provisioning:
```bash
docker-compose -f docker-compose.monitor.yml restart grafana
```

2. Open Grafana: http://localhost:3000

3. Navigate to: **Dashboards** → **MLOps** folder → **Rakuten API Monitoring**

## 📊 Dashboard Panels

The dashboard includes 9 panels:

1. **Total Predictions by Class** - Cumulative prediction counter per class
2. **Request Rate** - Predictions per second
3. **P50 Latency** - Median latency (50th percentile)
4. **P95 Latency** - 95th percentile latency
5. **P99 Latency** - 99th percentile latency
6. **Average Text Length** - Average characters in input text
7. **Prediction Distribution** - Pie chart of class distribution
8. **Latency Distribution Over Time** - P50/P95/P99/Average comparison
9. **Prediction Rate by Class** - Stacked area chart

## 🎯 Accessing the Services

- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090
- **Rakuten API**: http://localhost:8000
- **API Metrics**: http://localhost:8000/metrics
- **API Health**: http://localhost:8000/health

## 🔧 Available Metrics

### Custom Rakuten Metrics

- `rakuten_predictions_total` - Counter of predictions by class (label: `prdtypecode`)
- `rakuten_prediction_latency_seconds` - Histogram of prediction latency
- `rakuten_text_len_chars` - Histogram of input text length

### Standard Prometheus Metrics

- `process_*` - Process-level metrics (CPU, memory, file descriptors)
- `python_*` - Python runtime metrics (GC, threads)

## 📈 Useful Prometheus Queries

### Request Rate
```promql
# Total requests per second
sum(rate(rakuten_predictions_total[1m]))

# Requests per second by class
rate(rakuten_predictions_total[1m])
```

### Latency
```promql
# P50 (median)
histogram_quantile(0.5, rate(rakuten_prediction_latency_seconds_bucket[5m]))

# P95
histogram_quantile(0.95, rate(rakuten_prediction_latency_seconds_bucket[5m]))

# P99
histogram_quantile(0.99, rate(rakuten_prediction_latency_seconds_bucket[5m]))

# Average latency
rate(rakuten_prediction_latency_seconds_sum[5m]) / rate(rakuten_prediction_latency_seconds_count[5m])
```

### Distribution
```promql
# Total predictions by class
sum by (prdtypecode) (rakuten_predictions_total)

# Percentage by class
sum by (prdtypecode) (rakuten_predictions_total) / ignoring(prdtypecode) group_left sum(rakuten_predictions_total) * 100
```

### Text Length
```promql
# Average text length
rate(rakuten_text_len_chars_sum[5m]) / rate(rakuten_text_len_chars_count[5m])

# P95 text length
histogram_quantile(0.95, rate(rakuten_text_len_chars_bucket[5m]))
```

## 🧪 Generate Test Data

To see the dashboard in action, generate some predictions:

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "designation": "iPhone 13 Pro",
    "description": "Apple smartphone with OLED screen"
  }'

# Multiple predictions (bash loop)
for i in {1..10}; do
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d "{\"designation\": \"Product $i\", \"description\": \"Test description $i\"}"
  sleep 1
done
```

## 🎨 Customizing the Dashboard

### Edit Existing Panels
1. Click on the panel title
2. Select **Edit**
3. Modify the query or visualization
4. Click **Apply**
5. Save the dashboard (💾 icon)

### Add New Panels
1. Click **Add** → **Visualization**
2. Write your PromQL query
3. Choose visualization type
4. Configure thresholds, units, etc.
5. Click **Apply**

### Export Your Changes
1. Click **⚙️** (dashboard settings)
2. Select **JSON Model**
3. Copy the JSON
4. Save to a file

## 🔍 Troubleshooting

### No data in Grafana
1. Check Prometheus targets: http://localhost:9090/targets
   - Should show `rakuten_api` as **UP**
2. Verify API is running: `docker ps | grep rakuten_api`
3. Check metrics endpoint: `curl http://localhost:8000/metrics`
4. Ensure time range in Grafana covers when predictions were made

### Dashboard not auto-loaded
1. Check Grafana logs: `docker logs sep25_cmlops_rakuten-grafana-1`
2. Verify provisioning files are mounted: `docker exec sep25_cmlops_rakuten-grafana-1 ls -la /etc/grafana/provisioning/`
3. Restart Grafana: `docker-compose -f docker-compose.monitor.yml restart grafana`

### Metrics not updating
1. Check Prometheus scrape interval (default: 15s)
2. Verify Prometheus is scraping: http://localhost:9090/graph → Enter `up{job="rakuten_api"}`
3. Make a new prediction to generate fresh data

## 📚 Additional Resources

- [Prometheus Query Language](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Grafana Dashboard Guide](https://grafana.com/docs/grafana/latest/dashboards/)
- [FastAPI Prometheus Integration](https://github.com/trallnag/prometheus-fastapi-instrumentator)
- [Grafana Community Dashboards](https://grafana.com/grafana/dashboards/)

## 🏷️ Tags

- `monitoring`
- `observability`
- `mlops`
- `prometheus`
- `grafana`
- `fastapi`

