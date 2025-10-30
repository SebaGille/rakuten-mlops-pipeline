#!/bin/bash
# Test script to generate sample predictions for Grafana dashboard

echo "🚀 Generating test predictions for Rakuten API..."
echo "This will create metrics for your Grafana dashboard"
echo ""

API_URL="http://localhost:8000"

# Check if API is running
if ! curl -s "$API_URL/health" > /dev/null 2>&1; then
    echo "❌ Error: Rakuten API is not running at $API_URL"
    echo "Start it with: docker-compose -f docker-compose.api.yml up -d"
    exit 1
fi

echo "✅ API is running"
echo ""

# Sample products from different categories
declare -a products=(
    '{"designation": "iPhone 13 Pro", "description": "Apple smartphone with OLED display and 5G connectivity"}'
    '{"designation": "Samsung Galaxy S21", "description": "Android phone with high-resolution camera"}'
    '{"designation": "Harry Potter Book", "description": "Fantasy novel by J.K. Rowling"}'
    '{"designation": "Nike Running Shoes", "description": "Athletic footwear for running and sports"}'
    '{"designation": "Sony PlayStation 5", "description": "Next-gen gaming console"}'
    '{"designation": "MacBook Pro M2", "description": "Laptop computer for professionals"}'
    '{"designation": "Kindle Paperwhite", "description": "E-reader for digital books"}'
    '{"designation": "Adidas Soccer Ball", "description": "Official match ball for football"}'
    '{"designation": "LEGO Star Wars Set", "description": "Building blocks toy set"}'
    '{"designation": "Canon EOS R5", "description": "Professional mirrorless camera"}'
)

echo "📊 Making 10 predictions..."
echo ""

count=0
for product in "${products[@]}"; do
    count=$((count + 1))
    echo -n "[$count/10] Predicting... "
    
    response=$(curl -s -X POST "$API_URL/predict" \
        -H "Content-Type: application/json" \
        -d "$product")
    
    predicted_class=$(echo "$response" | grep -o '"predicted_prdtypecode":[0-9]*' | grep -o '[0-9]*')
    
    if [ -n "$predicted_class" ]; then
        echo "✅ Class: $predicted_class"
    else
        echo "❌ Failed"
    fi
    
    sleep 0.5  # Small delay to spread metrics
done

echo ""
echo "✨ Done! Generated $count predictions"
echo ""
echo "📈 View your metrics:"
echo "  - Grafana:    http://localhost:3000"
echo "  - Prometheus: http://localhost:9090"
echo "  - API Metrics: http://localhost:8000/metrics"
echo ""
echo "💡 Tip: Set Grafana time range to 'Last 5 minutes' to see the fresh data"

