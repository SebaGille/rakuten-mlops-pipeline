"""Live Prediction Interface Page"""
import streamlit as st
from pathlib import Path
import sys
import time
from PIL import Image
import io
import pandas as pd

# Add parent directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.utils.prediction_manager import PredictionManager
from streamlit_app.utils.constants import PRODUCT_CATEGORIES, API_URL

st.set_page_config(
    page_title="Predictions - Rakuten MLOps",
    page_icon="",
    layout="wide"
)

# Initialize prediction manager
@st.cache_resource
def get_prediction_manager():
    return PredictionManager(API_URL, PROJECT_ROOT)

prediction_manager = get_prediction_manager()

# Header
st.title("Live Prediction Interface")
st.markdown("Make real-time product category predictions using the deployed model")
st.markdown("---")

# Check API health
health_result = prediction_manager.check_api_health()
# Ensure we always get a tuple (defensive check for compatibility)
if isinstance(health_result, tuple) and len(health_result) == 2:
    api_healthy, error_msg = health_result
else:
    # Fallback: if function returns something unexpected, treat as unhealthy
    api_healthy = False
    error_msg = "API health check returned unexpected result"
if not api_healthy:
    st.error("⚠️ Prediction API is not accessible")
    st.info(f"**Expected API at:** `{API_URL}`")
    if error_msg:
        with st.expander("Error Details", expanded=True):
            st.text(error_msg)
    st.info("""
    **Troubleshooting Steps:**
    1. Check the ** Infrastructure** page to see if the API service is running
    2. For AWS deployments, verify the ECS service is healthy
    3. Check that the API_HOST environment variable is set correctly
    4. Verify network connectivity to the ALB
    """)
    st.stop()
else:
    st.success(f"✅ Connected to Prediction API: {API_URL}")

st.markdown("---")

# Main layout: Input and Results
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Product Information")
    
    # Text inputs
    designation = st.text_input(
        "Product Designation (Title):",
        placeholder="e.g., Sony PlayStation 5 Console Digital Edition",
        help="Enter the product title or name"
    )
    
    description = st.text_area(
        "Product Description:",
        placeholder="e.g., Next-gen gaming console with ultra-fast SSD, ray tracing, and 4K support...",
        height=150,
        help="Enter a detailed product description"
    )
    
    # Image upload
    st.markdown("---")
    st.markdown("### Product Image (Optional)")
    
    uploaded_image = st.file_uploader(
        "Upload product image:",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a product image (JPEG or PNG format)"
    )
    
    if uploaded_image:
        # Display uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Validate image
        image_bytes = uploaded_image.getvalue()
        is_valid, validation_msg = prediction_manager.validate_image(image_bytes)
        
        if is_valid:
            st.success(f"✅ {validation_msg}")
        else:
            st.error(f"❌ {validation_msg}")
    
    st.markdown("---")
    
    # Example inputs
    with st.expander("Use example inputs"):
        example_choice = st.selectbox(
            "Select an example:",
            [
                "Select...",
                "Video Game",
                "Book",
                "Kitchen Appliance",
                "Children's Toy",
                "Furniture"
            ]
        )
        
        if example_choice == "Video Game":
            if st.button("Load Video Game Example"):
                st.session_state.example_designation = "The Legend of Zelda: Breath of the Wild"
                st.session_state.example_description = "Explore the vast open world of Hyrule in this critically acclaimed adventure game. Features stunning graphics, dynamic weather, and challenging puzzles."
                st.rerun()
        elif example_choice == "Book":
            if st.button("Load Book Example"):
                st.session_state.example_designation = "The Midnight Library: A Novel"
                st.session_state.example_description = "Between life and death there is a library, and within that library, the shelves go on forever. A powerful novel about infinite choices and second chances."
                st.rerun()
        elif example_choice == "Kitchen Appliance":
            if st.button("Load Kitchen Example"):
                st.session_state.example_designation = "Ninja Air Fryer Max XL"
                st.session_state.example_description = "5.5-quart air fryer with Max Crisp Technology. Cook healthier meals with little to no oil. 450°F max heat. Ceramic-coated basket."
                st.rerun()
        elif example_choice == "Children's Toy":
            if st.button("Load Toy Example"):
                st.session_state.example_designation = "LEGO Star Wars Millennium Falcon"
                st.session_state.example_description = "Ultimate collector's edition with 7,541 pieces. Build the iconic ship with intricate details, rotating gun turrets, and detailed interior."
                st.rerun()
        elif example_choice == "Furniture":
            if st.button("Load Furniture Example"):
                st.session_state.example_designation = "Modern Velvet Accent Chair"
                st.session_state.example_description = "Mid-century modern design armchair with soft velvet upholstery, gold metal legs, and comfortable cushioning. Perfect for living room or bedroom."
                st.rerun()
    
    # Use example values if set
    if 'example_designation' in st.session_state:
        designation = st.session_state.example_designation
        description = st.session_state.example_description
    
    st.markdown("---")
    
    # Prediction button
    predict_button = st.button("PREDICT CATEGORY", type="primary")

with col2:
    st.markdown("### Prediction Results")
    
    if predict_button:
        if not designation or len(designation) < 3:
            st.error("⚠️ Please enter a product designation (at least 3 characters)")
        else:
            with st.spinner("🔄 Making prediction..."):
                start_time = time.time()
                
                # Make prediction
                image_bytes = uploaded_image.getvalue() if uploaded_image else None
                result = prediction_manager.predict(designation, description, image_bytes)
                
                prediction_time = time.time() - start_time
            
            if result and not result.get('error'):
                # Display prediction
                predicted_class = result.get('predicted_class')
                category_name = PRODUCT_CATEGORIES.get(predicted_class, f"Unknown Category ({predicted_class})")
                
                st.success("✅ Prediction Complete!")
                
                # Main prediction result
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 2rem; border-radius: 10px; color: white; text-align: center;'>
                    <h2 style='margin: 0; color: white;'> {category_name}</h2>
                    <p style='font-size: 1.5rem; margin: 0.5rem 0;'>Category Code: {predicted_class}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Prediction Time", f"{prediction_time:.3f}s")
                with col_b:
                    st.metric("Model Status", "✅ Production")
                
                st.markdown("---")
                
                # Confidence scores (if available)
                if 'confidence' in result:
                    st.markdown("#### Confidence Scores")
                    st.info("Top 3 predictions:")
                    # This would show top 3 predictions if API returns them
                    # For now, placeholder
                    confidence_data = result.get('confidence', {})
                    for i, (cat, conf) in enumerate(confidence_data.items(), 1):
                        st.progress(conf, text=f"{i}. {PRODUCT_CATEGORIES.get(cat, cat)}: {conf:.2%}")
                
                # Input summary
                with st.expander("View input details"):
                    st.markdown(f"""
                    **Designation:** {designation}
                    
                    **Description:** {description or "N/A"}
                    
                    **Image:** {"Provided" if uploaded_image else "Not provided"}
                    
                    **Text Length:** {len(designation) + len(description or '')} characters
                    """)
            
            else:
                st.error("❌ Prediction failed")
                st.error(result.get('message', 'Unknown error'))
                
                with st.expander("🔍 Debug information"):
                    st.json(result)
    
    else:
        # Show placeholder
        st.info("Fill in the product information and click 'PREDICT CATEGORY' to see results")
        
        # Show example of what results look like
        st.markdown("#### Expected Output:")
        st.markdown("""
        When you make a prediction, you'll see:
        - **Predicted Category** (with confidence)
        - **Prediction Time**
        - **Top 3 Category Predictions** (if available)
        - **Input Summary**
        """)

# Below the main columns
st.markdown("---")

# Prediction History
st.markdown("### Recent Predictions")

show_history = st.checkbox("Show prediction history", value=True)

if show_history:
    history = prediction_manager.get_prediction_history(limit=20)
    
    if not history.empty:
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        
        stats = prediction_manager.get_prediction_statistics()
        
        with col1:
            st.metric("Total Predictions", stats.get('total_predictions', 0))
        with col2:
            st.metric("Unique Categories", stats.get('unique_classes', 0))
        with col3:
            st.metric("Predictions Today", stats.get('predictions_today', 0))
        with col4:
            st.metric("Avg Text Length", f"{stats.get('avg_designation_length', 0):.0f} chars")
        
        st.markdown("---")
        
        # Format history for display
        history_display = history.copy()
        if 'timestamp' in history_display.columns:
            history_display['timestamp'] = pd.to_datetime(history_display['timestamp'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
        if 'predicted_class' in history_display.columns:
            history_display['category_name'] = history_display['predicted_class'].map(PRODUCT_CATEGORIES)
        
        # Determine which columns to display based on availability
        display_cols = []
        for col in ['timestamp', 'designation', 'predicted_class', 'category_name', 'designation_length']:
            if col in history_display.columns:
                display_cols.append(col)
        
        # Show table
        if display_cols:
            st.dataframe(
                history_display[display_cols].tail(20),
                width='stretch',
                height=300
            )
        else:
            st.warning("No valid data to display")
        
        # Class distribution chart
        if 'predicted_class' in history_display.columns:
            import plotly.express as px
            
            class_dist = history_display['predicted_class'].value_counts().head(10)
            df_chart = pd.DataFrame({
                'Category': [PRODUCT_CATEGORIES.get(c, f'Cat {c}') for c in class_dist.index],
                'Count': class_dist.values
            })
            
            fig = px.bar(
                df_chart,
                x='Category',
                y='Count',
                title='Top 10 Predicted Categories',
                color='Count',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, width='stretch')
        
        # Clear history button
        if st.button("Clear Prediction History", type="secondary"):
            if prediction_manager.clear_prediction_history():
                st.success("✅ Prediction history cleared")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Failed to clear history")
    
    else:
        st.info("No prediction history yet. Make your first prediction above!")

# Model Information
st.markdown("---")

with st.expander("Model Information"):
    st.markdown("""
    ### Current Production Model
    
    **Model Name:** Rakuten Multimodal Classifier
    
    **Model Type:** Logistic Regression with TF-IDF features
    
    **Input Features:**
    - Text: Product designation + description (TF-IDF vectorized)
    - Images: MobileNetV2 features (1280 dimensions) - *if provided*
    
    **Output:** Product category code (27 possible categories)
    
    **Training Data:** ~85,000 Rakuten products with multilingual text
    
    **Performance Metrics:**
    - Accuracy: ~82% (validation set)
    - F1-Weighted: ~0.80
    
    **Model Selection:** Auto-promoted "champion" model based on F1 score
    
    **Deployment:** Served via FastAPI with MLflow model registry
    
    **Monitoring:** All predictions logged for drift detection
    """)

# API Documentation link
st.markdown("---")
st.info(f"**For Developers:** Full API documentation available at [{API_URL}/docs]({API_URL}/docs)")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Tip:</strong> Try different product descriptions to see how the model performs</p>
    <p>All predictions are logged and monitored for data drift</p>
</div>
""", unsafe_allow_html=True)

