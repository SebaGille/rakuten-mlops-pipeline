"""Dataset Explorer Page"""
import streamlit as st
from pathlib import Path
import sys
import pandas as pd
import plotly.express as px

# Add parent directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.utils.training_manager import TrainingManager
from streamlit_app.utils.constants import PRODUCT_CATEGORIES

st.set_page_config(
    page_title="Dataset - Rakuten MLOps",
    page_icon="📊",
    layout="wide"
)

# Initialize training manager
@st.cache_resource
def get_training_manager():
    return TrainingManager(PROJECT_ROOT)

training_manager = get_training_manager()

# Header
st.title("Rakuten Dataset Explorer")
st.markdown("Explore the Rakuten product catalog dataset: statistics, distributions, and sample data")
st.markdown("---")

# ==================== Dataset Explorer ====================
st.markdown("### Dataset Statistics")

# Get dataset statistics
stats = training_manager.get_dataset_statistics()

if stats:
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", f"{stats.get('total_samples', 0):,}")
    with col2:
        st.metric("Categories", stats.get('num_categories', 0))
    with col3:
        st.metric("Avg Text Length", f"{stats.get('avg_text_length', 0):.0f} chars")
    with col4:
        missing_pct = (stats.get('missing_description', 0) / stats.get('total_samples', 1)) * 100
        st.metric("Missing Descriptions", f"{missing_pct:.1f}%")
    
    st.markdown("---")
    
    # Category distribution
    st.markdown("### Category Distribution")
    
    cat_dist = stats.get('category_distribution', {})
    if cat_dist:
        df_cat = pd.DataFrame([
            {
                'Category Code': code,
                'Category Name': PRODUCT_CATEGORIES.get(code, f'Unknown ({code})'),
                'Count': count
            }
            for code, count in cat_dist.items()
        ]).sort_values('Count', ascending=False)
        
        fig = px.bar(
            df_cat,
            x='Category Name',
            y='Count',
            title='Products per Category',
            color='Count',
            color_continuous_scale='Blues'
        )
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    
    # Sample data viewer
    st.markdown("### Sample Data Viewer")
    
    # Category filter
    categories_filter = st.multiselect(
        "Filter by categories:",
        options=list(PRODUCT_CATEGORIES.keys()),
        format_func=lambda x: PRODUCT_CATEGORIES[x],
        default=None
    )
    
    # Load button
    if st.button("Load 10 Samples", use_container_width=True):
        with st.spinner("Loading dataset sample..."):
            df_sample = training_manager.load_dataset_sample(
                sample_size=10,
                categories=categories_filter if categories_filter else None
            )
            
            if not df_sample.empty:
                st.success(f"Loaded {len(df_sample)} samples")
                
                # Handle both original and cleaned column names
                df_display = df_sample.head(10).copy()
                
                # Rename cleaned columns back to original names for display
                rename_map = {}
                if 'designation_clean' in df_display.columns:
                    rename_map['designation_clean'] = 'designation'
                if 'description_clean' in df_display.columns:
                    rename_map['description_clean'] = 'description'
                
                if rename_map:
                    df_display = df_display.rename(columns=rename_map)
                
                # Add category name for better readability
                if 'prdtypecode' in df_display.columns:
                    df_display['category_name'] = df_display['prdtypecode'].map(PRODUCT_CATEGORIES)
                
                # Display each sample with image
                st.markdown("---")
                images_path = PROJECT_ROOT / "data" / "raw" / "images" / "image_train"
                
                for idx, row in df_display.iterrows():
                    with st.container():
                        col_img, col_text = st.columns([1, 3])
                        
                        with col_img:
                            # Try to display image
                            if 'imageid' in row and 'productid' in row:
                                img_filename = f"image_{int(row['imageid'])}_product_{int(row['productid'])}.jpg"
                                img_path = images_path / img_filename
                                
                                if img_path.exists():
                                    st.image(str(img_path), use_column_width=True)
                                else:
                                    st.info("📷 Image not found")
                            else:
                                st.info("📷 No image ID")
                        
                        with col_text:
                            if 'category_name' in row:
                                st.caption(f"**Category:** {row['category_name']}")
                            if 'designation' in row and pd.notna(row['designation']):
                                st.markdown(f"**{row['designation']}**")
                            if 'description' in row and pd.notna(row['description']):
                                desc_preview = str(row['description'])[:200] + "..." if len(str(row['description'])) > 200 else str(row['description'])
                                st.write(desc_preview)
                        
                        st.markdown("---")
            else:
                st.warning("No data found with the selected filters.")
else:
    st.warning("Dataset not found. Please run data ingestion first: `python src/data/make_dataset.py`")

# Next steps
st.markdown("---")
st.info("Ready to train a model? Head to the **Training** page to configure and launch experiments!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>Built with ❤️ by <strong>Sébastien</strong></p>
    <p>ML Engineer + Product Manager | Seeking combined ML/Product roles</p>
    <p>
        <a href='https://github.com/sebagille' target='_blank'>GitHub</a> | 
        <a href='https://linkedin.com/in/seba-gille' target='_blank'>LinkedIn</a>
    </p>
</div>
""", unsafe_allow_html=True)

