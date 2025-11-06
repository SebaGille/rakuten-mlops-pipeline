"""Interactive Training Interface Page"""
import streamlit as st
from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.utils.training_manager import TrainingManager
from streamlit_app.utils.mlflow_manager import MLflowManager
from streamlit_app.utils.constants import (
    PRODUCT_CATEGORIES, MLFLOW_URL, MLFLOW_TRACKING_URI, DEFAULT_HYPERPARAMS, SAMPLE_SIZES
)

st.set_page_config(
    page_title="Training - Rakuten MLOps",
    page_icon="🎯",
    layout="wide"
)

# Initialize managers
@st.cache_resource
def get_managers():
    training_mgr = TrainingManager(PROJECT_ROOT)
    # Use MLFLOW_TRACKING_URI (path-prefixed for ALB routing) for client
    mlflow_mgr = MLflowManager(MLFLOW_TRACKING_URI)
    return training_mgr, mlflow_mgr

# Try to get managers, but clear cache if connection fails
try:
    training_manager, mlflow_manager = get_managers()
except Exception as e:
    st.error(f"Failed to initialize managers: {e}")
    st.cache_resource.clear()  # Clear cache on failure
    st.stop()

# Header
st.title("Model Training & Experiment Tracking")
st.markdown("Configure, train, and compare ML models with full experiment tracking")
st.markdown("---")

# Check MLflow connection
with st.spinner("Checking MLflow connection..."):
    mlflow_connected = mlflow_manager.check_connection()
    
if not mlflow_connected:
    st.error("MLflow server is not accessible. Please start the infrastructure first (Infrastructure page).")
    st.info(f"Attempted to connect to: `{MLFLOW_TRACKING_URI}`")
    # Clear cache to allow retry on next page load
    get_managers.clear()
    st.stop()

# ==================== Configure & Train ====================
st.markdown("### Configure & Train Model")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### Data Configuration")
    
    # Sample size selection
    sample_size = st.select_slider(
        "Training sample size:",
        options=SAMPLE_SIZES,
        value=1000,
        help="How many products to train on. Small = faster training, Large = better accuracy"
    )
    
    # Modality selection
    use_images = st.radio(
        "Model modality:",
        options=["Text Only", "Multimodal (Text + Images)"],
        index=1,
        help="Use only product descriptions (faster) or add product images (more accurate)"
    )
    use_images_bool = use_images == "Multimodal (Text + Images)"
    
    # Category selection (optional)
    train_all_categories = st.checkbox("Train on all categories", value=True)
    selected_categories = None
    if not train_all_categories:
        selected_categories = st.multiselect(
            "Select categories to train on:",
            options=list(PRODUCT_CATEGORIES.keys()),
            format_func=lambda x: PRODUCT_CATEGORIES[x],
            default=list(PRODUCT_CATEGORIES.keys())[:5]
        )

with col2:
    st.markdown("#### Hyperparameters")
    
    # TF-IDF parameters
    max_features = st.selectbox(
        "Vocabulary size:",
        options=[5000, 10000, 20000],
        index=2,
        help="How many words to learn. Small = faster but less detail, Large = slower but captures more nuances"
    )
    
    # N-gram parameters with better labels
    ngram_option = st.selectbox(
        "Text analysis depth:",
        options=["Single words only", "Single + Pairs of words", "Single + Pairs + Triplets"],
        index=1,
        help="How to analyze text. Pairs capture phrases like 'not good'. More = better understanding but slower"
    )
    # Map to actual ngram values
    ngram_mapping = {
        "Single words only": (1, 1),
        "Single + Pairs of words": (1, 2),
        "Single + Pairs + Triplets": (1, 3)
    }
    ngram_min, ngram_max = ngram_mapping[ngram_option]
    
    # Model parameters
    max_iter = st.selectbox(
        "Training effort:",
        options=[100, 200, 500],
        index=1,
        help="💡 How long to train. Low = fast but might not learn fully, High = slower but more thorough"
    )
    
    solver = st.selectbox(
        "Optimization method:",
        options=['lbfgs', 'saga'],
        index=0,
        format_func=lambda x: {
            'lbfgs': 'L-BFGS (Standard - good for most cases)',
            'saga': 'SAGA (Advanced - good for large datasets)'
        }[x],
        help="The algorithm used to train the model. L-BFGS is reliable and fast for most situations"
    )

st.markdown("---")

# Configuration summary
st.markdown("#### Training Configuration Summary")

config_summary = f"""
- **Sample Size**: {sample_size if sample_size != "Full Dataset" else "Full dataset (~85K samples)"}
- **Modality**: {"Text + Images" if use_images_bool else "Text only"}
- **Categories**: {len(selected_categories) if selected_categories else "All (27)"}
- **Vocabulary Size**: {max_features:,} words
- **Text Analysis**: {ngram_option}
- **Training Effort**: {max_iter} iterations
- **Optimization**: {solver.upper()}
"""
st.info(config_summary)

# Training button
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("START TRAINING", type="primary", use_container_width=True):
        # Create configuration
        config = {
            'sample_size': sample_size if sample_size != "Full Dataset" else None,
            'use_images': use_images_bool,
            'categories': selected_categories,
            'hyperparams': {
                'max_features': max_features,
                'ngram_range': (ngram_min, ngram_max),
                'max_iter': max_iter,
                'solver': solver,
                'random_state': DEFAULT_HYPERPARAMS['random_state']
            }
        }
        
        # Trigger training
        with st.spinner("Training in progress... This may take 2-10 minutes depending on sample size."):
            success, message, run_id = training_manager.trigger_training(config)
            
            if success:
                st.success(message)
                if run_id:
                    st.info(f"MLflow Run ID: `{run_id}`")
                    st.markdown(f"[View in MLflow]({MLFLOW_URL})")
                
                # Show next steps
                st.markdown("""
                ### Next Steps:
                1. Scroll down to **Experiment Tracking** section to compare runs
                2. Check **Model Registry** section to see champion/challenger status
                3. Go to **Predictions** page to test the model
                """)
            else:
                st.error(message)
                st.markdown("**Troubleshooting:**")
                st.markdown("- Make sure all services are running (Infrastructure page)")
                st.markdown("- Check that processed data exists: `data/processed/train_features.csv`")
                st.markdown("- Run data preparation: `python src/features/build_features.py`")

st.markdown("---")

# ==================== Experiment Tracking ====================
st.markdown("### Experiment Tracking")

st.info("""
🎯 **Automatic Model Promotion**: When you train a new model, it is automatically compared against the current production model. 
If the new model performs better (based on F1-weighted score), it will be promoted as the **Champion** and deployed to production. 
Otherwise, it becomes a **Challenger** for future comparison.
""")

# Get experiments
experiments = mlflow_manager.get_experiments()

if experiments:
    # Filter non-default experiments
    exp_names = [exp['name'] for exp in experiments if not exp['name'].startswith('Default')]
    if exp_names:
        # Get the main experiment (rakuten-multimodal-text-image)
        selected_exp = next((e for e in experiments if e['name'] == 'rakuten-multimodal-text-image'), experiments[0])
        
        # Get all runs (no limit)
        df_runs = mlflow_manager.get_runs(selected_exp['experiment_id'], max_results=1000)
        
        if not df_runs.empty:
            st.markdown(f"#### Recent Training Runs (Last 10 of {len(df_runs)} total)")
            
            # Format dataframe for display - show only last 10 runs
            df_display = df_runs.head(10).copy()
            if 'start_time' in df_display.columns:
                df_display['start_time'] = pd.to_datetime(df_display['start_time']).dt.strftime('%Y-%m-%d %H:%M')
            if 'duration_sec' in df_display.columns:
                df_display['duration_min'] = (df_display['duration_sec'] / 60).round(2)
            
            # Display runs table
            display_cols = [
                'run_name', 'status', 'start_time', 'accuracy', 
                'f1_weighted', 'model_type', 'auto_promotion_candidate', 
                'auto_promotion_reason', 'git_commit'
            ]
            # Only show columns that exist
            display_cols = [col for col in display_cols if col in df_display.columns]
            
            st.dataframe(
                df_display[display_cols],
                use_container_width=True
            )
            
            st.markdown("---")
            
            # Champion run section
            st.markdown("#### 🏆 Champion Model (Production)")
            
            # Get champion run_id from model registry
            champion_run_id = None
            models = mlflow_manager.get_registered_models()
            
            # Method 1: Try to get champion from model registry
            if models and len(models) > 0:
                model = models[0]  # Get the main model
                if model.get('champion_version'):
                    champion_details = mlflow_manager.get_model_version_details(
                        model['name'],
                        alias='champion'
                    )
                    if champion_details:
                        champion_run_id = champion_details.get('run_id')
            
            # Method 2: Fallback - Find champion from runs with auto_promotion_candidate='champion'
            if not champion_run_id and 'auto_promotion_candidate' in df_runs.columns:
                champion_runs = df_runs[df_runs['auto_promotion_candidate'] == 'champion']
                if not champion_runs.empty:
                    champion_run_id = champion_runs.iloc[0]['run_id']
            
            # Display champion run
            if champion_run_id:
                df_champion = df_runs[df_runs['run_id'] == champion_run_id].copy()
                if not df_champion.empty:
                    # Format the champion display
                    df_champion_display = df_champion.copy()
                    if 'start_time' in df_champion_display.columns:
                        df_champion_display['start_time'] = pd.to_datetime(df_champion_display['start_time']).dt.strftime('%Y-%m-%d %H:%M')
                    
                    st.dataframe(
                        df_champion_display[display_cols],
                        use_container_width=True
                    )
                else:
                    st.warning("Champion run not found in the current runs list. It may be older than the displayed runs.")
            else:
                st.info("No champion model found. The best performing model will be automatically promoted to champion after training.")
            
            st.markdown("---")
            
            # Metrics comparison chart
            st.markdown("#### Metrics Comparison")
            
            st.markdown("""
            Understanding these metrics helps you evaluate how well your model performs. 
            Both metrics are important, but they tell you different things about your model's quality.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Accuracy: The Percentage of Correct Predictions**")
                with st.expander("What does Accuracy mean?", expanded=False):
                    st.markdown("""
                    **Simple explanation:** Accuracy is the percentage of times your model gets it right.
                    
                    **Think of it like a test:** If you take a 100-question test and get 85 answers correct, 
                    your accuracy is 85%. That's it!
                    
                    **The catch:** Accuracy can be misleading when categories aren't balanced. 
                    For example, if 95% of products are in one category, a model that always guesses 
                    that category would get 95% accuracy—but it hasn't really learned anything useful!
                    
                    **When to use it:** Good when all categories are equally important and equally common.
                    """)
                
                # Accuracy over time
                fig_acc = px.line(
                    df_display,
                    x='start_time',
                    y='accuracy',
                    markers=True,
                    title='Accuracy Over Time',
                    labels={'start_time': 'Run Time', 'accuracy': 'Accuracy'}
                )
                fig_acc.update_layout(height=300)
                st.plotly_chart(fig_acc, use_container_width=True)
            
            with col2:
                st.markdown("**F1-Weighted: Balanced Performance Score**")
                with st.expander("What does F1-Weighted mean?", expanded=False):
                    st.markdown("""
                    **Simple explanation:** F1-weighted is a more balanced measure that considers both:
                    - **Precision**: How often you're right when you make a prediction (avoiding false alarms)
                    - **Recall**: How often you catch the correct answers (not missing important things)
                    
                    **Think of a security guard:** Imagine identifying suspicious packages:
                    - If you flag *everything* as suspicious → you'll catch all bad ones ✓ but annoy everyone with false alarms ✗
                    - If you flag *nothing* as suspicious → no false alarms ✓ but you'll miss dangerous ones ✗✗
                    
                    F1 score balances these two concerns. The **"weighted"** part means it accounts for 
                    the fact that some categories might be more common than others, giving a fair overall score.
                    
                    **When to use it:** Better when some categories are rare, or when both "catching things" 
                    and "being precise" matter. Often more reliable than accuracy for real-world problems!
                    """)
                
                # F1 score over time
                fig_f1 = px.line(
                    df_display,
                    x='start_time',
                    y='f1_weighted',
                    markers=True,
                    title='F1-Weighted Over Time',
                    labels={'start_time': 'Run Time', 'f1_weighted': 'F1 Score'},
                    color_discrete_sequence=['#e74c3c']
                )
                fig_f1.update_layout(height=300)
                st.plotly_chart(fig_f1, use_container_width=True)
            
        else:
            st.warning("No runs found for this experiment. Train a model first!")
    else:
        st.info("No experiments found. Train your first model!")
else:
    st.warning("Could not fetch experiments from MLflow. Check connection.")

st.markdown("---")

# ==================== Model Registry ====================
st.markdown("### Model Registry")

st.markdown("""
View all registered models with their versions, metadata, and performance metrics. 
Models promoted to **Champion** are currently deployed in production.
""")

# Get registered models
models = mlflow_manager.get_registered_models()

if models:
    for model in models:
        st.markdown(f"#### 📦 {model['name']}")
        
        # Model version overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Latest Version", model['latest_version'])
        with col2:
            champion_v = model['champion_version'] or "None"
            st.metric("🏆 Champion", f"v{champion_v}" if champion_v != "None" else champion_v)
        with col3:
            challenger_v = model['challenger_version'] or "None"
            st.metric("🔄 Challenger", f"v{challenger_v}" if challenger_v != "None" else challenger_v)
        
        # Get all versions for this model
        try:
            all_versions = mlflow_manager.client.search_model_versions(f"name='{model['name']}'")
            st.markdown(f"**Total Versions**: {len(all_versions)}")
        except:
            all_versions = []
        
        
        # Champion details
        if model['champion_version']:
            with st.expander("🏆 Champion Model (Production)", expanded=True):
                champion_details = mlflow_manager.get_model_version_details(
                    model['name'],
                    alias='champion'
                )
                
                if champion_details:
                    # Metadata
                    st.markdown("##### 📊 Model Information")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        **Version**: {champion_details['version']}  
                        **Status**: {champion_details['status']}  
                        **Created**: {champion_details['creation_timestamp'].strftime('%Y-%m-%d %H:%M')}
                        """)
                    
                    with col2:
                        acc = champion_details.get('accuracy', 0)
                        f1 = champion_details.get('f1_weighted', 0)
                        st.markdown(f"""
                        **Accuracy**: {acc:.4f}  
                        **F1-Weighted**: {f1:.4f}  
                        **Run ID**: `{champion_details['run_id'][:12]}...`
                        """)
                    
                    with col3:
                        promo_reason = champion_details.get('tags', {}).get('auto_promotion_reason', 'N/A')
                        st.markdown(f"""
                        **Promotion**: {promo_reason}  
                        **Aliases**: {', '.join(champion_details.get('aliases', []))}
                        """)
                    
                    # Parameters
                    st.markdown("##### ⚙️ Model Parameters")
                    params = champion_details.get('params', {})
                    if params:
                        params_df = pd.DataFrame([
                            {'Parameter': k, 'Value': v} 
                            for k, v in params.items()
                        ])
                        st.dataframe(params_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No parameters logged")
                    
                    # Tags & Metadata
                    st.markdown("##### 🏷️ Tags & Metadata")
                    tags = champion_details.get('tags', {})
                    if tags:
                        # Filter out mlflow internal tags for cleaner display
                        display_tags = {k: v for k, v in tags.items() if not k.startswith('mlflow.')}
                        if display_tags:
                            tags_df = pd.DataFrame([
                                {'Tag': k, 'Value': v} 
                                for k, v in display_tags.items()
                            ])
                            st.dataframe(tags_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No custom tags")
                    else:
                        st.info("No tags logged")
        
        # Challenger details
        if model['challenger_version']:
            with st.expander("🔄 Challenger Model", expanded=False):
                challenger_details = mlflow_manager.get_model_version_details(
                    model['name'],
                    alias='challenger'
                )
                
                if challenger_details:
                    # Metadata
                    st.markdown("##### 📊 Model Information")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        **Version**: {challenger_details['version']}  
                        **Status**: {challenger_details['status']}  
                        **Created**: {challenger_details['creation_timestamp'].strftime('%Y-%m-%d %H:%M')}
                        """)
                    
                    with col2:
                        acc = challenger_details.get('accuracy', 0)
                        f1 = challenger_details.get('f1_weighted', 0)
                        st.markdown(f"""
                        **Accuracy**: {acc:.4f}  
                        **F1-Weighted**: {f1:.4f}  
                        **Run ID**: `{challenger_details['run_id'][:12]}...`
                        """)
                    
                    with col3:
                        promo_reason = challenger_details.get('tags', {}).get('auto_promotion_reason', 'N/A')
                        st.markdown(f"""
                        **Promotion**: {promo_reason}  
                        **Aliases**: {', '.join(challenger_details.get('aliases', []))}
                        """)
                    
                    # Parameters
                    st.markdown("##### ⚙️ Model Parameters")
                    params = challenger_details.get('params', {})
                    if params:
                        params_df = pd.DataFrame([
                            {'Parameter': k, 'Value': v} 
                            for k, v in params.items()
                        ])
                        st.dataframe(params_df, use_container_width=True, hide_index=True)
                    
                    # Comparison with champion
                    if champion_details and challenger_details:
                        st.markdown("---")
                        st.markdown("##### ⚖️ Comparison vs Champion")
                        
                        comparison_data = {
                            'Metric': ['Accuracy', 'F1-Weighted'],
                            'Champion': [
                                champion_details.get('accuracy', 0),
                                champion_details.get('f1_weighted', 0)
                            ],
                            'Challenger': [
                                challenger_details.get('accuracy', 0),
                                challenger_details.get('f1_weighted', 0)
                            ]
                        }
                        
                        df_comp = pd.DataFrame(comparison_data)
                        df_comp['Difference'] = df_comp['Challenger'] - df_comp['Champion']
                        df_comp['% Change'] = ((df_comp['Challenger'] - df_comp['Champion']) / df_comp['Champion'] * 100).round(2)
                        
                        st.dataframe(df_comp, use_container_width=True, hide_index=True)
                        
                        # Recommendation
                        f1_diff = df_comp[df_comp['Metric'] == 'F1-Weighted']['Difference'].values[0]
                        if f1_diff > 0:
                            st.warning(f"⚠️ Challenger is performing better by {f1_diff:.4f}. Consider promoting to Champion!")
                        else:
                            st.success("✅ Champion is still the best model.")
        
else:
    st.info("No registered models yet. Train your first model to see it here!")

# Link to MLflow UI
st.markdown(f"[View full model registry in MLflow UI]({MLFLOW_URL}/#/models)")

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

