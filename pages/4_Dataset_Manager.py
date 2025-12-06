"""
Dataset Manager

Manage sharded datasets: browse, create, merge, collect large-scale data.

Features:
- Browse all datasets with statistics
- Large-scale data collection with progress tracking
- Merge and deduplicate datasets
- Dataset statistics and visualization

Author: Digital Foundry Materials Science Toolkit
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import os
import time
import io
import zipfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import modules
try:
    from dataset_manager import DatasetManager, print_dataset_info
    from large_scale_collection import LargeScaleCollector
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Page configuration
st.set_page_config(
    page_title="Dataset Manager",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .dataset-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 10px 0;
    }
    .stat-box {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("📊 Dataset Manager")
st.markdown("### Manage Large-Scale Materials Datasets")

# Check modules
if not MODULES_AVAILABLE:
    st.error(f"Error loading modules: {IMPORT_ERROR}")
    st.stop()

# Initialize manager
manager = DatasetManager()

# Helper function to create dataset zip
def create_dataset_zip(dataset_name: str) -> bytes:
    """Create a zip file of the entire dataset."""
    dataset_dir = manager.base_dir / dataset_name

    # Create in-memory zip file
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add manifest
        manifest_path = dataset_dir / "manifest.json"
        zip_file.write(manifest_path, f"{dataset_name}/manifest.json")

        # Add all shards
        shards_dir = dataset_dir / "shards"
        for shard_file in shards_dir.glob("*.pkl"):
            zip_file.write(shard_file, f"{dataset_name}/shards/{shard_file.name}")

    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# Sidebar
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📚 Browse Datasets",
    "🔬 Collect Data",
    "🔄 Merge Datasets",
    "ℹ️ Help"
])

# Tab 1: Browse Datasets
with tab1:
    st.header("Available Datasets")

    # Refresh button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    # List datasets
    datasets = manager.list_datasets()

    if not datasets:
        st.info("📭 No datasets found. Create one in the 'Collect Data' tab!")
    else:
        st.success(f"Found {len(datasets)} dataset(s)")

        # Dataset selector
        selected_dataset = st.selectbox(
            "Select dataset to view:",
            datasets,
            key="browse_dataset_selector"
        )

        if selected_dataset:
            try:
                # Get statistics
                stats = manager.get_statistics(selected_dataset)

                # Display overview
                st.markdown("### Overview")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Materials", f"{stats['total_materials']:,}")
                with col2:
                    st.metric("Shards", stats['num_shards'])
                with col3:
                    st.metric("Disk Usage", f"{stats['disk_usage_mb']:.1f} MB")
                with col4:
                    st.metric("CALPHAD", "✓" if stats['calphad_enhanced'] else "✗")

                # Detailed info
                st.markdown("### Details")

                info_col1, info_col2 = st.columns(2)

                with info_col1:
                    st.markdown("**Metadata**")
                    info_df = pd.DataFrame({
                        "Property": ["Name", "Description", "Created", "Version", "Shard Size"],
                        "Value": [
                            stats['name'],
                            stats['description'],
                            stats['created'][:10],
                            stats['version'],
                            f"{stats['shard_size']} materials/shard"
                        ]
                    })
                    st.dataframe(info_df, use_container_width=True, hide_index=True)

                with info_col2:
                    st.markdown("**Content**")
                    if stats['elements']:
                        st.write(f"**Elements ({len(stats['elements'])}):**")
                        st.write(", ".join(stats['elements'][:20]))
                        if len(stats['elements']) > 20:
                            st.write(f"... and {len(stats['elements']) - 20} more")

                    if stats['properties']:
                        st.write(f"**Properties:**")
                        st.write(", ".join(stats['properties']))

                # Visualizations
                st.markdown("### Visualizations")

                viz_col1, viz_col2 = st.columns(2)

                with viz_col1:
                    # Shard distribution
                    manifest_path = manager.base_dir / selected_dataset / "manifest.json"
                    import json
                    with open(manifest_path) as f:
                        manifest = json.load(f)

                    shard_sizes = [s['count'] for s in manifest['shards']]

                    fig = go.Figure(data=[
                        go.Bar(x=list(range(len(shard_sizes))), y=shard_sizes)
                    ])
                    fig.update_layout(
                        title="Materials per Shard",
                        xaxis_title="Shard Index",
                        yaxis_title="Material Count",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with viz_col2:
                    # Element distribution
                    if stats['elements']:
                        element_counts = {elem: 1 for elem in stats['elements'][:10]}

                        fig = go.Figure(data=[
                            go.Pie(labels=list(element_counts.keys()), values=list(element_counts.values()))
                        ])
                        fig.update_layout(
                            title="Element Distribution (Top 10)",
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # Actions
                st.markdown("### Actions")

                action_col1, action_col2, action_col3, action_col4 = st.columns(4)

                with action_col1:
                    if st.button("📥 Load Sample", use_container_width=True):
                        with st.spinner("Loading first shard..."):
                            shard_0 = manager.load_shard(selected_dataset, 0)
                            st.success(f"Loaded {len(shard_0)} materials from shard 0")
                            if shard_0:
                                sample = shard_0[0]
                                st.json({
                                    "material_id": getattr(sample, 'material_id', 'N/A'),
                                    "formula": getattr(sample, 'formula_str', 'N/A'),
                                    "num_nodes": sample.num_nodes,
                                    "num_edges": sample.num_edges
                                })

                with action_col2:
                    # Download button
                    try:
                        zip_data = create_dataset_zip(selected_dataset)
                        st.download_button(
                            label="💾 Download",
                            data=zip_data,
                            file_name=f"{selected_dataset}.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error creating zip: {e}")

                with action_col3:
                    if st.button("🔍 Deduplicate", use_container_width=True):
                        with st.spinner("Checking for duplicates..."):
                            removed = manager.deduplicate_dataset(selected_dataset)
                            if removed > 0:
                                st.success(f"Removed {removed} duplicate(s)")
                                st.rerun()
                            else:
                                st.info("No duplicates found")

                with action_col4:
                    if st.button("🗑️ Delete Dataset", type="secondary", use_container_width=True):
                        st.warning("⚠️ This will permanently delete the dataset!")
                        confirm = st.checkbox(f"Confirm delete '{selected_dataset}'")
                        if confirm and st.button("Confirm Delete", type="primary"):
                            manager.delete_dataset(selected_dataset, confirm=True)
                            st.success("Dataset deleted")
                            st.rerun()

            except Exception as e:
                st.error(f"Error loading dataset: {e}")
                st.exception(e)

# Tab 2: Collect Data
with tab2:
    st.header("Large-Scale Data Collection")

    st.markdown("""
    Collect large datasets from Materials Project with automatic sharding and checkpointing.
    """)

    # Check API key
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("MP_API_KEY")
        except:
            pass

    if not api_key:
        st.error("⚠️ **Materials Project API Key Required**")
        st.markdown("""
        Add `MP_API_KEY` to your environment or `.env` file.
        Get your key at: https://next-gen.materialsproject.org/api
        """)
        st.stop()

    st.success("✅ API Key Configured")

    # Collection form
    with st.form("collection_form"):
        st.markdown("### Dataset Configuration")

        col1, col2 = st.columns(2)

        with col1:
            dataset_name = st.text_input(
                "Dataset Name:",
                value="my_large_dataset",
                help="Unique name for this dataset"
            )

            data_mode = st.radio(
                "Filter by:",
                ["Chemical System", "Elements", "All Metallic"],
                horizontal=True
            )

            if data_mode == "Chemical System":
                chemsys = st.text_input("Chemical system:", value="Fe-Ni")
                elements = None
            elif data_mode == "Elements":
                elements_str = st.text_input("Elements (comma-separated):", value="Fe,Ni,Cr")
                elements = [e.strip() for e in elements_str.split(",")]
                chemsys = None
            else:
                elements = None
                chemsys = None

        with col2:
            target_size = st.number_input(
                "Target size (materials):",
                min_value=100,
                max_value=100000,
                value=1000,
                step=100,
                help="Number of materials to collect"
            )

            shard_size = st.number_input(
                "Shard size:",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100,
                help="Materials per shard file"
            )

            checkpoint_interval = st.number_input(
                "Checkpoint interval:",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Save checkpoint every N materials"
            )

        st.markdown("### Options")

        opt_col1, opt_col2 = st.columns(2)

        with opt_col1:
            metallic_only = st.checkbox("Metallic only (band gap = 0)", value=True)
            stable_only = st.checkbox("Stable materials only", value=False)

        with opt_col2:
            use_calphad = st.checkbox("Enable CALPHAD features", value=True)
            resume = st.checkbox("Resume from checkpoint", value=True)

        submitted = st.form_submit_button("🚀 Start Collection", type="primary", use_container_width=True)

    if submitted:
        # Check if dataset exists
        if dataset_name in manager.list_datasets() and not resume:
            st.error(f"Dataset '{dataset_name}' already exists! Enable 'Resume from checkpoint' or choose a different name.")
        else:
            # Initialize collector
            collector = LargeScaleCollector(api_key=api_key, dataset_manager=manager)

            # Progress tracking
            st.markdown("---")
            st.markdown("### 📊 Collection Progress")

            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_placeholder = st.empty()

            # Progress callback
            def progress_callback(progress):
                percent = progress['percent'] / 100.0
                progress_bar.progress(min(percent, 1.0))

                elapsed = progress['elapsed_time']
                rate = progress['collected'] / elapsed if elapsed > 0 else 0
                remaining_time = (progress['target'] - progress['collected']) / rate if rate > 0 else 0

                status_text.markdown(f"""
                **Status:** Collecting materials...
                **Collected:** {progress['collected']:,} / {progress['target']:,} ({progress['percent']:.1f}%)
                **Rate:** {rate:.1f} materials/second
                **Estimated remaining:** {remaining_time/60:.1f} minutes
                """)

            try:
                # Run collection
                summary = collector.collect_dataset(
                    dataset_name=dataset_name,
                    elements=elements,
                    chemsys=chemsys,
                    target_size=target_size,
                    shard_size=shard_size,
                    checkpoint_interval=checkpoint_interval,
                    metallic_only=metallic_only,
                    stable_only=stable_only,
                    use_calphad=use_calphad,
                    resume=resume,
                    callback=progress_callback
                )

                # Show summary
                progress_bar.progress(1.0)

                if summary['success']:
                    st.success("✅ **Collection Complete!**")
                else:
                    st.warning("⚠️ **Collection Incomplete**")

                summary_col1, summary_col2, summary_col3 = st.columns(3)
                with summary_col1:
                    st.metric("Materials Collected", f"{summary['materials_collected']:,}")
                with summary_col2:
                    st.metric("Time Elapsed", summary['elapsed_time_formatted'])
                with summary_col3:
                    st.metric("Failed Batches", summary['failed_batches'])

                st.info("💡 Go to 'Browse Datasets' to view your new dataset!")

            except Exception as e:
                st.error(f"❌ Collection failed: {e}")
                st.exception(e)

# Tab 3: Merge Datasets
with tab3:
    st.header("Merge Datasets")

    st.markdown("""
    Combine multiple datasets into one, with automatic deduplication.
    """)

    available_datasets = manager.list_datasets()

    if len(available_datasets) < 2:
        st.info("📭 Need at least 2 datasets to merge. Create more in the 'Collect Data' tab!")
    else:
        # Merge form
        with st.form("merge_form"):
            st.markdown("### Select Datasets to Merge")

            datasets_to_merge = st.multiselect(
                "Datasets:",
                available_datasets,
                help="Select 2 or more datasets"
            )

            output_name = st.text_input(
                "Output Dataset Name:",
                value="merged_dataset",
                help="Name for the merged dataset"
            )

            deduplicate = st.checkbox(
                "Remove duplicates",
                value=True,
                help="Remove materials with duplicate material_ids"
            )

            merge_submitted = st.form_submit_button("🔄 Merge Datasets", type="primary")

        if merge_submitted:
            if len(datasets_to_merge) < 2:
                st.error("⚠️ Select at least 2 datasets to merge")
            elif output_name in available_datasets:
                st.error(f"⚠️ Dataset '{output_name}' already exists!")
            else:
                try:
                    with st.spinner(f"Merging {len(datasets_to_merge)} datasets..."):
                        # Get stats before merge
                        total_before = sum(
                            manager.get_statistics(d)['total_materials']
                            for d in datasets_to_merge
                        )

                        # Merge
                        output_dir = manager.merge_datasets(
                            dataset_names=datasets_to_merge,
                            output_name=output_name,
                            deduplicate=deduplicate
                        )

                        # Get stats after
                        stats_after = manager.get_statistics(output_name)
                        total_after = stats_after['total_materials']

                    st.success("✅ **Merge Complete!**")

                    result_col1, result_col2, result_col3 = st.columns(3)
                    with result_col1:
                        st.metric("Total Before", f"{total_before:,}")
                    with result_col2:
                        st.metric("Total After", f"{total_after:,}")
                    with result_col3:
                        duplicates = total_before - total_after
                        st.metric("Duplicates Removed", f"{duplicates:,}" if duplicates > 0 else "0")

                    st.info(f"💡 View '{output_name}' in the 'Browse Datasets' tab!")

                except Exception as e:
                    st.error(f"❌ Merge failed: {e}")
                    st.exception(e)

# Tab 4: Help
with tab4:
    st.header("Dataset Manager Help")

    st.markdown("""
    ### 📚 How to Use

    #### Browse Datasets
    - View all available datasets
    - See statistics (size, elements, disk usage)
    - **Download datasets as zip files** for backup/sharing
    - Load samples, deduplicate, delete

    #### Collect Data
    - Large-scale collection from Materials Project
    - Automatic sharding (1000 materials/shard)
    - Checkpointing every 100 materials
    - Resume from failures

    #### Merge Datasets
    - Combine multiple datasets
    - Automatic deduplication by material_id
    - Preserve all metadata

    ---

    ### 🔧 Best Practices

    **Shard Size:**
    - Small datasets (< 1,000): 500 per shard
    - Medium datasets (1,000 - 10,000): 1,000 per shard
    - Large datasets (> 10,000): 2,000 per shard

    **Collection:**
    - Always enable "Resume from checkpoint"
    - Use checkpoint_interval = 100 for reliability
    - Enable CALPHAD features for better predictions

    **Merging:**
    - Always deduplicate when merging
    - Merge similar datasets (same properties)
    - Check total size before merging

    ---

    ### 📊 Dataset Structure

    ```
    datasets/
    └── my_dataset/
        ├── manifest.json          # Metadata & index
        ├── checkpoint.json        # Resume point (if collecting)
        └── shards/
            ├── shard_0000.pkl    # 1000 materials
            ├── shard_0001.pkl
            └── ...
    ```

    ---

    ### ⚡ Performance

    **Memory Usage:**
    - Small (1,000 mat): ~200 MB
    - Medium (10,000 mat): ~600 MB (sharded)
    - Large (50,000 mat): ~800 MB (sharded)

    **Collection Time:**
    - 1,000 materials: ~10 minutes
    - 10,000 materials: ~2 hours
    - 50,000 materials: ~8-10 hours

    **Disk Usage:**
    - ~50 KB per material
    - 10,000 materials: ~500 MB
    - 50,000 materials: ~2.5 GB

    ---

    ### 💾 Downloading & Sharing Datasets

    **Download:**
    - Click "💾 Download" button in Browse tab
    - Downloads dataset as `.zip` file
    - Includes all shards and manifest.json
    - Can be used for backup or sharing

    **Upload/Import:**
    - Extract downloaded zip to `datasets/` folder
    - Dataset structure:
      ```
      datasets/
      └── dataset_name/
          ├── manifest.json
          └── shards/
              └── shard_*.pkl
      ```
    - Refresh Browse tab to see imported dataset

    **Sharing with Others:**
    1. Download dataset as zip
    2. Share zip file
    3. Recipient extracts to their `datasets/` folder
    4. Ready to use immediately!
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("**📖 Documentation**")
st.sidebar.markdown("[Dataset Manager Guide](https://docs.claude.com)")
st.sidebar.markdown("[API Reference](https://materialsproject.org)")
