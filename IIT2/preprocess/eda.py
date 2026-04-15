import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

def perform_eda_on_folder(input_folder, output_base_folder):
    """
    Loops through all CSVs in the input_folder, generates EDA charts,
    and saves them in organized sub-folders within output_base_folder.
    """
    # 1. Find all CSV files in the target folder
    search_pattern = os.path.join(input_folder, "*.csv")
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f"No CSV files found in '{input_folder}'. Please check the path.")
        return
        
    print(f"Found {len(csv_files)} files. Starting Batch EDA...\n")
    
    # 2. Loop through each file
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        file_prefix = filename.replace('.csv', '')
        
        print(f"{'='*50}")
        print(f"Analyzing: {filename}")
        print(f"{'='*50}")
        
        # Create a specific output folder for this file's charts
        file_output_dir = os.path.join(output_base_folder, file_prefix)
        os.makedirs(file_output_dir, exist_ok=True)
        
        try:
            # Load the current file
            df = pd.read_csv(file_path)
            
            print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
            if 'device_aid' in df.columns:
                print(f"Unique Devices: {df['device_aid'].nunique():,}")
            
            # Set visualization style
            sns.set_theme(style="whitegrid")
            
            # --- CHART 1: OS Distribution ---
            if 'OS' in df.columns:
                plt.figure(figsize=(10, 6))
                sns.countplot(data=df, y='OS', order=df['OS'].value_counts().index, palette='viridis')
                plt.title(f'OS Distribution - {file_prefix}')
                plt.xlabel('Number of Pings')
                plt.tight_layout()
                plt.savefig(os.path.join(file_output_dir, '1_os_distribution.png'), dpi=300)
                plt.close()

            # --- CHART 2: Location Accuracy ---
            if 'horizontal_accuracy' in df.columns:
                plt.figure(figsize=(10, 6))
                acc_data = df[df['horizontal_accuracy'] < 200]['horizontal_accuracy']
                sns.histplot(acc_data, bins=50, kde=True, color='blue')
                plt.title(f'Horizontal Accuracy (<200m) - {file_prefix}')
                plt.xlabel('Accuracy in Meters')
                plt.tight_layout()
                plt.savefig(os.path.join(file_output_dir, '2_location_accuracy.png'), dpi=300)
                plt.close()

            # --- CHART 3: Geospatial Hexbin Map ---
            if 'latitude' in df.columns and 'longitude' in df.columns:
                plt.figure(figsize=(14, 8))
                valid_coords = df[(df['latitude'] >= -90) & (df['latitude'] <= 90) & 
                                  (df['longitude'] >= -180) & (df['longitude'] <= 180)]
                
                plt.hexbin(valid_coords['longitude'], valid_coords['latitude'], 
                           gridsize=80, cmap='YlOrRd', bins='log', mincnt=1)
                
                plt.colorbar(label='Density (log10 scale)')
                plt.title(f'Global Heatmap - {file_prefix}')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.tight_layout()
                plt.savefig(os.path.join(file_output_dir, '3_geospatial_heatmap.png'), dpi=300)
                plt.close()
                
            print(f"[SUCCESS] Charts saved in: {file_output_dir}/\n")
            
        except Exception as e:
            print(f"[ERROR] Could not process {filename}. Error: {e}\n")

    print("Batch EDA Complete!")

# ==========================================
# HOW TO USE THE SCRIPT
# ==========================================
# 1. CLEANED_DATA_FOLDER: The folder where you saved the output from the first script.
# 2. EDA_OUTPUT_FOLDER: The main folder where you want all the charts to go.

CLEANED_DATA_FOLDER = '../clean_data' 
EDA_OUTPUT_FOLDER = './EDA_Results'

# Run the function
perform_eda_on_folder(CLEANED_DATA_FOLDER, EDA_OUTPUT_FOLDER)