#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Extraction Script for NR Clinical Trial

This script extracts data from various modalities using the standardized extraction modules.

Author: medicabg
Updated: May 5, 2025
"""

import pandas as pd
import os
import sys
from pathlib import Path
import time
from datetime import date

# Import custom modules
sys.path.append(os.path.expanduser('~/Documents/Programs'))
from prcTools.prcGLY.GLY_extract     import GLY_extract
from prcTools.prcPWA.PWA_extract     import PWA_extract
from prcTools.prcLCI.LCI_extract     import LCI_extract
from prcTools.prcNIH.NIH_extract     import NIH_extract
from prcTools.prcMOSIO.MOSIO_extract import MOSIO_extract

# Define paths
ROOT = Path('/Users/medicabg/Library/CloudStorage/OneDrive-UniversityofOklahoma/OU GeroLab shared data/Projects/NR Clinical Trial/data')
STUDY_ID = 'NRA'

def main():
    """Run data extraction for all modalities"""
    print(f"Starting data extraction for {STUDY_ID} study...")
    start_time = time.time()
    
    # Get validation file
    validation_path = ROOT / '.validation_results'
    validation_files = sorted(os.listdir(validation_path))
    validation_maps = [f for f in validation_files if "validation_mappings" in f]
    validation_file = validation_path / validation_maps[-1]  # Get the latest file
    
    print(f"Using validation file: {validation_file}")
    file_maps = pd.read_csv(validation_file)
    
    # Set output directory
    timestamp = date.today().strftime('%Y%m%d')
    output_dir = ROOT / 'extracted' / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    results = {}
    
    # Extract GLY data
    print("\nExtracting GLY data...")
    gly_path = ROOT / 'GLY' / 'raws'
    gly_output = gly_path.parent / 'extn'
    gly_output.mkdir(parents=True, exist_ok=True)
    results['GLY'] = GLY_extract(
        raw_path=gly_path,
        file_maps=file_maps,
        output_path=gly_output
    )
    
    # Extract PWA data
    print("\nExtracting PWA data...")
    pwa_path = ROOT / 'PWA' / 'raws'
    pwa_output = pwa_path.parent / 'extn'
    pwa_output.mkdir(parents=True, exist_ok=True)
    results['PWA'] = PWA_extract(
        raw_path=pwa_path,
        file_maps=file_maps,
        output_path=pwa_output
    )
    
    # Extract LCI data
    print("\nExtracting LCI data...")
    lci_path = ROOT / 'LCI' / 'raws'
    segm_file = ROOT / 'LCI' / 'extn' / 'lci_segment2025-05-04.csv'
    lci_output = lci_path.parent / 'extn'
    lci_output.mkdir(parents=True, exist_ok=True)
    results['LCI'] = LCI_extract(
        raw_path=lci_path,
        file_maps=file_maps,
        output_path=lci_output,
        segm_file=segm_file
    )
    
    # Extract NIH data
    print("\nExtracting NIH Toolbox data...")
    nih_path = ROOT / 'NIH' / 'raws'
    nih_output = nih_path.parent / 'extn'
    nih_output.mkdir(parents=True, exist_ok=True)
    nih_result = NIH_extract(
        root_path=nih_path,
        assessment_path   = '2024-11-06 10.54.06 Assessment Scores.csv',
        registration_path = '2024-11-06 10.54.06 Registration Data.csv',
        study_id=STUDY_ID,
        output_path=nih_output
    )
    if nih_result and 'data' in nih_result:
        results['NIH'] = nih_result['data']
    
    # Extract MOSIO data
    print("\nExtracting MOSIO adherence data...")
    mosio_path = ROOT / 'MOSIO' / 'raws'
    export_files = sorted([f for f in os.listdir(mosio_path) if f.endswith('.csv')])
    mosio_file = mosio_path / export_files[-1]  # Get the latest file
    mosio_output = mosio_path.parent / 'extn'
    results['MOSIO'] = MOSIO_extract(
        input_file=mosio_file,
        output_path=mosio_output,
        study_id=STUDY_ID
    )
    
    # Create summary report
    summary = []
    for modality, df in results.items():
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            # Determine ID column
            id_col = next((col for col in ['ID', 'PIN', 'PatientID', 'Name'] 
                           if col in df.columns), None)
            
            # Count subjects if possible
            subjects = df[id_col].nunique() if id_col else 'Unknown'
            
            summary.append({
                'Modality': modality,
                'Records': len(df),
                'Subjects': subjects,
                'Variables': len(df.columns)
            })
    
    # Save and display summary
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(output_dir / 'extraction_summary.csv', index=False)
        
        print("\nExtraction Summary:")
        print(summary_df.to_string(index=False))
    
    # Display execution time
    execution_time = time.time() - start_time
    print(f"\nExtraction complete! Total time: {execution_time:.1f} seconds")
    
    return results

if __name__ == "__main__":
    main()