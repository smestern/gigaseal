#!/usr/bin/env python3
"""
Test script for testing the from_dataframe method with demo data
"""

import sys
import os
import pandas as pd
import pytest
import logging
logging.basicConfig(level=logging.INFO)

# Add the parent directory to sys.path to import gigaseal
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gigaseal.database.tsDatabase import tsDatabase

def test_demo_dataframe_import():
    """Test importing the demo arbitrary data CSV"""
    
    # Load the demo data
    demo_path = os.path.join(os.path.dirname(__file__), 'test_data', 'demo_arb_data.csv')
    
    if not os.path.exists(demo_path):
        pytest.skip(f"Demo data file not found: {demo_path}")
        
    df = pd.read_csv(demo_path, skiprows=2)
    print(f"Loaded demo data with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Create database instance
    db = tsDatabase()
    
    # Test auto-detection of protocol columns
    result = db.from_dataframe(
        df, 
        cell_id_col='CELL_ID',
        metadata_cols=['DATE', 'drug', 'NOTE', 'Burst Adex', 'Burst Cadex']
    )
    
    if result:
        print("\u2705 Successfully imported demo data with auto-detection")
        print(f"Database contains {len(db.cellindex)} cells")
        print(f"Protocol columns: {len(db.cellindex.columns)} total columns")

        # Show some sample data
        print("\nSample cells:")
        for i, (cell_name, cell_data) in enumerate(db.getCells().items()):
            if i >= 3:  # Show first 3 cells
                break
            print(f"  {cell_name}: {len([v for v in cell_data.values() if v is not None and v != ''])} protocols")

    assert result, "from_dataframe returned False on auto-detection"
    assert len(db.cellindex) > 0
    # IC1 is a known protocol column in the demo data
    assert "IC1" in db.cellindex.columns
    assert "IC1" in db.get_protocol_columns()

def test_specific_protocols():
    """Test with manually specified protocol columns"""
    
    demo_path = os.path.join(os.path.dirname(__file__), 'test_data', 'demo_arb_data.csv')
    df = pd.read_csv(demo_path, skiprows=2)
    
    # Create database instance
    db = tsDatabase()
    
    # Test with specific protocol columns
    specific_protocols = ['IC1', 'CTRL_PULSE', 'NET_PULSE', 'DYN_CFG1_EXP3', 'DYN_CFG1_EXP4']
    
    result = db.from_dataframe(
        df,
        filename_cols=specific_protocols,
        cell_id_col='CELL_ID', 
        metadata_cols=['DATE', 'drug', 'NOTE']
    )
    
    if result:
        print("\u2705 Successfully imported demo data with specific protocols")
        print(f"Database contains {len(db.cellindex)} cells")

        # Check that our specific protocols are there
        for protocol in specific_protocols:
            if protocol in db.cellindex.columns:
                non_empty = db.cellindex[protocol].notna().sum()
                print(f"  {protocol}: {non_empty} cells have this protocol")
            else:
                print(f"  {protocol}: NOT FOUND in database")

    assert result, "from_dataframe returned False with specific protocols"
    for protocol in specific_protocols:
        assert protocol in db.cellindex.columns, f"{protocol} missing"
        assert protocol in db.get_protocol_columns()

def test_indiv_rows():
    """Test that individual rows are correctly imported into the database"""
    demo_path = os.path.join(os.path.dirname(__file__), 'test_data', 'demo_arb_data.csv')
    if not os.path.exists(demo_path):
        pytest.skip(f"Demo data file not found: {demo_path}")
    df = pd.read_csv(demo_path, skiprows=2)
    db = tsDatabase()
    result = db.from_dataframe(
        df,
        cell_id_col='CELL_ID',
        metadata_cols=['DATE', 'drug', 'NOTE', 'Burst Adex', 'Burst Cadex']
    )
    assert result, "from_dataframe returned False in test_indiv_rows"
    # Check a few individual rows
    sample_rows = df.head(3)
    for idx, row in sample_rows.iterrows():
        cell_id = row['CELL_ID']
        assert cell_id in db.cellindex.index, (
            f"Cell ID {cell_id} not found in database index"
        )
if __name__ == "__main__":
    print("Testing tsDatabase.from_dataframe() with demo data...")
    print("=" * 60)
    
    # Test 1: Auto-detection
    print("Test 1: Auto-detection of protocol columns")
    success1 = test_demo_dataframe_import()
    
    print("\n" + "=" * 60)
    
    # Test 2: Specific protocols
    print("Test 2: Manually specified protocol columns")
    success2 = test_specific_protocols()
    
    print("\n" + "=" * 60)
    

