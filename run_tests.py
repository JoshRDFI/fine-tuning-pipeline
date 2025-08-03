#!/usr/bin/env python3
"""
Simple test runner for the Multi-Modal RAG Pipeline
"""

import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✓ SUCCESS")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("✗ FAILED")
        print(f"Error: {e}")
        if e.stdout:
            print("Stdout:")
            print(e.stdout)
        if e.stderr:
            print("Stderr:")
            print(e.stderr)
        return False

def main():
    print("Multi-Modal RAG Pipeline - Test Runner")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("scripts").exists():
        print("Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Test Phase 1
    success1 = run_command(
        "python tests/test_phase1_extraction.py",
        "Phase 1: PDF Extraction Tests"
    )
    
    # Test Phase 2 (if Phase 1 succeeded)
    success2 = False
    if success1:
        success2 = run_command(
            "python -c \"from scripts.enrich_metadata import MetadataEnricher; print('Phase 2 module imports successfully')\"",
            "Phase 2: Metadata Enrichment Module Test"
        )
    
    # Test Phase 3
    success3 = run_command(
        "python -c \"from scripts.build_db import DatabaseBuilder; print('Phase 3 module imports successfully')\"",
        "Phase 3: Database Construction Module Test"
    )
    
    # Test Phase 4
    success4 = run_command(
        "python -c \"from scripts.prepare_training import TrainingDataPreparer; print('Phase 4 module imports successfully')\"",
        "Phase 4: Training Data Preparation Module Test"
    )
    
    # Test Phase 5
    success5 = run_command(
        "python -c \"from scripts.fine_tune import ModelFineTuner; print('Phase 5 module imports successfully')\"",
        "Phase 5: Model Fine-Tuning Module Test"
    )
    
    # Test Phase 6
    success6 = run_command(
        "python -c \"from scripts.rag_pipeline import RAGPipeline; print('Phase 6 module imports successfully')\"",
        "Phase 6: RAG Pipeline Module Test"
    )
    
    # Test Phase 7
    success7 = run_command(
        "python -c \"from scripts.streamlit_app import StreamlitApp; print('Phase 7 module imports successfully')\"",
        "Phase 7: Streamlit UI Module Test"
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    results = [
        ("Phase 1: PDF Extraction", success1),
        ("Phase 2: Metadata Enrichment", success2),
        ("Phase 3: Database Construction", success3),
        ("Phase 4: Training Data Preparation", success4),
        ("Phase 5: Model Fine-Tuning", success5),
        ("Phase 6: RAG Pipeline", success6),
        ("Phase 7: Streamlit UI", success7)
    ]
    
    passed = 0
    for phase, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{phase:<30} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} phases passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The pipeline is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 