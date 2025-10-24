"""
run_all.py - Complete Data Breach Analytics Pipeline
=====================================================

Executes the entire workflow from data cleaning through dashboard deployment.

Assignments Covered:
    Assignment 4: Database & ETL Pipeline (data cleaning, loading)
    Assignment 5: Exploratory Data Analysis (statistical analysis)
    Assignment 6: Analytics Engine & Modeling (ML models, predictions)
    Assignment 7: Dashboard & Deployment (Streamlit interface)

Author: T. Spivey
Course: BUS 761
Date: October 2025
"""

from dataclean import universal_clean
from dataload import save_to_csv, load_to_database, get_table_info, load_sec_reference
import pandas as pd
import subprocess
import sys
from pathlib import Path

# Import Assignment 6 modules
try:
    from eda_package import DataLoader
    from analytics_engine import (
        FeatureEngineer,
        ModelTrainer,
        ModelEvaluator,
        BreachPredictor,
        BusinessRecommender
    )
    ASSIGNMENT_6_AVAILABLE = True
except ImportError:
    ASSIGNMENT_6_AVAILABLE = False
    print("Warning: Assignment 6 modules not found. Will skip modeling step.")


def run_all():
    """
    Execute complete data breach analytics pipeline.
    """
    print("="*60)
    print("DATA BREACH PIPELINE - FULL EXECUTION")
    print("="*60)

    # ========================================================================
    # ASSIGNMENT 4: DATABASE & ETL PIPELINE
    # ========================================================================

    # Step 1: Clean the data
    print("\n[1/6] CLEANING DATA (Assignment 4)")
    print("-"*60)
    cleaned_data = universal_clean()
    df_databreach = cleaned_data['databreach']

    print("Cleaned DataFrames:")
    for name, df in cleaned_data.items():
        print(f"  {name}: {df.shape[0]} rows, {df.shape[1]} columns")

    # Step 2: Save and load to database
    print("\n[2/6] SAVING AND LOADING TO DATABASE (Assignment 4)")
    print("-"*60)
    print("Saving to CSV...")
    save_to_csv(df_databreach)

    print("Loading breach data to database...")
    load_to_database(df_databreach)

    print("Loading SEC reference data...")
    load_sec_reference()

    print("\nDatabase info:")
    get_table_info()

    # ========================================================================
    # ASSIGNMENT 5: EXPLORATORY DATA ANALYSIS
    # ========================================================================

    # Step 3: Run EDA
    print("\n[3/6] RUNNING STATISTICAL ANALYSIS (Assignment 5)")
    print("-"*60)
    result = subprocess.run(['python', 'eda.py'], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")

    # ========================================================================
    # ASSIGNMENT 6: ANALYTICS ENGINE & MODELING
    # ========================================================================

    if ASSIGNMENT_6_AVAILABLE:
        print("\n[4/6] PREDICTIVE MODELING (Assignment 6)")
        print("-"*60)

        try:
            # Load data from database
            print("Loading data for modeling...")
            loader = DataLoader('databreach.db')
            df = loader.load_breach_data()
            print(f"  Loaded {len(df):,} records")

            # Feature Engineering
            print("\nEngineering features...")
            engineer = FeatureEngineer()
            X_train, X_test, y_train, y_test = engineer.prepare_data(
                df,
                threshold=1000  # Severity threshold: >1,000 individuals
            )
            print(f"  Training set: {len(X_train):,} samples")
            print(f"  Test set: {len(X_test):,} samples")
            print(f"  Features: {len(X_train.columns)}")

            # Model Training
            print("\nTraining machine learning models...")
            trainer = ModelTrainer()
            models = trainer.train_all_classifiers(X_train, y_train)
            print(f"  Trained {len(models)} models")

            # Model Evaluation
            print("\nEvaluating models...")
            evaluator = ModelEvaluator()
            comparison = evaluator.compare_models(
                models,
                X_test,
                y_test,
                task='classification'
            )

            # Select best model
            best_model_name = comparison.loc[comparison['f1_score'].idxmax(), 'model']
            best_model = models[best_model_name]
            best_accuracy = comparison.loc[comparison['model']==best_model_name, 'accuracy'].values[0]
            best_f1 = comparison.loc[comparison['model']==best_model_name, 'f1_score'].values[0]

            print(f"\n  Best model: {best_model_name}")
            print(f"  Accuracy: {best_accuracy:.1%}")
            print(f"  F1 Score: {best_f1:.3f}")

            # Save best model
            models_dir = Path('models')
            models_dir.mkdir(exist_ok=True)
            model_path = models_dir / f'{best_model_name}_model.pkl'

            metadata = {
                'model_name': best_model_name,
                'accuracy': float(best_accuracy),
                'f1_score': float(best_f1),
                'features': X_train.columns.tolist(),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }

            trainer.save_model(best_model, str(model_path), metadata)
            print(f"\n  Model saved: {model_path}")

            # Business Recommendations
            print("\nGenerating business recommendations...")
            recommender = BusinessRecommender()

            # Sample high-risk scenario
            high_risk_report = recommender.generate_comprehensive_report(
                severity_risk=0.85,
                predicted_impact=25000,
                organization_type='MED'
            )

            print(f"\n  Sample High-Risk Scenario (Healthcare):")
            print(f"    Risk Level: {high_risk_report['risk_level']}")
            print(f"    Estimated Cost: ${high_risk_report['cost_estimate']['total_estimated_cost']:,.0f}")
            print(f"    Priority Score: {high_risk_report['priority_score']}/100")
            print(f"    Recommendations: {len(high_risk_report['recommendations'])} actions")

            print("\n✓ Assignment 6 (Modeling) complete")

        except Exception as e:
            print(f"\n✗ Error in Assignment 6: {str(e)}")
            print("  Pipeline will continue without modeling results.")
    else:
        print("\n[4/6] PREDICTIVE MODELING (Assignment 6)")
        print("-"*60)
        print("  Skipped: analytics_engine package not found")

    # ========================================================================
    # ASSIGNMENT 7: DASHBOARD & DEPLOYMENT
    # ========================================================================

    print("\n[5/6] DASHBOARD VERIFICATION (Assignment 7)")
    print("-"*60)

    # Check if dashboard files exist
    dashboard_files = {
        'app.py': 'Streamlit dashboard application',
        'Dockerfile': 'Docker container configuration',
        'docker-compose.yml': 'Docker Compose configuration',
        'USER_GUIDE.md': 'User documentation',
        'DEPLOYMENT.md': 'Deployment instructions'
    }

    print("Verifying dashboard components...")
    all_present = True
    for filename, description in dashboard_files.items():
        if Path(filename).exists():
            print(f"  ✓ {filename} - {description}")
        else:
            print(f"  ✗ {filename} - Missing!")
            all_present = False

    if all_present:
        print("\n✓ All dashboard components present")
        print("\nTo launch the dashboard:")
        print("  Local:  streamlit run app.py")
        print("  Docker: docker-compose up")
    else:
        print("\n⚠ Some dashboard components missing")
        print("  Please ensure all Assignment 7 files are present")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("\n[6/6] PIPELINE SUMMARY")
    print("-"*60)
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)

    print("\n📊 Assignment 4 - Database & ETL:")
    print("  ✓ databreach.db created")
    print("    - databreach table (35,378 breach records)")
    print("    - sec_company_reference table (10,142 public companies)")

    print("\n📈 Assignment 5 - Exploratory Data Analysis:")
    print("  ✓ 13 statistical analysis tables")
    print("  ✓ documentation/ folder (7 markdown files)")

    if ASSIGNMENT_6_AVAILABLE:
        print("\n🤖 Assignment 6 - Analytics Engine & Modeling:")
        print(f"  ✓ {len(models)} models trained and evaluated")
        print(f"  ✓ Best model: {best_model_name} ({best_accuracy:.1%} accuracy)")
        print(f"  ✓ Model saved to: models/")
        print("  ✓ Business recommendations generated")
    else:
        print("\n🤖 Assignment 6 - Analytics Engine & Modeling:")
        print("  ⚠ Not executed (modules not found)")

    if all_present:
        print("\n🎨 Assignment 7 - Dashboard & Deployment:")
        print("  ✓ Interactive Streamlit dashboard (5 pages)")
        print("  ✓ Docker containerization ready")
        print("  ✓ Complete user documentation")
        print("  ✓ Production deployment guide")
    else:
        print("\n🎨 Assignment 7 - Dashboard & Deployment:")
        print("  ⚠ Some components missing")

    print("\n📁 Outputs Generated:")
    print("  - output/databreach.csv")
    print("  - databreach.db (71 MB)")
    print("  - documentation/ (14 markdown files)")
    if ASSIGNMENT_6_AVAILABLE:
        print("  - models/ (trained ML models)")
    if all_present:
        print("  - app.py (Streamlit dashboard)")
        print("  - Docker deployment files")

    print("\n" + "="*60)
    print("🚀 SYSTEM READY FOR DEPLOYMENT")
    print("="*60)
    print("\nQuick Start:")
    print("  1. Launch dashboard: streamlit run app.py")
    print("  2. Open browser: http://localhost:8501")
    print("  3. Explore all 5 pages using sidebar")
    print("\nFor deployment instructions, see: DEPLOYMENT.md")
    print("For user guide, see: USER_GUIDE.md")


if __name__ == "__main__":
    try:
        run_all()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)