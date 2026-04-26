"""
Feature Engineering for PA Approval Prediction
Converts extracted policy and request data into ML-ready features
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.nlp.policy_extractor import PolicyExtractor

class FeatureEngineer:
    """Transform PA requests and policies into ML features"""
    
    def __init__(self):
        self.extractor = PolicyExtractor()
        self.feature_names = []
    
    def create_features(self, pa_request, policy_rules):
        """
        Create feature vector for a PA request
        
        Args:
            pa_request: PA request dictionary
            policy_rules: List of extracted policy rules
            
        Returns:
            dict: Feature dictionary
        """
        features = {}
        
        # Match request to policy
        matching_rule = self.extractor.match_request_to_policy(pa_request, policy_rules)
        
        # Feature 1: Does procedure require PA?
        features['requires_pa'] = 1 if (matching_rule and matching_rule['requires_pa']) else 0
        
        # Feature 2: Number of prerequisites
        if matching_rule:
            features['num_prerequisites'] = len(matching_rule.get('prerequisites', []))
        else:
            features['num_prerequisites'] = 0
        
        # Feature 3: Prerequisite compliance rate
        if matching_rule and matching_rule.get('prerequisites'):
            compliance = self.extractor.check_prerequisite_compliance(pa_request, matching_rule)
            total_prereqs = len(compliance['details'])
            met_prereqs = sum(1 for d in compliance['details'] if d['met'])
            features['prerequisite_compliance_rate'] = met_prereqs / total_prereqs if total_prereqs > 0 else 0
        else:
            features['prerequisite_compliance_rate'] = 1.0  # No prereqs = 100% compliant
        
        # Feature 4-5: Patient demographics
        features['patient_age'] = pa_request.get('patient', {}).get('age', 0)
        features['patient_gender_male'] = 1 if pa_request.get('patient', {}).get('gender') == 'Male' else 0
        
        # Feature 6: BMI (if available)
        bmi = pa_request.get('clinical_info', {}).get('bmi')
        features['bmi'] = bmi if (bmi is not None and bmi > 0) else 30.0  
        
        # Feature 7: Conservative therapy duration
        therapy_weeks = pa_request.get('clinical_info', {}).get('conservative_therapy_duration_weeks')
        features['therapy_duration_weeks'] = therapy_weeks if (therapy_weeks is not None and therapy_weeks >= 0) else 0
        
        # Feature 8: Imaging completed
        imaging = pa_request.get('clinical_info', {}).get('imaging_completed')
        features['imaging_completed'] = 1 if imaging is True else 0
        
        # Feature 9-10: Procedure category (one-hot encoded)
        category = pa_request.get('requested_procedure', {}).get('category', '')
        features['category_orthopedic'] = 1 if 'Orthopedic' in category else 0
        features['category_radiology'] = 1 if 'Radiology' in category else 0
        
        # Feature 11: Check if meets numeric requirements
        features['meets_numeric_requirements'] = self._check_numeric_requirements(
            pa_request, matching_rule
        )
        
        # Feature 12-13: Time-based features
        features['day_of_week'] = pd.Timestamp(pa_request['timestamp']).dayofweek
        features['hour_of_day'] = pd.Timestamp(pa_request['timestamp']).hour
        
        # Feature 14: Insurer encoding
        insurer = pa_request.get('insurer', '')
        features['insurer_bupa'] = 1 if 'Bupa' in insurer else 0
        features['insurer_tawuniya'] = 1 if 'Tawuniya' in insurer else 0
        
        # Feature 15: Has diagnosis code
        features['has_diagnosis'] = 1 if pa_request.get('diagnosis', {}).get('icd10_code') else 0
        
        return features
    
    def _check_numeric_requirements(self, pa_request, matching_rule):
        """Check if request meets numeric requirements in policy"""
        if not matching_rule or not matching_rule.get('numeric_requirements'):
            return 1  # No requirements = pass
        
        requirements = matching_rule['numeric_requirements']
        patient_age = pa_request.get('patient', {}).get('age', 0) or 0
        patient_bmi = pa_request.get('clinical_info', {}).get('bmi')
        patient_bmi = patient_bmi if (patient_bmi is not None and patient_bmi > 0) else 30
        therapy_weeks = pa_request.get('clinical_info', {}).get('conservative_therapy_duration_weeks')
        therapy_weeks = therapy_weeks if (therapy_weeks is not None and therapy_weeks >= 0) else 0
        
        # Check age requirement
        if 'min_age' in requirements:
            if patient_age < requirements['min_age']:
                return 0
        
        # Check BMI requirement
        if 'max_bmi' in requirements:
            if patient_bmi > requirements['max_bmi']:
                return 0
        
        # Check therapy duration
        if 'duration_weeks' in requirements:
            if therapy_weeks < requirements['duration_weeks']:
                return 0
        
        return 1  # All requirements met
    
    def create_dataset(self, requests, policies):
        """
        Create complete feature dataset from requests and policies
        
        Args:
            requests: List of PA requests
            policies: List of policy documents
            
        Returns:
            pandas.DataFrame: Feature matrix with labels
        """
        print(f"Creating features for {len(requests)} requests...")
        
        # Extract rules from all policies
        all_rules = []
        for policy in policies:
            rules = self.extractor.extract_policy_rules(policy)
            all_rules.extend(rules)
        
        print(f"Extracted {len(all_rules)} policy rules")
        
        # Create features for each request
        feature_list = []
        labels = []
        
        for i, request in enumerate(requests):
            if (i + 1) % 20 == 0:
                print(f"  Processing request {i+1}/{len(requests)}...")
            
            features = self.create_features(request, all_rules)
            feature_list.append(features)
            
            # Create label from approval_likelihood
            likelihood = request.get('approval_likelihood_label', 'medium')
            if likelihood == 'high':
                label = 1  # Approve
            elif likelihood == 'medium':
                label = 0  # Review needed (we'll treat as denial for binary classification)
            else:  # low
                label = 0  # Deny
            
            labels.append(label)
        
        # Convert to DataFrame
        df = pd.DataFrame(feature_list)
        df['label'] = labels
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col != 'label']
        
        print(f"\n✓ Created dataset with {len(df)} samples and {len(self.feature_names)} features")
        print(f"  Features: {self.feature_names}")
        print(f"\n  Label distribution:")
        print(f"    Approve (1): {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
        print(f"    Deny (0): {len(labels)-sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")
        
        return df
    
    def save_dataset(self, df, output_path):
        """Save feature dataset to CSV"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"\n✓ Saved dataset to: {output_path}")
        
        return output_path


# Demo/Test code
if __name__ == "__main__":
    print("=" * 80)
    print("FEATURE ENGINEERING - DEMO")
    print("=" * 80)
    print()
    
    # Load synthetic data
    policies_file = Path("data/synthetic/synthetic_policies.json")
    requests_file = Path("data/synthetic/synthetic_pa_requests.json")
    
    print("Loading data...")
    with open(policies_file, 'r') as f:
        policies = json.load(f)
    
    with open(requests_file, 'r') as f:
        requests = json.load(f)
    
    print(f"✓ Loaded {len(policies)} policies and {len(requests)} requests")
    print()
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Test on single request
    print("=" * 80)
    print("Testing Single Request Feature Extraction")
    print("=" * 80)
    
    request = requests[0]
    print(f"Request: {request['request_id']}")
    print(f"Procedure: {request['requested_procedure']['name']}")
    print(f"Expected outcome: {request['approval_likelihood_label']}")
    print()
    
    # Extract policy rules
    extractor = PolicyExtractor()
    all_rules = []
    for policy in policies:
        rules = extractor.extract_policy_rules(policy)
        all_rules.extend(rules)
    
    # Create features
    features = engineer.create_features(request, all_rules)
    
    print("Extracted Features:")
    print("-" * 80)
    for key, value in features.items():
        print(f"  {key:35s}: {value}")
    
    # Create full dataset
    print("\n" + "=" * 80)
    print("Creating Full Feature Dataset")
    print("=" * 80)
    print()
    
    df = engineer.create_dataset(requests, policies)
    
    # Show sample
    print("\n" + "=" * 80)
    print("Sample of Feature Matrix (first 5 rows)")
    print("=" * 80)
    print(df.head())
    
    # Save dataset
    print("\n" + "=" * 80)
    print("Saving Dataset")
    print("=" * 80)
    
    output_path = Path("data/processed/pa_features.csv")
    engineer.save_dataset(df, output_path)
    
    # Statistics
    print("\n" + "=" * 80)
    print("Dataset Statistics")
    print("=" * 80)
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nFeature Statistics:")
    print(df.describe())
    
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 80)