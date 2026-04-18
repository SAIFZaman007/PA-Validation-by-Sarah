"""
NLP Policy Extractor
Extracts medical codes (CPT, ICD-10) and coverage rules from policy documents
"""

import re
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from core.config import CPT_PATTERN, ICD10_PATTERN, NLP_CONFIG

class PolicyExtractor:
    """Extract structured data from unstructured policy documents"""
    
    def __init__(self):
        self.cpt_pattern = re.compile(CPT_PATTERN)
        self.icd10_pattern = re.compile(ICD10_PATTERN)
        
        # Keywords that indicate prior authorization requirements
        self.pa_keywords = [
            'prior authorization', 'pre-authorization', 'preauthorization',
            'requires authorization', 'authorization required', 'authorization mandatory',
            'must be approved', 'approval required'
        ]
        
        # Keywords for prerequisites
        self.prerequisite_keywords = [
            'must', 'required', 'mandatory', 'necessary',
            'failed', 'documented', 'completed'
        ]
        
    def extract_cpt_codes(self, text):
        """
        Extract CPT codes from text
        
        Args:
            text: String containing policy text
            
        Returns:
            list: Unique CPT codes found
        """
        codes = self.cpt_pattern.findall(text)
        return list(set(codes))  # Remove duplicates
    
    def extract_icd10_codes(self, text):
        """
        Extract ICD-10 codes from text
        
        Args:
            text: String containing diagnosis text
            
        Returns:
            list: Unique ICD-10 codes found
        """
        codes = self.icd10_pattern.findall(text)
        return list(set(codes))
    
    def requires_prior_auth(self, text):
        """
        Determine if text indicates prior authorization is required
        
        Args:
            text: Policy rule text
            
        Returns:
            bool: True if PA is required
        """
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.pa_keywords)
    
    def extract_prerequisites(self, text):
        """
        Extract prerequisite requirements from policy text
        
        Args:
            text: Policy rule text
            
        Returns:
            list: Extracted prerequisite statements
        """
        prerequisites = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if sentence contains prerequisite keywords
            if any(keyword in sentence.lower() for keyword in self.prerequisite_keywords):
                prerequisites.append(sentence)
        
        return prerequisites
    
    def extract_numeric_requirements(self, text):
        """
        Extract numeric requirements (ages, BMI, durations, etc.)
        
        Args:
            text: Policy text
            
        Returns:
            dict: Numeric requirements found
        """
        requirements = {}
        
        # Age requirements (e.g., "Age >= 45", "patients over 50")
        age_pattern = r'(?:age|Age)\s*(?:>=|>|≥)\s*(\d+)'
        age_matches = re.findall(age_pattern, text)
        if age_matches:
            requirements['min_age'] = int(age_matches[0])
        
        # BMI requirements (e.g., "BMI < 40", "BMI under 35")
        bmi_pattern = r'(?:BMI|bmi)\s*(?:<=|<|≤)\s*(\d+)'
        bmi_matches = re.findall(bmi_pattern, text)
        if bmi_matches:
            requirements['max_bmi'] = int(bmi_matches[0])
        
        # Duration requirements (e.g., "6 months", "12 weeks")
        duration_pattern = r'(\d+)\s+(month|months|week|weeks|day|days)'
        duration_matches = re.findall(duration_pattern, text)
        if duration_matches:
            for value, unit in duration_matches:
                if 'month' in unit:
                    requirements['duration_months'] = int(value)
                elif 'week' in unit:
                    requirements['duration_weeks'] = int(value)
                elif 'day' in unit:
                    requirements['duration_days'] = int(value)
        
        return requirements
    
    def extract_policy_rules(self, policy_document):
        """
        Extract all rules from a policy document
        
        Args:
            policy_document: Dict containing policy data
            
        Returns:
            list: Extracted and structured rules
        """
        extracted_rules = []
        
        for rule in policy_document.get('coverage_rules', []):
            # Extract CPT code
            cpt_codes = self.extract_cpt_codes(str(rule))
            
            # Check if it's already in the rule dict
            if 'cpt_code' in rule:
                cpt_code = rule['cpt_code']
            elif cpt_codes:
                cpt_code = cpt_codes[0]
            else:
                cpt_code = None
            
            # Extract from coverage criteria text
            criteria_text = rule.get('coverage_criteria', '')
            
            extracted = {
                'cpt_code': cpt_code,
                'procedure_name': rule.get('procedure_name', ''),
                'category': rule.get('category', ''),
                'requires_pa': self.requires_prior_auth(criteria_text),
                'prerequisites': self.extract_prerequisites(criteria_text),
                'numeric_requirements': self.extract_numeric_requirements(criteria_text),
                'original_text': criteria_text
            }
            
            extracted_rules.append(extracted)
        
        return extracted_rules
    
    def match_request_to_policy(self, pa_request, policy_rules):
        """
        Match a PA request to relevant policy rules
        
        Args:
            pa_request: PA request dict
            policy_rules: List of extracted policy rules
            
        Returns:
            dict: Matching rule or None
        """
        request_cpt = pa_request.get('requested_procedure', {}).get('cpt_code')
        
        if not request_cpt:
            return None
        
        # Find matching rule by CPT code
        for rule in policy_rules:
            if rule['cpt_code'] == request_cpt:
                return rule
        
        return None
    
    def check_prerequisite_compliance(self, pa_request, policy_rule):
        """
        Check if a PA request meets policy prerequisites
        
        Args:
            pa_request: PA request dict
            policy_rule: Policy rule dict
            
        Returns:
            dict: Compliance status for each prerequisite
        """
        compliance = {
            'all_met': False,
            'details': []
        }
        
        prerequisites_met = pa_request.get('clinical_info', {}).get('prerequisites_met', [])
        required_prerequisites = policy_rule.get('prerequisites', [])
        
        if not required_prerequisites:
            compliance['all_met'] = True
            return compliance
        
        for prereq in required_prerequisites:
            is_met = prereq in prerequisites_met
            compliance['details'].append({
                'requirement': prereq,
                'met': is_met
            })
        
        compliance['all_met'] = all(item['met'] for item in compliance['details'])
        
        return compliance
    
    def extract_key_terms(self, text, top_n=10):
        """
        Extract most important terms from text using TF-IDF
        
        Args:
            text: Input text
            top_n: Number of top terms to return
            
        Returns:
            list: Top N important terms
        """
        vectorizer = TfidfVectorizer(
            max_features=NLP_CONFIG['max_features'],
            stop_words='english',
            ngram_range=NLP_CONFIG['ngram_range']
        )
        
        try:
            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get scores
            scores = tfidf_matrix.toarray()[0]
            
            # Get top N terms
            top_indices = np.argsort(scores)[-top_n:][::-1]
            top_terms = [feature_names[i] for i in top_indices if scores[i] > 0]
            
            return top_terms
        except:
            return []


# Demo/Test code
if __name__ == "__main__":
    print("=" * 80)
    print("NLP POLICY EXTRACTOR - DEMO")
    print("=" * 80)
    print()
    
    # Initialize extractor
    extractor = PolicyExtractor()
    
    # Load synthetic data
    policies_file = Path("data/synthetic/synthetic_policies.json")
    requests_file = Path("data/synthetic/synthetic_pa_requests.json")
    
    with open(policies_file, 'r') as f:
        policies = json.load(f)
    
    with open(requests_file, 'r') as f:
        requests = json.load(f)
    
    # Test on first policy
    print("📄 Testing on First Policy")
    print("-" * 80)
    policy = policies[0]
    print(f"Policy: {policy['policy_id']} ({policy['insurer']})")
    print()
    
    # Extract rules
    extracted_rules = extractor.extract_policy_rules(policy)
    
    print(f"Extracted {len(extracted_rules)} rules:")
    for i, rule in enumerate(extracted_rules, 1):
        print(f"\n  Rule {i}:")
        print(f"    CPT Code: {rule['cpt_code']}")
        print(f"    Procedure: {rule['procedure_name']}")
        print(f"    Requires PA: {rule['requires_pa']}")
        print(f"    Prerequisites: {len(rule['prerequisites'])} found")
        if rule['prerequisites']:
            for prereq in rule['prerequisites']:
                print(f"      - {prereq}")
        if rule['numeric_requirements']:
            print(f"    Numeric Requirements: {rule['numeric_requirements']}")
    
    # Test matching
    print("\n" + "=" * 80)
    print("📋 Testing Request Matching")
    print("-" * 80)
    
    request = requests[0]
    print(f"Request: {request['request_id']}")
    print(f"Procedure: {request['requested_procedure']['name']} (CPT: {request['requested_procedure']['cpt_code']})")
    print()
    
    # Find matching rule
    matching_rule = extractor.match_request_to_policy(request, extracted_rules)
    
    if matching_rule:
        print("✓ Found matching policy rule!")
        print(f"  Rule: {matching_rule['procedure_name']}")
        print(f"  Requires PA: {matching_rule['requires_pa']}")
        
        # Check compliance
        compliance = extractor.check_prerequisite_compliance(request, matching_rule)
        print(f"\n  Prerequisite Compliance: {'✓ ALL MET' if compliance['all_met'] else '✗ NOT ALL MET'}")
        
        for detail in compliance['details']:
            status = '✓' if detail['met'] else '✗'
            print(f"    {status} {detail['requirement']}")
    else:
        print("✗ No matching policy rule found")
    
    # Test key term extraction
    print("\n" + "=" * 80)
    print("🔑 Key Terms Extraction")
    print("-" * 80)
    
    sample_text = matching_rule['original_text'] if matching_rule else extracted_rules[0]['original_text']
    key_terms = extractor.extract_key_terms(sample_text, top_n=5)
    
    print(f"Sample text: {sample_text[:100]}...")
    print(f"\nTop 5 key terms:")
    for term in key_terms:
        print(f"  - {term}")
    
    # Statistics
    print("\n" + "=" * 80)
    print("📊 Extraction Statistics")
    print("-" * 80)
    
    total_rules = sum(len(extractor.extract_policy_rules(p)) for p in policies)
    total_pa_required = sum(
        1 for p in policies 
        for rule in extractor.extract_policy_rules(p) 
        if rule['requires_pa']
    )
    
    print(f"Total rules extracted: {total_rules}")
    print(f"Rules requiring PA: {total_pa_required} ({total_pa_required/total_rules*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("✅ NLP EXTRACTION DEMO COMPLETE")
    print("=" * 80)