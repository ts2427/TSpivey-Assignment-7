"""
Business Recommender Module
============================

Translates model predictions into actionable business recommendations.

Classes:
    BusinessRecommender: Recommendation engine based on predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


class BusinessRecommender:
    """
    Generate business recommendations based on breach predictions.
    
    Translates ML predictions into actionable security recommendations,
    budget allocations, and risk mitigation strategies.
    
    Attributes:
        recommendation_rules: Dictionary of business rules
        cost_matrix: Cost estimates for different scenarios
        
    Example:
        >>> recommender = BusinessRecommender()
        >>> recommendations = recommender.generate_recommendations(
        ...     severity_risk=0.85,
        ...     predicted_impact=25000,
        ...     organization_type='MED'
        ... )
    """
    
    def __init__(self):
        """Initialize the recommender with business rules."""
        self.recommendation_rules = self._load_recommendation_rules()
        self.cost_matrix = self._load_cost_estimates()
        
    def _load_recommendation_rules(self) -> Dict:
        """
        Load business recommendation rules.
        
        Returns:
            Dictionary of recommendation rules
        """
        return {
            'severity_thresholds': {
                'low': 0.3,
                'medium': 0.7
            },
            'impact_thresholds': {
                'small': 1000,
                'medium': 10000,
                'large': 100000
            },
            'priority_levels': {
                'critical': {'severity': 0.8, 'impact': 50000},
                'high': {'severity': 0.6, 'impact': 10000},
                'medium': {'severity': 0.4, 'impact': 5000},
                'low': {'severity': 0.0, 'impact': 0}
            }
        }
    
    def _load_cost_estimates(self) -> Dict:
        """
        Load cost estimates for breach scenarios.
        
        Based on industry research (Ponemon Institute):
        - Average cost per record: $165-$460 depending on industry
        - Notification costs: $150,000-$500,000
        - Legal/regulatory: $1M-$5M for large breaches
        
        Returns:
            Dictionary of cost estimates
        """
        return {
            'cost_per_record': {
                'MED': 460,  # Healthcare
                'BSF': 380,  # Financial
                'EDU': 245,  # Education
                'GOV': 425,  # Government
                'BSR': 175,  # Retail
                'BSO': 225,  # Other business
                'default': 250
            },
            'base_notification_cost': 200000,
            'legal_base_cost': 500000,
            'pr_recovery_cost': 300000
        }
    
    def classify_risk_level(self, severity_probability: float, 
                           predicted_impact: Optional[float] = None) -> str:
        """
        Classify overall risk level.
        
        Args:
            severity_probability: Probability of severe breach (0-1)
            predicted_impact: Optional predicted number affected
            
        Returns:
            Risk level: 'Low', 'Medium', 'High', or 'Critical'
        """
        thresholds = self.recommendation_rules['severity_thresholds']
        
        if severity_probability >= 0.8:
            return 'Critical'
        elif severity_probability >= thresholds['medium']:
            return 'High'
        elif severity_probability >= thresholds['low']:
            return 'Medium'
        else:
            return 'Low'
    
    def estimate_breach_cost(self, predicted_impact: float,
                            organization_type: str = 'default') -> Dict[str, float]:
        """
        Estimate total cost of a potential breach.
        
        Args:
            predicted_impact: Number of individuals affected
            organization_type: Type of organization
            
        Returns:
            Dictionary with cost breakdown
        """
        cost_per_record = self.cost_matrix['cost_per_record'].get(
            organization_type, 
            self.cost_matrix['cost_per_record']['default']
        )
        
        record_costs = predicted_impact * cost_per_record
        notification_costs = self.cost_matrix['base_notification_cost']
        
        # Scale legal costs by breach size
        if predicted_impact > 100000:
            legal_costs = self.cost_matrix['legal_base_cost'] * 3
        elif predicted_impact > 10000:
            legal_costs = self.cost_matrix['legal_base_cost']
        else:
            legal_costs = self.cost_matrix['legal_base_cost'] * 0.3
        
        pr_costs = self.cost_matrix['pr_recovery_cost']
        
        total_cost = record_costs + notification_costs + legal_costs + pr_costs
        
        return {
            'record_costs': record_costs,
            'notification_costs': notification_costs,
            'legal_costs': legal_costs,
            'pr_costs': pr_costs,
            'total_estimated_cost': total_cost
        }
    
    def generate_security_recommendations(self, severity_risk: float,
                                          organization_type: str,
                                          breach_type: Optional[str] = None) -> List[str]:
        """
        Generate specific security recommendations.
        
        Based on Assignment 5 EDA findings:
        - Healthcare: High DISC risk (+43%)
        - Financial: High PHYS risk (+169%)
        - Retail: High CARD risk (+400%)
        
        Args:
            severity_risk: Severity probability (0-1)
            organization_type: Type of organization
            breach_type: Optional specific breach type
            
        Returns:
            List of prioritized recommendations
        """
        recommendations = []
        
        # Risk level
        risk_level = self.classify_risk_level(severity_risk)
        
        # Universal recommendations for high risk
        if severity_risk > 0.7:
            recommendations.append(
                "🔴 CRITICAL: Immediate security audit required"
            )
            recommendations.append(
                "Activate incident response team and review protocols"
            )
        
        # Industry-specific recommendations (from EDA findings)
        if organization_type == 'MED':
            recommendations.extend([
                "Healthcare: Strengthen access controls (43% excess DISC breaches)",
                "Implement role-based access control (RBAC) for PHI",
                "Mandatory data handling training for staff",
                "Deploy automated audit logging for sensitive data access"
            ])
        
        elif organization_type == 'BSF':
            recommendations.extend([
                "Financial: Enhance physical document security (169% excess PHYS breaches)",
                "Audit document storage and disposal procedures",
                "Implement secure shredding protocols",
                "Transition to digital-first document management"
            ])
        
        elif organization_type == 'BSR':
            recommendations.extend([
                "Retail: Harden POS systems (400% excess CARD breaches)",
                "Accelerate EMV chip adoption across all terminals",
                "Implement end-to-end encryption (E2EE) for payments",
                "Deploy POS monitoring and anomaly detection"
            ])
        
        elif organization_type == 'GOV':
            recommendations.extend([
                "Government: Enhanced data protection for citizen records",
                "Multi-factor authentication for all system access",
                "Regular security awareness training",
                "Incident response drills and tabletop exercises"
            ])
        
        else:
            recommendations.extend([
                "Business/Tech: Cyber attack prevention (15% excess HACK)",
                "Deploy advanced threat detection systems",
                "Regular penetration testing and vulnerability assessments",
                "Implement zero-trust architecture principles"
            ])
        
        # Budget recommendations based on risk
        if severity_risk > 0.7:
            recommendations.append(
                f"💰 Recommended security budget: 8-12% of IT budget (high risk)"
            )
        elif severity_risk > 0.5:
            recommendations.append(
                f"💰 Recommended security budget: 5-8% of IT budget (medium risk)"
            )
        
        return recommendations
    
    def generate_comprehensive_report(self, severity_risk: float,
                                     predicted_impact: float,
                                     organization_type: str,
                                     breach_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive recommendation report.
        
        Args:
            severity_risk: Severity probability
            predicted_impact: Predicted individuals affected
            organization_type: Organization type
            breach_type: Optional breach type
            
        Returns:
            Dictionary with complete recommendation package
            
        Example:
            >>> recommender = BusinessRecommender()
            >>> report = recommender.generate_comprehensive_report(
            ...     severity_risk=0.85,
            ...     predicted_impact=25000,
            ...     organization_type='MED'
            ... )
            >>> print(report['executive_summary'])
        """
        # Risk classification
        risk_level = self.classify_risk_level(severity_risk, predicted_impact)
        
        # Cost estimation
        cost_estimate = self.estimate_breach_cost(predicted_impact, organization_type)
        
        # Security recommendations
        recommendations = self.generate_security_recommendations(
            severity_risk, organization_type, breach_type
        )
        
        # Executive summary
        executive_summary = f"""
BREACH RISK ASSESSMENT
{'='*60}

RISK LEVEL: {risk_level.upper()}
Severity Probability: {severity_risk:.1%}
Predicted Impact: {predicted_impact:,.0f} individuals affected

ESTIMATED COST IF BREACH OCCURS:
  Direct Costs (records): ${cost_estimate['record_costs']:,.0f}
  Notification Costs: ${cost_estimate['notification_costs']:,.0f}
  Legal/Regulatory: ${cost_estimate['legal_costs']:,.0f}
  PR/Recovery: ${cost_estimate['pr_costs']:,.0f}
  ---
  TOTAL ESTIMATED COST: ${cost_estimate['total_estimated_cost']:,.0f}

IMMEDIATE ACTIONS REQUIRED: {len(recommendations)}
"""
        
        # Compile report
        report = {
            'risk_level': risk_level,
            'severity_risk': severity_risk,
            'predicted_impact': predicted_impact,
            'cost_estimate': cost_estimate,
            'recommendations': recommendations,
            'executive_summary': executive_summary,
            'priority_score': self._calculate_priority_score(severity_risk, predicted_impact)
        }
        
        return report
    
    def _calculate_priority_score(self, severity_risk: float, 
                                  predicted_impact: float) -> int:
        """
        Calculate numerical priority score (0-100).
        
        Args:
            severity_risk: Severity probability
            predicted_impact: Predicted impact
            
        Returns:
            Priority score (0-100)
        """
        # Weighted combination of severity and impact
        severity_component = severity_risk * 60  # 60% weight on severity
        
        # Normalize impact to 0-40 scale
        impact_normalized = min(predicted_impact / 100000, 1.0) * 40
        
        priority = int(severity_component + impact_normalized)
        return min(priority, 100)
    
    def batch_recommendations(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate recommendations for batch of predictions.
        
        Args:
            predictions_df: DataFrame with predictions
            
        Returns:
            DataFrame with recommendations added
        """
        result = predictions_df.copy()
        
        # Generate recommendations for each row
        result['risk_level'] = result.apply(
            lambda row: self.classify_risk_level(
                row.get('severity_probability', row.get('severity_risk', 0)),
                row.get('predicted_impact', None)
            ),
            axis=1
        )
        
        result['priority_score'] = result.apply(
            lambda row: self._calculate_priority_score(
                row.get('severity_probability', row.get('severity_risk', 0)),
                row.get('predicted_impact', 10000)
            ),
            axis=1
        )
        
        return result.sort_values('priority_score', ascending=False)
    
    def print_recommendation_report(self, report: Dict[str, Any]):
        """
        Print formatted recommendation report.
        
        Args:
            report: Report dictionary from generate_comprehensive_report()
        """
        print(report['executive_summary'])
        print(f"\nPRIORITY SCORE: {report['priority_score']}/100\n")
        print("RECOMMENDED ACTIONS:")
        print("-" * 60)
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
        print("\n" + "="*60)
    
    def __repr__(self):
        """String representation."""
        return "BusinessRecommender(rules_loaded=True)"
