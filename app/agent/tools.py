from typing import Dict, List, Any
from datetime import datetime, date
from app.services.data import DataService
from app.utils.logger import logger


class CustomerAnalysisTools:
    """Tools for customer analysis that can be called by the agent"""

    def __init__(self, data_service: DataService):
        self.data_service = data_service

    def get_customer_purchase_summary(self, customer_id: str) -> Dict[str, Any]:
        """
        Tool: Get comprehensive customer purchase summary
        """
        logger.info("Getting customer purchase summary", customer_id=customer_id)

        try:
            customer_data = self.data_service.get_customer_data(customer_id)

            if not customer_data:
                return {
                    "error": f"Customer {customer_id} not found",
                    "summary": "No customer data available"
                }

            customer_info = customer_data["customer_info"]
            order_summary = customer_data["order_summary"]
            order_history = customer_data["order_history"]

            # Analyze purchase patterns
            if order_history:
                # Get most purchased products
                product_counts = {}
                for order in order_history:
                    sku = order["sku"]
                    qty = order["qty"]
                    product_counts[sku] = product_counts.get(sku, 0) + qty

                top_products = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)[:3]

                # Calculate purchase frequency
                if len(order_history) > 1:
                    # Fix datetime parsing issue
                    order_dates = []
                    for order in order_history:
                        order_date = order["order_date"]
                        # Handle both string and datetime objects
                        if isinstance(order_date, str):
                            parsed_date = datetime.fromisoformat(order_date[:10])
                        else:
                            # If it's already a datetime/Timestamp, convert to datetime
                            parsed_date = datetime(order_date.year, order_date.month, order_date.day)
                        order_dates.append(parsed_date)

                    order_dates.sort()
                    intervals = [(order_dates[i + 1] - order_dates[i]).days for i in range(len(order_dates) - 1)]
                    avg_days_between_orders = sum(intervals) / len(intervals) if intervals else None
                else:
                    avg_days_between_orders = None
            else:
                top_products = []
                avg_days_between_orders = None

            return {
                "customer_info": {
                    "name": customer_info["name"],
                    "segment": customer_info["segment"],
                    "territory": customer_info["territory"],
                    "credit_terms": customer_info["credit_terms"]
                },
                "purchase_behavior": {
                    "total_orders": order_summary["total_orders"],
                    "total_spent": order_summary["total_spent"],
                    "avg_order_value": order_summary["avg_order_value"],
                    "days_since_last_order": order_summary["days_since_last_order"],
                    "avg_days_between_orders": avg_days_between_orders,
                    "top_products": top_products
                },
                "summary": f"Customer {customer_info['name']} in {customer_info['segment']} segment has made {order_summary['total_orders']} orders totaling ${order_summary['total_spent']:.2f}"
            }

        except Exception as e:
            logger.error("Error getting customer summary", error=str(e), customer_id=customer_id)
            return {
                "error": str(e),
                "summary": "Error retrieving customer data"
            }

    def calculate_customer_scores(self, customer_id: str) -> Dict[str, Any]:
        """
        Tool: Calculate RFM score, churn risk, and priority level
        """
        logger.info("Calculating customer scores", customer_id=customer_id)

        try:
            # Get RFM score
            rfm_data = self.data_service.calculate_rfm_score(customer_id)

            # Get churn risk
            churn_risk = self.data_service.calculate_churn_risk(customer_id)

            # Calculate priority level (1-5, where 5 is highest priority)
            rfm_score = rfm_data["rfm_score"]

            if rfm_score >= 80:
                priority = 5  # High value, recent customers
            elif rfm_score >= 60:
                priority = 4  # Good customers
            elif rfm_score >= 40:
                priority = 3  # Average customers
            elif rfm_score >= 20:
                priority = 2  # At-risk customers
            else:
                priority = 1  # Low value customers

            # Adjust priority based on churn risk
            if churn_risk > 0.7:
                priority = max(1, priority - 1)  # Lower priority for high churn risk
            elif churn_risk < 0.3:
                priority = min(5, priority + 1)  # Higher priority for low churn risk

            return {
                "rfm_components": rfm_data,
                "scores": {
                    "rfm_score": rfm_score,
                    "churn_risk": churn_risk,
                    "priority": priority
                },
                "interpretation": {
                    "rfm_level": "High" if rfm_score >= 70 else "Medium" if rfm_score >= 40 else "Low",
                    "churn_level": "High" if churn_risk >= 0.7 else "Medium" if churn_risk >= 0.4 else "Low",
                    "priority_level": "Critical" if priority == 5 else "High" if priority == 4 else "Medium" if priority == 3 else "Low" if priority == 2 else "Minimal"
                }
            }

        except Exception as e:
            logger.error("Error calculating scores", error=str(e), customer_id=customer_id)
            return {
                "error": str(e),
                "scores": {
                    "rfm_score": 0,
                    "churn_risk": 1.0,
                    "priority": 1
                }
            }


class RecommendationTools:
    """Tools for generating recommendations"""

    def __init__(self, data_service: DataService):
        self.data_service = data_service

    def generate_action_recommendations(self, customer_data: Dict[str, Any], scores: Dict[str, Any]) -> List[
        Dict[str, str]]:
        """
        Tool: Generate specific action recommendations based on customer data and scores
        """
        logger.info("Generating action recommendations")

        recommendations = []

        try:
            rfm_score = scores.get("rfm_score", 0)
            churn_risk = scores.get("churn_risk", 1.0)
            priority = scores.get("priority", 1)

            purchase_behavior = customer_data.get("purchase_behavior", {})
            days_since_last_order = purchase_behavior.get("days_since_last_order", 999)
            total_orders = purchase_behavior.get("total_orders", 0)
            avg_order_value = purchase_behavior.get("avg_order_value", 0)

            # Rule-based recommendation engine

            # High churn risk customers
            if churn_risk > 0.7:
                if days_since_last_order > 45:
                    recommendations.append({
                        "action": "call",
                        "reason": f"High churn risk customer hasn't ordered in {days_since_last_order} days - needs immediate personal attention"
                    })
                else:
                    recommendations.append({
                        "action": "email",
                        "reason": "High churn risk customer needs re-engagement campaign with special offers"
                    })

            # High value customers (high RFM)
            if rfm_score > 70:
                if avg_order_value > 20:
                    recommendations.append({
                        "action": "offer_bundle",
                        "reason": f"High-value customer (${avg_order_value:.2f} AOV) - perfect candidate for premium product bundles"
                    })
                else:
                    recommendations.append({
                        "action": "call",
                        "reason": "High RFM score customer deserves personal attention to strengthen relationship"
                    })

            # Medium customers with potential
            elif rfm_score > 40:
                if total_orders < 3:
                    recommendations.append({
                        "action": "promo",
                        "reason": "Medium-value customer with few orders - promotional offers could increase frequency"
                    })
                else:
                    recommendations.append({
                        "action": "email",
                        "reason": "Consistent customer - maintain engagement with regular email communication"
                    })

            # Low engagement customers
            else:
                if days_since_last_order > 60:
                    recommendations.append({
                        "action": "promo",
                        "reason": f"Low engagement customer inactive for {days_since_last_order} days - win-back promotion needed"
                    })
                else:
                    recommendations.append({
                        "action": "email",
                        "reason": "Low RFM customer - basic email nurturing to build relationship"
                    })

            # Ensure we have at least 3 recommendations
            while len(recommendations) < 3:
                if len(recommendations) == 1:
                    recommendations.append({
                        "action": "email",
                        "reason": "Follow up with product updates and company news"
                    })
                elif len(recommendations) == 2:
                    recommendations.append({
                        "action": "promo",
                        "reason": "Offer seasonal promotion to drive additional purchases"
                    })

            # Limit to top 3
            return recommendations[:3]

        except Exception as e:
            logger.error("Error generating recommendations", error=str(e))
            return [
                {"action": "email", "reason": "Standard follow-up communication"},
                {"action": "call", "reason": "Personal check-in with customer"},
                {"action": "promo", "reason": "General promotional offer"}
            ]

    def get_daily_followup_list(self, target_date: date) -> List[str]:
        """
        Tool: Get prioritized list of customers to follow up today
        """
        logger.info("Getting daily followup list", date=target_date.isoformat())

        try:
            return self.data_service.get_top_followups_for_date(target_date, limit=5)
        except Exception as e:
            logger.error("Error getting followup list", error=str(e))
            return []