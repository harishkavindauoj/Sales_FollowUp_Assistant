from typing import Dict, Any
import json
from app.models import GraphState
from app.services.bedrock import BedrockService
from app.services.data import DataService
from app.agent.tools import CustomerAnalysisTools, RecommendationTools
from app.utils.logger import NodeLogger
from datetime import date


class GraphNodes:
    """LangGraph nodes for the Sales Follow-Up Assistant"""

    def __init__(self, bedrock_service: BedrockService, data_service: DataService):
        self.bedrock_service = bedrock_service
        self.data_service = data_service
        self.analysis_tools = CustomerAnalysisTools(data_service)
        self.recommendation_tools = RecommendationTools(data_service)

    def fetch_customer_data(self, state: GraphState) -> Dict[str, Any]:
        """Node: Fetch customer data and basic info"""
        logger = NodeLogger("fetch_customer_data")
        logger.log_start({"customer_id": state.customer_id})

        try:
            # Use tool to get customer data
            customer_data = self.analysis_tools.get_customer_purchase_summary(state.customer_id)

            if "error" in customer_data:
                state.errors.append(customer_data["error"])
                logger.log_error(Exception(customer_data["error"]))
                return {"customer_data": {}}

            logger.log_end({"customer_data": customer_data}, success=True)
            return {"customer_data": customer_data}

        except Exception as e:
            logger.log_error(e)
            return {"customer_data": {}, "errors": state.errors + [str(e)]}

    def analyze_rfm_parallel(self, state: GraphState) -> Dict[str, Any]:
        """Node: Calculate RFM analysis (runs in parallel)"""
        logger = NodeLogger("analyze_rfm_parallel")
        logger.log_start({"customer_id": state.customer_id})

        try:
            # Use tool to calculate scores
            score_data = self.analysis_tools.calculate_customer_scores(state.customer_id)

            if "error" in score_data:
                state.errors.append(score_data["error"])
                rfm_analysis = {"rfm_score": 0, "priority": 1}
            else:
                rfm_analysis = score_data

            logger.log_end({"rfm_analysis": rfm_analysis}, success=True)
            return {"rfm_analysis": rfm_analysis}

        except Exception as e:
            logger.log_error(e)
            return {"rfm_analysis": {"rfm_score": 0, "priority": 1}, "errors": state.errors + [str(e)]}

    def analyze_churn_parallel(self, state: GraphState) -> Dict[str, Any]:
        """Node: Calculate churn risk analysis (runs in parallel)"""
        logger = NodeLogger("analyze_churn_parallel")
        logger.log_start({"customer_id": state.customer_id})

        try:
            churn_risk = self.data_service.calculate_churn_risk(state.customer_id)

            churn_analysis = {
                "churn_risk": churn_risk,
                "churn_level": "High" if churn_risk >= 0.7 else "Medium" if churn_risk >= 0.4 else "Low",
                "urgency": "Immediate" if churn_risk >= 0.8 else "Soon" if churn_risk >= 0.6 else "Monitor"
            }

            logger.log_end({"churn_analysis": churn_analysis}, success=True)
            return {"churn_analysis": churn_analysis}

        except Exception as e:
            logger.log_error(e)
            return {"churn_analysis": {"churn_risk": 1.0, "churn_level": "Unknown"}, "errors": state.errors + [str(e)]}

    def generate_summary(self, state: GraphState) -> Dict[str, Any]:
        """Node: Generate AI-powered customer behavior summary with retry logic"""
        logger = NodeLogger("generate_summary")
        logger.log_start({"customer_id": state.customer_id})

        try:
            if not state.customer_data or "error" in state.customer_data:
                return {"summary": "Unable to generate summary due to missing customer data"}

            # Prepare prompt for summary generation
            customer_info = state.customer_data.get("customer_info", {})
            purchase_behavior = state.customer_data.get("purchase_behavior", {})
            rfm_scores = state.rfm_analysis.get("scores", {})
            churn_data = state.churn_analysis

            prompt = f"""
            Analyze this customer's purchase behavior and provide a concise summary (2-3 sentences):

            Customer: {customer_info.get('name', 'Unknown')} ({state.customer_id})
            Segment: {customer_info.get('segment', 'Unknown')}
            Territory: {customer_info.get('territory', 'Unknown')}

            Purchase Behavior:
            - Total Orders: {purchase_behavior.get('total_orders', 0)}
            - Total Spent: ${purchase_behavior.get('total_spent', 0):.2f}
            - Average Order Value: ${purchase_behavior.get('avg_order_value', 0):.2f}
            - Days Since Last Order: {purchase_behavior.get('days_since_last_order', 'N/A')}
            - Top Products: {purchase_behavior.get('top_products', [])}

            Scores:
            - RFM Score: {rfm_scores.get('rfm_score', 0)}/100
            - Churn Risk: {churn_data.get('churn_risk', 1.0):.2f}
            - Priority Level: {rfm_scores.get('priority', 1)}/5

            Provide a business-focused summary highlighting key insights about this customer's value, behavior patterns, and current status.
            Respond with ONLY the summary text, no JSON formatting.
            """

            system_prompt = "You are a sales analyst. Provide concise, actionable customer behavior summaries for sales representatives. Respond with plain text only, no formatting."

            # Use retry logic for summary generation
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    response = self.bedrock_service.invoke_with_monitoring(
                        [{"role": "user", "content": prompt}],
                        system_prompt
                    )

                    summary = response["content"].strip()

                    # Validate summary is not empty and reasonable length
                    if len(summary) < 10:
                        raise ValueError("Summary too short")
                    if len(summary) > 1000:
                        summary = summary[:1000] + "..."

                    logger.log_bedrock_call(
                        response["latency"],
                        response["tokens_used"],
                        response["cost_estimate"]
                    )

                    logger.log_end({"summary": len(summary)}, success=True)
                    return {"summary": summary}

                except Exception as e:
                    if attempt < max_retries:
                        logger.log_error(e, {"retry_attempt": attempt + 1})
                        continue
                    else:
                        logger.log_error(e, {"final_attempt": True})
                        raise e

        except Exception as e:
            logger.log_error(e)
            fallback_summary = f"Customer {state.customer_id}: Analysis shows {state.customer_data.get('purchase_behavior', {}).get('total_orders', 0)} orders totaling ${state.customer_data.get('purchase_behavior', {}).get('total_spent', 0):.2f}. Manual review recommended."
            return {"summary": fallback_summary, "errors": state.errors + [str(e)]}

    def generate_recommendations(self, state: GraphState) -> Dict[str, Any]:
        """Node: Generate action recommendations"""
        logger = NodeLogger("generate_recommendations")
        logger.log_start({"customer_id": state.customer_id})

        try:
            # Use tools to generate recommendations
            scores = state.rfm_analysis.get("scores", {})
            recommendations = self.recommendation_tools.generate_action_recommendations(
                state.customer_data, scores
            )

            logger.log_end({"recommendations": len(recommendations)}, success=True)
            return {"recommendations": recommendations}

        except Exception as e:
            logger.log_error(e)
            fallback_recommendations = [
                {"action": "email", "reason": "Standard follow-up required"},
                {"action": "call", "reason": "Personal check-in needed"},
                {"action": "promo", "reason": "Offer promotional discount"}
            ]
            return {"recommendations": fallback_recommendations, "errors": state.errors + [str(e)]}

    def get_top_followups(self, state: GraphState) -> Dict[str, Any]:
        """Node: Get today's top follow-up customers"""
        logger = NodeLogger("get_top_followups")
        logger.log_start({})

        try:
            today = date.today()
            top_followups = self.recommendation_tools.get_daily_followup_list(today)

            logger.log_end({"top_followups": len(top_followups)}, success=True)
            return {"top_followups": top_followups}

        except Exception as e:
            logger.log_error(e)
            return {"top_followups": [], "errors": state.errors + [str(e)]}

    def format_final_response(self, state: GraphState) -> Dict[str, Any]:
        """Node: Format the final JSON response with validation and retry logic"""
        logger = NodeLogger("format_final_response")
        logger.log_start({"customer_id": state.customer_id})

        try:
            # Extract scores from analyses
            rfm_scores = state.rfm_analysis.get("scores", {})
            churn_risk = state.churn_analysis.get("churn_risk", 1.0)

            # Build response structure
            response_data = {
                "customer_id": state.customer_id,
                "scores": {
                    "rfm_score": int(rfm_scores.get("rfm_score", 0)),
                    "churn_risk": float(churn_risk),
                    "priority": int(rfm_scores.get("priority", 1))
                },
                "summary": state.summary or "No summary available",
                "recommendations": state.recommendations[:3] if state.recommendations else [
                    {"action": "email", "reason": "Standard follow-up required"}
                ],
                "top_followups_today": state.top_followups or []
            }

            # Validate the response using Pydantic for strict schema compliance
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    from app.models import AnalysisResponse

                    # Validate against Pydantic model
                    validated_response = AnalysisResponse(**response_data)

                    # If validation succeeds, return the dict
                    final_dict = validated_response.dict()

                    logger.log_end({"response": len(str(final_dict))}, success=True)
                    return {"final_response": final_dict}

                except Exception as validation_error:
                    if attempt < max_retries:
                        logger.log_error(validation_error, {"retry_attempt": attempt + 1, "validation_error": True})

                        # Try to fix common validation issues
                        if "rfm_score" in str(validation_error):
                            response_data["scores"]["rfm_score"] = max(0, min(100, int(
                                response_data["scores"]["rfm_score"] or 0)))
                        if "churn_risk" in str(validation_error):
                            response_data["scores"]["churn_risk"] = max(0.0, min(1.0, float(
                                response_data["scores"]["churn_risk"] or 1.0)))
                        if "priority" in str(validation_error):
                            response_data["scores"]["priority"] = max(1, min(5, int(
                                response_data["scores"]["priority"] or 1)))
                        if "recommendations" in str(validation_error):
                            response_data["recommendations"] = [{"action": "email", "reason": "Standard follow-up"}]

                        continue
                    else:
                        logger.log_error(validation_error, {"final_attempt": True})
                        raise validation_error

        except Exception as e:
            logger.log_error(e)
            # Fallback response that definitely validates
            fallback_response = {
                "customer_id": state.customer_id,
                "scores": {"rfm_score": 0, "churn_risk": 1.0, "priority": 1},
                "summary": "Analysis could not be completed due to system error",
                "recommendations": [{"action": "email", "reason": "Manual review required"}],
                "top_followups_today": []
            }
            return {"final_response": fallback_response, "errors": state.errors + [str(e)]}

    def should_retry(self, state: GraphState) -> str:
        """Conditional edge: Determine if we should retry"""
        if state.retry_count >= 2:
            return "complete"
        elif state.errors and state.retry_count < 2:
            return "retry"
        else:
            return "complete"