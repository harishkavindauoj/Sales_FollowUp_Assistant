from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
from app.models import GraphState
from app.services.bedrock import BedrockService
from app.services.data import DataService
from app.agent.nodes import GraphNodes
from app.utils.logger import logger


class SalesFollowUpGraph:
    """LangGraph implementation for Sales Follow-Up Assistant"""

    def __init__(self, bedrock_service: BedrockService, data_service: DataService):
        self.bedrock_service = bedrock_service
        self.data_service = data_service
        self.nodes = GraphNodes(bedrock_service, data_service)
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledGraph:
        """Build the LangGraph workflow"""
        logger.info("Building LangGraph workflow")

        # Create the graph
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("fetch_customer_data", self._fetch_customer_data_wrapper)
        workflow.add_node("analyze_rfm", self._analyze_rfm_wrapper)
        workflow.add_node("analyze_churn", self._analyze_churn_wrapper)
        workflow.add_node("merge_analyses", self._merge_analyses_wrapper)
        workflow.add_node("generate_summary", self._generate_summary_wrapper)
        workflow.add_node("generate_recommendations", self._generate_recommendations_wrapper)
        workflow.add_node("get_top_followups", self._get_top_followups_wrapper)
        workflow.add_node("format_response", self._format_response_wrapper)

        # Set entry point
        workflow.set_entry_point("fetch_customer_data")

        # Sequential flow to parallel branches
        workflow.add_edge("fetch_customer_data", "analyze_rfm")
        workflow.add_edge("fetch_customer_data", "analyze_churn")

        # Parallel branches merge before continuing
        workflow.add_edge("analyze_rfm", "merge_analyses")
        workflow.add_edge("analyze_churn", "merge_analyses")

        # Continue sequential flow after merge
        workflow.add_edge("merge_analyses", "generate_summary")
        workflow.add_edge("generate_summary", "generate_recommendations")
        workflow.add_edge("generate_recommendations", "get_top_followups")
        workflow.add_edge("get_top_followups", "format_response")

        # End the workflow
        workflow.add_edge("format_response", END)

        # Compile the graph
        compiled_graph = workflow.compile()
        logger.info("LangGraph workflow compiled successfully")

        return compiled_graph

    def _fetch_customer_data_wrapper(self, state: GraphState) -> Dict[str, Any]:
        """Wrapper for fetch_customer_data node"""
        result = self.nodes.fetch_customer_data(state)
        return result  # Return the updates directly

    def _analyze_rfm_wrapper(self, state: GraphState) -> Dict[str, Any]:
        """Wrapper for analyze_rfm_parallel node"""
        result = self.nodes.analyze_rfm_parallel(state)
        return result  # Return the updates directly

    def _analyze_churn_wrapper(self, state: GraphState) -> Dict[str, Any]:
        """Wrapper for analyze_churn_parallel node"""
        result = self.nodes.analyze_churn_parallel(state)
        return result  # Return the updates directly

    def _merge_analyses_wrapper(self, state: GraphState) -> Dict[str, Any]:
        """Wrapper to merge parallel analysis results"""
        # Just return an empty dict since the state already contains both analyses
        # LangGraph will pass through without requiring an update
        return {"merge_complete": True}

    def _generate_summary_wrapper(self, state: GraphState) -> Dict[str, Any]:
        """Wrapper for generate_summary node"""
        result = self.nodes.generate_summary(state)
        return result  # Return the updates directly

    def _generate_recommendations_wrapper(self, state: GraphState) -> Dict[str, Any]:
        """Wrapper for generate_recommendations node"""
        result = self.nodes.generate_recommendations(state)
        return result  # Return the updates directly

    def _get_top_followups_wrapper(self, state: GraphState) -> Dict[str, Any]:
        """Wrapper for get_top_followups node"""
        result = self.nodes.get_top_followups(state)
        return result  # Return the updates directly

    def _format_response_wrapper(self, state: GraphState) -> Dict[str, Any]:
        """Wrapper for format_final_response node"""
        result = self.nodes.format_final_response(state)
        return result  # Return the updates directly

    def _update_state(self, current_state: GraphState, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update graph state with node results - return only the updates"""
        # Return only the updates, not the full state
        # LangGraph will merge these updates into the current state
        return updates

    async def analyze_customer(self, customer_id: str) -> Dict[str, Any]:
        """
        Main method to analyze a customer through the graph
        """
        logger.info("Starting customer analysis", customer_id=customer_id)

        try:
            # Initialize state
            initial_state = GraphState(customer_id=customer_id)

            # Run the graph
            result = await self.graph.ainvoke(initial_state.dict())

            # Extract final response
            if "final_response" in result:
                final_response = result["final_response"]
                logger.info("Customer analysis completed successfully", customer_id=customer_id)
                return final_response
            else:
                logger.error("No final response generated", customer_id=customer_id)
                raise ValueError("Graph execution did not produce final response")

        except Exception as e:
            logger.error("Customer analysis failed", customer_id=customer_id, error=str(e))
            # Return fallback response
            return {
                "customer_id": customer_id,
                "scores": {"rfm_score": 0, "churn_risk": 1.0, "priority": 1},
                "summary": f"Analysis failed for customer {customer_id}: {str(e)}",
                "recommendations": [
                    {"action": "email", "reason": "Manual review required due to system error"}
                ],
                "top_followups_today": []
            }

    def analyze_customer_sync(self, customer_id: str) -> Dict[str, Any]:
        """
        Synchronous version of customer analysis
        """
        logger.info("Starting synchronous customer analysis", customer_id=customer_id)

        try:
            # Initialize state
            initial_state = GraphState(customer_id=customer_id)

            # Run the graph synchronously
            result = self.graph.invoke(initial_state.dict())

            # Extract final response
            if "final_response" in result:
                final_response = result["final_response"]
                logger.info("Customer analysis completed successfully", customer_id=customer_id)
                return final_response
            else:
                logger.error("No final response generated", customer_id=customer_id)
                raise ValueError("Graph execution did not produce final response")

        except Exception as e:
            logger.error("Customer analysis failed", customer_id=customer_id, error=str(e))
            # Return fallback response
            return {
                "customer_id": customer_id,
                "scores": {"rfm_score": 0, "churn_risk": 1.0, "priority": 1},
                "summary": f"Analysis failed for customer {customer_id}: {str(e)}",
                "recommendations": [
                    {"action": "email", "reason": "Manual review required due to system error"}
                ],
                "top_followups_today": []
            }

    async def get_top_followups_for_date(self, target_date: str) -> Dict[str, Any]:
        """
        Get top customers to follow up on a specific date
        """
        logger.info("Getting top followups for date", date=target_date)

        try:
            from datetime import date
            parsed_date = date.fromisoformat(target_date)

            # Use recommendation tools directly for this simpler operation
            top_followups = self.nodes.recommendation_tools.get_daily_followup_list(parsed_date)

            return {
                "date": target_date,
                "top_followups_today": top_followups,
                "count": len(top_followups)
            }

        except Exception as e:
            logger.error("Failed to get top followups", date=target_date, error=str(e))
            return {
                "date": target_date,
                "top_followups_today": [],
                "count": 0,
                "error": str(e)
            }