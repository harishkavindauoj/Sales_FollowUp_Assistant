from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time
from contextlib import asynccontextmanager

from app.models import AnalysisResponse, AnalyzeRequest, TopFollowupsRequest
from app.services.bedrock import BedrockService
from app.services.data import DataService
from app.agent.graph import SalesFollowUpGraph
from app.utils.logger import logger

# Global variables for services
bedrock_service: BedrockService
data_service: DataService
sales_graph: SalesFollowUpGraph


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup"""
    global bedrock_service, data_service, sales_graph

    logger.info("Starting Sales Follow-Up Assistant API")

    try:
        # Initialize services
        bedrock_service = BedrockService()
        data_service = DataService()
        sales_graph = SalesFollowUpGraph(bedrock_service, data_service)

        logger.info("All services initialized successfully")
        yield

    except Exception as e:
        logger.error("Failed to initialize services", error=str(e))
        raise e
    finally:
        logger.info("Shutting down Sales Follow-Up Assistant API")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Sales Follow-Up Assistant",
    description="AI-powered sales follow-up recommendations using LangGraph and AWS Bedrock",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Sales Follow-Up Assistant API",
        "status": "healthy",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Test data service
        customer_count = len(data_service.customers_df) if data_service and data_service.customers_df is not None else 0

        return {
            "status": "healthy",
            "services": {
                "bedrock": bedrock_service is not None,
                "data": data_service is not None,
                "graph": sales_graph is not None
            },
            "data": {
                "customers_loaded": customer_count > 0,
                "customer_count": customer_count
            }
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_customer(request: AnalyzeRequest) -> JSONResponse:
    """
    PROBLEM STATEMENT REQUIREMENT 1, 2, 3:
    1. Answer analytical questions (who should the rep follow up today) ✅
    2. Summarize customer's purchase behavior ✅
    3. Recommend top 3 actions for the rep ✅

    This endpoint runs the full LangGraph workflow to analyze a customer's
    purchase behavior, calculate scores, and generate recommendations.
    """
    start_time = time.time()

    try:
        from app.utils.logger import ConsoleLogger
        ConsoleLogger.log_analysis_start(request.customer_id)

        logger.info("Received analyze request", customer_id=request.customer_id)

        if not sales_graph:
            raise HTTPException(status_code=503, detail="Sales analysis service not available")

        # Run the analysis through LangGraph
        result = sales_graph.analyze_customer_sync(request.customer_id)

        # Validate the result matches our Pydantic model for STRICT JSON
        try:
            validated_response = AnalysisResponse(**result)
        except Exception as validation_error:
            logger.error("Response validation failed", error=str(validation_error), result=result)
            raise HTTPException(status_code=500, detail=f"Invalid response format: {str(validation_error)}")

        processing_time = time.time() - start_time
        ConsoleLogger.log_analysis_complete(request.customer_id, processing_time)

        logger.info(
            "Analysis completed successfully",
            customer_id=request.customer_id,
            processing_time_seconds=round(processing_time, 3)
        )

        # Return the validated response with headers
        return JSONResponse(
            content=validated_response.dict(),
            headers={
                "X-Processing-Time": str(round(processing_time, 3)),
                "X-Customer-ID": request.customer_id,
                "X-Analysis-Complete": "true"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        from app.utils.logger import ConsoleLogger
        ConsoleLogger.log_analysis_error(request.customer_id, str(e))

        logger.error(
            "Analysis failed",
            customer_id=request.customer_id,
            error=str(e),
            processing_time_seconds=round(processing_time, 3)
        )
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/top-followups")
async def get_top_followups(request: TopFollowupsRequest) -> JSONResponse:
    """
    PROBLEM STATEMENT REQUIREMENT 1: Answer analytical questions
    "Who should the rep follow up today?" ✅

    This endpoint provides a prioritized list of customers that sales reps
    should focus on for the specified date.
    """
    start_time = time.time()

    try:
        logger.info("Received top-followups request", date=request.date)
        print(f"📅 Getting top follow-ups for date: {request.date}")

        if not sales_graph:
            raise HTTPException(status_code=503, detail="Sales analysis service not available")

        # Get top followups for the date
        result = await sales_graph.get_top_followups_for_date(request.date)

        processing_time = time.time() - start_time

        print(f"✅ Found {result.get('count', 0)} customers to follow up on {request.date}")

        logger.info(
            "Top followups retrieved successfully",
            date=request.date,
            count=result.get("count", 0),
            processing_time_seconds=round(processing_time, 3)
        )

        return JSONResponse(
            content=result,
            headers={
                "X-Processing-Time": str(round(processing_time, 3)),
                "X-Date": request.date,
                "X-Followup-Count": str(result.get("count", 0))
            }
        )

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            "Top followups request failed",
            date=request.date,
            error=str(e),
            processing_time_seconds=round(processing_time, 3)
        )
        raise HTTPException(status_code=500, detail=f"Failed to get top followups: {str(e)}")


@app.get("/analytics/question")
async def answer_analytical_question(question: str = "who should the rep follow up today?"):
    """
    PROBLEM STATEMENT REQUIREMENT 1: Answer analytical questions ✅

    Generic endpoint to answer analytical questions about customer data
    """
    try:
        logger.info("Received analytical question", question=question)

        if not data_service:
            raise HTTPException(status_code=503, detail="Data service not available")

        # Handle common analytical questions
        question_lower = question.lower()

        if "follow up" in question_lower or "contact" in question_lower:
            # Get today's top follow-ups
            from datetime import date
            today = date.today().isoformat()
            result = await sales_graph.get_top_followups_for_date(today)

            return {
                "question": question,
                "answer": f"Top customers to follow up today: {', '.join(result['top_followups_today'])}",
                "details": result,
                "answer_type": "followup_list"
            }

        elif "high value" in question_lower or "best customer" in question_lower:
            # Find high-value customers
            customers = data_service.customers_df['customer_id'].tolist()
            high_value_customers = []

            for customer_id in customers:
                rfm_data = data_service.calculate_rfm_score(customer_id)
                if rfm_data['rfm_score'] > 70:
                    high_value_customers.append(customer_id)

            return {
                "question": question,
                "answer": f"High-value customers (RFM > 70): {', '.join(high_value_customers) if high_value_customers else 'None found'}",
                "details": {"high_value_customers": high_value_customers},
                "answer_type": "customer_list"
            }

        elif "churn" in question_lower or "risk" in question_lower:
            # Find at-risk customers
            customers = data_service.customers_df['customer_id'].tolist()
            at_risk_customers = []

            for customer_id in customers:
                churn_risk = data_service.calculate_churn_risk(customer_id)
                if churn_risk > 0.7:
                    at_risk_customers.append({"customer_id": customer_id, "churn_risk": churn_risk})

            return {
                "question": question,
                "answer": f"At-risk customers (churn > 0.7): {', '.join([c['customer_id'] for c in at_risk_customers])}",
                "details": {"at_risk_customers": at_risk_customers},
                "answer_type": "risk_analysis"
            }

        else:
            return {
                "question": question,
                "answer": "I can answer questions about: customer follow-ups, high-value customers, and churn risk analysis.",
                "suggestions": [
                    "Who should the rep follow up today?",
                    "Which customers have high value?",
                    "Which customers are at risk of churning?"
                ],
                "answer_type": "help"
            }

    except Exception as e:
        logger.error("Failed to answer analytical question", question=question, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {str(e)}")


@app.get("/customers")
async def list_customers():
    """Get list of available customers"""
    try:
        if not data_service or data_service.customers_df is None:
            raise HTTPException(status_code=503, detail="Data service not available")

        customers = data_service.customers_df.to_dict('records')
        return {
            "customers": customers,
            "count": len(customers)
        }

    except Exception as e:
        logger.error("Failed to list customers", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list customers: {str(e)}")


@app.get("/customer/{customer_id}/summary")
async def get_customer_summary(customer_id: str):
    """Get basic customer summary without full analysis"""
    try:
        if not data_service:
            raise HTTPException(status_code=503, detail="Data service not available")

        customer_data = data_service.get_customer_data(customer_id)

        if not customer_data:
            raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

        return customer_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get customer summary", customer_id=customer_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get customer summary: {str(e)}")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "detail": "The requested resource was not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error("Internal server error", error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
