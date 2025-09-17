from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import time
from typing import Dict, Any
from contextlib import asynccontextmanager

from app.models import AnalysisResponse, AnalyzeRequest, TopFollowupsRequest
from app.services.bedrock import BedrockService
from app.services.data import DataService
from app.agent.graph import SalesFollowUpGraph
from app.utils.logger import logger

# Global variables for services
bedrock_service: BedrockService = None
data_service: DataService = None
sales_graph: SalesFollowUpGraph = None


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
    Analyze a customer and provide follow-up recommendations

    This endpoint runs the full LangGraph workflow to analyze a customer's
    purchase behavior, calculate scores, and generate recommendations.
    """
    start_time = time.time()

    try:
        logger.info("Received analyze request", customer_id=request.customer_id)

        if not sales_graph:
            raise HTTPException(status_code=503, detail="Sales analysis service not available")

        # Run the analysis through LangGraph
        result = sales_graph.analyze_customer_sync(request.customer_id)

        # Validate the result matches our Pydantic model
        try:
            validated_response = AnalysisResponse(**result)
        except Exception as validation_error:
            logger.error("Response validation failed", error=str(validation_error), result=result)
            raise HTTPException(status_code=500, detail=f"Invalid response format: {str(validation_error)}")

        processing_time = time.time() - start_time
        logger.info(
            "Analysis completed successfully",
            customer_id=request.customer_id,
            processing_time_seconds=round(processing_time, 3)
        )

        # Return the validated response
        return JSONResponse(
            content=validated_response.dict(),
            headers={"X-Processing-Time": str(round(processing_time, 3))}
        )

    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
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
    Get the top customers to follow up on a specific date

    This endpoint provides a prioritized list of customers that sales reps
    should focus on for the specified date.
    """
    start_time = time.time()

    try:
        logger.info("Received top-followups request", date=request.date)

        if not sales_graph:
            raise HTTPException(status_code=503, detail="Sales analysis service not available")

        # Get top followups for the date
        result = await sales_graph.get_top_followups_for_date(request.date)

        processing_time = time.time() - start_time
        logger.info(
            "Top followups retrieved successfully",
            date=request.date,
            count=result.get("count", 0),
            processing_time_seconds=round(processing_time, 3)
        )

        return JSONResponse(
            content=result,
            headers={"X-Processing-Time": str(round(processing_time, 3))}
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