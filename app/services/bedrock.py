import boto3
import json
import time
from typing import Dict, Any, Optional
from langchain_aws import ChatBedrock
from app.utils.logger import logger
import os


class BedrockService:
    """AWS Bedrock service wrapper with cost tracking and circuit breaker"""

    # Rough cost estimates per 1K tokens (input/output) for Nova models
    COST_PER_1K_TOKENS = {
        "input": 0.0008,  # $0.0008 per 1K input tokens
        "output": 0.0032  # $0.0032 per 1K output tokens
    }

    def __init__(self):
        self.timeout = int(os.getenv("BEDROCK_TIMEOUT", "8"))
        self.temperature = float(os.getenv("MODEL_TEMPERATURE", "0.2"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "2"))

        # Initialize Bedrock client
        self.bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        )

        # Initialize LangChain ChatBedrock with strict JSON mode
        self.llm = ChatBedrock(
            client=self.bedrock_client,
            model_id="amazon.nova-micro-v1:0",  # Using Nova family
            model_kwargs={
                "temperature": self.temperature,  # ≤ 0.3 for determinism
                "max_tokens": 2048,
                "top_p": 0.9,
                "stop_sequences": []
            }
        )

        # Strict JSON mode LLM for structured outputs
        self.json_llm = ChatBedrock(
            client=self.bedrock_client,
            model_id="amazon.nova-micro-v1:0",
            model_kwargs={
                "temperature": 0.1,  # Even lower for JSON consistency
                "max_tokens": 2048,
                "top_p": 0.8
            }
        )

    def invoke_with_monitoring(self, messages: list, system_prompt: str = None) -> Dict[str, Any]:
        """
        Invoke Bedrock with monitoring, timeout, and cost tracking
        """
        start_time = time.time()

        try:
            # Prepare messages
            if system_prompt:
                formatted_messages = [{"role": "system", "content": system_prompt}] + messages
            else:
                formatted_messages = messages

            # Convert to LangChain format
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

            lc_messages = []
            for msg in formatted_messages:
                if msg["role"] == "system":
                    lc_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user" or msg["role"] == "human":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    lc_messages.append(AIMessage(content=msg["content"]))

            # Invoke with timeout simulation (LangChain doesn't have native timeout)
            response = self.llm.invoke(lc_messages)

            # Calculate metrics
            latency = time.time() - start_time

            # Rough token estimation (1 token ≈ 4 characters)
            input_content = " ".join([msg["content"] for msg in formatted_messages])
            input_tokens = len(input_content) // 4
            output_tokens = len(response.content) // 4

            cost_estimate = (
                    (input_tokens / 1000) * self.COST_PER_1K_TOKENS["input"] +
                    (output_tokens / 1000) * self.COST_PER_1K_TOKENS["output"]
            )

            # Log metrics
            logger.info(
                "Bedrock call completed",
                latency_seconds=round(latency, 3),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_estimate_usd=round(cost_estimate, 6)
            )

            # Check timeout
            if latency > self.timeout:
                logger.warning("Bedrock call exceeded timeout", latency=latency, timeout=self.timeout)

            return {
                "content": response.content,
                "latency": latency,
                "tokens_used": input_tokens + output_tokens,
                "cost_estimate": cost_estimate
            }

        except Exception as e:
            latency = time.time() - start_time
            logger.error(
                "Bedrock call failed",
                error=str(e),
                latency_seconds=round(latency, 3)
            )
            raise e

    def invoke_with_json_mode(self, prompt: str, system_prompt: str = None, retry_count: int = 0) -> Dict[str, Any]:
        """
        Invoke Bedrock with strict JSON mode and auto-retry for malformed responses
        """
        json_system_prompt = """You MUST respond with valid JSON only. No additional text, explanations, or markdown formatting. 
        Your response must be a valid JSON object that can be parsed directly. Do not use code blocks or any other formatting.
        Ensure all JSON values are properly quoted and escaped. Numbers should not be quoted unless they are strings."""

        if system_prompt:
            combined_system_prompt = f"{system_prompt}\n\n{json_system_prompt}"
        else:
            combined_system_prompt = json_system_prompt

        messages = [{"role": "user", "content": f"{prompt}\n\nRemember: Respond with valid JSON only, no other text."}]

        try:
            # Use the JSON-specific LLM with lower temperature
            response = self._invoke_json_llm(messages, combined_system_prompt)

            # Clean the response content
            content = response["content"].strip()

            # Remove any markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            # Try to parse JSON
            try:
                parsed_json = json.loads(content)
                response["parsed_content"] = parsed_json
                return response
            except json.JSONDecodeError as e:
                if retry_count < self.max_retries:
                    logger.warning(
                        "Invalid JSON response, retrying",
                        retry_count=retry_count + 1,
                        error=str(e),
                        content_preview=content[:100]
                    )
                    return self.invoke_with_json_mode(prompt, system_prompt, retry_count + 1)
                else:
                    logger.error("Max retries exceeded for JSON parsing", error=str(e), content=content)
                    raise ValueError(f"Failed to get valid JSON after {self.max_retries} retries: {str(e)}")

        except Exception as e:
            if retry_count < self.max_retries:
                logger.warning(
                    "Bedrock call failed, retrying",
                    retry_count=retry_count + 1,
                    error=str(e)
                )
                return self.invoke_with_json_mode(prompt, system_prompt, retry_count + 1)
            else:
                logger.error("Max retries exceeded for Bedrock call", error=str(e))
                raise e

    def _invoke_json_llm(self, messages: list, system_prompt: str = None) -> Dict[str, Any]:
        """Internal method to invoke JSON-specific LLM"""
        start_time = time.time()

        try:
            # Prepare messages with system prompt
            if system_prompt:
                formatted_messages = [{"role": "system", "content": system_prompt}] + messages
            else:
                formatted_messages = messages

            # Convert to LangChain format
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

            lc_messages = []
            for msg in formatted_messages:
                if msg["role"] == "system":
                    lc_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user" or msg["role"] == "human":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    lc_messages.append(AIMessage(content=msg["content"]))

            # Invoke JSON LLM
            response = self.json_llm.invoke(lc_messages)

            # Calculate metrics
            latency = time.time() - start_time

            # Rough token estimation (1 token ≈ 4 characters)
            input_content = " ".join([msg["content"] for msg in formatted_messages])
            input_tokens = len(input_content) // 4
            output_tokens = len(response.content) // 4

            cost_estimate = (
                    (input_tokens / 1000) * self.COST_PER_1K_TOKENS["input"] +
                    (output_tokens / 1000) * self.COST_PER_1K_TOKENS["output"]
            )

            # Log metrics
            logger.info(
                "JSON Bedrock call completed",
                latency_seconds=round(latency, 3),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_estimate_usd=round(cost_estimate, 6),
                temperature=0.1
            )

            return {
                "content": response.content,
                "latency": latency,
                "tokens_used": input_tokens + output_tokens,
                "cost_estimate": cost_estimate
            }

        except Exception as e:
            latency = time.time() - start_time
            logger.error(
                "JSON Bedrock call failed",
                error=str(e),
                latency_seconds=round(latency, 3)
            )
            raise e