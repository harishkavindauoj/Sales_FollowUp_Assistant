import boto3
import json
import time
from typing import Dict, Any, Optional
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
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

        # Updated model identifiers based on AWS Bedrock documentation
        model_ids_to_try = [
            # Nova models (correct identifiers)
            "amazon.nova-micro-v1:0",  # Nova Micro - fastest and most cost-effective
            "amazon.nova-lite-v1:0",  # Nova Lite - balanced performance
            "amazon.nova-pro-v1:0",  # Nova Pro - highest capability

            # Fallback to Claude models if Nova isn't available
            "anthropic.claude-3-haiku-20240307-v1:0",  # Claude 3 Haiku
            "anthropic.claude-3-sonnet-20240229-v1:0",  # Claude 3 Sonnet
        ]

        self.model_id = None
        self.model_name = None

        for model_id in model_ids_to_try:
            try:
                logger.info(f"Testing model: {model_id}")

                # For Nova models, test with direct boto3 client first
                if "nova" in model_id.lower():
                    # Test Nova model with native API format
                    test_payload = {
                        "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
                        "inferenceConfig": {
                            "temperature": 0.1,
                            "maxTokens": 50
                        }
                    }

                    response = self.bedrock_client.invoke_model(
                        modelId=model_id,
                        body=json.dumps(test_payload),
                        contentType="application/json"
                    )

                    # If this succeeds, Nova model is available
                    self.model_id = model_id
                    self.model_name = model_id.split('.')[1] if '.' in model_id else model_id
                    self.is_nova_model = True
                    logger.info(f"Successfully configured Nova model: {model_id}")
                    break

                else:
                    # Test non-Nova models with LangChain
                    test_llm = ChatBedrock(
                        client=self.bedrock_client,
                        model_id=model_id,
                        model_kwargs={
                            "temperature": self.temperature,
                            "max_tokens": 100,
                        }
                    )

                    # Try a simple test message to verify the model works
                    test_message = [HumanMessage(content="Hello")]
                    test_response = test_llm.invoke(test_message)

                    # If we get here without error, the model works
                    self.model_id = model_id
                    self.model_name = model_id.split('.')[1] if '.' in model_id else model_id
                    self.is_nova_model = False
                    logger.info(f"Successfully configured LangChain model: {model_id}")
                    break

            except Exception as e:
                logger.warning(f"Failed to configure model {model_id}: {str(e)}")
                continue

        if not self.model_id:
            error_msg = (
                "No supported Bedrock models available. Check your AWS configuration and model access."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Initialize model configuration based on type
        if hasattr(self, 'is_nova_model') and self.is_nova_model:
            # Nova models use direct boto3 client
            self.llm = None
            self.json_llm = None
            logger.info(f"Configured for Nova model: {self.model_id}")
        else:
            # Non-Nova models use LangChain ChatBedrock
            self.llm = ChatBedrock(
                client=self.bedrock_client,
                model_id=self.model_id,
                model_kwargs={
                    "temperature": self.temperature,
                    "max_tokens": 2048,
                    "top_p": 0.9,
                    "stop_sequences": []
                }
            )

            self.json_llm = ChatBedrock(
                client=self.bedrock_client,
                model_id=self.model_id,
                model_kwargs={
                    "temperature": 0.1,
                    "max_tokens": 2048,
                    "top_p": 0.8
                }
            )
            logger.info(f"Configured LangChain ChatBedrock for: {self.model_id}")

    def get_model_info(self) -> Dict[str, str]:
        """Return information about the currently configured model"""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "timeout": self.timeout
        }

    def invoke_with_monitoring(self, messages: list, system_prompt: str = None) -> Dict[str, Any]:
        """
        REQUIREMENT 6: Invoke Bedrock with monitoring, timeout, and cost tracking
        """
        start_time = time.time()

        try:
            # Check if we're using Nova model or LangChain model
            if hasattr(self, 'is_nova_model') and self.is_nova_model:
                return self._invoke_nova_model(messages, system_prompt, start_time)
            else:
                return self._invoke_langchain_model(messages, system_prompt, start_time)

        except Exception as e:
            latency = time.time() - start_time
            logger.error(
                "Bedrock call failed",
                model_id=self.model_id,
                error=str(e),
                latency_seconds=round(latency, 3)
            )
            print(f"❌ BEDROCK CALL FAILED: {str(e)} (took {latency:.3f}s)")
            raise e

    def _invoke_nova_model(self, messages: list, system_prompt: str, start_time: float) -> Dict[str, Any]:
        """Invoke Nova model using direct boto3 client"""
        # Prepare messages for Nova format
        nova_messages = []

        # Add system message if provided
        if system_prompt:
            nova_messages.append({"role": "user", "content": [{"text": system_prompt}]})
            nova_messages.append({"role": "assistant", "content": [{"text": "I understand."}]})

        # Add user messages
        for msg in messages:
            if msg["role"] == "system":
                # Convert system messages to user messages for Nova
                nova_messages.append({"role": "user", "content": [{"text": msg["content"]}]})
                nova_messages.append({"role": "assistant", "content": [{"text": "Understood."}]})
            else:
                nova_messages.append({
                    "role": msg["role"] if msg["role"] in ["user", "assistant"] else "user",
                    "content": [{"text": msg["content"]}]
                })

        # Prepare Nova payload
        payload = {
            "messages": nova_messages,
            "inferenceConfig": {
                "temperature": self.temperature,
                "maxTokens": 2048,
                "topP": 0.9
            }
        }

        # Invoke Nova model
        response = self.bedrock_client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(payload),
            contentType="application/json"
        )

        # Parse response
        response_body = json.loads(response['body'].read())
        content = response_body['output']['message']['content'][0]['text']

        # Calculate metrics
        latency = time.time() - start_time

        # Rough token estimation (1 token ≈ 4 characters)
        input_content = " ".join([msg["content"] for msg in messages])
        if system_prompt:
            input_content = system_prompt + " " + input_content
        input_tokens = len(input_content) // 4
        output_tokens = len(content) // 4

        cost_estimate = (
                (input_tokens / 1000) * self.COST_PER_1K_TOKENS["input"] +
                (output_tokens / 1000) * self.COST_PER_1K_TOKENS["output"]
        )

        # Log metrics
        logger.info(
            "Nova Bedrock call completed",
            model_id=self.model_id,
            latency_seconds=round(latency, 3),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_estimate_usd=round(cost_estimate, 6),
            temperature=self.temperature
        )

        # Circuit breaker check
        if latency > self.timeout:
            logger.warning(
                "Nova Bedrock call exceeded timeout threshold",
                model_id=self.model_id,
                latency_seconds=round(latency, 3),
                timeout_threshold=self.timeout,
                status="circuit_breaker_warning"
            )
            print(f"⚠️ CIRCUIT BREAKER WARNING: Call took {latency:.3f}s, threshold is {self.timeout}s")

        return {
            "content": content,
            "latency": latency,
            "tokens_used": input_tokens + output_tokens,
            "cost_estimate": cost_estimate,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model_id": self.model_id
        }

    def _invoke_langchain_model(self, messages: list, system_prompt: str, start_time: float) -> Dict[str, Any]:
        """Invoke non-Nova model using LangChain"""
        # Prepare messages
        if system_prompt:
            formatted_messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            formatted_messages = messages

        # Convert to LangChain format
        lc_messages = []
        for msg in formatted_messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user" or msg["role"] == "human":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))

        # Invoke with LangChain
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
            "LangChain Bedrock call completed",
            model_id=self.model_id,
            latency_seconds=round(latency, 3),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_estimate_usd=round(cost_estimate, 6),
            temperature=self.temperature
        )

        # Circuit breaker check
        if latency > self.timeout:
            logger.warning(
                "LangChain Bedrock call exceeded timeout threshold",
                model_id=self.model_id,
                latency_seconds=round(latency, 3),
                timeout_threshold=self.timeout,
                status="circuit_breaker_warning"
            )
            print(f"⚠️ CIRCUIT BREAKER WARNING: Call took {latency:.3f}s, threshold is {self.timeout}s")

        return {
            "content": response.content,
            "latency": latency,
            "tokens_used": input_tokens + output_tokens,
            "cost_estimate": cost_estimate,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model_id": self.model_id
        }

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
            # Use the appropriate method based on model type
            if hasattr(self, 'is_nova_model') and self.is_nova_model:
                response = self._invoke_nova_json_mode(messages, combined_system_prompt)
            else:
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
                        model_id=self.model_id,
                        retry_count=retry_count + 1,
                        error=str(e),
                        content_preview=content[:100]
                    )
                    return self.invoke_with_json_mode(prompt, system_prompt, retry_count + 1)
                else:
                    logger.error("Max retries exceeded for JSON parsing",
                                 model_id=self.model_id,
                                 error=str(e),
                                 content=content)
                    raise ValueError(f"Failed to get valid JSON after {self.max_retries} retries: {str(e)}")

        except Exception as e:
            if retry_count < self.max_retries:
                logger.warning(
                    "Bedrock call failed, retrying",
                    model_id=self.model_id,
                    retry_count=retry_count + 1,
                    error=str(e)
                )
                return self.invoke_with_json_mode(prompt, system_prompt, retry_count + 1)
            else:
                logger.error("Max retries exceeded for Bedrock call",
                             model_id=self.model_id,
                             error=str(e))
                raise e

    def _invoke_nova_json_mode(self, messages: list, system_prompt: str) -> Dict[str, Any]:
        """Invoke Nova model in JSON mode using direct boto3 client"""
        start_time = time.time()

        try:
            # Prepare messages for Nova format
            nova_messages = []

            # Add system message if provided
            if system_prompt:
                nova_messages.append({"role": "user", "content": [{"text": system_prompt}]})
                nova_messages.append({"role": "assistant", "content": [{"text": "I understand. I will respond with valid JSON only."}]})

            # Add user messages
            for msg in messages:
                nova_messages.append({
                    "role": msg["role"] if msg["role"] in ["user", "assistant"] else "user",
                    "content": [{"text": msg["content"]}]
                })

            # Prepare Nova payload for JSON mode
            payload = {
                "messages": nova_messages,
                "inferenceConfig": {
                    "temperature": 0.1,  # Lower temperature for JSON consistency
                    "maxTokens": 2048,
                    "topP": 0.8
                }
            }

            # Invoke Nova model
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(payload),
                contentType="application/json"
            )

            # Parse response
            response_body = json.loads(response['body'].read())
            content = response_body['output']['message']['content'][0]['text']

            # Calculate metrics
            latency = time.time() - start_time

            # Rough token estimation
            input_content = " ".join([msg["content"] for msg in messages])
            if system_prompt:
                input_content = system_prompt + " " + input_content
            input_tokens = len(input_content) // 4
            output_tokens = len(content) // 4

            cost_estimate = (
                    (input_tokens / 1000) * self.COST_PER_1K_TOKENS["input"] +
                    (output_tokens / 1000) * self.COST_PER_1K_TOKENS["output"]
            )

            # Log metrics
            logger.info(
                "Nova JSON Bedrock call completed",
                model_id=self.model_id,
                latency_seconds=round(latency, 3),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_estimate_usd=round(cost_estimate, 6),
                temperature=0.1
            )

            return {
                "content": content,
                "latency": latency,
                "tokens_used": input_tokens + output_tokens,
                "cost_estimate": cost_estimate
            }

        except Exception as e:
            latency = time.time() - start_time
            logger.error(
                "Nova JSON Bedrock call failed",
                model_id=self.model_id,
                error=str(e),
                latency_seconds=round(latency, 3)
            )
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
                model_id=self.model_id,
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
                model_id=self.model_id,
                error=str(e),
                latency_seconds=round(latency, 3)
            )
            raise e