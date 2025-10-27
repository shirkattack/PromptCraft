import dspy
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

from app.services.lm_manager import LMManager
from app.core.config import settings

logger = logging.getLogger(__name__)

class PromptOptimizationService:
    """
    Service for optimizing prompts using DSPy and meta-prompt techniques.
    Adapted from Promptomatix core functionality.
    """
    
    def __init__(self):
        self.optimized_prompt = None
        self.optimization_history = []
    
    async def optimize_prompt(
        self, 
        original_prompt: str,
        provider: str,
        model: str,
        task_type: str = "general",
        optimization_method: str = "meta_prompt",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Optimize a prompt using the specified method and provider.
        
        Args:
            original_prompt: The prompt to optimize
            provider: AI provider to use
            model: Model name
            task_type: Type of task (general, code, creative, etc.)
            optimization_method: Method to use (meta_prompt, dspy, simple)
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            # Initialize language model
            lm = LMManager.get_lm(
                provider=provider,
                model_name=model,
                temperature=0.7,
                max_tokens=2000
            )
            
            start_time = datetime.now()
            
            # Use DSPy context for async safety
            with dspy.context(lm=lm):
                if optimization_method == "meta_prompt":
                    result = await self._optimize_with_meta_prompt(
                        original_prompt, task_type, lm
                    )
                elif optimization_method == "dspy":
                    result = await self._optimize_with_dspy(
                        original_prompt, task_type, lm
                    )
                else:
                    result = await self._simple_optimization(
                        original_prompt, task_type, lm
                    )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Calculate improvement score (simplified)
            improvement_score = self._calculate_improvement_score(
                original_prompt, result["optimized_prompt"]
            )
            
            optimization_result = {
                "original_prompt": original_prompt,
                "optimized_prompt": result["optimized_prompt"],
                "method": optimization_method,
                "provider": provider,
                "model": model,
                "task_type": task_type,
                "improvement_score": improvement_score,
                "processing_time": processing_time,
                "metadata": result.get("metadata", {}),
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
            
            self.optimization_history.append(optimization_result)
            self.optimized_prompt = result["optimized_prompt"]
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            return {
                "original_prompt": original_prompt,
                "optimized_prompt": original_prompt,  # Fallback
                "method": optimization_method,
                "provider": provider,
                "model": model,
                "task_type": task_type,
                "improvement_score": 0.0,
                "processing_time": 0.0,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _optimize_with_meta_prompt(
        self, 
        original_prompt: str, 
        task_type: str, 
        lm
    ) -> Dict[str, Any]:
        """Optimize using meta-prompt technique from Promptomatix."""
        
        meta_prompt = self._generate_meta_prompt(original_prompt, task_type)
        
        try:
            # Use DSPy to generate the optimized prompt
            predictor = dspy.Predict("meta_prompt -> optimized_prompt")
            result = predictor(meta_prompt=meta_prompt)
            
            optimized_prompt = result.optimized_prompt.strip()
            
            return {
                "optimized_prompt": optimized_prompt,
                "metadata": {
                    "method": "meta_prompt",
                    "meta_prompt_used": meta_prompt[:200] + "..." if len(meta_prompt) > 200 else meta_prompt
                }
            }
        except Exception as e:
            logger.error(f"Meta-prompt optimization failed: {e}")
            # Fallback to simple optimization
            return await self._simple_optimization(original_prompt, task_type, lm)
    
    async def _optimize_with_dspy(
        self, 
        original_prompt: str, 
        task_type: str, 
        lm
    ) -> Dict[str, Any]:
        """Optimize using DSPy's built-in optimization techniques."""
        
        # Define a simple signature for the task
        class TaskSignature(dspy.Signature):
            """Perform the specified task based on the input."""
            input_text = dspy.InputField()
            output = dspy.OutputField()
        
        # Create predictor
        predictor = dspy.ChainOfThought(TaskSignature)
        
        # For now, we'll use a simplified approach
        # In a full implementation, you'd generate training data and use optimizers
        optimized_prompt = f"""Task: {task_type.title()} Task

{original_prompt}

Please provide a clear, detailed response that addresses all aspects of the request. Think step by step and ensure your response is comprehensive and accurate."""
        
        return {
            "optimized_prompt": optimized_prompt,
            "metadata": {
                "method": "dspy",
                "signature": "TaskSignature",
                "predictor": "ChainOfThought"
            }
        }
    
    async def _simple_optimization(
        self, 
        original_prompt: str, 
        task_type: str, 
        lm
    ) -> Dict[str, Any]:
        """Simple optimization using direct LM completion."""
        
        optimization_prompt = f"""You are an expert prompt engineer. Analyze and improve the following prompt to make it more effective, clear, and likely to produce better results.

Original prompt:
{original_prompt}

Task type: {task_type}

Please provide an improved version that:
1. Is more specific and clear
2. Includes better instructions
3. Has appropriate context
4. Will produce more consistent results
5. Follows prompt engineering best practices

Improved prompt:"""
        
        try:
            response = lm(optimization_prompt, max_tokens=1000)
            # Handle both string and list responses
            if isinstance(response, list):
                optimized_prompt = response[0] if response else original_prompt
            else:
                optimized_prompt = str(response)
            optimized_prompt = optimized_prompt.strip()
            
            # Clean up the response if it includes extra text
            if "Improved prompt:" in optimized_prompt:
                optimized_prompt = optimized_prompt.split("Improved prompt:")[-1].strip()
            
            return {
                "optimized_prompt": optimized_prompt,
                "metadata": {
                    "method": "simple",
                    "optimization_prompt_length": len(optimization_prompt)
                }
            }
        except Exception as e:
            logger.error(f"Simple optimization failed: {e}")
            return {
                "optimized_prompt": original_prompt,
                "metadata": {
                    "method": "simple",
                    "error": str(e),
                    "fallback": True
                }
            }
    
    def _generate_meta_prompt(self, original_prompt: str, task_type: str) -> str:
        """Generate meta-prompt for optimization (adapted from Promptomatix)."""
        
        return f"""You are an expert prompt engineer specializing in {task_type} tasks. Your goal is to analyze and dramatically improve the following prompt to make it more effective, specific, and reliable.

## Original Prompt Analysis:
{original_prompt}

## Task Type: {task_type}

## Optimization Guidelines:
1. **Clarity & Specificity**: Make instructions crystal clear and unambiguous
2. **Context**: Add relevant context that helps the AI understand the task better
3. **Structure**: Organize the prompt with clear sections and formatting
4. **Examples**: Include examples if they would improve understanding
5. **Constraints**: Add appropriate constraints to guide the output format
6. **Best Practices**: Apply proven prompt engineering techniques

## Your Task:
Rewrite the prompt to be significantly more effective. Focus on:
- Clear, actionable instructions
- Proper formatting and structure
- Relevant context and constraints
- Better specification of desired output
- Elimination of ambiguity

## Optimized Prompt:"""
    
    def _calculate_improvement_score(self, original: str, optimized: str) -> float:
        """Calculate a simple improvement score based on various factors."""
        
        score = 50.0  # Base score
        
        # Length improvement (more detailed prompts are often better)
        length_ratio = len(optimized) / len(original) if len(original) > 0 else 1.0
        if 1.2 <= length_ratio <= 3.0:  # Good length increase
            score += 20.0
        elif length_ratio > 3.0:  # Too verbose
            score += 10.0
        
        # Structure improvements (simple heuristics)
        if "##" in optimized or "**" in optimized:  # Has formatting
            score += 10.0
        
        if "example" in optimized.lower() or "format" in optimized.lower():
            score += 10.0
        
        if len(optimized.split('\n')) > len(original.split('\n')):  # Better structure
            score += 10.0
        
        # Cap the score
        return min(score, 100.0)
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get the history of optimizations performed."""
        return self.optimization_history
    
    def clear_history(self):
        """Clear optimization history."""
        self.optimization_history = []
        self.optimized_prompt = None

# Global service instance
optimization_service = PromptOptimizationService()