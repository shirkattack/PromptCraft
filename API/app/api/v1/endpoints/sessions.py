from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import uuid
from app.core.database import get_db
from app.models.optimization import OptimizationSession
from app.schemas.optimization import (
    OptimizationSessionCreate,
    OptimizationSessionResponse,
    OptimizationSessionUpdate,
    PerformanceMetrics
)
from app.services.ollama_service import ollama_service
from app.services.optimization_service import optimization_service

router = APIRouter()

@router.post("/", response_model=OptimizationSessionResponse)
async def create_session(
    session_data: OptimizationSessionCreate,
    db: Session = Depends(get_db)
):
    """Create a new optimization session"""
    session_id = str(uuid.uuid4())
    
    db_session = OptimizationSession(
        id=session_id,
        name=session_data.name,
        original_prompt=session_data.original_prompt,
        provider=session_data.provider,
        model=session_data.model,
        task_type=session_data.task_type
    )
    
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    
    return db_session

@router.get("/optimization-methods")
async def get_optimization_methods():
    """Get available optimization methods"""
    return {
        "methods": [
            {
                "id": "meta_prompt",
                "name": "Meta-Prompt Optimization",
                "description": "Uses meta-prompting techniques to improve prompt effectiveness",
                "recommended_for": ["general", "creative", "analysis"]
            },
            {
                "id": "dspy", 
                "name": "DSPy Optimization",
                "description": "Uses DSPy framework for systematic prompt optimization",
                "recommended_for": ["structured", "reasoning", "code"]
            },
            {
                "id": "simple",
                "name": "Simple Optimization", 
                "description": "Basic prompt improvement using direct language model feedback",
                "recommended_for": ["quick", "basic", "testing"]
            }
        ]
    }

@router.get("/", response_model=List[OptimizationSessionResponse])
def get_sessions(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Get all optimization sessions"""
    sessions = db.query(OptimizationSession).offset(skip).limit(limit).all()
    return sessions

@router.get("/{session_id}", response_model=OptimizationSessionResponse)
def get_session(session_id: str, db: Session = Depends(get_db)):
    """Get a specific optimization session"""
    session = db.query(OptimizationSession).filter(OptimizationSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@router.put("/{session_id}", response_model=OptimizationSessionResponse)
def update_session(
    session_id: str,
    session_update: OptimizationSessionUpdate,
    db: Session = Depends(get_db)
):
    """Update an optimization session"""
    session = db.query(OptimizationSession).filter(OptimizationSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    update_data = session_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(session, field, value)
    
    db.commit()
    db.refresh(session)
    return session

@router.delete("/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db)):
    """Delete an optimization session"""
    session = db.query(OptimizationSession).filter(OptimizationSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    db.delete(session)
    db.commit()
    return {"message": "Session deleted successfully"}

@router.post("/{session_id}/optimize")
async def optimize_prompt(
    session_id: str, 
    optimization_method: str = "meta_prompt",
    db: Session = Depends(get_db)
):
    """Optimize a prompt using Promptomatix algorithms"""
    session = db.query(OptimizationSession).filter(OptimizationSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Update session status to running
    session.status = "running"
    db.commit()
    
    try:
        # Use the optimization service with Promptomatix algorithms
        optimization_result = await optimization_service.optimize_prompt(
            original_prompt=session.original_prompt,
            provider=session.provider,
            model=session.model,
            task_type=session.task_type,
            optimization_method=optimization_method
        )
        
        if optimization_result["success"]:
            # Update session with optimization results
            session.optimized_prompt = optimization_result["optimized_prompt"]
            session.performance_score = optimization_result["improvement_score"]
            session.status = "completed"
            
            db.commit()
            db.refresh(session)
            
            return {
                "message": "Prompt optimized successfully",
                "session": session,
                "optimization_details": {
                    "method": optimization_result["method"],
                    "improvement_score": optimization_result["improvement_score"],
                    "processing_time": optimization_result["processing_time"],
                    "metadata": optimization_result.get("metadata", {})
                }
            }
        else:
            # Optimization failed, update session status
            session.status = "failed"
            db.commit()
            
            raise HTTPException(
                status_code=500, 
                detail=f"Optimization failed: {optimization_result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        # Update session status to failed
        session.status = "failed"
        db.commit()
        
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

@router.get("/analytics/performance", response_model=PerformanceMetrics)
def get_performance_metrics(db: Session = Depends(get_db)):
    """Get overall performance metrics"""
    sessions = db.query(OptimizationSession).all()
    
    total_optimizations = len(sessions)
    completed_sessions = [s for s in sessions if s.status == "completed"]
    
    if completed_sessions:
        average_improvement = sum(s.performance_score for s in completed_sessions) / len(completed_sessions)
        success_rate = len(completed_sessions) / total_optimizations * 100
    else:
        average_improvement = 0.0
        success_rate = 0.0
    
    return PerformanceMetrics(
        total_optimizations=total_optimizations,
        average_improvement=average_improvement,
        success_rate=success_rate,
        total_processing_time=120.0,  # Mock data
        cost_savings=50.0  # Mock data
    )