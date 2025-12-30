from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any
from app.core.prompts import prompt_manager

router = APIRouter(prefix="/api", tags=["prompts"])

@router.get("/system/prompts", response_model=Dict[str, Any])
async def get_prompts():
    """
    获取所有提示词配置
    """
    return prompt_manager.get_all_prompts()

@router.post("/system/prompts/{key}")
async def update_prompt(key: str, data: Dict[str, Any] = Body(...)):
    """
    更新指定提示词配置
    """
    prompts = prompt_manager.get_all_prompts()
    if key not in prompts:
        raise HTTPException(status_code=404, detail="Prompt key not found")
    
    # Update via manager
    prompt_manager.update_prompt(key, data)
    
    return {"status": "success", "data": prompt_manager.get_prompt(key)}
