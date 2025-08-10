"""
Standalone LoRA merge script for Qwen3-8B model
병합 실패 문제를 해결하기 위한 독립 실행 스크립트
"""

import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def merge_lora_model():
    """LoRA 모델 병합"""
    
    # 경로 설정
    script_dir = Path(__file__).parent
    base_model_path = "Qwen/Qwen3-8B"
    lora_path = script_dir / "qwen3_lora_ttalkkac_8b"
    merged_path = script_dir / "8B_merged_qwen3_lora_model"
    
    # 병합된 모델이 이미 있는지 확인
    if merged_path.exists():
        logger.info(f"✅ 병합된 모델이 이미 존재: {merged_path}")
        return str(merged_path)
    
    # LoRA 어댑터가 있는지 확인
    if not lora_path.exists():
        logger.error(f"❌ LoRA 어댑터가 없음: {lora_path}")
        return None
    
    logger.info("=" * 60)
    logger.info("LoRA 병합 시작")
    logger.info("=" * 60)
    
    try:
        # 베이스 모델 로딩
        logger.info(f"📥 베이스 모델 로딩: {base_model_path}")
        logger.info("메모리 사용량을 줄이기 위해 CPU에서 로드 후 병합합니다...")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,  # bfloat16 대신 float16 사용
            device_map="cpu",  # CPU에서 병합
            trust_remote_code=True,
            low_cpu_mem_usage=True  # 메모리 사용량 최소화
        )
        
        logger.info(f"📥 LoRA 어댑터 로딩: {lora_path}")
        model = PeftModel.from_pretrained(
            base_model, 
            str(lora_path),
            device_map="cpu"
        )
        
        logger.info("🔀 모델 병합 중...")
        merged_model = model.merge_and_unload()
        
        logger.info(f"💾 병합된 모델 저장: {merged_path}")
        merged_path.mkdir(exist_ok=True)
        merged_model.save_pretrained(
            str(merged_path),
            safe_serialization=True,  # safetensors 형식으로 저장
            max_shard_size="4GB"  # 큰 모델을 여러 파일로 분할
        )
        
        # 토크나이저도 저장
        logger.info("💾 토크나이저 저장 중...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(str(merged_path))
        
        logger.info("✅ LoRA 병합 완료!")
        logger.info(f"📂 저장 위치: {merged_path}")
        
        # 메모리 정리
        del base_model
        del model
        del merged_model
        torch.cuda.empty_cache()
        
        return str(merged_path)
        
    except Exception as e:
        logger.error(f"❌ LoRA 병합 실패: {e}")
        logger.error("가능한 원인:")
        logger.error("1. 메모리 부족 - 8B 모델 병합에는 최소 32GB RAM 필요")
        logger.error("2. 디스크 공간 부족 - 약 15GB 여유 공간 필요")
        logger.error("3. 모델 다운로드 실패 - 인터넷 연결 확인")
        return None

if __name__ == "__main__":
    logger.info("Qwen3-8B LoRA 병합 스크립트")
    result = merge_lora_model()
    
    if result:
        logger.info(f"✅ 성공! 병합된 모델 경로: {result}")
    else:
        logger.error("❌ 병합 실패!")