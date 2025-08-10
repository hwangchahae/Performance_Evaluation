"""
LoRA 병합 테스트 스크립트
병합 실패 문제 디버깅용
"""

import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging
import gc

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_lora_merge():
    """LoRA 병합 테스트"""
    
    # 경로 설정
    script_dir = Path(__file__).parent
    base_model_path = "Qwen/Qwen3-8B"
    lora_path = script_dir / "qwen3_lora_ttalkkac_8b"
    
    # LoRA 어댑터 파일 확인
    logger.info("=" * 60)
    logger.info("LoRA 어댑터 파일 확인")
    logger.info("=" * 60)
    
    if not lora_path.exists():
        logger.error(f"❌ LoRA 폴더 없음: {lora_path}")
        return False
    
    # 필수 파일 확인
    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors"
    ]
    
    for file in required_files:
        file_path = lora_path / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ {file}: {size_mb:.2f} MB")
        else:
            logger.error(f"❌ {file} 없음")
            return False
    
    # adapter_config.json 내용 확인
    import json
    with open(lora_path / "adapter_config.json", 'r') as f:
        adapter_config = json.load(f)
    
    logger.info("\nAdapter 설정:")
    logger.info(f"  - base_model: {adapter_config.get('base_model_name_or_path', 'N/A')}")
    logger.info(f"  - r (rank): {adapter_config.get('r', 'N/A')}")
    logger.info(f"  - lora_alpha: {adapter_config.get('lora_alpha', 'N/A')}")
    logger.info(f"  - target_modules: {adapter_config.get('target_modules', 'N/A')}")
    
    # 베이스 모델이 맞는지 확인
    expected_base = "Qwen/Qwen3-8B"
    actual_base = adapter_config.get('base_model_name_or_path', '')
    
    if expected_base not in actual_base and "Qwen3-8B" not in actual_base:
        logger.warning(f"⚠️ 베이스 모델 불일치!")
        logger.warning(f"   예상: {expected_base}")
        logger.warning(f"   실제: {actual_base}")
    
    logger.info("\n" + "=" * 60)
    logger.info("LoRA 병합 시도")
    logger.info("=" * 60)
    
    try:
        # 메모리 정리
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info(f"📥 베이스 모델 로딩: {base_model_path}")
        logger.info("   (CPU 로딩, float16 사용)")
        
        # 베이스 모델 로드 - 메모리 최적화
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        logger.info(f"📥 LoRA 어댑터 로딩: {lora_path}")
        
        # LoRA 어댑터 로드
        model = PeftModel.from_pretrained(
            base_model, 
            str(lora_path),
            device_map="cpu"
        )
        
        logger.info("🔀 모델 병합 중...")
        merged_model = model.merge_and_unload()
        
        logger.info("✅ 병합 성공!")
        
        # 간단한 테스트 생성
        logger.info("\n테스트 생성:")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        
        test_prompt = "안녕하세요. 오늘 회의 주제는"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        # CPU에서 간단히 테스트
        with torch.no_grad():
            outputs = merged_model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"생성된 텍스트: {generated_text}")
        
        # 메모리 정리
        del base_model
        del model  
        del merged_model
        gc.collect()
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 병합 실패: {e}")
        logger.error(f"   에러 타입: {type(e).__name__}")
        
        import traceback
        logger.error("상세 에러:")
        logger.error(traceback.format_exc())
        
        return False

if __name__ == "__main__":
    logger.info("LoRA 병합 테스트 시작")
    success = test_lora_merge()
    
    if success:
        logger.info("\n✅ 테스트 성공! 병합 가능합니다.")
    else:
        logger.error("\n❌ 테스트 실패! 병합에 문제가 있습니다.")
        logger.info("\n가능한 해결 방법:")
        logger.info("1. LoRA 어댑터 파일 재다운로드")
        logger.info("2. 베이스 모델과 LoRA 버전 확인")
        logger.info("3. 메모리 확보 (최소 32GB RAM)")