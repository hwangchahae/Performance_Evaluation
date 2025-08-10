import json
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import gc

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelConfig:
    """모델 설정 관리 클래스"""
    MODEL_PATH: str = "Qwen/Qwen3-4B-AWQ"
    MAX_NEW_TOKENS: int = 2048
    TEMPERATURE: float = 0.2
    TOP_P: float = 0.9
    REPETITION_PENALTY: float = 1.1
    CHUNK_SIZE: int = 5000
    CHUNK_OVERLAP: int = 512
    TEST_FILE_LIMIT: int = 0  # 0이면 전체 파일 처리, 양수면 해당 개수만 처리
    
    # vLLM 전용 설정
    TENSOR_PARALLEL_SIZE: int = 1  # GPU 수
    GPU_MEMORY_UTILIZATION: float = 0.9  # GPU 메모리 사용률
    MAX_MODEL_LEN: int = 16384  # 최대 컨텍스트 길이
    DTYPE: str = "auto"  # 또는 "float16", "bfloat16"
    TRUST_REMOTE_CODE: bool = True  # Qwen 모델에 필요
    ENFORCE_EAGER: bool = False  # CUDA graphs 사용
    MAX_NUM_SEQS: int = 256  # 동시 처리 시퀀스 수
    
    # 배치 처리 설정
    BATCH_SIZE: int = 3  # 한 번에 처리할 파일 수
    MAX_CHUNKS_PER_BATCH: int = 30  # 배치당 최대 청크 수


@dataclass
class MeetingData:
    """회의 데이터 구조체"""
    transcript: Optional[str] = None
    chunks: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_chunked(self) -> bool:
        return self.chunks is not None


@dataclass
class ProcessingStats:
    """처리 통계 관리"""
    total: int = 0
    processed: int = 0
    success: int = 0
    failed: int = 0
    chunked: int = 0
    
    @property
    def success_rate(self) -> float:
        if self.processed == 0:
            return 0.0
        return (self.success / self.processed) * 100


class QwenVLLMMeetingProcessor:
    """vLLM을 사용한 Qwen AWQ 모델 회의록 처리기"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        생성자
        
        Args:
            config: 모델 설정 객체
        """
        self.config = config or ModelConfig()
        self.model = None
        self.tokenizer = None
        self.sampling_params = None
        
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """vLLM 모델 초기화"""
        try:
            logger.info(f"🔧 vLLM 모델 초기화 시작...")
            logger.info(f"🚀 선택된 모델: {self.config.MODEL_PATH}")
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.MODEL_PATH,
                trust_remote_code=self.config.TRUST_REMOTE_CODE
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # vLLM 모델 초기화
            self.model = LLM(
                model=self.config.MODEL_PATH,
                quantization="awq" if "AWQ" in self.config.MODEL_PATH else None,
                tensor_parallel_size=self.config.TENSOR_PARALLEL_SIZE,
                gpu_memory_utilization=self.config.GPU_MEMORY_UTILIZATION,
                max_model_len=self.config.MAX_MODEL_LEN,
                dtype=self.config.DTYPE,
                trust_remote_code=self.config.TRUST_REMOTE_CODE,
                enforce_eager=self.config.ENFORCE_EAGER,
                max_num_seqs=self.config.MAX_NUM_SEQS,
            )
            
            # 샘플링 파라미터 설정
            self.sampling_params = SamplingParams(
                temperature=self.config.TEMPERATURE,
                top_p=self.config.TOP_P,
                repetition_penalty=self.config.REPETITION_PENALTY,
                max_tokens=self.config.MAX_NEW_TOKENS,
                skip_special_tokens=True,
            )
            
            logger.info("✅ vLLM 모델 초기화 완료!")
            
        except Exception as e:
            logger.error(f"❌ vLLM 모델 초기화 실패: {e}")
            raise
    
    def find_meeting_files(self, base_dir: str) -> List[Path]:
        """
        회의 파일 검색
        
        Args:
            base_dir: 검색할 기본 디렉토리
            
        Returns:
            발견된 파일 경로 리스트
        """
        base_path = Path(base_dir)
        if not base_path.exists():
            logger.warning(f"디렉토리가 존재하지 않음: {base_dir}")
            return []
        
        target_files = list(base_path.rglob("05_final_result.json"))
        logger.info(f"📁 {len(target_files)}개의 회의 파일 발견")
        return target_files
    
    def chunk_text(self, text: str, chunk_size: int = 5000, overlap: int = 512) -> List[str]:
        """텍스트를 청킹하여 나누기 (문자 단위)"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end >= len(text):
                chunk = text[start:]
            else:
                chunk = text[start:end]
                
                # 마지막 완전한 문장에서 끊기 시도
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:
                    chunk = text[start:break_point + 1]
                    end = break_point + 1
            
            chunks.append(chunk.strip())
            
            if end >= len(text):
                break
                
            start = end - overlap
        
        logger.info(f"📊 {len(chunks)}개 청크 생성 (문자 기반 5000자)")
        return chunks
    
    def load_meeting_data(self, file_path: Path) -> Optional[MeetingData]:
        """
        회의 데이터 로드
        
        Args:
            file_path: 파일 경로
            
        Returns:
            MeetingData 객체 또는 None
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 텍스트 변환
            meeting_lines = []
            speakers = set()
            
            for item in data:
                timestamp = item.get('timestamp', 'Unknown')
                speaker = item.get('speaker', 'Unknown')
                text = item.get('text', '')
                if text:  # 텍스트가 있는 경우만 추가
                    speakers.add(speaker)
                    meeting_lines.append(f"[{timestamp}] {speaker}: {text}")
            
            if not meeting_lines:
                logger.warning(f"텍스트가 없는 파일: {file_path}")
                return None
            
            full_text = '\n'.join(meeting_lines)
            
            # 메타데이터 생성
            metadata = {
                "source_file": str(file_path),
                "utterance_count": len(data),
                "speakers": list(speakers),
                "original_length": len(full_text)
            }
            
            # 청킹 여부 결정
            if len(full_text) > self.config.CHUNK_SIZE:
                logger.info(f"긴 텍스트 감지 ({len(full_text)}자) - 청킹 처리")
                chunks = self.chunk_text(full_text, self.config.CHUNK_SIZE, self.config.CHUNK_OVERLAP)
                metadata["chunking_info"] = {
                    "is_chunked": True,
                    "total_chunks": len(chunks)
                }
                return MeetingData(chunks=chunks, metadata=metadata)
            else:
                metadata["chunking_info"] = {
                    "is_chunked": False,
                    "total_chunks": 1
                }
                return MeetingData(transcript=full_text, metadata=metadata)
                
        except Exception as e:
            logger.error(f"파일 로드 오류 ({file_path}): {e}")
            return None
    
    def generate_notion_project_prompt(self, meeting_transcript: str) -> str:
        """노션 기획안 생성 프롬프트"""
        return f"""다음 회의 전사본을 바탕으로 노션에 업로드할 프로젝트 기획안을 작성하세요.

**회의 전사본:**
{meeting_transcript}

**작성 지침:**
1. 회의에서 논의된 내용을 바탕으로 체계적인 기획안을 작성
2. 프로젝트명은 회의 내용을 바탕으로 적절히 명명
3. 목적과 목표는 명확하고 구체적으로 작성
4. 실행 계획은 실현 가능한 단계별로 구성
5. 기대 효과는 정량적/정성적 결과를 포함
6. 모든 내용은 한국어로 작성

**응답 형식:**
다음 JSON 형식으로 응답하세요:
{{
    "project_name": "프로젝트명",
    "project_purpose": "프로젝트의 주요 목적",
    "project_period": "예상 수행 기간",
    "project_manager": "담당자명",
    "core_objectives": ["목표 1", "목표 2", "목표 3"],
    "core_idea": "핵심 아이디어",
    "idea_description": "아이디어 설명",
    "execution_plan": "실행 계획",
    "expected_effects": ["효과 1", "효과 2", "효과 3"]
}}"""
    
    def generate_batch_responses(self, prompts: List[str]) -> List[Optional[str]]:
        """
        배치 처리를 위한 vLLM 생성
        
        Args:
            prompts: 프롬프트 리스트
            
        Returns:
            생성된 응답 리스트
        """
        try:
            if not prompts:
                return []
            
            logger.info(f"🚀 {len(prompts)}개 프롬프트 배치 처리 중...")
            
            # vLLM 배치 생성
            outputs = self.model.generate(
                prompts=prompts,
                sampling_params=self.sampling_params
            )
            
            # 결과 추출
            results = []
            for output in outputs:
                if output.outputs:
                    results.append(output.outputs[0].text.strip())
                else:
                    results.append("{}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 배치 응답 생성 오류: {e}")
            return ["{}" for _ in prompts]
    
    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        JSON 응답 파싱
        
        Args:
            response: 모델 응답 문자열
            
        Returns:
            파싱된 딕셔너리
        """
        try:
            # JSON 블록 추출
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
            else:
                return {"error": "No JSON found", "raw_text": response}
            
            # JSON 파싱
            parsed = json.loads(json_str)
            
            # 필수 필드 검증
            required_fields = ["project_name", "project_purpose"]
            for field in required_fields:
                if field not in parsed:
                    parsed[field] = "미정"
            
            return parsed
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 파싱 실패: {e}")
            return {"error": "JSON parsing failed", "raw_text": response}
        except Exception as e:
            logger.warning(f"예상치 못한 오류: {e}")
            return {"error": str(e), "raw_text": response}
    
    def process_batch(self, 
                     batch_data: List[Tuple[str, Path, MeetingData]],
                     output_dir: Path) -> Tuple[int, int]:
        """
        배치 단위로 파일 처리
        
        Args:
            batch_data: (폴더명, 파일경로, 회의데이터) 튜플 리스트
            output_dir: 출력 디렉토리
            
        Returns:
            (성공 수, 실패 수) 튜플
        """
        success_count = 0
        fail_count = 0
        
        # 모든 프롬프트 준비
        all_prompts = []
        metadata_list = []
        
        for folder_name, file_path, meeting_data in batch_data:
            if meeting_data.is_chunked:
                # 청킹된 데이터
                for chunk_idx, chunk_text in enumerate(meeting_data.chunks):
                    prompt = self.generate_notion_project_prompt(chunk_text)
                    all_prompts.append(prompt)
                    metadata_list.append({
                        "folder_name": folder_name,
                        "file_path": file_path,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(meeting_data.chunks),
                        "meeting_metadata": meeting_data.metadata
                    })
            else:
                # 단일 텍스트
                prompt = self.generate_notion_project_prompt(meeting_data.transcript)
                all_prompts.append(prompt)
                metadata_list.append({
                    "folder_name": folder_name,
                    "file_path": file_path,
                    "chunk_idx": 0,
                    "total_chunks": 1,
                    "meeting_metadata": meeting_data.metadata
                })
        
        # 배치 생성
        responses = self.generate_batch_responses(all_prompts)
        
        # 결과 저장
        for response, meta in zip(responses, metadata_list):
            try:
                # 디렉토리 결정
                if meta["total_chunks"] == 1:
                    save_dir = output_dir / meta["folder_name"]
                    save_id = meta["folder_name"]
                else:
                    save_dir = output_dir / f"{meta['folder_name']}_chunk_{meta['chunk_idx']+1}"
                    save_id = f"{meta['folder_name']}_chunk_{meta['chunk_idx']+1}"
                
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # 결과 데이터 구성
                result_data = {
                    "id": save_id,
                    "source_dir": meta["folder_name"],
                    "notion_output": self.parse_json_response(response),
                    "metadata": {
                        **meta["meeting_metadata"],
                        "is_chunk": meta["total_chunks"] > 1,
                        "chunk_index": meta["chunk_idx"] + 1 if meta["total_chunks"] > 1 else None,
                        "total_chunks": meta["total_chunks"],
                        "processing_date": datetime.now().isoformat(),
                        "model_used": self.config.MODEL_PATH
                    }
                }
                
                # JSON 저장
                with open(save_dir / "result.json", 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2)
                
                success_count += 1
                
                if meta["total_chunks"] > 1:
                    logger.info(f"✅ 청크 {meta['chunk_idx']+1}/{meta['total_chunks']} 저장 완료: {save_id}")
                else:
                    logger.info(f"✅ 저장 완료: {save_id}")
                    
            except Exception as e:
                logger.error(f"❌ 저장 실패 ({save_id}): {e}")
                fail_count += 1
        
        return success_count, fail_count
    
    def process_all_files(self, 
                         meeting_files: List[Path],
                         output_dir: Path) -> ProcessingStats:
        """
        모든 파일 처리
        
        Args:
            meeting_files: 처리할 파일 리스트
            output_dir: 출력 디렉토리
            
        Returns:
            처리 통계
        """
        stats = ProcessingStats(total=len(meeting_files))
        
        # 배치 준비
        current_batch = []
        current_chunk_count = 0
        batch_num = 1
        
        for file_path in meeting_files:
            parent_folder = file_path.parent.name
            
            try:
                # 데이터 로드
                meeting_data = self.load_meeting_data(file_path)
                if not meeting_data:
                    stats.failed += 1
                    stats.processed += 1
                    logger.warning(f"⚠️ 데이터 로드 실패: {file_path}")
                    continue
                
                # 청킹 통계
                if meeting_data.is_chunked:
                    stats.chunked += 1
                    chunk_count = len(meeting_data.chunks)
                else:
                    chunk_count = 1
                
                # 배치 크기 확인
                if (current_chunk_count + chunk_count > self.config.MAX_CHUNKS_PER_BATCH 
                    and current_batch):
                    # 현재 배치 처리
                    logger.info(f"\n📦 배치 {batch_num} 처리 중... ({current_chunk_count}개 청크)")
                    success, fail = self.process_batch(current_batch, output_dir)
                    stats.success += success
                    stats.failed += fail
                    stats.processed += success + fail
                    
                    # 메모리 정리
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # 새 배치 시작
                    batch_num += 1
                    current_batch = [(parent_folder, file_path, meeting_data)]
                    current_chunk_count = chunk_count
                else:
                    # 현재 배치에 추가
                    current_batch.append((parent_folder, file_path, meeting_data))
                    current_chunk_count += chunk_count
                    
            except Exception as e:
                logger.error(f"❌ 파일 처리 실패 ({file_path}): {e}")
                stats.failed += 1
                stats.processed += 1
        
        # 마지막 배치 처리
        if current_batch:
            logger.info(f"\n📦 배치 {batch_num} 처리 중... ({current_chunk_count}개 청크)")
            success, fail = self.process_batch(current_batch, output_dir)
            stats.success += success
            stats.failed += fail
            stats.processed += success + fail
        
        return stats


def main():
    """메인 실행 함수"""
    logger.info("=" * 60)
    logger.info("🚀 vLLM을 사용한 Qwen AWQ 모델 회의록 처리 시작!")
    logger.info("=" * 60)
    
    # 설정 로드
    config = ModelConfig()
    
    # 입출력 경로 설정
    input_dir = "../Raw_Data_val"
    output_dir = Path("4B_awq_model_results_structured")
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"📂 입력: {input_dir}")
    logger.info(f"📂 출력: {output_dir}")
    logger.info(f"⚙️ 배치 크기: {config.BATCH_SIZE}, 최대 청크/배치: {config.MAX_CHUNKS_PER_BATCH}")
    
    # 모델 초기화
    try:
        processor = QwenVLLMMeetingProcessor(config)
    except Exception as e:
        logger.error(f"모델 초기화 실패: {e}")
        return
    
    # 회의 파일 검색
    meeting_files = processor.find_meeting_files(input_dir)
    
    if not meeting_files:
        logger.error(f"{input_dir} 폴더에서 파일을 찾을 수 없습니다.")
        return
    
    # 테스트 모드 처리
    if config.TEST_FILE_LIMIT > 0:
        meeting_files = meeting_files[:config.TEST_FILE_LIMIT]
        logger.info(f"🧪 테스트 모드: {len(meeting_files)}개 파일만 처리")
    else:
        logger.info(f"📋 전체 파일 처리 모드: {len(meeting_files)}개 파일")
    
    # 처리 시작
    start_time = datetime.now()
    stats = processor.process_all_files(meeting_files, output_dir)
    elapsed_time = datetime.now() - start_time
    
    # 결과 출력
    logger.info("=" * 60)
    logger.info("✅ 처리 완료 통계:")
    logger.info(f"  📁 전체 파일: {stats.total}개")
    logger.info(f"  ✔️ 처리 완료: {stats.processed}개")
    logger.info(f"  ✅ 성공: {stats.success}개")
    logger.info(f"  ❌ 실패: {stats.failed}개")
    logger.info(f"  📊 청킹 처리: {stats.chunked}개")
    logger.info(f"  📈 성공률: {stats.success_rate:.1f}%")
    logger.info(f"  ⏱️ 소요 시간: {elapsed_time}")
    logger.info(f"\n🎉 모든 처리 완료! 결과는 {output_dir}에 저장되었습니다.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()