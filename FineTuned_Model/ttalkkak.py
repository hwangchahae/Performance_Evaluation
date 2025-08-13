import json
import os
from typing import List, Dict, Any
from openai import OpenAI
from datetime import datetime
from pathlib import Path
from meeting_analysis_prompts import generate_meeting_analysis_system_prompt, generate_meeting_analysis_user_prompt

class TtalkkakGoldStandardGenerator:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def get_meeting_files(self, base_dir: str = "batch_triplet_results") -> List[str]:
        """05_final_result.json 파일들을 찾아서 반환"""
        target_files = []
        
        if os.path.exists(base_dir):
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if file == "05_final_result.json":
                        target_files.append(os.path.join(root, file))
        
        return target_files
    
    def chunk_text(self, text: str, chunk_size: int = 5000, overlap: int = 512) -> List[str]:
        """텍스트를 청킹하여 나누기"""
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
        
        return chunks

    def load_meeting_data(self, file_path: str) -> Dict[str, Any]:
        """회의 데이터를 로드하고 텍스트로 변환"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 회의 내용을 텍스트로 변환
            meeting_text = ""
            for item in data:
                timestamp = item.get('timestamp', 'Unknown')
                speaker = item.get('speaker', 'Unknown')
                text = item.get('text', '')
                meeting_text += f"[{timestamp}] {speaker}: {text}\n"
            
            full_text = meeting_text.strip()
            
            # 텍스트 길이 체크 및 청킹
            if len(full_text) > 5000:
                print(f"      📏 긴 텍스트 감지 ({len(full_text)}자) - 청킹 처리 중...")
                chunks = self.chunk_text(full_text, chunk_size=5000, overlap=512)
                print(f"      ✂️ {len(chunks)}개 청크로 분할")
                
                # 모든 청크를 리턴 (각각 별도 처리용)
                chunk_info = {
                    "is_chunked": True,
                    "total_chunks": len(chunks),
                    "original_length": len(full_text)
                }
                
                return {
                    "transcript": None,  # 청크된 경우 transcript는 None
                    "all_chunks": chunks,  # 모든 청크들
                    "metadata": {
                        "source_file": file_path,
                        "utterance_count": len(data),
                        "original_transcript_length": len(full_text),
                        "speakers": list(set(item.get('speaker', 'Unknown') for item in data)),
                        "chunking_info": chunk_info
                    }
                }
            else:
                chunk_info = {
                    "is_chunked": False,
                    "total_chunks": 1,
                    "original_length": len(full_text)
                }
                
                return {
                    "transcript": full_text,
                    "all_chunks": None,
                    "metadata": {
                        "source_file": file_path,
                        "utterance_count": len(data),
                        "transcript_length": len(full_text),
                        "original_transcript_length": len(full_text),
                        "speakers": list(set(item.get('speaker', 'Unknown') for item in data)),
                        "chunking_info": chunk_info
                    }
                }
        except Exception as e:
            print(f"파일 로드 오류 ({file_path}): {e}")
            return None
    
    def generate_initial_notion_response(self, meeting_transcript: str) -> Dict[str, Any]:
        """초기 노션 프로젝트 생성"""
        try:
            system_prompt = generate_meeting_analysis_system_prompt()
            user_prompt = generate_meeting_analysis_user_prompt(meeting_transcript)
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            
            result = response.choices[0].message.content
            
            # JSON 파싱 시도
            try:
                parsed_result = json.loads(result)
                return {"success": True, "result": parsed_result}
            except json.JSONDecodeError:
                return {"success": True, "result": result}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def evaluate_notion_response(self, meeting_transcript: str, initial_response: Any) -> Dict[str, Any]:
        """노션 응답 평가"""
        try:
            evaluation_prompt = f"""
당신은 회의록 기반 노션 기획안의 품질을 평가하는 전문 평가자입니다.

**원본 회의록:**
{meeting_transcript}

**생성된 노션 기획안:**
{json.dumps(initial_response, ensure_ascii=False, indent=2)}

**평가 기준 (각 1-10점 척도):**

1. **회의 내용 반영도 (Meeting Fidelity)**: 
   - 회의에서 논의된 내용이 정확히 반영되었는가?
   - 회의의 핵심 주제와 결정사항이 포함되어 있는가?

2. **프로젝트 정보 완성도 (Project Completeness)**: 
   - 프로젝트명, 목적, 담당자 등 필수 정보가 적절히 추출되었는가?
   - 실행 계획과 기대 효과가 구체적으로 작성되었는가?

3. **실무 활용성 (Practical Usability)**: 
   - 실제 업무에서 사용할 수 있는 수준의 기술과 지식이 포함되어 있는가?
   - 구체적이고 실행 가능한 계획인가?

4. **논리적 일관성 (Logical Consistency)**: 
   - 프로젝트 목적과 실행 계획이 논리적으로 연결되는가?
   - 기대 효과가 프로젝트 내용과 일치하는가?

5. **한국어 품질 및 비즈니스 적합성 (Korean Business Quality)**: 
   - 자연스럽고 전문적인 한국어 표현인가?

**응답 형식 (JSON):**
{{
    "scores": {{
        "meeting_fidelity": 점수,
        "project_completeness": 점수,
        "practical_usability": 점수,
        "logical_consistency": 점수,
        "korean_business_quality": 점수
    }},
    "overall_score": 평균점수,
    "strengths": ["구체적인 강점1", "구체적인 강점2"],
    "critical_issues": ["반드시 수정해야 할 문제1", "반드시 수정해야 할 문제2"],
    "improvement_suggestions": [
        "구체적인 개선 방안1 (어떤 부분을 어떻게 수정할지)",
        "구체적인 개선 방안2",
        "구체적인 개선 방안3"
    ],
    "missing_information": ["회의에서 언급됐지만 누락된 정보들"],
    "needs_refinement": true/false
}}
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "당신은 회의록 기반 프로젝트 기획안을 평가하는 전문가입니다."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            result = response.choices[0].message.content
            print(f"      🔍 평가 응답 (처음 300자): {result[:300]}...")
            
            # JSON 마크다운 블록 제거
            if result.startswith("```json"):
                result = result[7:]  # ```json 제거
            if result.startswith("```"):
                result = result[3:]   # ``` 제거
            if result.endswith("```"):
                result = result[:-3]  # 끝의 ``` 제거
            result = result.strip()
            
            try:
                parsed_result = json.loads(result)
                actual_score = parsed_result.get("overall_score", 6.0)
                print(f"      📊 파싱 성공! 실제 점수: {actual_score}")
                return {"success": True, "result": parsed_result}
            except json.JSONDecodeError as e:
                print(f"      ⚠️ JSON 파싱 실패: {str(e)}")
                print(f"      🔍 정리된 응답: {result[:200]}...")
                return {"success": True, "result": {"evaluation": result, "overall_score": 6.0}}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def refine_notion_response(self, meeting_transcript: str, initial_response: Any, evaluation: Dict) -> Dict[str, Any]:
        """노션 응답 개선"""
        try:
            refinement_prompt = f"""
회의록을 기반으로 한 노션 프로젝트 기획안을 개선해주세요.

**회의록:**
{meeting_transcript}

**현재 기획안:**
{json.dumps(initial_response, ensure_ascii=False, indent=2)}

**평가 결과:**
- 전체 점수: {evaluation.get('overall_score', 0)}/10
- 강점: {evaluation.get('strengths', [])}
- 심각한 문제: {evaluation.get('critical_issues', [])}
- 누락 정보: {evaluation.get('missing_information', [])}
- 개선 제안: {evaluation.get('improvement_suggestions', [])}

위 평가를 바탕으로 기획안을 개선해주세요.

**필수 개선 사항:**
1. 심각한 문제들을 우선적으로 해결하세요
2. 누락된 정보를 회의 내용에서 찾아 보완하세요
3. 개선 제안사항을 반영해서 더 구체적으로 작성하세요
4. 프로젝트명, 목적, 실행계획을 더 명확하게 다시 작성하세요
5. 회의에서 언급된 구체적인 내용들을 더 많이 포함시키세요

**목표**: 이전 버전보다 반드시 더 나은 기획안을 만들어주세요.

다음 JSON 형식으로 **완전히 새로운 개선된 기획안**을 응답하세요:
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "당신은 회의록을 분석해서 완벽한 노션 프로젝트 기획안을 만드는 전문가입니다."},
                    {"role": "user", "content": refinement_prompt}
                ],
                temperature=0.3,  # 사실 기반 회의록 생성을 위해 낮게 유지
                max_tokens=4000
            )
            
            result = response.choices[0].message.content
            try:
                parsed_result = json.loads(result)
                return {"success": True, "result": parsed_result}
            except json.JSONDecodeError:
                return {"success": True, "result": result}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    
    def generate_gold_standard_sample(self, meeting_transcript: str, metadata: Dict) -> Dict[str, Any]:
        """하나의 회의록에 대한 노션 프로젝트 골드 스탠다드 생성 (PRD 제외)"""
        
        # Stage 1: 초기 노션 응답 생성
        print("      🎯 Stage 1: 초기 노션 프로젝트 생성...", flush=True)
        initial_notion = self.generate_initial_notion_response(meeting_transcript)
        if not initial_notion["success"]:
            return {"success": False, "error": f"Stage 1 실패: {initial_notion['error']}"}
        
        # Stage 2: 초기 응답 평가
        print("      📊 Stage 2: 초기 응답 품질 평가...")
        evaluation = self.evaluate_notion_response(meeting_transcript, initial_notion["result"])
        if not evaluation["success"]:
            return {"success": False, "error": f"Stage 2 실패: {evaluation['error']}"}
        
        iterations_used = 1
        final_notion = initial_notion["result"]
        current_score = evaluation["result"].get("overall_score", 0)
        
        # Stage 3: 7점 이상 될 때까지 반복 개선 (최대 3회)
        max_iterations = 3
        while current_score < 7.0 and iterations_used < max_iterations:
            print(f"      ⚡ Stage 3: 응답 개선 중... (시도 {iterations_used}, 현재 점수: {current_score}/10)")
            refined_notion = self.refine_notion_response(
                meeting_transcript, 
                final_notion, 
                evaluation["result"]
            )
            if not refined_notion["success"]:
                print(f"      ⚠️ 개선 실패, 기존 결과 사용: {refined_notion.get('error', 'Unknown error')}")
                break
            
            final_notion = refined_notion["result"]
            iterations_used += 1
            
            # 개선된 결과 재평가
            print(f"      📊 개선 결과 재평가 중...")
            re_evaluation = self.evaluate_notion_response(meeting_transcript, final_notion)
            if re_evaluation["success"]:
                current_score = re_evaluation["result"].get("overall_score", 0)
                print(f"      📈 개선 후 점수: {current_score}/10")
            else:
                print(f"      ⚠️ 재평가 실패, 이전 점수 유지")
                break
        
        if current_score >= 7.0:
            print(f"      ✅ 목표 품질 달성! (최종 점수: {current_score}/10, 반복 횟수: {iterations_used})")
        else:
            print(f"      ⚠️ 최대 시도 후에도 목표 미달 (최종 점수: {current_score}/10, 반복 횟수: {iterations_used})")
        
        # 7점 이상이면 추가 검증 없이 바로 저장
        final_quality_result = {
            "final_score": current_score,
            "is_production_ready": current_score >= 7.0,
            "quality_breakdown": evaluation["result"] if evaluation["success"] else {},
            "skip_final_check": current_score >= 7.0
        }
        
        return {
            "success": True,
            "notion_result": final_notion,
            "final_quality": final_quality_result,
            "iterations_used": iterations_used
        }

def split_train_val(file_paths: List[str], train_ratio: float = 0.8) -> tuple:
    """파일을 학습/검증용으로 분할"""
    import random
    random.seed(42)  # 재현 가능한 결과
    
    shuffled_paths = file_paths.copy()
    random.shuffle(shuffled_paths)
    
    split_idx = int(len(shuffled_paths) * train_ratio)
    train_files = shuffled_paths[:split_idx]
    val_files = shuffled_paths[split_idx:]
    
    return train_files, val_files

def get_dir_name_from_path(file_path: str) -> str:
    """파일 경로에서 디렉토리 이름 추출"""
    return os.path.basename(os.path.dirname(file_path))

def print_processing_summary(processing_stats: Dict):
    """처리 요약 출력"""
    if processing_stats['processed'] > 0:
        print(f"   📈 처리 완료: {processing_stats['processed']}개")
        print(f"   ✅ 성공: {processing_stats['success']}개")
        print(f"   ❌ 실패: {processing_stats['failed']}개")
        print(f"   📈 성공률: {processing_stats['success']/processing_stats['processed']*100:.1f}%")

def process_dataset_batch(generator: TtalkkakGoldStandardGenerator, 
                         data_dirs: List[str], 
                         dataset_type: str,
                         stats: Dict,
                         main_results_dir: Path) -> List[Dict]:
    """데이터셋 배치 처리 - 각 파일별 개별 결과 폴더 생성"""
    dataset = []
    
    for i, file_path in enumerate(data_dirs):
        dir_name = get_dir_name_from_path(file_path)
        print(f"   📁 [{i+1}/{len(data_dirs)}] {dir_name} 처리 중...", flush=True)
        
        # 각 파일별 개별 결과 폴더 생성
        file_result_dir = main_results_dir / f"{dataset_type}_{i+1:03d}_{dir_name}"
        file_result_dir.mkdir(exist_ok=True)
        
        try:
            meeting_data = generator.load_meeting_data(file_path)
            if not meeting_data:
                stats["failed"] += 1
                print(f"      💥 데이터 로드 실패")
                continue
            
            metadata = meeting_data["metadata"]
            
            # 청킹된 데이터인지 확인
            if meeting_data["all_chunks"]:
                # 청킹된 경우: 각 청크별로 개별 처리
                chunks = meeting_data["all_chunks"]
                print(f"      ✂️ {len(chunks)}개 청크 각각 별도 처리")
                
                for chunk_idx, chunk_text in enumerate(chunks):
                    chunk_result_dir = main_results_dir / f"{dataset_type}_{i+1:03d}_{dir_name}_chunk_{chunk_idx+1}"
                    chunk_result_dir.mkdir(exist_ok=True)
                    
                    print(f"      🔄 청크 {chunk_idx+1}/{len(chunks)} 처리 중...")
                    
                    result = generator.generate_gold_standard_sample(chunk_text, metadata)
                    stats["processed"] += 1
                    
                    if result["success"]:
                        quality_score = result["final_quality"].get("final_score", 0)
                        is_production_ready = result["final_quality"].get("is_production_ready", False)
                        is_high_quality = quality_score >= 7.0
                        
                        # 청킹된 데이터의 개별 결과 저장
                        chunk_result = {
                            "id": f"{dataset_type}_{i+1:03d}_chunk_{chunk_idx+1}",
                            "source_dir": dir_name,
                            "notion_output": result["notion_result"],
                            "quality_metrics": {
                                "final_score": quality_score,
                                "is_production_ready": is_production_ready,
                                "iterations_used": result["iterations_used"],
                                "is_high_quality": is_high_quality
                            },
                            "metadata": {
                                **metadata,
                                "is_chunk": True,
                                "chunk_info": {
                                    "chunk_index": chunk_idx + 1,
                                    "total_chunks": len(chunks),
                                    "chunk_length": len(chunk_text)
                                },
                                "dataset_type": dataset_type,
                                "processing_date": datetime.now().isoformat()
                            }
                        }
                        
                        # 청크별 결과 파일 저장
                        chunk_file = chunk_result_dir / "result.json"
                        with open(chunk_file, 'w', encoding='utf-8') as f:
                            json.dump(chunk_result, f, ensure_ascii=False, indent=2)
                        
                        dataset.append(chunk_result)
                        stats["success"] += 1
                        
                        quality_status = "✅ 고품질" if is_high_quality else "⚠️ 저품질"
                        print(f"        {quality_status} 청크 {chunk_idx+1} 저장 완료 (품질: {quality_score}/10)")
                    else:
                        stats["failed"] += 1
                        print(f"        💥 청크 {chunk_idx+1} 생성 실패: {result.get('error', 'Unknown error')}")
            else:
                # 일반 데이터: 기존 처리 방식
                meeting_transcript = meeting_data["transcript"]
                print(f"      🔄 전체 회의록 처리 중...")
                
                result = generator.generate_gold_standard_sample(meeting_transcript, metadata)
                stats["processed"] += 1
                
                if result["success"]:
                    quality_score = result["final_quality"].get("final_score", 0)
                    is_production_ready = result["final_quality"].get("is_production_ready", False)
                    is_high_quality = quality_score >= 7.0
                    
                    # 개별 파일 결과 저장 (노션 출력만)
                    individual_result = {
                        "id": f"{dataset_type}_{i+1:03d}",
                        "source_dir": dir_name,
                        "notion_output": result["notion_result"],
                        "quality_metrics": {
                            "final_score": quality_score,
                            "is_production_ready": is_production_ready,
                            "iterations_used": result["iterations_used"],
                            "is_high_quality": is_high_quality
                        },
                        "metadata": {
                            **metadata,
                            "is_chunk": False,
                            "dataset_type": dataset_type,
                            "processing_date": datetime.now().isoformat()
                        }
                    }
                    
                    # 개별 결과 파일 저장
                    individual_file = file_result_dir / "result.json"
                    with open(individual_file, 'w', encoding='utf-8') as f:
                        json.dump(individual_result, f, ensure_ascii=False, indent=2)
                    
                    # 전체 데이터셋에도 추가
                    dataset.append(individual_result)
                    stats["success"] += 1
                    
                    quality_status = "✅ 고품질" if is_high_quality else "⚠️ 저품질"
                    print(f"      {quality_status} 저장 완료 (품질: {quality_score}/10)")
                else:
                    stats["failed"] += 1
                    print(f"      💥 생성 실패: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            stats["failed"] += 1
            stats["processed"] += 1
            print(f"      💥 처리 중 오류: {str(e)}")
    
    return dataset

def save_final_datasets(train_dataset: List[Dict], val_dataset: List[Dict], stats: Dict, results_dir: Path):
    """최종 데이터셋 및 통계 저장"""
    
    train_file = results_dir / "ttalkkac_train_gold_standard.json"
    val_file = results_dir / "ttalkkac_val_gold_standard.json"
    
    # 학습 데이터 저장
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_dataset, f, ensure_ascii=False, indent=2)
    
    # 검증 데이터 저장
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_dataset, f, ensure_ascii=False, indent=2)
    
    # 통계 계산 및 저장
    if train_dataset:
        train_quality_avg = sum(s["quality_metrics"]["final_score"] for s in train_dataset) / len(train_dataset)
        train_lengths = [s["metadata"].get("transcript_length", s["metadata"].get("original_transcript_length", 0)) for s in train_dataset]
    else:
        train_quality_avg = 0
        train_lengths = []
        
    if val_dataset:
        val_quality_avg = sum(s["quality_metrics"]["final_score"] for s in val_dataset) / len(val_dataset)
        val_lengths = [s["metadata"].get("transcript_length", s["metadata"].get("original_transcript_length", 0)) for s in val_dataset]
    else:
        val_quality_avg = 0
        val_lengths = []
    
    final_stats = {
        **stats,
        "end_time": datetime.now().isoformat(),
        "final_train_count": len(train_dataset),
        "final_val_count": len(val_dataset),
        "train_quality_average": train_quality_avg,
        "val_quality_average": val_quality_avg,
        "train_length_stats": {
            "min": min(train_lengths) if train_lengths else 0,
            "max": max(train_lengths) if train_lengths else 0,
            "avg": sum(train_lengths) / len(train_lengths) if train_lengths else 0
        },
        "val_length_stats": {
            "min": min(val_lengths) if val_lengths else 0,
            "max": max(val_lengths) if val_lengths else 0,
            "avg": sum(val_lengths) / len(val_lengths) if val_lengths else 0
        }
    }
    
    stats_file = results_dir / "ttalkkac_generation_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 저장된 파일들 ({results_dir}):")
    print(f"   📚 ttalkkac_train_gold_standard.json ({len(train_dataset)}개 샘플)")
    print(f"   🔍 ttalkkac_val_gold_standard.json ({len(val_dataset)}개 샘플)")
    print(f"   📊 ttalkkac_generation_statistics.json")
    print(f"   📂 개별 결과 폴더들: train_xxx_*, val_xxx_*")
    print(f"\n📈 품질 통계:")
    print(f"   학습 데이터 평균 품질: {train_quality_avg:.2f}/10")
    print(f"   검증 데이터 평균 품질: {val_quality_avg:.2f}/10")

if __name__ == "__main__":
    # Windows 콘솔 인코딩 설정 (안전한 방식)
    import sys
    import locale
    
    # 출력 버퍼링 비활성화로 실시간 로그 출력
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    # 한글 출력을 위한 인코딩 설정
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except:
            # fallback: 기본 인코딩 사용
            pass
    
    # OpenAI API 키 설정
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        # 직접 API 키 설정 (임시)
        API_KEY = ""
        print("API key set directly in code")
    
    if not API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Windows: set OPENAI_API_KEY=your_api_key_here")
        print("Linux/Mac: export OPENAI_API_KEY=your_api_key_here")
        exit(1)
    
    print("🚀 TtalKkac 골드 스탠다드 데이터셋 생성 시작!")
    print("=" * 60)
    
    # 메인 결과 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_results_dir = Path(f"ttalkkac_gold_standard_results_{timestamp}")
    main_results_dir.mkdir(exist_ok=True)
    print(f"📂 메인 결과 폴더: {main_results_dir}")
    
    # 처리 통계
    stats = {
        "start_time": datetime.now().isoformat(),
        "processed": 0,
        "success": 0,
        "failed": 0
    }
    
    # 생성기 초기화
    generator = TtalkkakGoldStandardGenerator(api_key=API_KEY)
    
    # 회의 파일 목록 가져오기 (상대 경로로 변경)
    meeting_files = generator.get_meeting_files("../batch_triplet_results")
    if not meeting_files:
        print("❌ batch_triplet_results 폴더에서 05_final_result.json 파일을 찾을 수 없습니다.")
        exit(1)
    
    print(f"📊 발견된 회의 파일: {len(meeting_files)}개")
    
    # 학습/검증 데이터 분할
    train_dirs, val_dirs = split_train_val(meeting_files, train_ratio=0.8)
    print(f"📚 학습용: {len(train_dirs)}개")
    print(f"🔍 검증용: {len(val_dirs)}개")
    
    # 결과 저장용
    train_dataset = []
    val_dataset = []
    
    print(f"\n📚 학습 데이터 생성 중...")
    print("=" * 40)
    train_dataset = process_dataset_batch(generator, train_dirs, "train", stats, main_results_dir)
    
    print(f"\n🔍 검증 데이터 생성 중...")
    print("=" * 40)
    val_dataset = process_dataset_batch(generator, val_dirs, "val", stats, main_results_dir)
    
    # 최종 저장
    print(f"\n💾 최종 데이터셋 저장 중...")
    save_final_datasets(train_dataset, val_dataset, stats, main_results_dir)
    
    print(f"\n✅ 골드 스탠다드 생성 완료!")
    print("=" * 60)