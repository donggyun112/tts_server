from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
import numpy as np
from utils.audio_converter import AudioConverter
import time
from MeloTTS.melo.utils import get_text_for_tts_infer
from MeloTTS.melo.api import TTS
import re

class TTSModelAdapter(ABC):
    
    @abstractmethod
    def initialize(self) -> bool:
        pass
    
    @abstractmethod
    def warmup(self) -> bool:
        pass
    
    @abstractmethod
    def list_voices(self) -> List[str]:
        pass
    
    @abstractmethod
    def generate_audio(
        self, 
        text: str, 
        voice: str, 
        speed: float, 
        target_sr: int, 
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """텍스트에서 오디오 생성 (전체 텍스트 처리)
        
        Returns:
            Tuple[np.ndarray, int]: (오디오 데이터, 샘플레이트)
        """
        pass
    
    @abstractmethod
    def generate_audio_by_sentence(
        self,
        sentence: str,
        voice: str,
        speed: float,
        model_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """단일 문장에 대한 오디오 생성
        
        Returns:
            np.ndarray: 생성된 오디오 데이터
        """
        pass
    
    @abstractmethod
    def split_sentences(self, text: str, lang: str) -> List[str]:
        """텍스트를 문장으로 분리
            text: 분리할 텍스트
            lang: 언어 코드
        Returns:
            List[str]: 문장 목록
        """
        pass
    
    @property
    @abstractmethod
    def default_sample_rate(self) -> int:
        """모델의 기본 샘플레이트"""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """모델 이름"""
        pass
    

class MeloTTSAdapter(TTSModelAdapter):
    """MeloTTS 모델 어댑터"""
    
    def __init__(self, device: str = "mps"):
        self.device = device
        self.models = {}  # 언어별 모델 저장
        self.voices = {}  # 사용 가능한 화자 목록
        self._model_sr = 24000  # MeloTTS 기본 샘플레이트
    
    def initialize(self) -> bool:
        """모델 초기화"""
        try:
            # MeloTTS 라이브러리 임포트
            
            
            # 기본 모델(KR) 로드
            tts = TTS(language="KR", device=self.device)
            self.models["KR"] = tts
            self._model_sr = tts.hps.data.sampling_rate
            
            # 화자 목록 설정
            for name, idx in tts.hps.data.spk2id.items():
                self.voices[name] = idx
            
            return True
            
        except ImportError:
            return False
        except Exception:
            return False
    
    def warmup(self) -> bool:
        """모델 워밍업"""
        try:
            
            import torch
            
            # 각 로드된 모델마다 워밍업 수행
            for lang, tts in self.models.items():
                # 언어별 적절한 워밍업 텍스트 선택
                if lang == "KR":
                    warmup_text = "안녕하세요. 이것은 모델 워밍업을 위한 텍스트입니다."
                elif lang == "EN":
                    warmup_text = "Hello. This is a text for model warmup."
                elif lang == "JP":
                    warmup_text = "こんにちは。これはモデルのウォームアップ用のテキストです。"
                elif lang == "ZH":
                    warmup_text = "你好。这是模型预热的文本。"
                else:
                    warmup_text = "Hello. This is a text for model warmup."
                
                # 대표 화자 선택 (첫 번째 화자)
                speaker = next(iter(tts.hps.data.spk2id.items()))[1]
                
                # 텍스트 처리
                bert, ja_bert, phones, tones, lang_ids = get_text_for_tts_infer(
                    warmup_text, tts.language, tts.hps, tts.device, tts.symbol_to_id
                )
                speakers = torch.LongTensor([speaker]).to(tts.device)
                
                # 워밍업 추론 수행
                with torch.no_grad():
                    start_time = time.time()
                    out = tts.model.infer(
                        phones.unsqueeze(0).to(tts.device),
                        torch.LongTensor([phones.size(0)]).to(tts.device),
                        speakers,
                        tones.unsqueeze(0).to(tts.device),
                        lang_ids.unsqueeze(0).to(tts.device),
                        bert.unsqueeze(0).to(tts.device),
                        ja_bert.unsqueeze(0).to(tts.device),
                        sdp_ratio=0.2, noise_scale=0.6,
                        noise_scale_w=0.8, length_scale=1.0
                    )
                
                # 리소스 해제
                del bert, ja_bert, phones, tones, lang_ids, speakers
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            
            return True
            
        except Exception:
            return False
    
    def load_model(self, lang: str) -> bool:
        """지정된 언어의 TTS 모델 로드"""
        if lang in self.models:
            return True
            
        try:
            
            tts = TTS(language=lang, device=self.device)
            self.models[lang] = tts
            
            # 화자 목록 업데이트
            for name, idx in tts.hps.data.spk2id.items():
                self.voices[name] = idx
            
            return True
            
        except Exception:
            return False
    
    def list_voices(self) -> List[str]:
        """사용 가능한 음성 목록 반환"""
        return list(self.voices.keys())
    
    def generate_audio(
        self, 
        text: str, 
        voice: str = "KR", 
        speed: float = 1.0, 
        target_sr: int = 24000,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """텍스트에서 오디오 생성 (전체 텍스트)"""
        try:
            # 특수 문자 제거 및 공백 정리
            text = text.strip()
            # 문장 분리
            sentences = self.split_sentences(text, voice)
            if not sentences:
                # 빈 배열 대신 무음 생성
                silent_audio = np.zeros(int(0.5 * target_sr), dtype=np.float32)  # 0.5초 무음
                return silent_audio, target_sr
            
            # 전체 오디오를 담을 배열
            full_audio = np.array([], dtype=np.float32)
            
            for s_idx, sent in enumerate(sentences, 1):
                # 단일 문장에 대한 오디오 생성
                audio = self.generate_audio_by_sentence(
                    sentence=sent,
                    voice=voice,
                    speed=speed,
                    model_sr=self._model_sr,
                    target_sr=target_sr
                )
                
                # 생성된 오디오가 없으면 건너뜀
                if audio.size == 0:
                    continue
                
                # 오디오 결합
                full_audio = np.concatenate([full_audio, audio])
            
            # 최종 결과가 빈 배열이면 무음 반환
            if full_audio.size == 0:
                silent_audio = np.zeros(int(0.5 * target_sr), dtype=np.float32)  # 0.5초 무음
                return silent_audio, target_sr
                
            return full_audio, target_sr
            
        except Exception:
            # 오류 발생 시 무음 반환
            silent_audio = np.zeros(int(0.5 * target_sr), dtype=np.float32)  # 0.5초 무음
            return silent_audio, target_sr
        
    def generate_audio_by_sentence(
        self,
        sentence: str,
        voice: str,
        speed: float,
        model_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """단일 문장에 대한 오디오 생성"""
        try:
            from melo.utils import get_text_for_tts_infer
            import torch
            
            # 문장 유효성 검사
            if not sentence or len(sentence.strip()) == 0:
                # 빈 문장은 짧은 무음으로 대체
                return np.zeros(int(0.1 * target_sr), dtype=np.float32)  # 0.1초 무음
            
            # 화자 유효성 검사
            if voice not in self.voices:
                voice = "KR"
            
            # 언어 코드 추출 및 모델 로드
            lang = voice.split("-")[0] if "-" in voice else voice
            if lang not in self.models:
                if not self.load_model(lang):
                    raise ValueError(f"Failed to load {lang} model")
            
            # 모델 및 화자 ID 가져오기
            tts = self.models[lang]
            spk = self.voices[voice]
            
            # MeloTTS 텍스트 처리
            bert, ja_bert, phones, tones, lang_ids = get_text_for_tts_infer(
                sentence, tts.language, tts.hps, tts.device, tts.symbol_to_id
            )
            speakers = torch.LongTensor([spk]).to(tts.device)
            
            # 오디오 생성
            with torch.no_grad():
                out = tts.model.infer(
                    phones.unsqueeze(0).to(tts.device),
                    torch.LongTensor([phones.size(0)]).to(tts.device),
                    speakers,
                    tones.unsqueeze(0).to(tts.device),
                    lang_ids.unsqueeze(0).to(tts.device),
                    bert.unsqueeze(0).to(tts.device),
                    ja_bert.unsqueeze(0).to(tts.device),
                    sdp_ratio=0.2, noise_scale=0.6,
                    noise_scale_w=0.8, length_scale=1.0/speed
                )
            
            # 리소스 해제
            del bert, ja_bert, phones, tones, lang_ids, speakers
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # 오디오 추출
            audio = out[0][0, 0].cpu().numpy()
            
            # 오디오 유효성 검사
            if audio.size == 0:
                return np.zeros(int(0.1 * target_sr), dtype=np.float32)  # 0.1초 무음
            
            # 리샘플링 (필요한 경우)
            if model_sr != target_sr:
                audio = self.safe_resample(audio, model_sr, target_sr)
            
            return audio
            
        except Exception:
            # 오류 시 무음 반환
            return np.zeros(int(0.1 * target_sr), dtype=np.float32)  # 0.1초 무음
    
    def safe_resample(self, audio: np.ndarray, orig_sr: int, tgt_sr: int) -> np.ndarray:
        """안전한 리샘플링 (빈 배열 처리)"""
        if audio.size == 0:
            return np.zeros(1, dtype=np.float32)  # 빈 배열이면 작은 무음 반환
            
        try:
            from scipy.signal import resample_poly
            g = np.gcd(orig_sr, tgt_sr)
            return resample_poly(audio, tgt_sr // g, orig_sr // g, padtype="line")
        except Exception:
            # 리샘플링 오류 시 원본 반환
            return audio
    
    def split_sentences(self, text: str, lang: str) -> List[str]:
        """텍스트를 문장으로 분리"""
        try:
            # 텍스트 유효성 검사
            if not text or len(text.strip()) == 0:
                return []
                
            # 언어 코드 추출
            lang_code = lang.split("-")[0] if "-" in lang else lang
            
            # 모델 로드
            if lang_code not in self.models:
                if not self.load_model(lang_code):
                    # 모델 로드 실패 시 기본 분리 규칙 사용
                    return self._fallback_split_sentences(text)
            
            # MeloTTS 문장 분리 사용
            tts = self.models[lang_code]
            sentences = tts.split_sentences_into_pieces(text, tts.language)
            
            # 빈 문장 제거
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # 결과 유효성 검사
            if not sentences:
                return self._fallback_split_sentences(text)
                
            return sentences
            
        except Exception:
            # 오류 발생 시 기본 분리 규칙 사용
            return self._fallback_split_sentences(text)
    
    def _fallback_split_sentences(self, text: str) -> List[str]:
        """기본 문장 분리 메서드 (MeloTTS 분리 실패 시 사용)"""
        try:
            import re
            
            # 문장 종결 기호 기준 분리
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            # 빈 문장 제거
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # 결과 확인
            if not sentences:
                return [text.strip()]  # 분리 실패 시 원본 텍스트 반환
                
            return sentences
            
        except Exception:
            # 최후의 수단: 원본 텍스트 그대로 반환
            return [text.strip()] if text and text.strip() else []
    
    @property
    def default_sample_rate(self) -> int:
        """모델의 기본 샘플레이트"""
        return self._model_sr
    
    @property
    def model_name(self) -> str:
        """모델 이름"""
        return "MeloTTS"

class KokoroTTSAdapter(TTSModelAdapter):
    """Kokoro TTS 모델 어댑터"""
    
    def __init__(self, model_path: str = "./KoKoro_models/kokoro-v1.0-q4f16.onnx", 
                voice_path: str = "./KoKoro_models/voices-v1.0.bin"):
        self.model_path = model_path
        self.voice_path = voice_path
        self.kokoro = None
        self.g2p = None
        self._model_sr = 24000  # Kokoro 기본 샘플레이트
        self._voices = ["af_sarah", "en_anna", "en_michael", "jf_alpha"]
    
    def initialize(self) -> bool:
        """모델 초기화"""
        try:
            # Kokoro 라이브러리 임포트
            from kokoro_onnx import Kokoro
            from misaki.espeak import EspeakG2P
            
            self.kokoro = Kokoro(self.model_path, self.voice_path)
            self.g2p = EspeakG2P(language="ko")
            
            return True
            
        except ImportError:
            return False
        except Exception:
            return False
            
    
    def warmup(self) -> bool:
        """모델 워밍업"""
        try:
            # 워밍업 텍스트
            warmup_text = "안녕하세요. 이것은 모델 워밍업을 위한 텍스트입니다."
            
            # G2P 변환
            phonemes, _ = self.g2p(warmup_text)
            
            # 워밍업 실행
            samples, sample_rate = self.kokoro.create(
                text=phonemes,
                voice="af_sarah",  # 기본 화자
                speed=1.0,
                lang="ko",
                is_phonemes=True
            )
            
            # 샘플레이트 확인
            self._model_sr = sample_rate
            
            return True
            
        except Exception:
            return False
    
    def list_voices(self) -> List[str]:
        """사용 가능한 음성 목록 반환"""
        return self._voices
    
    def generate_audio(
        self, 
        text: str, 
        voice: str = "af_sarah", 
        speed: float = 1.0, 
        target_sr: int = 24000,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """텍스트에서 오디오 생성 (전체 텍스트)"""
        try:
            text = self.clean_markdown_for_tts(text)
            # 기본 구현은 단일 문장으로 처리하지만, 실제로는 각 문장별로 처리 권장
            sentences = self.split_sentences(text, "ko")
            combined_audio = np.array([], dtype=np.float32)
            
            for sentence in sentences:
                audio = self.generate_audio_by_sentence(
                    sentence=sentence,
                    voice=voice,
                    speed=speed,
                    model_sr=self._model_sr,
                    target_sr=target_sr
                )
                combined_audio = np.concatenate([combined_audio, audio])
            
            return combined_audio, target_sr
            
        except Exception:
            # 오류 발생 시 빈 오디오 반환
            return np.array([], dtype=np.float32), target_sr
        

    def clean_markdown_for_tts(self, text: str) -> str:
        """
        TTS용 마크다운 특수문자 제거 및 정리
        마크다운 문법을 자연스러운 텍스트로 변환
        """
        if not text:
            return ""
        
        # 1. 코드 블록 제거 (```코드```)
        text = re.sub(r'```[\s\S]*?```', '', text)
        
        # 2. 인라인 코드 제거 (`코드`)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # 3. 링크 문법 처리 [텍스트](URL) -> 텍스트만 남김
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # 4. 이미지 링크 제거 ![alt](URL)
        text = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', text)
        
        # 5. 강조 문법 제거 **텍스트**, *텍스트*
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **굵게**
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *기울임*
        text = re.sub(r'__([^_]+)__', r'\1', text)      # __굵게__
        text = re.sub(r'_([^_]+)_', r'\1', text)        # _기울임_
        
        # 6. 헤더 제거 (# ## ### ...)
        text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
        
        # 7. 순서 있는 리스트 (1. 2. 3. ...)
        text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)
        
        # 8. 순서 없는 리스트 (- * +)
        text = re.sub(r'^\s*[-*+]\s*', '', text, flags=re.MULTILINE)
        
        # 9. 인용문 (> 텍스트)
        text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)
        
        # 10. 수평선 제거 (--- *** ___)
        text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
        
        # 11. 테이블 구분자 제거 (|)
        text = re.sub(r'\|', ' ', text)
        
        # 12. 남은 특수문자들 제거/변환
        # 대괄호, 중괄호 제거
        text = re.sub(r'[\[\]{}]', '', text)
        
        # 백슬래시 이스케이프 제거
        text = re.sub(r'\\(.)', r'\1', text)
        
        # 13. 여러 공백을 하나로 통합
        text = re.sub(r'\s+', ' ', text)
        
        # 14. 여러 줄바꿈을 하나로 통합
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # 15. 앞뒤 공백 제거
        text = text.strip()
        
        return text
    
    def generate_audio_by_sentence(
        self,
        sentence: str,
        voice: str,
        speed: float,
        model_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """단일 문장에 대한 오디오 생성"""
        if not self.kokoro or not self.g2p:
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        if not sentence or len(sentence.strip()) == 0:
            return np.array([], dtype=np.float32)
            
        if voice not in self._voices:
            voice = "af_sarah"  # 기본 화자
        
        try:
            # G2P 변환
            phonemes, _ = self.g2p(sentence)
            
            # 오디오 생성
            samples, sample_rate = self.kokoro.create(
                text=phonemes,
                voice=voice,
                speed=float(speed),
                lang="ko",
                is_phonemes=True
            )
            
            # 샘플레이트 확인 및 업데이트
            if sample_rate != self._model_sr:
                self._model_sr = sample_rate
            
            # 리샘플링 (필요한 경우)
            if sample_rate != target_sr:
                samples = AudioConverter.resample(samples, sample_rate, target_sr)
            
            return samples
            
        except Exception:
            raise
    
    def split_sentences(self, text: str, lang: str) -> List[str]:
        """텍스트를 문장으로 분리"""
        if not text:
            return []
            
        try:
            import re
            
            # 마침표, 물음표, 느낌표 뒤에 공백이나 줄바꿈이 있으면 분리
            pattern = r'(?<=[.!?])\s+'
            
            # 문장 분리
            sentences = re.split(pattern, text)
            
            # 빈 문장 제거 및 공백 정리
            result = [s.strip() for s in sentences if s.strip()]
            
            # 분리된 문장이 없으면 원본 텍스트 반환
            if not result:
                result = [text]
                
            return result
            
        except Exception:
            # 에러 발생 시 원본 텍스트를 단일 문장으로 반환
            return [text]
    
    @property
    def default_sample_rate(self) -> int:
        """모델의 기본 샘플레이트"""
        return self._model_sr
    
    @property
    def model_name(self) -> str:
        """모델 이름"""
        return "KokoroTTS"