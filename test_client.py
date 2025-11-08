import zmq
import json
import time
import threading
import numpy as np
import argparse
import wave
from typing import Dict, Any, Optional
import queue

class TTSClient:
    """TTS 서버 클라이언트 with 레이턴시 측정"""
    
    def __init__(self, cmd_addr: str = "tcp://localhost:5555", 
                 audio_addr: str = "tcp://localhost:5556"):
        # ZMQ 소켓 설정
        self.ctx = zmq.Context()
        
        # 명령 소켓 (REQ)
        self.cmd_sock = self.ctx.socket(zmq.REQ)
        self.cmd_sock.connect(cmd_addr)
        
        # 오디오 소켓 (PULL)
        self.audio_sock = self.ctx.socket(zmq.PULL)
        self.audio_sock.connect(audio_addr)
        
        # 레이턴시 측정 변수
        self.request_time = None
        self.first_audio_time = None
        self.latency = None
        
        # 오디오 버퍼
        self.audio_queue = queue.Queue()
        self.meta_info = None
        
    def list_voices(self) -> list:
        """사용 가능한 음성 목록 조회"""
        self.cmd_sock.send_json({
            "command": "list_voices"
        })
        
        response = self.cmd_sock.recv_json()
        return response.get("voices", [])
    
    def generate(self, text: str, job_id: str, **kwargs) -> Dict[str, Any]:
        """TTS 생성 요청"""
        # 레이턴시 측정 시작
        self.request_time = time.time()
        self.first_audio_time = None
        self.latency = None
        
        # 생성 요청
        request = {
            "command": "generate",
            "job_id": job_id,
            "text": text,
            **kwargs
        }
        
        self.cmd_sock.send_json(request)
        response = self.cmd_sock.recv_json()
        
        # 응답 상태 확인
        if response.get("status") == "started":
            # 오디오 수신 스레드 시작
            self.receiver_thread = threading.Thread(
                target=self._receive_audio, 
                args=(job_id,),
                daemon=True
            )
            self.receiver_thread.start()
        
        return response
    
    def _receive_audio(self, job_id: str):
        """오디오 데이터 수신 스레드"""
        try:
            while True:
                msg = self.audio_sock.recv_multipart()
                
                if len(msg) < 3:
                    continue
                
                recv_jid = msg[0].decode()
                msg_type = msg[1]
                data = msg[2]
                
                # 올바른 job_id 확인
                if recv_jid != job_id:
                    continue
                
                if msg_type == b"meta":
                    # 메타데이터 수신
                    self.meta_info = json.loads(data.decode())
                    print(f"메타데이터 수신: {self.meta_info}")
                    
                elif msg_type == b"data":
                    # 첫 오디오 데이터 수신 시간 기록
                    if self.first_audio_time is None:
                        self.first_audio_time = time.time()
                        self.latency = self.first_audio_time - self.request_time
                        print(f"첫 오디오 수신! 레이턴시: {self.latency*1000:.2f} ms")
                    
                    # 오디오 데이터 저장
                    self.audio_queue.put(data)
                    
                elif msg_type == b"end":
                    # 완료 메시지
                    end_time = time.time()
                    total_time = end_time - self.request_time
                    print(f"완료: {data.decode()}")
                    print(f"총 소요 시간: {total_time*1000:.2f} ms")
                    break
                    
                elif msg_type == b"error":
                    # 오류 처리
                    print(f"오류 발생: {data.decode()}")
                    break
                    
        except Exception as e:
            print(f"수신 오류: {e}")
    
    def interrupt(self, job_id: Optional[str] = None):
        """작업 중단"""
        self.cmd_sock.send_json({
            "command": "interrupt",
            "job_id": job_id
        })
        
        response = self.cmd_sock.recv_json()
        return response
    
    def wait_for_audio(self, timeout: float = 30.0) -> bytes:
        """모든 오디오 데이터 대기 및 수집"""
        audio_chunks = []
        start_time = time.time()
        
        while True:
            try:
                # 타임아웃 체크
                if time.time() - start_time > timeout:
                    break
                
                # 오디오 데이터 가져오기
                data = self.audio_queue.get(timeout=1.0)
                audio_chunks.append(data)
                
            except queue.Empty:
                # 수신 스레드가 종료되었는지 확인
                if not self.receiver_thread.is_alive():
                    break
        
        return b''.join(audio_chunks)
    
    def save_audio(self, audio_data: bytes, filename: str):
        """오디오 데이터를 WAV 파일로 저장"""
        if not self.meta_info:
            print("메타데이터가 없습니다.")
            return
        
        # WAV 파일 저장
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(self.meta_info['channels'])
            wav_file.setsampwidth(2)  # int16 = 2 bytes
            wav_file.setframerate(self.meta_info['sample_rate'])
            wav_file.writeframes(audio_data)
        
        print(f"오디오 저장됨: {filename}")
    
    def close(self):
        """리소스 정리"""
        self.cmd_sock.close()
        self.audio_sock.close()
        self.ctx.term()


def main():
    # 클라이언트 생성
    client = TTSClient()
    
    try:
        # 사용 가능한 음성 목록 조회
        voices = client.list_voices()
        print(f"사용 가능한 음성: {voices}")
        
        # 일반적인 대화 문장으로 TTS 테스트
        test_texts = [
            # 한국어 짧은 문장
            "네, 알겠습니다.",
            "잠시만요.",
            
            # 영어 짧은 문장
            "Hello there.",
            "Thank you.",
            
            # 한국어 중간 문장
            "오늘 날씨가 정말 좋네요.",
            "점심 메뉴는 뭐가 좋을까요?",
            "지금 무엇을 도와드릴까요?",
            
            # 영어 중간 문장
            "How can I help you today?",
            "What would you like for lunch?",
            "The weather is really nice today.",
            
            # 한국어 긴 문장
            "어제 말씀하신 프로젝트 건에 대해서 좀 더 자세히 설명해 주실 수 있으신가요?",
            "내일 오전 10시 회의가 있는데 참석 가능하신지 확인 부탁드립니다.",
            "지난번에 추천해주신 레스토랑에 가봤는데 정말 맛있었어요. 다시 한번 감사드려요.",
            
            # 영어 긴 문장
            "Could you please provide more details about the project you mentioned yesterday?",
            "I have a meeting tomorrow at 10 AM, could you please confirm if you can attend?",
            "I went to the restaurant you recommended last time and it was absolutely delicious. Thank you again.",
            
            # 혼합된 대화 문장
            "스마트폰 배터리가 20%밖에 안 남았어요. 충전기 있으신가요?",
            "Can you send me the report by end of day? 오늘까지 필요해요.",
            "회의 시간이 3시에서 4시로 변경되었습니다. Is that okay with you?",
        ]
        
        # 문장 길이별 통계를 위한 카테고리
        short_latencies = []  # 1-4 단어
        medium_latencies = []  # 5-10 단어
        long_latencies = []   # 11+ 단어
        
        korean_latencies = []
        english_latencies = []
        mixed_latencies = []
        
        for i, text in enumerate(test_texts):
            print(f"\n테스트 {i+1}: {text}")
            
            # 단어 수 계산
            word_count = len(text.split())
            print(f"단어 수: {word_count}")
            
            # 언어 감지 및 음성 선택
            has_korean = any('\uac00' <= char <= '\ud7a3' for char in text)
            has_english = any('a' <= char.lower() <= 'z' for char in text)
            
            if has_korean and has_english:
                voice = "KR"  # 혼합 문장은 한국어 음성 사용
                lang_type = "mixed"
            elif has_korean:
                voice = "KR"
                lang_type = "korean"
            else:
                voice = "EN"
                lang_type = "english"
            
            # TTS 생성 요청
            job_id = f"test_{i}_{int(time.time()*1000)}"
            response = client.generate(
                text=text,
                job_id=job_id,
                voice=voice,
                speed=1.0,
                target_sample_rate=16000,
                chunk_size=1024
            )
            
            if response.get("status") == "started":
                # 오디오 수신 대기
                audio_data = client.wait_for_audio()
                
                # 오디오 저장
                client.save_audio(audio_data, f"output_{i}.wav")
                
                # 레이턴시 수집
                if client.latency:
                    latency_ms = client.latency * 1000
                    print(f"레이턴시: {latency_ms:.2f} ms")
                    
                    # 길이별 분류
                    if word_count <= 4:
                        short_latencies.append(latency_ms)
                    elif word_count <= 10:
                        medium_latencies.append(latency_ms)
                    else:
                        long_latencies.append(latency_ms)
                    
                    # 언어별 분류
                    if lang_type == "korean":
                        korean_latencies.append(latency_ms)
                    elif lang_type == "english":
                        english_latencies.append(latency_ms)
                    else:
                        mixed_latencies.append(latency_ms)
            
            time.sleep(0.5)  # 다음 요청 전 대기
        
        # 상세 레이턴시 통계
        print("\n=== 레이턴시 통계 ===")
        
        # 전체 통계
        all_latencies = short_latencies + medium_latencies + long_latencies
        if all_latencies:
            print(f"\n전체 평균: {sum(all_latencies)/len(all_latencies):.2f} ms")
            print(f"전체 최소: {min(all_latencies):.2f} ms")
            print(f"전체 최대: {max(all_latencies):.2f} ms")
        
        # 길이별 통계
        print("\n길이별 통계:")
        if short_latencies:
            print(f"짧은 문장 (1-4 단어): 평균 {sum(short_latencies)/len(short_latencies):.2f} ms")
        if medium_latencies:
            print(f"중간 문장 (5-10 단어): 평균 {sum(medium_latencies)/len(medium_latencies):.2f} ms")
        if long_latencies:
            print(f"긴 문장 (11+ 단어): 평균 {sum(long_latencies)/len(long_latencies):.2f} ms")
        
        # 언어별 통계
        print("\n언어별 통계:")
        if korean_latencies:
            print(f"한국어: 평균 {sum(korean_latencies)/len(korean_latencies):.2f} ms")
        if english_latencies:
            print(f"영어: 평균 {sum(english_latencies)/len(english_latencies):.2f} ms")
        if mixed_latencies:
            print(f"혼합: 평균 {sum(mixed_latencies)/len(mixed_latencies):.2f} ms")
        
    except KeyboardInterrupt:
        print("\n중단됨")
    except Exception as e:
        print(f"오류: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS Client with Latency Measurement")
    parser.add_argument("--cmd-addr", default="tcp://localhost:5555",
                      help="Command socket address")
    parser.add_argument("--audio-addr", default="tcp://localhost:5556", 
                      help="Audio socket address")
    parser.add_argument("--text", type=str, help="Text to generate")
    parser.add_argument("--voice", type=str, default="KR", help="Voice to use")
    
    args = parser.parse_args()
    
    if args.text:
        # 단일 테스트
        client = TTSClient(args.cmd_addr, args.audio_addr)
        
        job_id = f"test_{int(time.time()*1000)}"
        response = client.generate(
            text=args.text,
            job_id=job_id,
            voice=args.voice
        )
        
        if response.get("status") == "started":
            audio_data = client.wait_for_audio()
            client.save_audio(audio_data, "output.wav")
            print(f"레이턴시: {client.latency*1000:.2f} ms")
        
        client.close()
    else:
        # 여러 테스트 실행
        main()