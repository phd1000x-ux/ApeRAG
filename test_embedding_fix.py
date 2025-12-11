#!/usr/bin/env python3
"""
테스트 스크립트: 임베딩 컨텍스트 창 제한 문제 해결 확인
"""

import asyncio
import sys
import os
import json

# 프로젝트 경로를 Python 경로에 추가
sys.path.insert(0, os.path.abspath('.'))

from aperag.llm.embed.embedding_service import EmbeddingService
from aperag.llm.llm_error_types import EmbeddingError
from aperag.db.models import Collection, User
from aperag.schema.utils import parseCollectionConfig, dumpCollectionConfig

async def test_embedding_with_long_text():
    """긴 텍스트에 대한 임베딩 처리 테스트"""
    
    # 테스트용 컬렉션 및 사용자 생성
    user = User(id=1, username="test_user")
    
    # 테스트용 컬렉션 설정
    config = {
        "embedding": {
            "model_service_provider": "cohere",
            "model": "embed-multilingual-v3.0",
            "api_key": "test-key",
            "base_url": "https://api.cohere.ai/v1"
        },
        "completion": {
            "model_service_provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1"
        }
    }
    
    # 컬렉션 설정 생성
    config_json = {
        "enable_vector": True,
        "enable_fulltext": True,
        "enable_knowledge_graph": True,
        "enable_summary": False,
        "enable_vision": False,
        "embedding": {
            "model_service_provider": "cohere",
            "model": "embed-multilingual-v3.0",
            "api_key": "test-key",
            "base_url": "https://api.cohere.ai/v1"
        },
        "completion": {
            "model_service_provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1"
        },
        "knowledge_graph_config": {
            "language": "English",
            "entity_types": [
                "organization",
                "person",
                "geo",
                "event",
                "product",
                "technology",
                "date",
                "category"
            ]
        }
    }
    
    collection_config = parseCollectionConfig(json.dumps(config_json))
    
    collection = Collection(
        id=1,
        title="test_collection",
        user=user,
        config=dumpCollectionConfig(collection_config)
    )
    
    try:
        # 임베딩 서비스 생성
        from aperag.llm.embed.base_embedding import get_collection_embedding_service_sync
        embedding_svc, dim = get_collection_embedding_service_sync(collection)
        
        # 2048 토큰을 초과하는 긴 텍스트 (약 2167 토큰)
        long_text = """
        이것은 매우 긴 텍스트입니다. 이 텍스트는 임베딩 모델의 최대 토큰 길이(2048)를 초과하도록 설계되었습니다.
        이 텍스트는 다양한 주제를 포함하고 있으며, 기술적인 내용과 비기술적인 내용이 모두 포함되어 있습니다.
        예를 들어, 이 텍스트는 인공지능, 머신러닝, 딥러닝, 자연어 처리, 컴퓨터 비전, 데이터베이스,
        웹 개발, 클라우드 컴퓨팅, 마이크로서비스, 컨테이너 기술, 데브옵스, CI/CD 파이프라인,
        모니터링, 로깅, 보안, 네트워킹, 스토리지 시스템, 분산 시스템, 병렬 컴퓨팅,
        성능 최적화, 알고리즘, 데이터 구조, 소프트웨어 아키텍처, 디자인 패턴,
        테스트 주도 개발, 애자일 방법론, 스크럼 개발, 프로젝트 관리, 버전 관리,
        코드 리뷰, 리팩토링, 디버깅, 문서화, 협업 도구, Git, GitHub,
        도커, 쿠버네티스, 헬름, 프로메테우스, 그라파나, 엘라스틱서치,
        키바나, 로그스태시, 플루언트, 재닉스, 시스템 디자인, 마이크로서비스 아키텍처,
        서버리스 컴퓨팅, API 디자인, REST API, GraphQL, WebSocket, gRPC,
        데이터 직렬화, JSON, XML, YAML, Protocol Buffers, Avro, MessagePack,
        데이터베이스 설계, 정규화, 인덱싱, 트랜잭션, 동시성 제어,
        분산 데이터베이스, 샤딩, 복제, 파티셔닝, 일관성, 가용성,
        내결함성, 분산 트랜잭션, 2단계 커밋, 3단계 커밋, 낙관적 동시성 제어,
        분산 락, 리더 선정, 쿼럼, Raft, Paxos, 분산 합의,
        블록체인, 스마트 계약, 탈중앙화, 증명 작업, 해시 함수,
        디지털 서명, 공개키 암호화, 개인키 암호화, 대칭키 암호화,
        양자 컴퓨팅, 양자 암호화, 양자 통신, 양자 내성성,
        양자 알고리즘, Shor 알고리즘, Grover 알고리즘, 양자 오류 정정,
        양자 키 분배, 양자 암호 프로토콜, 양자 안전 통신,
        머신러닝 알고리즘, 선형 회귀, 로지스틱 회귀, 결정 트리,
        랜덤 포레스트, 서포트 벡터 머신, k-최근접 이웃, k-평균,
        클러스터링, 계층적 클러스터링, 밀도 기반 클러스터링,
        그래프 기반 클러스터링, 차원 축소, 주성분 분석,
        독립 성분 분석, 요인 분석, 특이값 분해,
        비음수 행렬 분해, 잠재 디리클레 할당, 가우시안 혼합 모델,
        음수 미만 분포, 다항 분포, 베타 분포, 감마 분포,
        디리클레 분포, 와이블 분포, 학습 가능한 파라미터 추정,
        최대 가능도 추정, 베이즈 추정, 마르코프 연쇄 몬테카를로 시뮬레이션,
        깁스-샘플링, 깁스 필터링, 입자 필터링,
        순차 몬테카를로 방법, 메트로폴리스-해스팅, 해밀턴ian 몬테카를로 방법,
        경사 하강법, 확률적 경사 하강법, 모멘텀 기반 최적화,
        Adam 옵티마이저, RMSprop 옵티마이저, AdaGrad 옵티마이저,
        학습률 스케줄링, 학습률 감소, 가중치 감쇄,
        정규화, 드롭아웃, L1 정규화, L2 정규화,
        배치 정규화, 레이어 정규화, 활성화 함수,
        ReLU, Sigmoid, Tanh, Softmax, Leaky ReLU, ELU, SELU,
        합성곱 신경망, 순환 신경망, LSTM, GRU,
        어텐션 메커니즘, 트랜스포머, 셀프 어텐션, 멀티헤드 어텐션,
        위치 인코딩, 상대적 위치 인코딩, 회전 위치 인코딩,
        생성적 적대 신경망, 생성자, 판별자, 변분 자동 인코더,
        변이 오토인코더, 잠재 공간, 잠재 변수,
        VAE, GAN, Diffusion 모델, 안정적 확산,
        클래스 조건부 생성, 노이즈 스케줄링, 분류기 없는 생성,
        전이 모델, 순환 전이 모델, 플로우 기반 모델,
        에너지 기반 모델, 스코어링, 샘플링,
        온도 스케줄링, 탐욱적 샘플링, 핵 샘플링,
        분산 샘플링, 라플라스 샘플링, 해밀턴ian 샘플링,
        메트로폴리스-해스팅, 점프 MCM, 해밀턴ian 몬테카를로,
        변분 추론, 재매개변수화, 증분 자동 인코더,
        정규화 흐름, 정규화 계층, 스타일 전이,
        스타일GAN, CycleGAN, Pix2Pix, StyleGAN,
        대립적 생성 네트워크, WGAN, WGAN-GP, StyleGAN2,
        신경망 아키텍처, ResNet, DenseNet, MobileNet, EfficientNet,
        Vision Transformer, Swin Transformer, ConvNeXt,
        자연어 처리 모델, BERT, GPT, T5, Transformer,
        어텐션 메커니즘, 셀프 어텐션, 멀티헤드 어텐션,
        위치 인코딩, 세그먼트 임베딩, 문장 임베딩,
        토크나이저, WordPiece, BPE, 유니코드 처리,
        언어 모델, 다국어 모델, 번역, 다국어 이해,
        제로샷 학습, 프롬프트 학습, 지도 학습,
        자기 지도 학습, 준지도 학습, 강화 학습,
        정책 그래디언트, 가치 함수, 행동-가치 함수,
        Advantage Actor-Critic, Proximal Policy Optimization, Trust Region Policy Optimization,
        다중 에이전트 시스템, 협동 학습, 중앙화된 학습,
        분산 강화 학습, 연결 강화 학습, 계층적 강화 학습,
        메타 학습, 메타 강화 학습, 학습 학습,
        전이 학습, 도메인 적응, 다작업 학습,
        연속 학습, 평생 학습, 재생 신경망,
        드림 기반 모델, 세계 모델, 이미지 생성,
        텍스트 생성, 대화 시스템, 챗봇,
        지식 증강 생성, 검색 증강 생성, 검색 엔진,
        벡터 데이터베이스, FAISS, Annoys, HNSW,
        근사 최근접 이웃 검색, LSH, 로컬티 해시,
        제품 추천 시스템, 협업 필터링, 행렬 분해,
        특이값 분해, 비음수 행렬 분해, 트리플 분해,
        CP 분해, Tucker 분해, PARAFAC 분해,
        텐서 분해, 텐서 트레인, 텐서플로우,
        지식 그래프, 엔티티 추출, 관계 추출,
        그래프 임베딩, 그래프 신경망, 그래프 컨볼루션 네트워크,
        그래프 트랜스포머, 그래프 어텐션, 그래프 표현 학습,
        이상 탐지, 이상 감지, 통계적 방법, 머신러닝 방법,
        고립 포레스트, 원-클래스 SVM, 로컬 이상팩터,
        자동 인코더, 변분 자동 인코더, 생성적 적대 신경망,
        시계열 분석, ARIMA 모델, 시계열 예측,
        추천 시스템, 협업 필터링, 행렬 분해,
        특이값 분해, 비음수 행렬 분해, 트리플 분해,
        A/B 테스팅, 다중 팔 문제, 밴딧 문제,
        탐욱적 최적화, 다중 팔 문제, 밴딧 문제,
        시뮬레이티드 어닐링, 몬테카를로 시뮬레이션,
        강화 학습, 정책 그래디언트, 가치 함수,
        행동-가치 함수, Advantage Actor-Critic,
        Proximal Policy Optimization, Trust Region Policy Optimization,
        다중 에이전트 시스템, 협동 학습, 중앙화된 학습,
        분산 강화 학습, 연결 강화 학습, 계층적 강화 학습,
        메타 학습, 메타 강화 학습, 학습 학습,
        전이 학습, 도메인 적응, 다작업 학습,
        연속 학습, 평생 학습, 재생 신경망,
        드림 기반 모델, 세계 모델, 이미지 생성,
        텍스트 생성, 대화 시스템, 챗봇,
        지식 증강 생성, 검색 증강 생성, 검색 엔진,
        벡터 데이터베이스, FAISS, Annoys, HNSW,
        근사 최근접 이웃 검색, LSH, 로컬티 해시,
        제품 추천 시스템, 협업 필터링, 행렬 분해,
        특이값 분해, 비음수 행렬 분해, 트리플 분해,
        CP 분해, Tucker 분해, PARAFAC 분해,
        텐서 분해, 텐서 트레인, 텐서플로우,
        지식 그래프, 엔티티 추출, 관계 추출,
        그래프 임베딩, 그래프 신경망, 그래프 컨볼루션 네트워크,
        그래프 트랜스포머, 그래프 어텐션, 그래프 표현 학습,
        이상 탐지, 이상 감지, 통계적 방법, 머신러닝 방법,
        고립 포레스트, 원-클래스 SVM, 로컬 이상팩터,
        자동 인코더, 변분 자동 인코더, 생성적 적대 신경망,
        시계열 분석, ARIMA 모델, 시계열 예측,
        추천 시스템, 협업 필터링, 행렬 분해,
        특이값 분해, 비음수 행렬 분해, 트리플 분해,
        A/B 테스팅, 다중 팔 문제, 밴딧 문제,
        탐욱적 최적화, 다중 팔 문제, 밴딧 문제,
        시뮬레이티드 어닐링, 몬테카를로 시뮬레이션,
        강화 학습, 정책 그래디언트, 가치 함수,
        행동-가치 함수, Advantage Actor-Critic,
        Proximal Policy Optimization, Trust Region Policy Optimization,
        다중 에이전트 시스템, 협동 학습, 중앙화된 학습,
        분산 강화 학습, 연결 강화 학습, 계층적 강화 학습,
        메타 학습, 메타 강화 학습, 학습 학습,
        전이 학습, 도메인 적응, 다작업 학습,
        연속 학습, 평생 학습, 재생 신경망,
        드림 기반 모델, 세계 모델, 이미지 생성,
        텍스트 생성, 대화 시스템, 챗봇,
        지식 증강 생성, 검색 증강 생성, 검색 엔진,
        벡터 데이터베이스, FAISS, Annoys, HNSW,
        근사 최근접 이웃 검색, LSH, 로컬티 해시,
        제품 추천 시스템, 협업 필터링, 행렬 분해,
        특이값 분해, 비음수 행렬 분해, 트리플 분해,
        CP 분해, Tucker 분해, PARAFAC 분해,
        텐서 분해, 텐서 트레인, 텐서플로우,
        지식 그래프, 엔티티 추출, 관계 추출,
        그래프 임베딩, 그래프 신경망, 그래프 컨볼루션 네트워크,
        그래프 트랜스포머, 그래프 어텐션, 그래프 표현 학습,
        이상 탐지, 이상 감지, 통계적 방법, 머신러닝 방법,
        고립 포레스트, 원-클래스 SVM, 로컬 이상팩터,
        자동 인코더, 변분 자동 인코더, 생성적 적대 신경망,
        시계열 분석, ARIMA 모델, 시계열 예측,
        추천 시스템, 협업 필터링, 행렬 분해,
        특이값 분해, 비음수 행렬 분해, 트리플 분해,
        A/B 테스팅, 다중 팔 문제, 밴딧 문제,
        탐욱적 최적화, 다중 팔 문제, 밴딧 문제,
        시뮬레이티드 어닐링, 몬테카를로 시뮬레이션,
        강화 학습, 정책 그래디언트, 가치 함수,
        행동-가치 함수, Advantage Actor-Critic,
        Proximal Policy Optimization, Trust Region Policy Optimization,
        다중 에이전트 시스템, 협동 학습, 중앙화된 학습,
        분산 강화 학습, 연결 강화 학습, 계층적 강화 학습,
        메타 학습, 메타 강화 학습, 학습 학습,
        전이 학습, 도메인 적응, 다작업 학습,
        연속 학습, 평생 학습, 재생 신경망,
        드림 기반 모델, 세계 모델, 이미지 생성,
        텍스트 생성, 대화 시스템, 챗봇,
        지식 증강 생성, 검색 증강 생성, 검색 엔진,
        벡터 데이터베이스, FAISS, Annoys, HNSW,
        근사 최근접 이웃 검색, LSH, 로컬티 해시,
        제품 추천 시스템, 협업 필터링, 행렬 분해,
        특이값 분해, 비음수 행렬 분해, 트리플 분해,
        CP 분해, Tucker 분해, PARAFAC 분해,
        텐서 분해, 텐서 트레인, 텐서플로우,
        지식 그래프, 엔티티 추출, 관계 추출,
        그래프 임베딩, 그래프 신경망, 그래프 컨볼루션 네트워크,
        그래프 트랜스포머, 그래프 어텐션, 그래프 표현 학습,
        이상 탐지, 이상 감지, 통계적 방법, 머신러닝 방법,
        고립 포레스트, 원-클래스 SVM, 로컬 이상팩터,
        자동 인코더, 변분 자동 인코더, 생성적 적대 신경망,
        시계열 분석, ARIMA 모델, 시계열 예측,
        추천 시스템, 협업 필터링, 행렬 분해,
        특이값 분해, 비음수 행렬 분해, 트리플 분해,
        A/B 테스팅, 다중 팔 문제, 밴딧 문제,
        탐욱적 최적화, 다중 팔 문제, 밴딧 문제,
        시뮬레이티드 어닐링, 몬테카를로 시뮬레이션.
        """
        
        print("테스트 시작: 긴 텍스트 임베딩 시도...")
        print(f"텍스트 길이: {len(long_text)} 문자")
        
        # 임베딩 시도
        embeddings = await embedding_svc.aembed_documents([long_text])
        
        print(f"성공! 임베딩 shape: {embeddings.shape}")
        print("테스트 성공: 임베딩 컨텍스트 창 제한 문제가 해결되었습니다.")
        return True
        
    except EmbeddingError as e:
        print(f"임베딩 오류 발생: {str(e)}")
        return False
    except Exception as e:
        print(f"예상치 못한 오류 발생: {str(e)}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_embedding_with_long_text())
    if result:
        print("\n✅ 테스트 통과: 임베딩 컨텍스트 창 제한 문제가 성공적으로 해결되었습니다.")
        sys.exit(0)
    else:
        print("\n❌ 테스트 실패: 임베딩 컨텍스트 창 제한 문제가 여전히 존재합니다.")
        sys.exit(1)