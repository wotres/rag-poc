# 로컬 실행
* $ python mock_llm_server.py
* $ python app.py

# 기능
* ALL Python 구성
* pdf text 업로드
* pdf text 업로드 결과 확인
* 문서 조회
* 문서 삭제
* 삭제 결과
* PDF 다운로드
* 일반 질문하기 
* RAG 참조 하여 질문하기


# 구현 할것
* 문서 저장 
* gke 배포
* 카프카 레빗앤큐 
* s3 연결
* 자주묻는 질문 redis 로 호출


# 로컬 배포
```bash
cd mock-llm-server
docker build --no-cache -t mock-llm-server:local .

cd rag-app
docker build --no-cache -t rag-app:local .

# -d 선택
docker run --rm -p 8001:8001 mock-llm-server:local
docker run --rm -p 7860:7860 rag-app:local
```

# k8s 배포
```bash
# project ID 입력

# artifact Repository 생성 (같은 리전으로)
gcloud artifacts repositories create rag-poc \
  --repository-format=docker \
  --location=us-central1 \
  --description="RAG POC Docker images"


# build 및 push
cd mock-llm-server
docker build --no-cache -t mock-llm-server:latest .
docker tag mock-llm-server:latest us-central1-docker.pkg.dev/$PROJECTID/rag-poc/mock-llm-server:latest
docker push us-central1-docker.pkg.dev/$PROJECTID/rag-poc/mock-llm-server:latest

cd rag-app
docker build --no-cache -t rag-app:latest .
docker tag rag-app:latest us-central1-docker.pkg.dev/$PROJECTID/rag-poc/rag-app:latest
docker push us-central1-docker.pkg.dev/$PROJECTID/rag-poc/rag-app:latest

# image



kubectl apply -f k8s/mock-llm-deployment.yaml
kubectl apply -f k8s/mock-llm-service.yaml

kubectl apply -f k8s/rag-app-deployment.yaml
kubectl apply -f k8s/rag-app-service.yaml
```


# hpa
```bash
# HPA가 CPU/메모리 사용량을 모니터링하려면 클러스터 내에 Metrics Server가 배포되야함
kubectl get deployment metrics-server -n kube-system

# Metrics Server가 없다면 설치
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

kubectl apply -f rag-app-hpa.yaml

kubectl get hpa

# Pod 수를 모니터링
watch kubectl get pods -l app=rag-app
```

# milvus cluster 설치
* https://milvus.io/docs/ko/install_cluster-milvusoperator.md
* https://milvus.io/docs/ko/gcp.md#Deploy-a-Milvus-Cluster-on-GKE



