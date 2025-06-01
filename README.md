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
cd mock-llm-server
docker build --no-cache -t gcr.io/$PROJECTID/mock-llm-server:latest .
docker push gcr.io/$PROJECTID/mock-llm-server:latest

cd rag-app
docker build --no-cache -t gcr.io/$PROJECTID/rag-app:latest .
docker push gcr.io/$PROJECTID/rag-app:latest

kubectl apply -f k8s/mock-llm-deployment.yaml
kubectl apply -f k8s/mock-llm-service.yaml

kubectl apply -f k8s/rag-app-deployment.yaml
kubectl apply -f k8s/rag-app-service.yaml
```
