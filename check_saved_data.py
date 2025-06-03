# check_milvus_collections.py

from pymilvus import connections, utility, Collection

# Milvus 연결 정보 (필요에 따라 수정)
HOST = "localhost"    # 도커 환경에서 host.docker.internal:19530 등을 쓸 수도 있습니다.
PORT = "19530"

# 점검할 컬렉션 이름들
COLLECTIONS_TO_CHECK = [
    "user_question_collection",  # 질문 저장용
    "rag_docs_collection"        # RAG 문서 청크 저장용
]


def connect_milvus():
    try:
        connections.connect(alias="default", host=HOST, port=PORT, timeout=5)
        print(f"✅ Milvus 연결 성공 ({HOST}:{PORT})\n")
    except Exception as e:
        print("❌ Milvus 연결 실패:", e)
        exit(1)


def inspect_collection(name: str):
    print(f"==============================")
    print(f"🔍 컬렉션 확인: '{name}'")
    print(f"==============================")

    # 1) 컬렉션 존재 여부 확인
    if not utility.has_collection(name):
        print(f"❌ 컬렉션 '{name}' 이(가) 존재하지 않습니다.\n")
        return

    # 2) 컬렉션 로드
    try:
        coll = Collection(name)
        coll.load()
        print(f"✅ 컬렉션 '{name}' 로드 완료")
    except Exception as e:
        print(f"❌ 컬렉션 로드 실패: {e}\n")
        return

    # 3) 총 엔티티 수 조회
    try:
        total = coll.num_entities
        print(f"📊 총 레코드 수: {total}")
    except Exception as e:
        print(f"❌ num_entities 조회 실패: {e}")

    # 4) 스키마 필드 목록 출력
    try:
        fields = [field.name for field in coll.schema.fields]
        print(f"\n🔍 스키마 필드 목록:")
        for idx, f in enumerate(fields, start=1):
            print(f"   {idx}. {f}")
    except Exception as e:
        print(f"❌ 스키마 필드 조회 실패: {e}")

    # 5) 샘플 레코드 출력 (최대 3개)
    #    ▶ user_question_collection: ['id', 'user_id', 'question', 'embedding']
    #    ▶ rag_docs_collection: ['id', 'filename', 'chunk_id', 'chunk_text', 'embedding']
    output_fields = fields
    # Milvus expr은 None을 허용하지 않으므로, “항상 참”인 조건을 넣어야 합니다.
    expr = "id >= 0"

    try:
        print(f"\n📋 샘플 레코드 ({min(total, 3)}개):")
        results = coll.query(expr=expr, output_fields=output_fields, limit=3)
        if not results:
            print("  (레코드가 없습니다.)\n")
        else:
            for i, rec in enumerate(results, start=1):
                print(f"  ─── 레코드 {i} ───")
                for f in fields:
                    val = rec.get(f)
                    # embedding 필드는 길어서 길이 정보만 출력
                    if f == "embedding" and isinstance(val, list):
                        print(f"    • {f}: [ ... {len(val)}차원 벡터 ... ]")
                    else:
                        print(f"    • {f}: {val}")
                print()
    except Exception as e:
        print(f"❌ 샘플 레코드 조회 실패: {e}")

    print("\n")


if __name__ == "__main__":
    connect_milvus()

    for cname in COLLECTIONS_TO_CHECK:
        inspect_collection(cname)
