-- 테이블 생성 (자동 실행되더라도 안전하게 IF NOT EXISTS 붙임)
CREATE TABLE IF NOT EXISTS users (
  username VARCHAR PRIMARY KEY,
  password VARCHAR NOT NULL,
  role     VARCHAR NOT NULL,
  "group"  VARCHAR NOT NULL
);

-- 초기 데이터 삽입
INSERT INTO users (username, password, role, "group")
VALUES
  ('manager1','manager1','manager','A'),
  ('user1','user1','user','A'),
  ('manager2','manager2','manager','B'),
  ('user2','user2','user','B')
ON CONFLICT (username) DO NOTHING;
