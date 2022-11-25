### 해당 프로젝트는 FASTAPI를 이용한 영화 추천서비스 웹을 개발하는 프로젝트입니다

- 먼저 Content Based Filtering을 적용한 뒤 웹으로 배포할 예정임.
- 추후 적용한 기법을 추가 및 수정할 예정
  - counter vectorizer -> tf-idf </br>
      cosine similariy -> pearson correlation coefficient)
- 10.28일에 프로젝트를 시작하였으나, Github에 수정 및 오류 사항 발생 후 대처가 미흡하여 repository 삭제 후 재시작하였음.

### Git 적용 순서

- 추후 협업 시 어색해 하지 않도록 이슈생성 ~ 소스코드 컨펌까지 진행
- 이슈 템플릿을 생성한 뒤 템플릿에 맞춰서 이슈 생성
- 커밋 시 git convention에 맞춰서 commit 작성

```
1. Issue Create
2. Pull Request Create
3. source code confirm & merge (self)


** git remote prune origin 명령어 unreachable한 git object들을 "local" 에서 clean 하는 작업진행
```

### 개발환경

```
OS : Window10(WSL2 : ubuntu20.04)
DB : postgresql
language : Python
WebFrameWork : FAST-API
VCS : git
```

### 업데이트 이력

- 10.28 ~ 10.29   : project initialize & recommendation System Create
- 10.29 ~ 11.01   : FAST API Base File Create
  - DataBase(postgresql) Connect
  - User Table Create
  - DataValidation(pydantic) Create
  - User PWD hashing apply
  - "create user" API router Create
  - Config File Create
  - DI(dependency injection) Create
- 11.02 ~ 11.02   : Recommend API Create
  - Recommend API Create
  - Data Validation schemas Create
  - "recommend_sys" insert into "recommend_web"
    - because recommend function import error problem

  - 11.25 ~ 11.25 : Search function modify