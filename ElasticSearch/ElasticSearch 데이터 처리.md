## CRUD 

`http://<host>:<port>/<index>/_doc/<document id>`

- ES 에서는 하나의 문서는 고유한 URL을 가짐
- Dev Tools 에서는 `http://<host>:port>/` 가 생략

<br>
<br>


### PUT
데이터를 추가할 때 사용 
- 인덱스 생성

    <br>

    ```
    PUT <index>
    PUT test
    ```

    <br>

- 데이터 삽입

    <br>

    ```
    PUT index/_doc/<doc_id>
    PUT test/_doc/1
    {
    "name" : "JH Yoo",
    "message" : "TEST PUT"
    }
    ```

    <br>

    - 이미 존재하는 문서의 id에 PUT 명령어를 쓰면 내용이 덮어씌어짐 

        <br>

        ```
        PUT test/_doc/1
        {
        "age" : 40
        }
        ```

        <br>

### GET
데이터를 조회할 때 사용
- 인덱스의 특정 id를 가진 문서 조회

    <br>

    ```
    GET <index>/_doc/<doc id>
    GET test/_doc/1
    ```

    <br>

- 인덱스의 특정 id를 가진 문서의 내용만 보고 싶은 경우

    <br>

    ```
    GET <index>/_source/<doc id>
    GET test/_source/1
    ```

    <br>

<br>

### DELETE
문서 또는 인덱스 삭제
- 특정 id를 가진 문서 삭제

    <br>

    ```
    DELETE <index>/_doc/<doc id>
    DELETE test/_doc/1
    ```

    <br>
   
- 인덱스 삭제

    <br>

    ```
    DELETE <index>
    DELETE test
    ```

    <br>

<br>

### POST
- 데이터 입력 

    <br>

    ```
    POST <index>/_doc
    
    POST test/_doc
    {
    "name" : "JH Yoo",
    "message" : "TEST POST"
    }
    ```

    <br>

    - POSE 매서드로 데이터를 입력하는 경우 문서 id를 기입하지 않음
        - 랜덤으로 id 생성

<br>

- 데이터 업데이트

    <br>

    ```
    POST <index>/_update/<doc_id>
    
    POST test/_update/7Ii3vYgBt-VcEOe7a8x3
    {
        "doc":{
            "message" : "TEST POST UPDATE"
        }
    }
    ```

    <br>

    - _doc 대신에 _update를 사용 
    - doc 지정자 안에 업데이트 할 내용 작성

<br> 
<br> 

## Bulk API
- 단일 API 호출로 여러 인덱싱 도는 삭제 작업 수행 가능
- 오버헤드 줄이고 인덱싱 속도 증가

<br>

 - bulk

    <br>

    ```
    POST /_bulk



    POST _bulk
    { "index" : { "_index" : "test", "_id" : "1" } }
    { "field1" : "value1" }
    { "delete" : { "_index" : "test", "_id" : "2" } }
    { "create" : { "_index" : "test", "_id" : "3" } }
    { "field1" : "value3" }
    { "update" : {"_id" : "1", "_index" : "test"} }
    { "doc" : {"field2" : "value2"} }
    ```

    <br>    

    - 1~2 : test 인덱스의 문서 id 1에 데이터 추가
        - field1에 "value1"
    - 3 : test 인덱스 문서 id 2 삭제
    - 4~5 : test 인덱스 문서 id 3애 데이터 추가
        - field1에 "value3"
    - 6~7 : test 인덱스 문서 id 1의 내용을 업데이트
        - field2에 "value2"

    <br>

    - 명령에 줄 바꿈이 있으면 안됨

<br>
<br>

## Search API
- URI 검색
    - 뒤에 q option 필요
    - 많이 사용되지는 않음

    <br>

    ```
    GET test/_search?q=value
    ```

    <br>

    - q 뒤에 query문을 넣어서 사용 가능

<br>

- Data Body 검색

    <br>

    ```
    GET test/_search
    {
        "query" : {
            "match" : {
            "field1" : "value"
            }
        }
    }
    ```

    <br>

    - 사용할 query의 종류
        - match
    - filed명과 값