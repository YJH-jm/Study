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

<br>
<br>

## Query DSL(Domain Specific Language)
### Full Text Query
- **match_all**
    - 인덱스의 모든 문서를 검색하는 쿼리

    <br>

    ```
    GET <index>/_search
    {
        "query":{
            "match_all":{}
        }
    }


    GET test/_search
    {
        "query":{
            "match_all":{}
        }
    }
    ```

    <br>

    - 아래와 같음

        <br>

    ```
    GET <index명>/_search
    
    GET test/_search
    ```

    <br>

- **match**
    - 일반적으로 가장 많이 사용되는 query

        ```
        GET <index명>/_search
        {
            "query": {
                "match": {
                    "<field명>": <검색어>
                }
            }
        }

        GET test/_search
        {
            "query" : {
                "match" : {
                    "message" : "dog"
                }
            }
        }
        ```
        - field에 검색어가 존재하는지 아닌지 검색
    
    <br>


    - 여러 개의 검색어를 검색

        ```
        GET test/_search
        {
            "query" : {
                "match" : {
                    "message" : "quick dog"
                }
            }
        }
        ```
        - 여러 개의 검색어가 들어가면 OR로 검색
            - 여러 개의 검색어 중 하나라도 포함된 것이 있으면 검색

    <br>

    - 여러 검색어를 모두 포함하여 검색
        - operator 옵션 사용 

            ```
            GET test/_search
            {
                "query": {
                    "match": {
                        "message": {
                            "query": "quick dog",
                            "operator": "and"
                        }
                    }
                }
            }
            ```

<br>

- **match_phrase**
    - 압력된 검색어 순서까지 고려하여 검색 
    - 구문을 공백을 포함한 정확한 내용 검색

        ```
        GET test/_search
        {
            "query" : {
                "match_phrase" : {
                    "message" : "lazy dog"
                }
            }
        }
        ```

        <br>

    - slop 옵션을 이용하면 지정된 값 만큼 단어 사이에 다른 검색어가 끼어드는 것을 허용

        ```
        GET test/_search
        {
            "query": {
                "match_phrase": {
                    "message": {
                        "query": "lazy dog",
                        "slop": 1
                    }
                }   
            }
        }
        ```
        - slop이 1이기 때문에 두 단어 사이에 하나의 단어가 끼어드는 것 가능 

<br>

- **query_string**
    - 루씬 쿼리를 사용 
    ```
    GET test/_search
    {
        "query": {
            "query_string": {
            "default_field": "FIELD",
            "query": "this AND that OR thus"
            }
        }
    }

    GET test/_search
    {
        "query": {
            "query_string": {
                "default_field": "message",
                "query": "(jumping AND lazy) OR \"quick dog\""
            }
        }
    }
    ```

<br>


<br>

### Relevancy
- ES는 풀 텍스트 검색엔진이기 때문에 검색 조건과 결과가 얼마나 일치하는지를 계산 할 수 있음
    - 이 일치하는 정도를 relevancy 라 함

<br>



- **Score**
    - 검색된 결과가 얼마나 검색 조건과 일치하는지 점수로 나타냄
    - _score 항목으로 확인 가능 
    - 높은 점수가 더 좋은 검색 결과
    - 이 점수를 계산하기  위해 **BM25** 라는 식을 계산
        - BM25를 계산하기 위해서 TF, IDF, Field Length 필요
    
    <br>

- **TF (Term Frequency)**
    - 검색할 때 이용하는 단어를 term 이라고 함
    - 문서 내에 검색된 term이 많을 수록 점수가 높아짐
        - 내가 원하는 정보가 있을 확률이 높기 때문

    <br>

- **IDF (Inverse Document Frequency)**
    - 여러 단어를 검색할 때, 총 문서에서 단어마다 나오는 빈도가 다름
    - A 라는 단어가 포함된 문서는 5개, B라는 단어가 포함된 문서가 100개라면 A 라는 단어 더 희소한 단어이고 중요한 힌트일 확률이 높음
    - 검색한 텀을 포함한 문서의 개수가 많을 수록 점수가 감소 

    <br>

- **Field length**
    - 문서에서 길이가 긴 필드보다는 길이가 짧은 필드의 점수를 높임 



<br>
<br>


### Bool 복합 Query
- 여러개의 query를 같이 쓰기 위해 Bool query 제공
- Bool query 안에 여러개의 query 같이 넣음
- ES에서는 4가지의 option 선택
    - **must**
        
    - **must_not**
        - Query가 거짓인 문서 거색
    - **should**
        - 검색 결과 중 이 query에 해당하는 문서의 점수 높임
    - **filter**
        - Query가 참이니 문서를 검색하지만 스코어 계산하지 않음
            - 속도 증가


    <br>
    
    ```
    GET <index>/_search
    {
        "query": {
            "bool": {
                "must": [
                    {<query>}, 
                ],
                "must_not": [
                    {<query>}, 
                ],
                "should": [
                    {<query>}, 
                ],
                "filter": [
                    {<query>}, 
                ]
            }
        }
    }
    ```

<br>

- **must**
    - Query가 참인 문서 검색
    - 배열 안이 and로 연산됨
        
        ```
        GET test/_search
        {
        "query": {
            "bool": {
            "must": [
                {
                "match": {
                    "message": "quick"
                }
                },
                {
                "match_phrase": {
                    "message": "lazy dog"
                }
                }
            ]
            }
        }
        }
        ```
        - "quick" 이라는 단어도 있고 "lazy dog" 이라는 구문도 포함하는 경우

<br>

- **must_not**
    - Query가 거짓인 문서 검색

        <br>

        ```
        GET test/_search
        {
            "query": {
                "bool": {
                    "must_not": [
                        {
                            "match": {
                                "message": "quick"
                            }
                        },
                        {
                            "match_phrase": {
                                "message": "lazy dog"
                            }
                        }
                    ]
                }
            }
        }    
        ```
        - "quick"을 포함하지 않고, "lazy dog"도 포함하지 않는 문서 검색 

        <br>
        <br>



        ```
        GET test/_search
        {
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "message": "quick"
                            }
                        }
                    ],
                    "must_not": [
                        {
                            "match_phrase": {
                                "message": "lazy dog"
                            }
                        }
                    ]
                }
            }
        }
        ```

        - "quick"이 포함되어 있지만 "lazy dog" 은 포함이 되지 않는 경우

<br>

- **should**
    - 검색 점수를 조정하기 위해 사용
        - 검색 결과 중 이 query에 해당하는 문서의 점수 높임

        <br>

        ```
        GET test/_search
        {
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "message": "fox"
                            }
                        }
                    ],
                    "should": [
                        {
                            "match_phrase": {
                                "message": "lazy"
                            }
                        }
                    ]
                }
            }
        }
        ```
        - "fox"를 포함한 문서중에 "lazy" 라는 단어가 포함되면 가중치를 줌

        <br>

<br>
<br>

- **filter**
    - Query가 참이먄 문서를 검색하지만 스코어 계산하지 않음
        - 속도 증가
    - 즉, Query 결과에는 영향을 미치나 score에는 영향을 주지 않음
    

        <br>

        ```
        GET test/_search
        {
        "query": {
            "bool" : {
            "must": [
                {
                "match": {
                    "message": "fox"
                } 
                }
            ],
            "filter": [
                {
                "match" : {
                    "message" : "quick"
                }
                }
            ]
            }
        }
        }
        ```
        - "fox"만 검색한 결과와 _score는 같게 나오지만 결과는 "quick"도 포함한 결과가 나옴

<br>

### Range Query
- 숫자, 날짜 등의 범위를 검색할 때 사용

    <br>

    ```
    GET <index>/_search
    {
        "query": {
                "range": {
                "<field>": {
                    "<parameter>": <value>,
                    "<parameter>": <value>
                }
            }
        }
    }
    ```
    - parameter
        - **gte**
            - 이상
        - **gt**
            - 초과
        - **lte**
            - 이하
        - **lt**
            - 미만
    
    <br>
    <br>
    
    - 숫자 검색
        ```
        GET phones/_search
        {
            "query": {
                "range": {
                    "price": {
                        "gte":700,
                        "lt": 900
                    }
                }
            }
        }
        ```
        - price가 700 이상이고 900 미만인 문서 검색

        <br>

    - 날짜 검색
        - 날짜 형식은 ISO8601 형식 따름
            - 2016-01-01 또는 2016-01-01T10:15:30 
        - 날짜 형식을 다르게 하고 싶으면 format 옵션 값 이용 

        
        <br>
        <br>

        
        ```
        GET phones/_search
        {
            "query": {
                "range": {
                    "date": {
                        "gt": "2016-01-01"
                    }
                }
            }
        }
        ```
        - 2016년 1월 1일 이후 날짜 데이터의 문서 검색

        <br>

        ```
        GET phones/_search
        {
            "query": {
                "range": {
                    "date": {
                        "gte":"31/12/2015",
                        "lte": "2018",
                        "format": "dd/MM/yyyy||yyyy"
                    }
                }
            }
        }
        ```
        - y(년), M(월), d(일), h(시), m(분), s(초), w(주) 
        - || 이용하면 여러 값  입력 가능
        - 2015년 12월 31일 이후부터 2018년도 이전의 문서 검색

        <br>

        ```
        GET phones/_search
        {
            "query": {
                "range": {
                    "date": {
                        "gt": "2016-01-01||+6M",
                        "lt": "now-365d"
                    }
                }
            }
        }
        ```
        - now를 이용하여 현재시간을 가져옴
        - 2016 년 1월 1일에 6개월을 더한 2016년 6월 1일 후 부터 오늘의 365일 이전 값들을 가지는 문서 검색 